"""Vertex AI & Gemini 3.8 Flash TTS Client using the official `google-genai` SDK.

Supports:
  1. Google Cloud Vertex AI (`use_vertex_ai=True`) with Service Account JSON key file
     (`SERVICE_ACCOUNT_KEY_PATH` / `GOOGLE_APPLICATION_CREDENTIALS`) or Application Default
     Credentials (`gcloud auth application-default login`).
  2. Gemini API Key (`use_vertex_ai=False` or `GEMINI_API_KEY`).

Includes SHA-256 disk caching (`cache/tts_cache/`), multi-threaded batch synthesis,
exponential backoff retry logic, and automatic PCM/WAV formatting.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import os
from pathlib import Path
import time
from typing import Any
import wave

import google.auth
from google.auth.transport.requests import Request
from google import genai
from google.genai import types
from google.oauth2 import service_account

from config import PipelineConfig


def create_genai_client(cfg: PipelineConfig) -> tuple[genai.Client, str]:
    """Initialize a `google.genai.Client` using Vertex AI or Gemini API key."""
    http_opts = types.HttpOptions(timeout=cfg.tts_http_timeout_ms)

    # 1. Explicit Gemini API Key Mode when `use_vertex_ai=False` (or when API key is provided and no GCP project is configured)
    api_key = (cfg.gemini_api_key or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")).strip()
    if not cfg.use_vertex_ai and api_key:
        return genai.Client(api_key=api_key, http_options=http_opts), "Gemini API Key (`google-genai`)"

    # 2. Vertex AI Mode (Service Account Key JSON -> Application Default Credentials)
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    credentials = None
    project_id = (cfg.project_id or os.getenv("GOOGLE_CLOUD_PROJECT", "")).strip()
    if project_id == "YOUR_GCP_PROJECT_ID":
        project_id = ""
    location = (cfg.location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")).strip()
    auth_desc = "Vertex AI ADC"

    sa_path = (
        cfg.service_account_key_path
        or os.getenv("SERVICE_ACCOUNT_KEY_PATH", "")
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    ).strip()

    if sa_path and Path(sa_path).expanduser().exists():
        resolved_sa = Path(sa_path).expanduser().resolve()
        credentials = service_account.Credentials.from_service_account_file(
            str(resolved_sa), scopes=scopes
        )
        if not project_id:
            try:
                key_data = json.loads(resolved_sa.read_text(encoding="utf-8"))
                project_id = key_data.get("project_id", "")
            except Exception:
                project_id = getattr(credentials, "project_id", "") or ""
        auth_desc = f"Vertex AI Service Account ({resolved_sa.name}, project={project_id})"
        return (
            genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
                credentials=credentials,
                http_options=http_opts,
            ),
            auth_desc,
        )

    # Try standard Google Cloud Application Default Credentials (ADC)
    try:
        adc_creds, adc_proj = google.auth.default(scopes=scopes)
        adc_creds.refresh(Request())
        final_proj = project_id or adc_proj or ""
        if final_proj:
            return (
                genai.Client(
                    vertexai=True,
                    project=final_proj,
                    location=location,
                    credentials=adc_creds,
                    http_options=http_opts,
                ),
                f"Vertex AI Application Default Credentials (project={final_proj}, location={location})",
            )
    except Exception as adc_err:
        # If ADC fails and a Gemini API key is available, seamlessly fall back to API key
        if api_key:
            return (
                genai.Client(api_key=api_key, http_options=http_opts),
                "Gemini API Key Fallback (`google-genai`)",
            )
        raise RuntimeError(
            f"Vertex AI authentication failed ({adc_err}). "
            "Please configure `service_account_key_path` / `GOOGLE_APPLICATION_CREDENTIALS`, "
            "run `gcloud auth application-default login`, or set `GEMINI_API_KEY`."
        ) from adc_err

    if api_key:
        return genai.Client(api_key=api_key, http_options=http_opts), "Gemini API Key (`google-genai`)"

    raise RuntimeError(
        "No valid Vertex AI project/credentials or GEMINI_API_KEY found. "
        "Update `configs/pipeline_config.yaml` or set environment variables."
    )


def tts_cache_path(cache_dir: Path, model_id: str, voice: str, text: str) -> Path:
    """Deterministic SHA-256 cache filename for a (model, voice, text) tuple."""
    norm_model = model_id.replace("models/", "")
    digest = hashlib.sha256(f"{norm_model}|{voice}|{text.strip()}".encode("utf-8")).hexdigest()[:20]
    return cache_dir / f"{voice}_{digest}.wav"


def _ensure_valid_wav_bytes(raw_bytes: bytes, mime_type: str = "", default_sr: int = 24000) -> bytes:
    """Ensure audio payload returned by Gemini TTS is wrapped in a standard RIFF WAVE header."""
    if len(raw_bytes) >= 12 and raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WAVE":
        return raw_bytes

    # Parse sample rate from mime_type if present (e.g. "audio/L16;codec=pcm;rate=24000")
    sr = default_sr
    if "rate=" in (mime_type or ""):
        for part in mime_type.split(";"):
            part = part.strip()
            if part.startswith("rate="):
                try:
                    sr = int(part.split("=", 1)[1])
                except ValueError:
                    pass

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sr)
        wf.writeframes(raw_bytes)
    return buf.getvalue()


def synthesize_utterance(
    client: genai.Client,
    cfg: PipelineConfig,
    voice: str,
    text: str,
    cache_dir: Path,
    prosody_style: str = "normal",
) -> Path:
    """Synthesize a single spoken utterance via Gemini 3.8 Flash TTS (`gemini-3.8-flash-tts`)
    and save as a 24 kHz 16-bit mono WAV file in `cache_dir`."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = tts_cache_path(cache_dir, cfg.tts_model_id, voice, text)
    if out_path.exists() and out_path.stat().st_size > 1024:
        return out_path

    # Optionally prepend TTS director / style instructions if enabled in config
    prompt_text = text.strip()
    if cfg.enable_tts_style_prompting:
        style_hint = cfg.tts_prosody_style_instructions.get(prosody_style, "")
        director = (cfg.tts_director_system_prompt or "").strip()
        prefix = " ".join(p for p in (director, style_hint) if p).strip()
        if prefix:
            prompt_text = f"[{prefix}]\n{prompt_text}"

    models_to_try = [cfg.tts_model_id] + [
        m for m in cfg.fallback_tts_models if m != cfg.tts_model_id
    ]
    last_error: Exception | None = None

    for model_id in models_to_try:
        for attempt in range(cfg.tts_max_retries):
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=prompt_text,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=voice
                                )
                            )
                        ),
                    ),
                )
                audio_part = None
                for cand in response.candidates or []:
                    if cand.content and cand.content.parts:
                        for part in cand.content.parts:
                            if part.inline_data and part.inline_data.data:
                                audio_part = part.inline_data
                                break
                    if audio_part:
                        break

                if audio_part is None:
                    raise RuntimeError(f"Model {model_id} returned no inline audio bytes.")

                wav_bytes = _ensure_valid_wav_bytes(
                    audio_part.data, getattr(audio_part, "mime_type", "")
                )
                tmp_path = out_path.with_suffix(f".tmp.{os.getpid()}.{time.time_ns()}.wav")
                tmp_path.write_bytes(wav_bytes)
                tmp_path.replace(out_path)
                return out_path

            except Exception as err:
                last_error = err
                err_str = str(err)
                # If model name is not found in region/endpoint, move to fallback model
                if "404" in err_str or "NOT_FOUND" in err_str:
                    break
                delay = min(30.0, cfg.tts_retry_base_delay_s * (1.8 ** attempt))
                time.sleep(delay)

    raise RuntimeError(
        f"TTS synthesis failed for voice={voice!r}, text={text[:45]!r}: {last_error}"
    )


def batch_synthesize_utterances(
    scenarios: list[Any],
    cfg: PipelineConfig,
) -> tuple[Path, dict[str, Any]]:
    """Pre-synthesize all unique `(voice, text)` pairs required by `scenarios` in parallel."""
    cache_dir = Path(cfg.tts_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    jobs: set[tuple[str, str]] = set()
    for sc in scenarios:
        for t in sc.turns:
            jobs.add((sc.primary_voice, t.text))
            if t.overlap_bgv_text:
                jobs.add((t.overlap_bgv_voice, t.overlap_bgv_text))
            if t.pause_bgv_text:
                jobs.add((t.pause_bgv_voice, t.pause_bgv_text))
            if t.pause_bgv2_text:
                jobs.add((t.pause_bgv2_voice, t.pause_bgv2_text))

    pending = [
        (v, txt)
        for v, txt in sorted(jobs)
        if not (
            tts_cache_path(cache_dir, cfg.tts_model_id, v, txt).exists()
            and tts_cache_path(cache_dir, cfg.tts_model_id, v, txt).stat().st_size > 1024
        )
    ]

    cached_count = len(jobs) - len(pending)
    print(
        f"[TTS] Unique utterances: {len(jobs)} "
        f"({cached_count} cached, {len(pending)} to synthesize via {cfg.tts_model_id})"
    )

    auth_mode = "Disk Cache (0 API calls needed)"
    if pending:
        client, auth_mode = create_genai_client(cfg)
        print(f"[TTS] Auth Mode: {auth_mode}")
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, cfg.tts_max_workers)) as pool:
            futs = {
                pool.submit(synthesize_utterance, client, cfg, v, txt, cache_dir): (v, txt)
                for v, txt in pending
            }
            for fut in as_completed(futs):
                fut.result()
                done += 1
                if done % 5 == 0 or done == len(pending):
                    print(f"  [TTS] Synthesized {done}/{len(pending)} utterances...")

    return cache_dir, {
        "tts_model_id": cfg.tts_model_id,
        "auth_mode": auth_mode,
        "total_unique_utterances": len(jobs),
        "newly_synthesized": len(pending),
        "reused_from_cache": cached_count,
    }
