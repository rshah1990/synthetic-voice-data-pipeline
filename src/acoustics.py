"""Signal processing, Room Impulse Response (RIR) acoustics, speech island extraction,
telephony downsampling, and ground-truth gap annotation.

Completely standalone: depends only on `numpy`, `scipy`, and `soundfile`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, correlate, filtfilt, resample_poly
import soundfile as sf

from config import PipelineConfig


@dataclass
class NoiseClip:
    category: str
    tag: str
    source_id: str
    audio: np.ndarray  # 16 kHz float32, RMS-normalized to 1.0


# ==============================================================================
# 1. RESAMPLING & TELEPHONY CONVERSION (ZERO-PHASE POLYPHASE FIR)
# ==============================================================================

def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x.mean(axis=1) if x.ndim == 2 else x


def resample_audio(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Delay-compensated zero-phase polyphase resampling (`scipy.signal.resample_poly`)."""
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    g = gcd(int(sr_in), int(sr_out))
    return resample_poly(x, int(sr_out) // g, int(sr_in) // g).astype(np.float32)


def float_to_int16(x: np.ndarray) -> tuple[np.ndarray, int]:
    """Convert [-1.0, 1.0] float32 waveform to int16 PCM and count clipped samples."""
    y = np.round(x * 32767.0)
    n_clipped = int(np.count_nonzero(np.abs(y) > 32767))
    return np.clip(y, -32768, 32767).astype(np.int16), n_clipped


def to_telephony_pcm(x: np.ndarray, sr_in: int, telephony_sr: int = 8000) -> tuple[np.ndarray, int]:
    """Downsample native-rate waveform to 8 kHz mono int16 PCM."""
    return float_to_int16(resample_audio(to_mono(x), sr_in, telephony_sr))


def measure_lag_ms(ref: np.ndarray, sr_ref: int, test: np.ndarray, sr_test: int) -> float:
    """Measure lag of `test` relative to `ref` in ms via FFT cross-correlation."""
    r = resample_audio(to_mono(ref), sr_ref, sr_test)
    t = np.asarray(test)
    t = to_mono(t / 32768.0 if t.dtype == np.int16 else t)
    n = min(len(r), len(t))
    r, t = r[:n] - r[:n].mean(), t[:n] - t[:n].mean()
    c = correlate(t, r, mode="full", method="fft")
    lag = int(np.argmax(c)) - (n - 1)
    return 1000.0 * lag / sr_test


# ==============================================================================
# 2. CLEAN TTS SPEECH ISLAND EXTRACTION & PROSODY VARIATIONS
# ==============================================================================

def expand_internal_hesitation_pauses(
    x: np.ndarray,
    sr: int,
    add_pause_s: float = 0.380,
) -> np.ndarray:
    """For `hesitation_stutter` / `slow_hesitant` turns, locate up to 2 natural low-energy
    valleys inside the utterance (between 22%–48% and 54%–80% of duration) and insert
    `add_pause_s` (380 ms) of clean silence with 10 ms cosine tapers so the speaker
    has genuine >350 ms mid-thought hesitation pauses."""
    hop = int(0.010 * sr)
    n_frames = len(x) // hop
    if n_frames < 80:
        return x
    frames = x[: n_frames * hop].reshape(n_frames, hop)
    frms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    f_a0, f_a1 = int(n_frames * 0.22), int(n_frames * 0.48)
    f_b0, f_b1 = int(n_frames * 0.54), int(n_frames * 0.80)
    idx1 = f_a0 + int(np.argmin(frms[f_a0:f_a1]))
    idx2 = f_b0 + int(np.argmin(frms[f_b0:f_b1]))
    s1 = idx1 * hop
    s2 = idx2 * hop
    pad = np.zeros(int(round(add_pause_s * sr)), dtype=np.float32)
    p1 = x[:s1].copy()
    p2 = x[s1:s2].copy()
    p3 = x[s2:].copy()
    fn = min(int(0.010 * sr), len(p1) // 4, len(p2) // 4, len(p3) // 4)
    if fn > 1:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fn, dtype=np.float32)))
        p1[-fn:] *= ramp[::-1]
        p2[:fn] *= ramp
        p2[-fn:] *= ramp[::-1]
        p3[:fn] *= ramp
    return np.concatenate([p1, pad, p2, pad, p3]).astype(np.float32)


def load_and_trim_clean_tts(
    wav_path: Path,
    target_sr: int = 16000,
    merge_gap_s: float = 0.340,
    min_island_s: float = 0.120,
    gt_merge_gap_s: float = 0.200,
    speed_rate: float = 1.0,
    inject_hesitation_pauses: bool = False,
    hesitation_pause_s: float = 0.380,
) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]:
    """Load 24 kHz TTS WAV, resample to `target_sr` (16 kHz), apply speaking-rate speed
    multiplier and optional mid-utterance hesitation pauses, trim outer digital silence,
    normalize active speech RMS to 1.0, and return:
      `(trimmed_normalized_audio, relative_voice_segments_s, relative_speech_spans_s)`.
    """
    x_raw, sr_in = sf.read(str(wav_path), dtype="float32")
    x = resample_audio(to_mono(x_raw), sr_in, target_sr)
    if abs(speed_rate - 1.0) > 1e-3:
        x = resample_audio(x, int(round(target_sr * speed_rate)), target_sr)
    if inject_hesitation_pauses:
        x = expand_internal_hesitation_pauses(x, target_sr, add_pause_s=hesitation_pause_s)

    hop = int(0.010 * target_sr)
    n_frames = len(x) // hop
    if n_frames < 5:
        rms = float(np.sqrt(np.mean(x**2) + 1e-12))
        full = [(0.0, round(len(x) / target_sr, 3))]
        return (x / rms).astype(np.float32), full, full

    frames = x[: n_frames * hop].reshape(n_frames, hop)
    frms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    peak_rms = float(np.max(frms))
    db_floor = -20.0 if min_island_s < 0.100 else -36.0
    pad_frames = 0 if min_island_s < 0.100 else 2
    thresh = max(peak_rms * (10.0 ** (db_floor / 20.0)), 1e-4)
    active = frms >= thresh

    raw_spans: list[list[float]] = []
    in_run = False
    s0 = 0
    for i, a in enumerate(active):
        if a and not in_run:
            s0 = i
            in_run = True
        elif not a and in_run:
            raw_spans.append([
                max(0.0, (s0 - pad_frames) * 0.010),
                min(len(x) / target_sr, (i + pad_frames) * 0.010),
            ])
            in_run = False
    if in_run:
        raw_spans.append([max(0.0, (s0 - pad_frames) * 0.010), len(x) / target_sr])

    if not raw_spans:
        rms = float(np.sqrt(np.mean(x**2) + 1e-12))
        full = [(0.0, round(len(x) / target_sr, 3))]
        return (x / rms).astype(np.float32), full, full

    merged: list[list[float]] = [raw_spans[0]]
    for s, e in raw_spans[1:]:
        if s - merged[-1][1] < merge_gap_s:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    merged = [se for se in merged if (se[1] - se[0]) >= min_island_s]
    if not merged:
        merged = [[raw_spans[0][0], raw_spans[-1][1]]]

    trim_start = max(0.0, merged[0][0] - 0.020)
    trim_end = min(len(x) / target_sr, merged[-1][1] + 0.020)
    i0 = int(round(trim_start * target_sr))
    i1 = int(round(trim_end * target_sr))
    x_trim = x[i0:i1].copy()

    fade_n = min(int(0.015 * target_sr), len(x_trim) // 4)
    if fade_n > 1:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_n, dtype=np.float32)))
        x_trim[:fade_n] *= ramp
        x_trim[-fade_n:] *= ramp[::-1]

    rel_spans: list[tuple[float, float]] = []
    active_samples: list[np.ndarray] = []
    max_dur = len(x_trim) / target_sr
    for s, e in merged:
        rs = round(max(0.0, s - trim_start), 3)
        re_ = round(min(max_dur, e - trim_start), 3)
        if re_ - rs >= min_island_s:
            rel_spans.append((rs, re_))
            active_samples.append(x_trim[int(rs * target_sr) : int(re_ * target_sr)])

    cat_active = np.concatenate(active_samples) if active_samples else x_trim
    active_rms_val = float(np.sqrt(np.mean(cat_active**2) + 1e-12))
    x_norm = (x_trim / active_rms_val).astype(np.float32)

    def _rel(s: float, e: float) -> tuple[float, float]:
        return (round(max(0.0, s - trim_start), 3), round(min(max_dur, e - trim_start), 3))

    speech_rel = [_rel(s, e) for s, e in raw_spans if e > trim_start and s < trim_end]
    speech_rel = [(a, b) for a, b in speech_rel if b > a]
    gt_merged: list[list[float]] = []
    for a, b in speech_rel:
        if gt_merged and a - gt_merged[-1][1] < gt_merge_gap_s:
            gt_merged[-1][1] = max(gt_merged[-1][1], b)
        else:
            gt_merged.append([a, b])
    gt_voice = [(a, b) for a, b in gt_merged if (b - a) >= min_island_s]
    if not gt_voice:
        gt_voice = list(rel_spans)
    return x_norm, gt_voice, speech_rel


def annotate_gaps(segments: list[dict[str, Any]], speech_spans: list[list[float]]) -> None:
    """Add `gap_from_prev_voice_s` and `gap_to_next_voice_s` to each ground-truth segment in place."""
    spans = sorted((float(a), float(b)) for a, b in speech_spans if float(b) > float(a))
    for seg in segments:
        s0, s1 = float(seg["start"]), float(seg["end"])
        prev_ends = [b for a, b in spans if b <= s0 + 1e-6 and a < s0 - 1e-6]
        next_starts = [a for a, b in spans if a >= s1 - 1e-6 and b > s1 + 1e-6]
        seg["gap_from_prev_voice_s"] = round(s0 - max(prev_ends), 3) if prev_ends else None
        seg["gap_to_next_voice_s"] = round(min(next_starts) - s1, 3) if next_starts else None


# ==============================================================================
# 3. ROOM IMPULSE RESPONSE (RIR) & BACKGROUND SPEAKER ACOUSTICS
# ==============================================================================

def far_field_rir(
    sr: int,
    rt60: float,
    rng: np.random.Generator,
    direct_gain: float = 0.65,
) -> np.ndarray:
    """Synthetic room impulse response with early reflections and exponential late reverberation."""
    n = int(sr * max(0.15, rt60) * 1.25)
    t = np.arange(n) / sr
    decay = np.exp(-6.9 * t / max(0.12, rt60))
    late = rng.standard_normal(n) * decay
    h = 0.55 * late
    h[0] += direct_gain
    for delay_ms, gain in ((12, 0.35), (23, 0.25), (37, 0.18)):
        idx = int(delay_ms * 0.001 * sr)
        if idx < len(h):
            h[idx] += gain
    pre = int(0.008 * sr)
    h = np.concatenate([np.zeros(pre), h])
    return (h / np.sqrt(np.sum(h**2) + 1e-12)).astype(np.float32)


def apply_far_field_acoustics(
    speech_norm: np.ndarray,
    sr: int,
    rt60: float,
    rng: np.random.Generator,
    direct_gain: float = 0.65,
    lowpass_hz: float = 2600.0,
) -> np.ndarray:
    """Convolve normalized background speech with room RIR + off-axis air absorption low-pass filter."""
    rir = far_field_rir(sr, rt60, rng, direct_gain=direct_gain)
    wet = np.convolve(speech_norm, rir)[: len(speech_norm) + int(0.18 * sr)].astype(np.float32)
    b, a = butter(2, lowpass_hz / (0.5 * sr), btype="low")
    wet = filtfilt(b, a, wet).astype(np.float32)
    rms = float(np.sqrt(np.mean(wet**2) + 1e-12))
    return (wet / rms).astype(np.float32)


def apply_bgv_acoustics(
    speech_norm: np.ndarray,
    sr: int,
    rt60: float,
    rng: np.random.Generator,
    cfg: PipelineConfig,
    mode: str = "far_field",
    speech2_norm: np.ndarray | None = None,
) -> np.ndarray:
    """Apply scenario-specific background speaker acoustics using parameters from `cfg.acoustic_modes`."""
    mcfg = cfg.acoustic_modes.get(mode, cfg.acoustic_modes.get("far_field", {}))

    if mode in ("near_field_loud", "bystander_backchannel", "bystander_laugh_cough"):
        rt = float(mcfg.get("rt60_override_s", 0.16 if mode != "bystander_laugh_cough" else 0.20))
        dg = float(mcfg.get("direct_gain", 1.35 if mode != "bystander_laugh_cough" else 1.25))
        lp = float(mcfg.get("lowpass_hz", 3500.0 if mode != "bystander_laugh_cough" else 3600.0))
        rir = far_field_rir(sr, rt, rng, direct_gain=dg)
        wet = np.convolve(speech_norm, rir)[: len(speech_norm) + int(0.09 * sr)].astype(np.float32)
        b, a = butter(2, lp / (0.5 * sr), btype="low")
        wet = filtfilt(b, a, wet).astype(np.float32)
        rms = float(np.sqrt(np.mean(wet**2) + 1e-12))
        return (wet / rms).astype(np.float32)

    if mode in ("bot_echo_leak", "tv_radio_playback"):
        bp = mcfg.get("bandpass_hz", [400.0, 3200.0] if mode == "bot_echo_leak" else [320.0, 3300.0])
        drive = float(mcfg.get("tanh_drive", 1.85 if mode == "bot_echo_leak" else 1.60))
        dg = float(mcfg.get("direct_gain", 0.95 if mode == "bot_echo_leak" else 0.80))
        b, a = butter(3 if mode == "bot_echo_leak" else 2, [bp[0] / (0.5 * sr), bp[1] / (0.5 * sr)], btype="band")
        spkr = filtfilt(b, a, speech_norm).astype(np.float32)
        spkr = np.tanh(drive * spkr).astype(np.float32)
        eff_rt60 = max(0.22, rt60 * 0.75) if mode == "bot_echo_leak" else rt60
        rir = far_field_rir(sr, eff_rt60, rng, direct_gain=dg)
        wet = np.convolve(spkr, rir)[: len(spkr) + int(0.14 * sr)].astype(np.float32)
        rms = float(np.sqrt(np.mean(wet**2) + 1e-12))
        return (wet / rms).astype(np.float32)

    if mode == "cocktail_dual_talker" and speech2_norm is not None:
        stagger_s = float(mcfg.get("stagger_s", 0.25))
        w2 = float(mcfg.get("secondary_speaker_weight", 0.92))
        wet1 = apply_far_field_acoustics(speech_norm, sr, rt60, rng)
        wet2 = apply_far_field_acoustics(speech2_norm, sr, max(0.25, rt60 * 0.80), rng)
        stagger = int(stagger_s * sr)
        tot_len = max(len(wet1), stagger + len(wet2))
        mix = np.zeros(tot_len, dtype=np.float32)
        mix[: len(wet1)] += wet1
        mix[stagger : stagger + len(wet2)] += w2 * wet2
        rms = float(np.sqrt(np.mean(mix**2) + 1e-12))
        return (mix / rms).astype(np.float32)

    dg = float(mcfg.get("direct_gain", 0.65))
    lp = float(mcfg.get("lowpass_hz", 2600.0))
    return apply_far_field_acoustics(speech_norm, sr, rt60, rng, direct_gain=dg, lowpass_hz=lp)


# ==============================================================================
# 4. NOISE BANK LOADER & COMPOUND MULTI-NOISE COMPOSER
# ==============================================================================

def synth_wind_mic_rub(
    n_samples: int,
    sr: int,
    rng: np.random.Generator,
    bandpass_hz: list[float] | None = None,
    tanh_drive: float = 2.20,
) -> np.ndarray:
    """Synthesize low-frequency outdoor wind buffeting (22–210 Hz) + microphone handling gusts."""
    bp = bandpass_hz or [22.0, 210.0]
    white = rng.standard_normal(n_samples).astype(np.float32)
    b, a = butter(3, [bp[0] / (0.5 * sr), bp[1] / (0.5 * sr)], btype="band")
    low = filtfilt(b, a, white).astype(np.float32)
    t = np.arange(n_samples, dtype=np.float32) / sr
    env = (
        0.45
        + 0.55 * np.sin(2.0 * np.pi * 1.7 * t) ** 2
        + 0.35 * np.sin(2.0 * np.pi * 0.65 * t + 0.8) ** 2
    )
    gust = np.tanh(tanh_drive * low * env).astype(np.float32)
    rms = float(np.sqrt(np.mean(gust**2) + 1e-12))
    return (gust / rms).astype(np.float32)


def load_noise_bank(noise_stems_dir: str | Path, target_sr: int = 16000) -> dict[str, list[NoiseClip]]:
    """Load speech-free noise stems from `assets/noise_stems/` (or any user-configured folder)."""
    base = Path(noise_stems_dir)
    bank: dict[str, list[NoiseClip]] = {}

    manifest_files: set[str] = set()
    manifest_path = base / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for cat, entries in manifest.get("categories", {}).items():
            for item in entries:
                wav_p = base / item["file"]
                if not wav_p.exists():
                    continue
                manifest_files.add(str(wav_p.resolve()))
                x, sr = sf.read(str(wav_p), dtype="float32")
                x = resample_audio(to_mono(x), sr, target_sr)
                rms = float(np.sqrt(np.mean(x**2) + 1e-12))
                if rms < 1e-5:
                    continue
                bank.setdefault(cat, []).append(
                    NoiseClip(
                        category=cat,
                        tag=item.get("tag", f"<{cat}>"),
                        source_id=item.get("source_id", wav_p.stem),
                        audio=(x / rms).astype(np.float32),
                    )
                )

    # Also scan any custom subfolder `.wav` files added by the customer outside manifest.json
    if base.exists():
        for cat_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            cat = cat_dir.name
            for wav_p in sorted(cat_dir.glob("*.wav")):
                if str(wav_p.resolve()) in manifest_files:
                    continue
                x, sr = sf.read(str(wav_p), dtype="float32")
                x = resample_audio(to_mono(x), sr, target_sr)
                rms = float(np.sqrt(np.mean(x**2) + 1e-12))
                if rms < 1e-5:
                    continue
                bank.setdefault(cat, []).append(
                    NoiseClip(
                        category=cat,
                        tag=f"<{wav_p.stem}>",
                        source_id=wav_p.stem,
                        audio=(x / rms).astype(np.float32),
                    )
                )

    return bank


def compose_compound_noise_burst(
    bank: dict[str, list[NoiseClip]],
    categories: list[str],
    target_dur_s: float,
    sr: int,
    rng: np.random.Generator,
    cfg: PipelineConfig,
) -> tuple[np.ndarray, str, str, list[str]]:
    """Combine multiple noise stems simultaneously into a multi-noise burst of length `target_dur_s`."""
    n_samples = max(int(0.25 * sr), int(round(target_dur_s * sr)))
    canvas = np.zeros(n_samples, dtype=np.float32)
    used_tags: list[str] = []
    primary_cat = categories[0] if categories else "vehicle_traffic"

    for i, cat in enumerate(categories):
        if cat == "wind_mic_rub":
            gust = synth_wind_mic_rub(
                n_samples,
                sr,
                rng,
                bandpass_hz=cfg.wind_bandpass_hz,
                tanh_drive=cfg.wind_tanh_drive,
            )
            canvas += 1.10 * gust
            tag_label = "wind_mic_rub:<wind_buffeting>"
            if tag_label not in used_tags:
                used_tags.append(tag_label)
            continue

        pool = bank.get(cat) or (bank[next(iter(bank))] if bank else [])
        if not pool:
            gust = synth_wind_mic_rub(n_samples, sr, rng)
            canvas += gust
            continue

        cursor = int(rng.uniform(0.0, min(0.25, target_dur_s * 0.2)) * sr) if i > 0 else 0
        while cursor < n_samples - int(0.15 * sr):
            clip = pool[int(rng.integers(0, len(pool)))]
            snip = clip.audio
            weight = float(rng.uniform(0.65, 1.15))
            if len(snip) >= n_samples - cursor:
                seg = snip[: n_samples - cursor].copy()
            else:
                seg = snip.copy()
            fn = min(int(0.020 * sr), len(seg) // 4)
            if fn > 1:
                ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fn, dtype=np.float32)))
                seg[:fn] *= ramp
                seg[-fn:] *= ramp[::-1]
            canvas[cursor : cursor + len(seg)] += weight * seg
            tag_label = f"{cat}:{clip.tag}"
            if tag_label not in used_tags:
                used_tags.append(tag_label)
            cursor += len(seg) + int(rng.uniform(0.05, 0.25) * sr)

    rms = float(np.sqrt(np.mean(canvas**2) + 1e-12))
    canvas = (canvas / rms).astype(np.float32)
    short_tags = []
    for ut in used_tags:
        t_only = ut.split(":", 1)[1]
        if t_only not in short_tags:
            short_tags.append(t_only)
    combined_tag = "+".join(short_tags[:3]) if short_tags else "<noise>"
    return canvas, primary_cat, combined_tag, used_tags


def build_continuous_ambient_bed(
    bank: dict[str, list[NoiseClip]],
    categories: list[str],
    total_samples: int,
    sr: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a continuous crossfaded low-level ambient bed across the full call."""
    bed = np.zeros(total_samples, dtype=np.float32)
    pool: list[NoiseClip] = []
    for c in categories:
        pool.extend(bank.get(c, []))
    if not pool:
        for v in bank.values():
            pool.extend(v)
    if not pool:
        return synth_wind_mic_rub(total_samples, sr, rng) * 0.2

    cursor = 0
    xfade = int(0.060 * sr)
    while cursor < total_samples:
        clip = pool[int(rng.integers(0, len(pool)))]
        snip = clip.audio.copy()
        b, a = butter(2, 1800.0 / (0.5 * sr), btype="low")
        snip = filtfilt(b, a, snip).astype(np.float32)
        rms = float(np.sqrt(np.mean(snip**2) + 1e-12))
        snip = snip / rms

        fn = min(xfade, len(snip) // 3)
        if fn > 1:
            ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fn, dtype=np.float32)))
            snip[:fn] *= ramp
            snip[-fn:] *= ramp[::-1]

        end = min(total_samples, cursor + len(snip))
        bed[cursor:end] += snip[: end - cursor]
        cursor += max(1, len(snip) - fn)

    rms = float(np.sqrt(np.mean(bed**2) + 1e-12))
    return (bed / rms).astype(np.float32)
