"""4-Stem Synthetic Voice & Multi-Hazard Audio Composer.

Assembles synchronized 16 kHz and 8 kHz telephony calls from 4 independent stems:
  1. Primary Caller Voice (`label="voice"`, `should_trigger=True`)
  2. Competing Background Human Voice (`label="background_voice"`, `should_trigger=False` when alone)
  3. Compound Multi-Noise & Wind Buffeting Bursts (`label="noise"`, `should_trigger=False` when alone)
  4. Continuous Ambient Noise Bed

Outputs exact sub-10ms ground-truth JSON annotations (`ground_truth/<id>.json`),
`manifest.jsonl`, `data_summary.md`, and WAV files (`audio_16k/*.wav`, `audio_8k/*.wav`,
plus optional `stems/<id>_*.wav`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from config import PipelineConfig
from src.acoustics import (
    NoiseClip,
    annotate_gaps,
    apply_bgv_acoustics,
    apply_far_field_acoustics,
    build_continuous_ambient_bed,
    compose_compound_noise_burst,
    load_and_trim_clean_tts,
    measure_lag_ms,
    to_telephony_pcm,
)
from src.scenarios import CallTurnSpec, ScenarioSpec
from src.tts_client import tts_cache_path


def build_scenario_call(
    sc: ScenarioSpec,
    bank: dict[str, list[NoiseClip]],
    tts_cache_dir: Path,
    cfg: PipelineConfig,
    seed: int,
) -> dict[str, Any]:
    """Build a single multi-stem synthetic call and compute exact ground-truth segment timestamps."""
    rng = np.random.default_rng(seed)
    sr = cfg.native_sr
    target_voice_rms = cfg.target_voice_rms

    timeline_events: list[dict[str, Any]] = []
    cursor_s = cfg.initial_lead_in_s

    for turn_idx, tspec in enumerate(sc.turns):
        # --- 1. Primary User Voice Turn (should_trigger = True) ---
        p_wav = tts_cache_path(tts_cache_dir, cfg.tts_model_id, sc.primary_voice, tspec.text)
        p_speed = float(cfg.prosody_speed_multipliers.get(tspec.prosody_style, 1.0))
        p_min_isl = (
            cfg.micro_min_island_s
            if tspec.prosody_style == "short_micro"
            else cfg.default_min_island_s
        )
        p_hes = tspec.prosody_style in ("hesitation_stutter", "slow_hesitant")

        p_audio_norm, rel_islands, rel_speech = load_and_trim_clean_tts(
            p_wav,
            target_sr=sr,
            merge_gap_s=cfg.audio_trim_merge_gap_s,
            min_island_s=p_min_isl,
            gt_merge_gap_s=cfg.voice_merge_gap_s,
            speed_rate=p_speed,
            inject_hesitation_pauses=p_hes,
            hesitation_pause_s=cfg.hesitation_pause_s,
        )
        turn_dur_s = len(p_audio_norm) / sr
        t_turn_start = cursor_s
        t_turn_end = cursor_s + turn_dur_s

        timeline_events.append({
            "kind": "primary_turn",
            "turn_idx": turn_idx,
            "start_s": t_turn_start,
            "end_s": t_turn_end,
            "audio_norm": p_audio_norm,
            "rel_islands": rel_islands,
            "rel_speech": rel_speech,
            "spec": tspec,
        })
        cursor_s = t_turn_end + cfg.post_voice_guard_s

        # --- 2. Listening Window: Compound Multi-Noise Burst (should_trigger = False) ---
        if tspec.pause_noise_cats:
            burst_dur_s = float(
                rng.uniform(cfg.pause_noise_burst_min_s, cfg.pause_noise_burst_max_s)
            )
            burst_norm, p_cat, c_tag, u_tags = compose_compound_noise_burst(
                bank, tspec.pause_noise_cats, burst_dur_s, sr, rng, cfg
            )
            n_start = cursor_s
            n_end = cursor_s + burst_dur_s
            timeline_events.append({
                "kind": "pause_noise",
                "turn_idx": turn_idx,
                "start_s": n_start,
                "end_s": n_end,
                "audio_norm": burst_norm,
                "snr_db": tspec.pause_noise_snr_db,
                "category": p_cat,
                "tag": c_tag,
                "used_tags": u_tags,
            })
            cursor_s = n_end + cfg.inter_event_guard_s

        # --- 3. Listening Window: Competing Background Human Voice + Noise (should_trigger = False) ---
        if tspec.pause_bgv_text:
            bg_wav = tts_cache_path(
                tts_cache_dir, cfg.tts_model_id, tspec.pause_bgv_voice, tspec.pause_bgv_text
            )
            bg_min_isl = (
                cfg.micro_min_island_s
                if tspec.pause_bgv_acoustics == "bystander_backchannel"
                else cfg.default_min_island_s
            )
            bg_clean_norm, _, _ = load_and_trim_clean_tts(
                bg_wav,
                target_sr=sr,
                merge_gap_s=cfg.audio_trim_merge_gap_s,
                min_island_s=bg_min_isl,
                gt_merge_gap_s=cfg.voice_merge_gap_s,
            )
            bg2_clean_norm = None
            if tspec.pause_bgv2_text:
                bg2_wav = tts_cache_path(
                    tts_cache_dir, cfg.tts_model_id, tspec.pause_bgv2_voice, tspec.pause_bgv2_text
                )
                bg2_clean_norm, _, _ = load_and_trim_clean_tts(
                    bg2_wav,
                    target_sr=sr,
                    merge_gap_s=cfg.audio_trim_merge_gap_s,
                    min_island_s=cfg.default_min_island_s,
                    gt_merge_gap_s=cfg.voice_merge_gap_s,
                )

            bg_wet_norm = apply_bgv_acoustics(
                bg_clean_norm,
                sr,
                sc.rt60_s,
                rng,
                cfg,
                mode=tspec.pause_bgv_acoustics,
                speech2_norm=bg2_clean_norm,
            )
            bg_dur_s = len(bg_wet_norm) / sr

            co_noise_norm = None
            co_tags: list[str] = []
            if tspec.pause_bgv_with_noise_cats:
                co_noise_norm, _, _, co_tags = compose_compound_noise_burst(
                    bank, tspec.pause_bgv_with_noise_cats, bg_dur_s, sr, rng, cfg
                )

            b_start = cursor_s
            b_end = cursor_s + bg_dur_s
            full_bg_text = (
                f"{tspec.pause_bgv_text} || {tspec.pause_bgv2_text}"
                if tspec.pause_bgv2_text
                else tspec.pause_bgv_text
            )
            timeline_events.append({
                "kind": "pause_bgv",
                "turn_idx": turn_idx,
                "start_s": b_start,
                "end_s": b_end,
                "audio_norm": bg_wet_norm,
                "sir_db": tspec.pause_bgv_sir_db,
                "acoustics": tspec.pause_bgv_acoustics,
                "voice_name": (
                    f"{tspec.pause_bgv_voice}+{tspec.pause_bgv2_voice}"
                    if tspec.pause_bgv2_text
                    else tspec.pause_bgv_voice
                ),
                "text": full_bg_text,
                "co_noise_norm": co_noise_norm,
                "co_tags": co_tags,
            })
            cursor_s = b_end + cfg.pre_voice_guard_s
        else:
            cursor_s += (
                (cfg.pre_voice_guard_s - cfg.inter_event_guard_s)
                if tspec.pause_noise_cats
                else cfg.pre_voice_guard_s
            )

    # Enforce minimum and maximum conversation duration range (`min_conversation_duration_s` .. `max_conversation_duration_s`)
    min_dur_s = max(float(cfg.min_conversation_duration_s), float(cfg.min_call_duration_s))
    max_dur_s = float(cfg.max_conversation_duration_s) if cfg.max_conversation_duration_s > 0 else 0.0
    extra_idx = 0
    bgv_turns = [t for t in sc.turns if t.pause_bgv_text]
    while min_dur_s > 0 and (cursor_s + cfg.final_tail_s) < min_dur_s:
        ref_t = sc.turns[extra_idx % len(sc.turns)]
        ncats = ref_t.pause_noise_cats or sc.ambient_categories
        burst_dur_s = float(
            rng.uniform(cfg.pause_noise_burst_min_s, cfg.pause_noise_burst_max_s)
        )
        if max_dur_s > 0 and (cursor_s + burst_dur_s + cfg.final_tail_s) > max_dur_s:
            break
        burst_norm, p_cat, c_tag, u_tags = compose_compound_noise_burst(
            bank, ncats, burst_dur_s, sr, rng, cfg
        )
        n_start = cursor_s
        n_end = cursor_s + burst_dur_s
        timeline_events.append({
            "kind": "pause_noise",
            "turn_idx": len(sc.turns) + extra_idx,
            "start_s": n_start,
            "end_s": n_end,
            "audio_norm": burst_norm,
            "snr_db": ref_t.pause_noise_snr_db,
            "category": p_cat,
            "tag": c_tag,
            "used_tags": u_tags,
        })
        cursor_s = n_end + cfg.inter_event_guard_s

        if bgv_turns and (cursor_s + cfg.final_tail_s) < min_dur_s:
            bt = bgv_turns[extra_idx % len(bgv_turns)]
            bg_wav = tts_cache_path(tts_cache_dir, cfg.tts_model_id, bt.pause_bgv_voice, bt.pause_bgv_text)
            bg_min_isl = (
                cfg.micro_min_island_s
                if bt.pause_bgv_acoustics == "bystander_backchannel"
                else cfg.default_min_island_s
            )
            bg_clean_norm, _, _ = load_and_trim_clean_tts(
                bg_wav, target_sr=sr, min_island_s=bg_min_isl
            )
            bg_wet_norm = apply_bgv_acoustics(
                bg_clean_norm, sr, sc.rt60_s, rng, cfg, mode=bt.pause_bgv_acoustics
            )
            bg_dur_s = len(bg_wet_norm) / sr
            if max_dur_s > 0 and (cursor_s + bg_dur_s + cfg.final_tail_s) > max_dur_s:
                extra_idx += 1
                break
            b_start = cursor_s
            b_end = cursor_s + bg_dur_s
            timeline_events.append({
                "kind": "pause_bgv",
                "turn_idx": len(sc.turns) + extra_idx,
                "start_s": b_start,
                "end_s": b_end,
                "audio_norm": bg_wet_norm,
                "sir_db": bt.pause_bgv_sir_db,
                "acoustics": bt.pause_bgv_acoustics,
                "voice_name": bt.pause_bgv_voice,
                "text": bt.pause_bgv_text,
                "co_noise_norm": None,
                "co_tags": [],
            })
            cursor_s = b_end + cfg.pre_voice_guard_s
        extra_idx += 1

    # If natural turns exceeded max_conversation_duration_s, trim trailing non-primary pause events so call stays within max_dur_s
    while (
        max_dur_s > 0
        and len(timeline_events) > 1
        and timeline_events[-1]["kind"] in ("pause_noise", "pause_bgv")
        and (timeline_events[-1]["end_s"] + cfg.final_tail_s) > max_dur_s
    ):
        timeline_events.pop()
        cursor_s = timeline_events[-1]["end_s"] + cfg.post_voice_guard_s

    total_dur_s = round(cursor_s + cfg.final_tail_s, 3)
    total_samples = int(round(total_dur_s * sr))

    # Initialize 4 synchronized stems
    stem_primary = np.zeros(total_samples, dtype=np.float32)
    stem_bgv = np.zeros(total_samples, dtype=np.float32)
    stem_noise = np.zeros(total_samples, dtype=np.float32)
    stem_ambient = build_continuous_ambient_bed(
        bank, sc.ambient_categories, total_samples, sr, rng
    )
    ambient_gain = target_voice_rms * (10.0 ** (sc.ambient_level_db / 20.0))
    stem_ambient *= ambient_gain * 0.30

    segments: list[dict[str, Any]] = []
    speech_segments: list[list[float]] = []
    all_speech_spans: list[list[float]] = []
    words_meta: list[dict[str, Any]] = []
    interferer_windows: list[dict[str, Any]] = []
    noise_events_meta: list[dict[str, Any]] = []
    transcript_parts: list[str] = []

    # Track per-turn exact audio timestamps (`start_audio_sec` / `end_audio_sec`)
    turns_with_timestamps: list[dict[str, Any]] = [t.to_dict() if hasattr(t, "to_dict") else dict(t.__dict__) for t in sc.turns]
    for td in turns_with_timestamps:
        td["audio_timestamps"] = {
            "primary_voice": None,
            "overlap_noise": None,
            "overlap_background_voice": None,
            "pause_noise": None,
            "pause_background_voice": None,
        }

    for ev in timeline_events:
        i0 = int(round(ev["start_s"] * sr))
        t_idx = ev["turn_idx"]
        turn_ts = turns_with_timestamps[t_idx]["audio_timestamps"] if t_idx < len(turns_with_timestamps) else None
        if ev["kind"] == "primary_turn":
            tspec: CallTurnSpec = ev["spec"]
            p_gain = target_voice_rms * (10.0 ** (tspec.primary_gain_db / 20.0))
            p_sig = (ev["audio_norm"] * p_gain).copy()

            overlap_tags: list[str] = []
            if tspec.primary_gain_db < -2.0:
                overlap_tags.append(f"soft_user_voice:{tspec.primary_gain_db:g} dB")
            elif tspec.primary_gain_db > 2.0:
                overlap_tags.append(f"loud_user_voice:+{tspec.primary_gain_db:g} dB")
            if tspec.prosody_style != "normal":
                overlap_tags.append(f"prosody:{tspec.prosody_style}")

            # Apply microphone / telephony channel quirks to primary speech stem
            if tspec.channel_quirk == "mic_clipping":
                pk = float(np.max(np.abs(p_sig)) + 1e-9)
                drv = cfg.mic_clipping_drive
                p_sig = (np.tanh(drv * (p_sig / pk)) / np.tanh(drv) * pk).astype(np.float32)
                overlap_tags.append("channel_quirk:mic_clipping")
            elif tspec.channel_quirk == "packet_loss" and len(p_sig) > int(0.40 * sr):
                drop_n = int(round(cfg.packet_loss_dropout_s * sr))
                for frac in cfg.packet_loss_positions:
                    d0 = int(len(p_sig) * float(frac))
                    d1 = min(len(p_sig), d0 + drop_n)
                    p_sig[d0:d1] = 0.0
                overlap_tags.append(
                    f"channel_quirk:packet_loss_{int(round(cfg.packet_loss_dropout_s * 1000))}ms"
                )

            i1 = min(total_samples, i0 + len(p_sig))
            stem_primary[i0:i1] += p_sig[: i1 - i0]
            transcript_parts.append(tspec.text)

            turn_dur = (i1 - i0) / sr

            # Overlapping compound noise during user speech
            if tspec.voice_mode in ("overlap_noise", "overlap_bgv_and_noise") and tspec.overlap_noise_cats:
                ov_noise_norm, ov_cat, ov_tag, ov_utags = compose_compound_noise_burst(
                    bank, tspec.overlap_noise_cats, turn_dur, sr, rng, cfg
                )
                ov_gain = target_voice_rms * (10.0 ** (-tspec.overlap_snr_db / 20.0))
                stem_noise[i0:i1] += ov_noise_norm[: i1 - i0] * ov_gain
                overlap_tags.extend(ov_utags)
                ov_s = round(ev["start_s"], 3)
                ov_e = round(ev["end_s"], 3)
                noise_events_meta.append({
                    "category": ov_cat,
                    "tag": ov_tag,
                    "start": ov_s,
                    "end": ov_e,
                    "start_audio_sec": ov_s,
                    "end_audio_sec": ov_e,
                    "event_type": "overlap_with_voice",
                })
                if turn_ts is not None:
                    turn_ts["overlap_noise"] = {
                        "label": "overlap_noise_with_voice",
                        "start_audio_sec": ov_s,
                        "end_audio_sec": ov_e,
                        "duration_s": round(ov_e - ov_s, 3),
                        "category": ov_cat,
                        "tag": ov_tag,
                        "snr_db": tspec.overlap_snr_db,
                    }

            # Overlapping competing background speaker during user speech
            if tspec.voice_mode == "overlap_bgv_and_noise" and tspec.overlap_bgv_text:
                ov_bg_wav = tts_cache_path(
                    tts_cache_dir, cfg.tts_model_id, tspec.overlap_bgv_voice, tspec.overlap_bgv_text
                )
                ov_bg_clean, _, _ = load_and_trim_clean_tts(ov_bg_wav, target_sr=sr)
                ov_bg_wet = apply_far_field_acoustics(ov_bg_clean, sr, sc.rt60_s, rng)
                bg_gain = target_voice_rms * (10.0 ** (-tspec.overlap_sir_db / 20.0))
                b1 = min(i1, i0 + len(ov_bg_wet))
                stem_bgv[i0:b1] += ov_bg_wet[: b1 - i0] * bg_gain
                overlap_tags.append(f"background_voice:SIR {tspec.overlap_sir_db:g} dB")
                ov_bs = round(ev["start_s"], 3)
                ov_be = round(b1 / sr, 3)
                interferer_windows.append({
                    "type": "interferer_overlap",
                    "start": ov_bs,
                    "end": ov_be,
                    "start_audio_sec": ov_bs,
                    "end_audio_sec": ov_be,
                    "sir_db": tspec.overlap_sir_db,
                    "speaker": tspec.overlap_bgv_voice,
                })
                if turn_ts is not None:
                    turn_ts["overlap_background_voice"] = {
                        "label": "overlap_background_voice_with_voice",
                        "start_audio_sec": ov_bs,
                        "end_audio_sec": ov_be,
                        "duration_s": round(ov_be - ov_bs, 3),
                        "sir_db": tspec.overlap_sir_db,
                        "speaker": tspec.overlap_bgv_voice,
                        "transcript": tspec.overlap_bgv_text,
                    }

            # Ground-truth segment tag classification
            if tspec.prosody_style == "short_micro":
                v_tag = "short_utterance"
            elif tspec.prosody_style == "angry_shout":
                v_tag = "angry_clipped_speech"
            elif tspec.prosody_style == "fast_panicked":
                v_tag = "fast_panicked_speech"
            elif tspec.prosody_style == "slow_hesitant":
                v_tag = "slow_hesitant_speech"
            elif tspec.prosody_style == "laugh_chuckle":
                v_tag = "laugh_speech"
            elif tspec.prosody_style == "hesitation_stutter":
                v_tag = "hesitant_speech"
            elif tspec.channel_quirk == "packet_loss":
                v_tag = "packet_loss_speech"
            else:
                v_tag = (
                    "soft_noisy_speech"
                    if (tspec.primary_gain_db < -2.0 and overlap_tags)
                    else (
                        "soft_clean_speech"
                        if tspec.primary_gain_db < -2.0
                        else ("clean_speech" if not overlap_tags else "noisy_speech")
                    )
                )

            for rs, re_ in ev["rel_speech"]:
                all_speech_spans.append([ev["start_s"] + rs, ev["start_s"] + re_])

            turn_islands_ts: list[dict[str, float]] = []
            for rs, re_ in ev["rel_islands"]:
                abs_s = round(ev["start_s"] + rs, 3)
                abs_e = round(ev["start_s"] + re_, 3)
                speech_segments.append([abs_s, abs_e])
                turn_islands_ts.append({
                    "start_audio_sec": abs_s,
                    "end_audio_sec": abs_e,
                    "duration_s": round(abs_e - abs_s, 3),
                })
                seg_dict: dict[str, Any] = {
                    "label": "voice",
                    "should_trigger": True,
                    "category": "user_voice",
                    "tag": v_tag,
                    "overlap_tags": list(overlap_tags),
                    "start": abs_s,
                    "end": abs_e,
                    "start_audio_sec": abs_s,
                    "end_audio_sec": abs_e,
                    "duration_s": round(abs_e - abs_s, 3),
                    "clip_start": False,
                    "rel_loudness_db": round(float(tspec.primary_gain_db), 1),
                    "sir_db": tspec.overlap_sir_db if tspec.voice_mode == "overlap_bgv_and_noise" else None,
                    "transcript": tspec.text,
                }
                if tspec.prosody_style != "normal":
                    seg_dict["prosody_style"] = tspec.prosody_style
                if tspec.channel_quirk != "none":
                    seg_dict["channel_quirk"] = tspec.channel_quirk
                segments.append(seg_dict)
                words_meta.append({
                    "text": tspec.text,
                    "start": abs_s,
                    "end": abs_e,
                    "start_audio_sec": abs_s,
                    "end_audio_sec": abs_e,
                })
            if turn_ts is not None and turn_islands_ts:
                turn_ts["primary_voice"] = {
                    "label": "voice",
                    "should_trigger": True,
                    "start_audio_sec": turn_islands_ts[0]["start_audio_sec"],
                    "end_audio_sec": turn_islands_ts[-1]["end_audio_sec"],
                    "duration_s": round(
                        turn_islands_ts[-1]["end_audio_sec"] - turn_islands_ts[0]["start_audio_sec"], 3
                    ),
                    "voice_islands": turn_islands_ts,
                }

        elif ev["kind"] == "pause_noise":
            n_sig = ev["audio_norm"] * target_voice_rms * (10.0 ** (-ev["snr_db"] / 20.0))
            i1 = min(total_samples, i0 + len(n_sig))
            stem_noise[i0:i1] += n_sig[: i1 - i0]
            s0 = round(ev["start_s"], 3)
            s1 = round(i1 / sr, 3)
            segments.append({
                "label": "noise",
                "should_trigger": False,
                "category": ev["category"],
                "tag": ev["tag"],
                "overlap_tags": ev["used_tags"],
                "start": s0,
                "end": s1,
                "start_audio_sec": s0,
                "end_audio_sec": s1,
                "duration_s": round(s1 - s0, 3),
                "clip_start": False,
                "rel_loudness_db": round(-float(ev["snr_db"]), 1),
                "sir_db": None,
            })
            noise_events_meta.append({
                "category": ev["category"],
                "tag": ev["tag"],
                "start": s0,
                "end": s1,
                "start_audio_sec": s0,
                "end_audio_sec": s1,
                "event_type": "pure_noise",
            })
            if turn_ts is not None:
                turn_ts["pause_noise"] = {
                    "label": "noise",
                    "should_trigger": False,
                    "start_audio_sec": s0,
                    "end_audio_sec": s1,
                    "duration_s": round(s1 - s0, 3),
                    "category": ev["category"],
                    "tag": ev["tag"],
                }

        elif ev["kind"] == "pause_bgv":
            bg_sig = ev["audio_norm"] * target_voice_rms * (10.0 ** (-ev["sir_db"] / 20.0))
            i1 = min(total_samples, i0 + len(bg_sig))
            stem_bgv[i0:i1] += bg_sig[: i1 - i0]
            if ev["co_noise_norm"] is not None:
                cn = ev["co_noise_norm"] * target_voice_rms * (10.0 ** (-10.0 / 20.0))
                stem_noise[i0:i1] += cn[: i1 - i0]

            s0 = round(ev["start_s"], 3)
            s1 = round(i1 / sr, 3)
            ac_mode = ev.get("acoustics", "far_field")
            segments.append({
                "label": "background_voice",
                "should_trigger": False,
                "category": "background_voice",
                "tag": f"SIR {ev['sir_db']:g} dB ({ac_mode})",
                "acoustics": ac_mode,
                "overlap_tags": ev["co_tags"],
                "start": s0,
                "end": s1,
                "start_audio_sec": s0,
                "end_audio_sec": s1,
                "duration_s": round(s1 - s0, 3),
                "clip_start": False,
                "rel_loudness_db": round(-float(ev["sir_db"]), 1),
                "sir_db": float(ev["sir_db"]),
                "transcript": ev["text"],
            })
            interferer_windows.append({
                "type": "interferer_only",
                "acoustics": ac_mode,
                "start": s0,
                "end": s1,
                "start_audio_sec": s0,
                "end_audio_sec": s1,
                "sir_db": float(ev["sir_db"]),
                "speaker": ev["voice_name"],
            })
            if turn_ts is not None:
                turn_ts["pause_background_voice"] = {
                    "label": "background_voice",
                    "should_trigger": False,
                    "start_audio_sec": s0,
                    "end_audio_sec": s1,
                    "duration_s": round(s1 - s0, 3),
                    "acoustics": ac_mode,
                    "sir_db": float(ev["sir_db"]),
                    "speaker": ev["voice_name"],
                    "transcript": ev["text"],
                }

    # Combine all 4 stems and apply peak limiter guard
    mix = stem_primary + stem_bgv + stem_noise + stem_ambient
    peak = float(np.max(np.abs(mix)))
    peak_rescaled = False
    if peak > cfg.peak_ceiling:
        scale = cfg.peak_ceiling / peak
        mix = (mix * scale).astype(np.float32)
        stem_primary = (stem_primary * scale).astype(np.float32)
        stem_bgv = (stem_bgv * scale).astype(np.float32)
        stem_noise = (stem_noise * scale).astype(np.float32)
        stem_ambient = (stem_ambient * scale).astype(np.float32)
        peak_rescaled = True

    segments.sort(key=lambda seg: (seg["start"], seg["end"]))
    annotate_gaps(segments, all_speech_spans)
    for idx, seg in enumerate(segments):
        sid = f"s{idx:02d}"
        seg["segment_id"] = sid
        seg["event_id"] = sid
        seg["event_type"] = seg["label"]

    pcm_8k, n_clip_8k = to_telephony_pcm(mix, sr, cfg.telephony_sr)
    lag_ms = measure_lag_ms(mix, sr, pcm_8k, cfg.telephony_sr)

    gt = {
        "id": sc.cid,
        "kind": "synthetic_multi_variation_call",
        "title": sc.title,
        "failure_focus": sc.failure_focus,
        "split": "test",
        "source": {
            "tts_model": cfg.tts_model_id,
            "primary_voice": sc.primary_voice,
            "noise_source": "Bundled CC-BY-4.0 VAANI Stems + Procedural Acoustics",
            "language": sc.language,
        },
        "pilot_category": sc.pilot_category,
        "native_sr": sr,
        "telephony_sr": cfg.telephony_sr,
        "duration_s": total_dur_s,
        "target_duration_range_s": {
            "min_s": cfg.min_conversation_duration_s,
            "max_s": cfg.max_conversation_duration_s,
            "target_avg_s": cfg.target_avg_conversation_duration_s,
        },
        "noise_floor_db": round(sc.ambient_level_db - 10.0, 1),
        "transcript_raw": " ".join(transcript_parts),
        "transcript_clean": " ".join(transcript_parts),
        "words": words_meta,
        "speech_segments": speech_segments,
        "pure_noise_intervals": [n for n in noise_events_meta if n["event_type"] == "pure_noise"],
        "noise_events": noise_events_meta,
        "turns_with_timestamps": turns_with_timestamps,
        "segments": segments,
        "events": segments,
        "interferer": {
            "source_id": f"{cfg.tts_model_id}-multimode",
            "rt60_s": sc.rt60_s,
            "windows": interferer_windows,
        },
        "checks": {
            "warnings": [],
            "resample_lag_ms": round(lag_ms, 3),
            "n_clipped_8k": n_clip_8k,
            "peak_rescaled": peak_rescaled,
        },
    }
    return {
        "cid": sc.cid,
        "audio_16k": mix,
        "audio_8k": pcm_8k,
        "stems_16k": {
            "primary_voice": stem_primary,
            "background_voice": stem_bgv,
            "noise_bursts": stem_noise,
            "ambient_bed": stem_ambient,
        },
        "gt": gt,
    }


def write_dataset_outputs(
    out_dir: Path,
    items: list[dict[str, Any]],
    cfg: PipelineConfig,
) -> dict[str, Any]:
    """Write 16 kHz / 8 kHz WAVs, optional isolated stems, ground-truth JSONs, manifest, and summary."""
    if cfg.export_16khz_native:
        (out_dir / "audio_16k").mkdir(parents=True, exist_ok=True)
    if cfg.export_8khz_telephony:
        (out_dir / "audio_8k").mkdir(parents=True, exist_ok=True)
    if cfg.export_isolated_stems:
        (out_dir / "stems").mkdir(parents=True, exist_ok=True)
    (out_dir / "ground_truth").mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    tot_voice = tot_noise = tot_bgv = 0
    tot_dur = 0.0

    for item in items:
        cid = item["cid"]
        gt = item["gt"]
        if cfg.export_16khz_native and item.get("audio_16k") is not None:
            sf.write(
                str(out_dir / "audio_16k" / f"{cid}.wav"),
                item["audio_16k"],
                cfg.native_sr,
                subtype="PCM_16",
            )
        if cfg.export_8khz_telephony and item.get("audio_8k") is not None:
            sf.write(
                str(out_dir / "audio_8k" / f"{cid}.wav"),
                item["audio_8k"],
                cfg.telephony_sr,
                subtype="PCM_16",
            )
        if cfg.export_isolated_stems and item.get("stems_16k"):
            for stem_name, stem_arr in item["stems_16k"].items():
                sf.write(
                    str(out_dir / "stems" / f"{cid}_{stem_name}.wav"),
                    stem_arr,
                    cfg.native_sr,
                    subtype="PCM_16",
                )

        (out_dir / "ground_truth" / f"{cid}.json").write_text(
            json.dumps(gt, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        segs = gt["segments"]
        nv = sum(1 for s in segs if s["label"] == "voice")
        nn = sum(1 for s in segs if s["label"] == "noise")
        nb = sum(1 for s in segs if s["label"] == "background_voice")
        tot_voice += nv
        tot_noise += nn
        tot_bgv += nb
        tot_dur += float(gt["duration_s"])

        manifest_rows.append({
            "id": cid,
            "kind": gt["kind"],
            "language": gt["source"]["language"],
            "primary_voice": gt["source"]["primary_voice"],
            "failure_focus": gt.get("failure_focus", ""),
            "duration_s": gt["duration_s"],
            "native_sr": cfg.native_sr,
            "telephony_sr": cfg.telephony_sr,
            "audio_16k": f"audio_16k/{cid}.wav" if cfg.export_16khz_native else None,
            "audio_8k": f"audio_8k/{cid}.wav" if cfg.export_8khz_telephony else None,
            "ground_truth": f"ground_truth/{cid}.json",
            "n_voice": nv,
            "n_noise": nn,
            "n_background_voice": nb,
            "n_total_segments": len(segs),
        })

    (out_dir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in manifest_rows) + "\n",
        encoding="utf-8",
    )

    # Also write `scenarios_with_timestamps.json` so users can see the turn-by-turn scenario config
    # side-by-side with exact `start_audio_sec` and `end_audio_sec` for every voice, noise, and background_voice event
    scenarios_with_ts = []
    for item in items:
        cid = item["cid"]
        gt = item["gt"]
        scenarios_with_ts.append({
            "cid": cid,
            "title": gt["title"],
            "pilot_category": gt["pilot_category"],
            "language": gt["source"]["language"],
            "primary_voice": gt["source"]["primary_voice"],
            "duration_s": gt["duration_s"],
            "audio_16k": f"audio_16k/{cid}.wav" if cfg.export_16khz_native else None,
            "audio_8k": f"audio_8k/{cid}.wav" if cfg.export_8khz_telephony else None,
            "ground_truth_file": f"ground_truth/{cid}.json",
            "turns": gt.get("turns_with_timestamps", []),
            "segments": [
                {
                    "segment_id": s["segment_id"],
                    "label": s["label"],
                    "should_trigger": s["should_trigger"],
                    "start_audio_sec": s["start_audio_sec"],
                    "end_audio_sec": s["end_audio_sec"],
                    "duration_s": s["duration_s"],
                    "category": s["category"],
                    "tag": s["tag"],
                    "overlap_tags": s.get("overlap_tags", []),
                    "transcript": s.get("transcript", ""),
                }
                for s in gt["segments"]
            ],
        })
    (out_dir / "scenarios_with_timestamps.json").write_text(
        json.dumps({"scenarios": scenarios_with_ts}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    avg_dur_s = tot_dur / max(1, len(items))
    min_actual_s = min((float(r["duration_s"]) for r in manifest_rows), default=0.0)
    max_actual_s = max((float(r["duration_s"]) for r in manifest_rows), default=0.0)

    lines = [
        f"# Synthetic Multi-Variation Voice Dataset Summary ({len(items)} Calls)",
        "",
        f"- **Generated Calls**: **{len(items)}**",
        f"- **Total Duration**: **{tot_dur:.1f} s ({tot_dur / 60.0:.2f} mins)**",
        f"- **Configured Conversation Duration Range**: **`{cfg.min_conversation_duration_s:.1f}s – {cfg.max_conversation_duration_s:.1f}s`** (Target Avg: **`{cfg.target_avg_conversation_duration_s:.1f}s`**)",
        f"- **Actual Conversation Duration Range**: **`{min_actual_s:.1f}s – {max_actual_s:.1f}s`** (Actual Avg: **`{avg_dur_s:.1f}s`**)",
        f"- **TTS Engine**: **{cfg.tts_model_id}** (Google GenAI SDK `google-genai`)",
        f"- **Scenario Generator LLM**: **{cfg.scenario_generator_model_id}**",
        f"- **Sample Rates Exported**: `16 kHz` wideband (`audio_16k/*.wav`) & `8 kHz` telephony (`audio_8k/*.wav`)",
        "",
        "## Ground-Truth Segment Breakdown",
        "",
        "| Segment `label` | `should_trigger` | Count |",
        "| :--- | :---: | ---: |",
        f"| `voice` (Primary User Speech — clean, noisy, overlapped, micro-utterances, laughs, hesitations, angry/fast/slow, packet-loss) | **`true` (`1`)** | {tot_voice} |",
        f"| `noise` (Compound Multi-Noise & Wind Buffeting Bursts in silence) | **`false` (`0`)** | {tot_noise} |",
        f"| `background_voice` (Far-Field, Near-Field Loud, TV/Radio, 2-Speaker Babble, Bot Echo, Bystander Backchannels & Coughs) | **`false` (`0`)** | {tot_bgv} |",
        f"| **Total Segments** | — | **{tot_voice + tot_noise + tot_bgv}** |",
        "",
        "## Per-Call Summary",
        "",
        "| Call ID | Language | Voice | Variations Tested | Duration (s) | `voice` (`1`) | `noise` (`0`) | `background_voice` (`0`) |",
        "| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: |",
    ]
    for row in manifest_rows:
        lines.append(
            f"| `{row['id']}` | {row['language']} | `{row['primary_voice']}` | {row['failure_focus']} | {row['duration_s']:.1f}s | {row['n_voice']} | {row['n_noise']} | {row['n_background_voice']} |"
        )
    lines.append("")
    lines.append("## Per-Call Ground-Truth Segment Timelines (`start_audio_sec` → `end_audio_sec`)")
    lines.append("")
    for item in items:
        cid = item["cid"]
        gt = item["gt"]
        lines.append(f"### `{cid}` ({gt['source']['language']}, `{gt['duration_s']:.2f}s`)")
        lines.append(
            f"- **Audio Files**: [`audio_16k/{cid}.wav`](audio_16k/{cid}.wav) | [`audio_8k/{cid}.wav`](audio_8k/{cid}.wav) | **Ground Truth JSON**: [`ground_truth/{cid}.json`](ground_truth/{cid}.json)"
        )
        lines.append("")
        lines.append(
            "| Seg ID | `start_audio_sec` | `end_audio_sec` | `duration_s` | `label` | `should_trigger` | Category / Tag | Transcript / Overlapping Events |"
        )
        lines.append("| :--- | ---: | ---: | ---: | :--- | :---: | :--- | :--- |")
        for s in gt["segments"]:
            trig = "**`true` (`1`)**" if s["should_trigger"] else "`false` (`0`)"
            detail = s.get("transcript", "") or ", ".join(s.get("overlap_tags", []))
            if s.get("transcript") and s.get("overlap_tags"):
                detail = f"{s['transcript']} *(overlaps: {', '.join(s['overlap_tags'])})*"
            lines.append(
                f"| `{s['segment_id']}` | `{s['start_audio_sec']:.3f}s` | `{s['end_audio_sec']:.3f}s` | `{s['duration_s']:.3f}s` | `{s['label']}` | {trig} | `{s['category']}` / `{s['tag']}` | {detail} |"
            )
        lines.append("")

    (out_dir / "data_summary.md").write_text("\n".join(lines), encoding="utf-8")

    return {
        "total_calls": len(items),
        "total_duration_s": round(tot_dur, 2),
        "avg_duration_s": round(avg_dur_s, 2),
        "min_duration_s": round(min_actual_s, 2),
        "max_duration_s": round(max_actual_s, 2),
        "total_voice_segments": tot_voice,
        "total_noise_segments": tot_noise,
        "total_background_voice_segments": tot_bgv,
        "manifest": manifest_rows,
    }
