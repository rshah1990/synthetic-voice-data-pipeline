"""Unit tests for the standalone Synthetic Voice & Multi-Hazard Audio Data Pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf

from config import PipelineConfig
from src.acoustics import (
    annotate_gaps,
    apply_bgv_acoustics,
    compose_compound_noise_burst,
    load_and_trim_clean_tts,
    load_noise_bank,
    measure_lag_ms,
    to_telephony_pcm,
)
from src.composer import build_scenario_call
from src.scenarios import CallTurnSpec, ScenarioSpec, get_test_showcase_scenarios
from src.tts_client import tts_cache_path


def test_config_loads_yaml():
    cfg = PipelineConfig.from_file(Path(__file__).resolve().parents[1] / "configs" / "pipeline_config.yaml")
    assert cfg.tts_model_id == "gemini-3.8-flash-tts"
    assert cfg.native_sr == 16000
    assert cfg.telephony_sr == 8000
    assert "short_micro" in cfg.prosody_speed_multipliers
    assert "bot_echo_leak" in cfg.acoustic_modes
    assert "conversational audio dataset architect" in cfg.scenario_generation_system_prompt
    assert "multilingual voice actor" in cfg.tts_director_system_prompt
    assert "angry_shout" in cfg.tts_prosody_style_instructions
    assert cfg.min_conversation_duration_s == 35.0
    assert cfg.max_conversation_duration_s == 60.0
    assert cfg.target_avg_conversation_duration_s == 45.0


def test_noise_bank_and_acoustics():
    cfg = PipelineConfig()
    bank = load_noise_bank(cfg.noise_stems_dir, target_sr=cfg.native_sr)
    assert len(bank) == 7
    rng = np.random.default_rng(42)

    burst, p_cat, c_tag, u_tags = compose_compound_noise_burst(
        bank, ["vehicle_traffic", "wind_mic_rub"], 2.0, cfg.native_sr, rng, cfg
    )
    assert len(burst) == 2 * cfg.native_sr
    assert any("wind_mic_rub" in t for t in u_tags)

    # Test all 7 background voice acoustic modes
    dummy_speech = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, cfg.native_sr, dtype=np.float32))
    for mode in (
        "far_field",
        "near_field_loud",
        "bystander_backchannel",
        "bystander_laugh_cough",
        "bot_echo_leak",
        "tv_radio_playback",
        "cocktail_dual_talker",
    ):
        wet = apply_bgv_acoustics(
            dummy_speech, cfg.native_sr, 0.50, rng, cfg, mode=mode, speech2_norm=dummy_speech
        )
        assert len(wet) >= len(dummy_speech)
        assert np.isfinite(wet).all()


def test_end_to_end_call_composition_with_cached_wav():
    cfg = PipelineConfig()
    bank = load_noise_bank(cfg.noise_stems_dir, target_sr=cfg.native_sr)
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp)
        sc = ScenarioSpec(
            cid="unit_test_call",
            title="Unit Test Call",
            pilot_category="vehicle_traffic",
            language="Hindi",
            primary_voice="Kore",
            ambient_categories=["vehicle_traffic"],
            ambient_level_db=-20.0,
            rt60_s=0.45,
            turns=[
                CallTurnSpec(
                    text="Test turn one",
                    voice_mode="overlap_noise",
                    prosody_style="angry_shout",
                    channel_quirk="mic_clipping",
                    primary_gain_db=4.0,
                    overlap_noise_cats=["vehicle_traffic"],
                    pause_noise_cats=["wind_mic_rub"],
                    pause_bgv_text="Test background turn",
                    pause_bgv_voice="Aoede",
                    pause_bgv_acoustics="bot_echo_leak",
                )
            ],
        )
        # Seed synthetic 24kHz WAVs in cache
        t = np.linspace(0, 1.2, 24000, dtype=np.float32)
        tone = (0.4 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        for v, txt in [("Kore", "Test turn one"), ("Aoede", "Test background turn")]:
            p = tts_cache_path(cache_dir, cfg.tts_model_id, v, txt)
            sf.write(str(p), tone, 24000, subtype="PCM_16")

        res = build_scenario_call(sc, bank, cache_dir, cfg, seed=123)
        gt = res["gt"]
        assert abs(gt["checks"]["resample_lag_ms"]) < 2.0
        labels = [s["label"] for s in gt["segments"]]
        assert "voice" in labels
        assert "noise" in labels
        assert "background_voice" in labels
