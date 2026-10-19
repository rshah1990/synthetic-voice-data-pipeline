#!/usr/bin/env python3
"""Main CLI Entrypoint for the Synthetic Voice & Multi-Hazard Audio Data Creation Pipeline.

Usage:
  # 1. Generate the 5 built-in multi-variation showcase examples:
  python generate_dataset.py --config configs/pipeline_config.yaml --preset showcase

  # 2. Generate only specific scenario IDs:
  python generate_dataset.py --config configs/pipeline_config.yaml --scenarios example_01_hindi_micro_bot_echo_backchannels

  # 3. Generate custom scenarios from a JSON file:
  python generate_dataset.py --config configs/pipeline_config.yaml --scenarios-file examples/custom_scenarios.json

  # 4. Dynamically generate new scenarios via LLM using the configured system prompt & conversation duration range:
  python generate_dataset.py --config configs/pipeline_config.yaml --generate-scenarios 3 --min-duration 35 --max-duration 60

  # 5. Print active configuration:
  python generate_dataset.py --config configs/pipeline_config.yaml --dump-config
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from config import PipelineConfig
from src.acoustics import load_noise_bank
from src.composer import build_scenario_call, write_dataset_outputs
from src.scenarios import (
    generate_scenarios_via_llm,
    get_core_27_scenarios,
    get_edge_10_scenarios,
    get_full_37_scenarios,
    get_quick_5_scenarios,
    get_test_showcase_scenarios,
    load_scenarios_from_json,
)
from src.tts_client import batch_synthesize_utterances


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Synthetic Voice & Multi-Hazard Audio Data Creation Pipeline (Gemini 3.8 Flash TTS)"
    )
    ap.add_argument(
        "--config",
        default="configs/pipeline_config.yaml",
        help="Path to YAML or JSON configuration file (default: configs/pipeline_config.yaml)",
    )
    ap.add_argument(
        "--preset",
        choices=["full", "showcase", "core27", "edge10", "quick5"],
        default="full",
        help="Built-in scenario preset to run: 'full' (all 37 calls, default), 'core27' (27 24-lang calls), 'edge10' (10 edge-case calls), 'quick5' (5 short clips)",
    )
    ap.add_argument(
        "--scenarios-file",
        default="",
        help="Optional path to custom scenarios JSON file (e.g. examples/custom_scenarios.json)",
    )
    ap.add_argument(
        "--scenarios",
        nargs="*",
        default=[],
        help="Optional filter of scenario IDs (cid) to build",
    )
    ap.add_argument("--out-dir", default="", help="Override output directory from config")
    ap.add_argument("--tts-cache", default="", help="Override TTS cache directory from config")
    ap.add_argument("--noise-dir", default="", help="Override noise stems directory from config")
    ap.add_argument("--seed", type=int, default=0, help="Override random seed from config")
    ap.add_argument(
        "--min-duration",
        type=float,
        default=-1.0,
        help="Override min_conversation_duration_s (minimum duration in seconds per voice conversation)",
    )
    ap.add_argument(
        "--max-duration",
        type=float,
        default=-1.0,
        help="Override max_conversation_duration_s (maximum duration in seconds per voice conversation)",
    )
    ap.add_argument(
        "--target-avg-duration",
        type=float,
        default=-1.0,
        help="Override target_avg_conversation_duration_s (target average duration in seconds)",
    )
    ap.add_argument(
        "--generate-scenarios",
        type=int,
        default=0,
        help="Dynamically generate N new scenarios using the LLM and SCENARIO_GENERATION_SYSTEM_PROMPT",
    )
    ap.add_argument(
        "--dump-config",
        action="store_true",
        help="Print resolved configuration JSON and exit",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config) if args.config and Path(args.config).exists() else None
    cfg = PipelineConfig.from_file(cfg_path)

    if args.out_dir:
        cfg.output_dir = args.out_dir
    if args.tts_cache:
        cfg.tts_cache_dir = args.tts_cache
    if args.noise_dir:
        cfg.noise_stems_dir = args.noise_dir
    if args.seed:
        cfg.seed = args.seed
    if args.min_duration >= 0:
        cfg.min_conversation_duration_s = args.min_duration
    if args.max_duration >= 0:
        cfg.max_conversation_duration_s = args.max_duration
    if args.target_avg_duration >= 0:
        cfg.target_avg_conversation_duration_s = args.target_avg_duration

    if args.dump_config:
        print(json.dumps(cfg.to_dict(), indent=2))
        return

    if args.generate_scenarios > 0:
        print(
            f"[LLM Scenario Generator] Generating {args.generate_scenarios} scenarios via {cfg.scenario_generator_model_id}..."
        )
        scenarios = generate_scenarios_via_llm(cfg, num_scenarios=args.generate_scenarios)
    elif args.scenarios_file:
        scenarios = load_scenarios_from_json(args.scenarios_file)
    elif args.preset == "core27":
        scenarios = get_core_27_scenarios()
    elif args.preset == "edge10":
        scenarios = get_edge_10_scenarios()
    elif args.preset == "quick5":
        scenarios = get_quick_5_scenarios()
    else:
        scenarios = get_full_37_scenarios()

    if args.scenarios:
        wanted = set(args.scenarios)
        scenarios = [s for s in scenarios if s.cid in wanted]
        if not scenarios:
            raise ValueError(f"No matching scenarios found for filter: {args.scenarios}")

    t0 = time.time()
    print("=" * 76)
    print(" Synthetic Voice & Multi-Hazard Audio Data Creation Pipeline")
    print(f" TTS Model      : {cfg.tts_model_id} (Style Prompting={cfg.enable_tts_style_prompting})")
    print(f" Scenarios      : {len(scenarios)} calls (LLM Model={cfg.scenario_generator_model_id})")
    print(
        f" Target Duration: {cfg.min_conversation_duration_s:.1f}s - {cfg.max_conversation_duration_s:.1f}s "
        f"(Target Avg: {cfg.target_avg_conversation_duration_s:.1f}s)"
    )
    print(f" Output Dir     : {cfg.output_dir}")
    print(f" Sample Rates   : 16 kHz={cfg.export_16khz_native}, 8 kHz={cfg.export_8khz_telephony}, Stems={cfg.export_isolated_stems}")
    print("=" * 76)

    print("\n[1/3] Synthesizing Primary & Competing Background Voices via Gemini 3.8 Flash TTS...")
    tts_cache_dir, tts_stats = batch_synthesize_utterances(scenarios, cfg)

    print("\n[2/3] Loading Speech-Free Multi-Category Noise Stems...")
    bank = load_noise_bank(cfg.noise_stems_dir, target_sr=cfg.native_sr)
    for cat, clips in sorted(bank.items()):
        tags = sorted({c.tag for c in clips})
        print(f"  - {cat:<20}: {len(clips)} stems ({', '.join(tags)})")

    print("\n[3/3] Composing 4-Stem Audio & Computing Sub-10ms Ground-Truth Segments...")
    out_dir = Path(cfg.output_dir)
    built_items = []
    for idx, sc in enumerate(scenarios):
        call_seed = cfg.seed + idx * 1009
        item = build_scenario_call(sc, bank, tts_cache_dir, cfg, seed=call_seed)
        built_items.append(item)
        gt = item["gt"]
        segs = gt["segments"]
        nv = sum(1 for s in segs if s["label"] == "voice")
        nn = sum(1 for s in segs if s["label"] == "noise")
        nb = sum(1 for s in segs if s["label"] == "background_voice")
        print(
            f"  [OK] {sc.cid} ({sc.language}, {gt['duration_s']:.1f}s) -> "
            f"{nv} voice, {nn} noise, {nb} background_voice segments "
            f"(lag={gt['checks']['resample_lag_ms']:.2f}ms)"
        )

    summary = write_dataset_outputs(out_dir, built_items, cfg)
    elapsed = round(time.time() - t0, 2)
    print("\n" + "=" * 76)
    print(
        f" Pipeline Complete in {elapsed}s! "
        f"Total Audio: {summary['total_duration_s']:.1f}s across {summary['total_calls']} calls "
        f"(Actual Avg: {summary['avg_duration_s']:.1f}s, Range: {summary['min_duration_s']:.1f}s - {summary['max_duration_s']:.1f}s)."
    )
    print(f" Summary Report : {out_dir / 'data_summary.md'}")
    print(f" Manifest JSONL : {out_dir / 'manifest.jsonl'}")
    print("=" * 76)


if __name__ == "__main__":
    main()
