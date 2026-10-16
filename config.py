"""Centralized Configuration File for the Synthetic Voice & Multi-Hazard Audio Data Pipeline.

All authentication, TTS model selection, LLM scenario generation & TTS director system prompts,
target conversation duration range (`min_conversation_duration_s` to `max_conversation_duration_s`),
sample rate, acoustic Room Impulse Response (RIR), prosody/emotion variations, microphone/telephony
channel quirks, background voice modes, and noise mixing variables live here and can be overridden
via YAML/JSON (`configs/pipeline_config.yaml`) or environment variables.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parent


# ==============================================================================
# 1. VERTEX AI & GEMINI 3.8 FLASH TTS AUTHENTICATION VARIABLES
# ==============================================================================
# Set USE_VERTEX_AI = True to authenticate via Google Cloud Vertex AI
# (Service Account JSON key file, GOOGLE_APPLICATION_CREDENTIALS, or gcloud ADC).
# Set USE_VERTEX_AI = False if authenticating with a GEMINI_API_KEY / GOOGLE_API_KEY.
USE_VERTEX_AI: bool = os.getenv("USE_VERTEX_AI", "true").lower() in ("1", "true", "yes")
GOOGLE_CLOUD_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "YOUR_GCP_PROJECT_ID")
GOOGLE_CLOUD_LOCATION: str = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
SERVICE_ACCOUNT_KEY_PATH: str = os.getenv(
    "SERVICE_ACCOUNT_KEY_PATH",
    os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""),
)
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))

# Primary TTS Model: Gemini 3.8 Flash TTS (`gemini-3.8-flash-tts`)
TTS_MODEL_ID: str = os.getenv("TTS_MODEL_ID", "gemini-3.8-flash-tts")
FALLBACK_TTS_MODELS: list[str] = [
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts",
]

# LLM Model used when dynamically generating new multi-turn conversation scripts
SCENARIO_GENERATOR_MODEL_ID: str = os.getenv("SCENARIO_GENERATOR_MODEL_ID", "gemini-3.8-flash")
FALLBACK_SCENARIO_MODELS: list[str] = [
    "gemini-3.5-flash",
    "gemini-2.5-flash",
]

# Concurrency & Retry Settings for TTS Generation
TTS_MAX_WORKERS: int = int(os.getenv("TTS_MAX_WORKERS", "3"))
TTS_MAX_RETRIES: int = int(os.getenv("TTS_MAX_RETRIES", "6"))
TTS_RETRY_BASE_DELAY_S: float = 2.0
TTS_HTTP_TIMEOUT_MS: int = 90_000


# ==============================================================================
# 2. SYSTEM PROMPTS (SCENARIO SCRIPT GENERATOR & TTS ACOUSTIC DIRECTOR)
# ==============================================================================
# System Prompt 1: Guides the LLM when generating realistic multi-turn voice conversation
# scripts with embedded prosody styles, micro-utterances, paralinguistic cues, competing
# background voices, and telephony channel quirks that fit the configured duration range.
SCENARIO_GENERATION_SYSTEM_PROMPT: str = """You are an expert conversational audio dataset architect specializing in realistic Indian and enterprise customer-support telephony conversations.

Your objective is to generate natural, domain-authentic multi-turn voice conversation scripts designed to stress-test Voice Activity Detection (VAD), noise-cancellation / voice-isolation models, and conversational AI agents across all real-world variations:
1. Primary Caller Speech Variations:
   - Mix full multi-clause explanations with ultra-short 1-word micro-utterances (`prosody_style: "short_micro"`, 0.15s–0.45s, e.g., "हाँ जी", "हूँ", "अच्छा", "बिल्कुल", "Yes", "Okay", "ஆமா").
   - Include emotional and cadence variations where realistic: `angry_shout` (frustrated customer with +5.5 dB gain & `mic_clipping`), `fast_panicked` (1.20x urgent rate), `slow_hesitant` (0.85x soft/elderly speaker at -4.0 dB), `laugh_chuckle` (mid-turn chuckle), and `hesitation_stutter` (stuttered thought with >350ms mid-sentence pauses).
   - Include hardware/channel quirks (`channel_quirk: "mic_clipping"` or `"packet_loss"`).
2. Competing Background Human Voices (`should_trigger: false`):
   - Place realistic secondary speech both overlapping the caller (`overlap_bgv_text`) and during caller listening pauses (`pause_bgv_text`) across all 7 acoustic modes:
     * `far_field`: Family member or coworker talking across the room.
     * `near_field_loud`: Bystander standing right next to the caller (`SIR -3 to +3 dB`).
     * `bystander_backchannel`: Brief 1-2 word bystander reaction in silence ("अच्छा भाई!", "हाँ सही है!").
     * `bystander_laugh_cough`: Close-mic cough, throat-clear, sneeze, or laugh in silence.
     * `bot_echo_leak`: Speakerphone AI agent prompt leaking back into the microphone (`SIR 14–16 dB`).
     * `tv_radio_playback`: TV news or public announcement playing in the room.
     * `cocktail_dual_talker`: Two simultaneous competing talkers (`pause_bgv_text` + `pause_bgv2_text`).
3. Multi-Hazard Background Noise Bursts (`should_trigger: false`):
   - Layer 2 to 3 simultaneous noise categories per turn/pause from: `vehicle_traffic`, `animal`, `baby_child`, `human_non_speech`, `phone_signal_alarm`, `appliance_machine`, `singing_music`, and `wind_mic_rub`.
4. Target Duration Compliance:
   - Pace the number of turns and listening windows so the rendered conversation audio lands strictly within the configured target duration range (`min_conversation_duration_s` to `max_conversation_duration_s`)."""

# System Prompt 2: Directs `gemini-3.8-flash-tts` on vocal delivery, natural Indian/regional
# inflection, conversational pacing, and emotional authenticity for each prosody style.
TTS_DIRECTOR_SYSTEM_PROMPT: str = """You are a native multilingual voice actor recording a real-world telephone call.
Deliver the spoken text with natural, unscripted human conversational cadence, authentic native pronunciation for the script's language, and realistic breath/syllable phrasing. Do not read punctuation marks aloud or add extra words outside the provided transcript."""

# Per-Prosody Style Vocal Delivery Prompts (prepended when `enable_tts_style_prompting: true`)
TTS_PROSODY_STYLE_INSTRUCTIONS: dict[str, str] = {
    "normal": "Speak in a natural, conversational phone-call tone at a steady pace.",
    "short_micro": "Speak as a quick, crisp, natural single-word phone acknowledgment.",
    "angry_shout": "Speak with high urgency, sharp frustration, and loud projected vocal energy.",
    "fast_panicked": "Speak rapidly with anxious, hurried urgency as if racing against a deadline.",
    "slow_hesitant": "Speak slowly, softly, and tentatively with gentle elderly/hesitant cadence.",
    "laugh_chuckle": "Deliver with a warm, natural chuckle woven into the start of the phrase.",
    "hesitation_stutter": "Speak with natural mid-thought hesitation, pausing briefly as if recalling a detail.",
}

# Set True to prepend `TTS_DIRECTOR_SYSTEM_PROMPT` / `TTS_PROSODY_STYLE_INSTRUCTIONS` to TTS requests,
# or False to send the raw utterance text directly (default False preserves exact cached hashes).
ENABLE_TTS_STYLE_PROMPTING: bool = False


# ==============================================================================
# 3. CONVERSATION DURATION RANGE & TIMING GUARD VARIABLES
# ==============================================================================
# Target duration range (in seconds) for each generated voice conversation:
#   - Default Showcase / Short Call Range: `35.0s` to `60.0s` (avg ~`45.0s`)
#   - Standard Production Call Range: set e.g. `120.0s` to `240.0s` (2 to 4 minutes, avg ~`180.0s`)
MIN_CONVERSATION_DURATION_S: float = float(os.getenv("MIN_CONVERSATION_DURATION_S", "35.0"))
MAX_CONVERSATION_DURATION_S: float = float(os.getenv("MAX_CONVERSATION_DURATION_S", "60.0"))
TARGET_AVG_CONVERSATION_DURATION_S: float = float(
    os.getenv("TARGET_AVG_CONVERSATION_DURATION_S", "45.0")
)

# Duration range for each pure multi-noise listening window (`pause_noise`) between turns
PAUSE_NOISE_BURST_MIN_S: float = 2.40
PAUSE_NOISE_BURST_MAX_S: float = 3.60

# Scene timing guards between user speech turns, noise bursts, and bystander voice windows
INITIAL_AMBIENT_LEAD_IN_S: float = 1.80
FINAL_AMBIENT_TAIL_S: float = 1.50
POST_VOICE_GUARD_S: float = 0.750
INTER_EVENT_GUARD_S: float = 0.650
PRE_VOICE_GUARD_S: float = 0.650


# ==============================================================================
# 4. AUDIO SAMPLE RATES, LEVELS & GROUND-TRUTH EXTRACTION VARIABLES
# ==============================================================================
NATIVE_SAMPLE_RATE_HZ: int = 16000
TELEPHONY_SAMPLE_RATE_HZ: int = 8000
EXPORT_8KHZ_TELEPHONY: bool = True
EXPORT_16KHZ_NATIVE: bool = True
EXPORT_ISOLATED_STEMS: bool = True  # Saves isolated primary/bgv/noise/ambient WAVs

# Nominal RMS for primary user speech (-19.2 dBFS) and peak limiter ceiling
TARGET_PRIMARY_VOICE_RMS: float = 0.11
PEAK_LIMITER_CEILING: float = 0.96

# Ground-truth syllable/word island extraction parameters
VOICE_MERGE_GAP_S: float = 0.200  # Pauses < 200 ms merge into a single continuous voice segment
AUDIO_TRIM_MERGE_GAP_S: float = 0.340
DEFAULT_MIN_ISLAND_DURATION_S: float = 0.120
MICRO_UTTERANCE_MIN_ISLAND_DURATION_S: float = 0.050
OUTER_TRIM_PAD_S: float = 0.020
EDGE_FADE_DURATION_S: float = 0.015

# Supported Prebuilt Voices in Gemini 3.8 Flash TTS
AVAILABLE_VOICES: list[str] = [
    "Kore",
    "Charon",
    "Fenrir",
    "Puck",
    "Leda",
    "Orus",
    "Zephyr",
    "Aoede",
    "Callirrhoe",
    "Alnilam",
    "Algenib",
    "Rasalgethi",
    "Schedar",
    "Achird",
    "Gacrux",
    "Pulcherrima",
    "Sulafat",
    "Vindemiatrix",
]

# Supported 24 Indian & South Asian Languages (auto-detected from script/text by Gemini 3.8 Flash TTS)
SUPPORTED_LANGUAGES: list[str] = [
    "Hindi",
    "Indian English",
    "Hinglish",
    "Bengali",
    "Telugu",
    "Marathi",
    "Tamil",
    "Urdu",
    "Gujarati",
    "Kannada",
    "Malayalam",
    "Odia",
    "Punjabi",
    "Assamese",
    "Maithili",
    "Nepali",
    "Santali",
    "Kashmiri",
    "Sindhi",
    "Manipuri",
    "Bhojpuri",
    "Awadhi",
    "Chhattisgarhi",
    "Magahi",
    "Mizo",
]


# ==============================================================================
# 5. PROSODY, EMOTION & SPEAKING-RATE VARIATION VARIABLES
# ==============================================================================
PROSODY_SPEED_MULTIPLIERS: dict[str, float] = {
    "normal": 1.00,
    "short_micro": 1.65,        # Sub-500ms 1-word backchannels ("Haan", "Hmm", "Jee", "Yes")
    "fast_panicked": 1.20,      # Rapid urgent customer speech
    "slow_hesitant": 0.85,      # Slow / elderly speaker cadence
    "angry_shout": 1.05,        # Loud frustrated speech (often paired with +5.5 dB gain & mic clipping)
    "laugh_chuckle": 1.00,      # Mid-turn chuckle/laughter
    "hesitation_stutter": 1.00, # Stuttered speech with >350ms mid-sentence silence valleys
}

HESITATION_PAUSE_INJECTION_S: float = 0.380  # Inserted silence valley duration for hesitation_stutter


# ==============================================================================
# 6. MICROPHONE & TELEPHONY CHANNEL QUIRK VARIABLES
# ==============================================================================
MIC_CLIPPING_TANH_DRIVE: float = 2.80         # Nonlinear saturation factor for angry/shouted turns
PACKET_LOSS_DROPOUT_DURATION_S: float = 0.042 # 42 ms RTP VoIP frame zeroing
PACKET_LOSS_FRACTION_POSITIONS: list[float] = [0.34, 0.68]

WIND_BUFFETING_BANDPASS_HZ: list[float] = [22.0, 210.0]
WIND_BUFFETING_TANH_DRIVE: float = 2.20


# ==============================================================================
# 7. BACKGROUND VOICE & ROOM IMPULSE RESPONSE (RIR) ACOUSTIC VARIABLES
# ==============================================================================
ACOUSTIC_MODES_CONFIG: dict[str, dict[str, Any]] = {
    "far_field": {
        "description": "Distant family member / coworker in same room with room reverb and air absorption",
        "direct_gain": 0.65,
        "lowpass_hz": 2600.0,
    },
    "near_field_loud": {
        "description": "Bystander standing right next to the caller's phone (high direct path, wide bandwidth)",
        "rt60_override_s": 0.16,
        "direct_gain": 1.35,
        "lowpass_hz": 3500.0,
    },
    "bystander_backchannel": {
        "description": "Short 1-2 word bystander interjection during caller silence",
        "rt60_override_s": 0.16,
        "direct_gain": 1.35,
        "lowpass_hz": 3500.0,
    },
    "bystander_laugh_cough": {
        "description": "Close-mic or room cough, throat-clear, sneeze, or bystander laugh",
        "rt60_override_s": 0.20,
        "direct_gain": 1.25,
        "lowpass_hz": 3600.0,
    },
    "bot_echo_leak": {
        "description": "Speakerphone AI bot voice leaking back into the mic (bandpass + nonlinear saturation + RIR)",
        "bandpass_hz": [400.0, 3200.0],
        "tanh_drive": 1.85,
        "direct_gain": 0.95,
    },
    "tv_radio_playback": {
        "description": "Television, radio, or public announcement speaker in the background",
        "bandpass_hz": [320.0, 3300.0],
        "tanh_drive": 1.60,
        "direct_gain": 0.80,
    },
    "cocktail_dual_talker": {
        "description": "Two simultaneous competing background speakers staggered by 250ms",
        "stagger_s": 0.25,
        "secondary_speaker_weight": 0.92,
    },
}


# ==============================================================================
# 8. NOISE CATEGORIES & DIRECTORY PATHS
# ==============================================================================
NOISE_CATEGORIES: list[str] = [
    "vehicle_traffic",
    "animal",
    "baby_child",
    "human_non_speech",
    "phone_signal_alarm",
    "appliance_machine",
    "singing_music",
    "wind_mic_rub",  # Procedural low-frequency outdoor wind buffeting & mic handling noise
]

DEFAULT_NOISE_STEMS_DIR: str = str(ROOT_DIR / "assets" / "noise_stems")
DEFAULT_TTS_CACHE_DIR: str = str(ROOT_DIR / "cache" / "tts_cache")
DEFAULT_OUTPUT_DIR: str = str(ROOT_DIR / "output_dataset")
DEFAULT_RANDOM_SEED: int = 20261016


@dataclass
class PipelineConfig:
    """Structured runtime configuration loaded from defaults, YAML/JSON file, and CLI overrides."""

    # Auth & TTS / Scenario LLM Models
    use_vertex_ai: bool = USE_VERTEX_AI
    project_id: str = GOOGLE_CLOUD_PROJECT
    location: str = GOOGLE_CLOUD_LOCATION
    service_account_key_path: str = SERVICE_ACCOUNT_KEY_PATH
    gemini_api_key: str = GEMINI_API_KEY
    tts_model_id: str = TTS_MODEL_ID
    fallback_tts_models: list[str] = field(default_factory=lambda: list(FALLBACK_TTS_MODELS))
    scenario_generator_model_id: str = SCENARIO_GENERATOR_MODEL_ID
    fallback_scenario_models: list[str] = field(
        default_factory=lambda: list(FALLBACK_SCENARIO_MODELS)
    )
    tts_max_workers: int = TTS_MAX_WORKERS
    tts_max_retries: int = TTS_MAX_RETRIES
    tts_retry_base_delay_s: float = TTS_RETRY_BASE_DELAY_S
    tts_http_timeout_ms: int = TTS_HTTP_TIMEOUT_MS

    # System Prompts & Style Prompting
    scenario_generation_system_prompt: str = SCENARIO_GENERATION_SYSTEM_PROMPT
    tts_director_system_prompt: str = TTS_DIRECTOR_SYSTEM_PROMPT
    tts_prosody_style_instructions: dict[str, str] = field(
        default_factory=lambda: dict(TTS_PROSODY_STYLE_INSTRUCTIONS)
    )
    enable_tts_style_prompting: bool = ENABLE_TTS_STYLE_PROMPTING

    # Target Conversation Duration Range (Min / Max / Avg in seconds)
    min_conversation_duration_s: float = MIN_CONVERSATION_DURATION_S
    max_conversation_duration_s: float = MAX_CONVERSATION_DURATION_S
    target_avg_conversation_duration_s: float = TARGET_AVG_CONVERSATION_DURATION_S
    pause_noise_burst_min_s: float = PAUSE_NOISE_BURST_MIN_S
    pause_noise_burst_max_s: float = PAUSE_NOISE_BURST_MAX_S

    # Audio Rates & Output Flags
    native_sr: int = NATIVE_SAMPLE_RATE_HZ
    telephony_sr: int = TELEPHONY_SAMPLE_RATE_HZ
    export_8khz_telephony: bool = EXPORT_8KHZ_TELEPHONY
    export_16khz_native: bool = EXPORT_16KHZ_NATIVE
    export_isolated_stems: bool = EXPORT_ISOLATED_STEMS

    # Levels & Guards
    target_voice_rms: float = TARGET_PRIMARY_VOICE_RMS
    peak_ceiling: float = PEAK_LIMITER_CEILING
    initial_lead_in_s: float = INITIAL_AMBIENT_LEAD_IN_S
    final_tail_s: float = FINAL_AMBIENT_TAIL_S
    post_voice_guard_s: float = POST_VOICE_GUARD_S
    inter_event_guard_s: float = INTER_EVENT_GUARD_S
    pre_voice_guard_s: float = PRE_VOICE_GUARD_S

    # Ground-truth & Island Detection
    voice_merge_gap_s: float = VOICE_MERGE_GAP_S
    audio_trim_merge_gap_s: float = AUDIO_TRIM_MERGE_GAP_S
    default_min_island_s: float = DEFAULT_MIN_ISLAND_DURATION_S
    micro_min_island_s: float = MICRO_UTTERANCE_MIN_ISLAND_DURATION_S

    # Prosody, Quirks & Acoustics
    prosody_speed_multipliers: dict[str, float] = field(
        default_factory=lambda: dict(PROSODY_SPEED_MULTIPLIERS)
    )
    hesitation_pause_s: float = HESITATION_PAUSE_INJECTION_S
    mic_clipping_drive: float = MIC_CLIPPING_TANH_DRIVE
    packet_loss_dropout_s: float = PACKET_LOSS_DROPOUT_DURATION_S
    packet_loss_positions: list[float] = field(
        default_factory=lambda: list(PACKET_LOSS_FRACTION_POSITIONS)
    )
    wind_bandpass_hz: list[float] = field(default_factory=lambda: list(WIND_BUFFETING_BANDPASS_HZ))
    wind_tanh_drive: float = WIND_BUFFETING_TANH_DRIVE
    acoustic_modes: dict[str, dict[str, Any]] = field(
        default_factory=lambda: dict(ACOUSTIC_MODES_CONFIG)
    )

    # Paths & Seed
    noise_stems_dir: str = DEFAULT_NOISE_STEMS_DIR
    tts_cache_dir: str = DEFAULT_TTS_CACHE_DIR
    output_dir: str = DEFAULT_OUTPUT_DIR
    seed: int = DEFAULT_RANDOM_SEED

    @property
    def min_call_duration_s(self) -> float:
        """Backwards-compatible alias for `min_conversation_duration_s`."""
        return self.min_conversation_duration_s

    @classmethod
    def from_file(cls, path: str | Path | None = None) -> "PipelineConfig":
        """Load configuration from a YAML or JSON file, merging with defaults and env vars."""
        cfg = cls()
        if path is None:
            return cfg
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Configuration file not found: {p}")
        raw_text = p.read_text(encoding="utf-8")
        data = json.loads(raw_text) if p.suffix.lower() == ".json" else yaml.safe_load(raw_text)
        if not isinstance(data, dict):
            return cfg

        # Flatten nested YAML sections if present
        flat: dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, dict) and k in (
                "auth",
                "tts",
                "prompts",
                "system_prompts",
                "conversation_duration",
                "audio",
                "timing",
                "ground_truth",
                "variations",
                "paths",
            ):
                flat.update(v)
            else:
                flat[k] = v

        # Support legacy `min_call_duration_s` key if present in YAML
        if "min_call_duration_s" in flat and "min_conversation_duration_s" not in flat:
            flat["min_conversation_duration_s"] = flat["min_call_duration_s"]

        valid_fields = set(asdict(cfg).keys())
        for k, v in flat.items():
            if k in valid_fields and v is not None:
                setattr(cfg, k, v)

        # Environment variable overrides take highest priority when explicitly set
        if os.getenv("GOOGLE_CLOUD_PROJECT"):
            cfg.project_id = os.environ["GOOGLE_CLOUD_PROJECT"]
        if os.getenv("GOOGLE_CLOUD_LOCATION"):
            cfg.location = os.environ["GOOGLE_CLOUD_LOCATION"]
        if os.getenv("SERVICE_ACCOUNT_KEY_PATH") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            cfg.service_account_key_path = os.getenv(
                "SERVICE_ACCOUNT_KEY_PATH", os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
            )
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            cfg.gemini_api_key = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
        if os.getenv("USE_VERTEX_AI"):
            cfg.use_vertex_ai = os.environ["USE_VERTEX_AI"].lower() in ("1", "true", "yes")
        if os.getenv("MIN_CONVERSATION_DURATION_S"):
            cfg.min_conversation_duration_s = float(os.environ["MIN_CONVERSATION_DURATION_S"])
        if os.getenv("MAX_CONVERSATION_DURATION_S"):
            cfg.max_conversation_duration_s = float(os.environ["MAX_CONVERSATION_DURATION_S"])
        if os.getenv("TARGET_AVG_CONVERSATION_DURATION_S"):
            cfg.target_avg_conversation_duration_s = float(
                os.environ["TARGET_AVG_CONVERSATION_DURATION_S"]
            )
        return cfg

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d.get("gemini_api_key"):
            d["gemini_api_key"] = "***REDACTED***"
        return d
