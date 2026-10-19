# Synthetic Voice & Multi-Hazard Audio Data Creation Pipeline

A standalone, config-driven pipeline for generating realistic multi-stem conversational and telephony audio datasets with exact sub-10ms ground-truth timestamps using **Gemini 3.8 Flash TTS (`gemini-3.8-flash-tts`)** on **Google Cloud Vertex AI** (or Gemini API).

Designed to stress-test Voice Activity Detection (VAD), noise-cancellation / voice-isolation front-ends, speaker diarization, and conversational AI agents across **all real-world acoustic, prosodic, emotional, paralinguistic, and telephony channel variations**.

---

## Architecture: 4-Stem Synchronized Audio Composition

Every generated call is composed from **4 independent 16 kHz stems** before zero-phase polyphase downsampling (`scipy.signal.resample_poly`) to **8 kHz mono `int16` telephony PCM**:

```
┌────────────────────────────────────────────────────────────────────────────┐
│ 1. Primary Caller Stem (`label="voice"`, `should_trigger=True`)            │
│    • Gemini 3.8 Flash TTS (`gemini-3.8-flash-tts`) across 24 languages     │
│    • Prosody & speed variations: normal, short_micro (0.15–0.45s),         │
│      angry_shout (+5.5 dB), fast_panicked (1.20x), slow_hesitant (0.85x),  │
│      laugh_chuckle, hesitation_stutter (>350ms mid-sentence silence)       │
│    • Hardware/channel quirks: nonlinear `mic_clipping` (`tanh` saturation) │
│      & VoIP RTP `packet_loss` (42ms zeroed frame dropouts)                 │
├────────────────────────────────────────────────────────────────────────────┤
│ 2. Competing Background Voice Stem (`label="background_voice"`, `False`)   │
│    • Secondary TTS speakers convolved with synthetic Room Impulse Response │
│    • 7 Acoustic Modes: `far_field`, `near_field_loud`, `tv_radio_playback`,│
│      `cocktail_dual_talker` (2 simultaneous talkers), `bot_echo_leak`      │
│      (speakerphone AI bot residual), `bystander_backchannel`, and          │
│      `bystander_laugh_cough` at controlled SIR (-3 dB to +16 dB)           │
├────────────────────────────────────────────────────────────────────────────┤
│ 3. Compound Multi-Noise Burst Stem (`label="noise"`, `should_trigger=False`)│
│    • 2–4 simultaneous staggered noise events across 8 categories:          │
│      `vehicle_traffic`, `animal`, `baby_child`, `human_non_speech`,        │
│      `phone_signal_alarm`, `appliance_machine`, `singing_music`, plus      │
│      procedural outdoor `wind_mic_rub` (22–210 Hz gusts)                   │
├────────────────────────────────────────────────────────────────────────────┤
│ 4. Continuous Ambient Bed Stem                                             │
│    • Crossfaded low-pass room/street ambiance across the full call         │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
       ┌──────────────────────────────────────────────────────────────┐
       │ Outputs (`output_dataset/`)                                  │
       │  ├── audio_16k/<id>.wav      (16 kHz wideband PCM_16)        │
       │  ├── audio_8k/<id>.wav       (8 kHz telephony PCM_16)        │
       │  ├── stems/<id>_*.wav        (4 isolated stems per call)     │
       │  ├── ground_truth/<id>.json  (Exact sub-10ms segments & gaps)│
       │  ├── manifest.jsonl          (Dataset index)                 │
       │  └── data_summary.md         (Markdown overview table)       │
       └──────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```text
synthetic-voice-data-pipeline/
├── config.py                        # Central Python config with all pipeline variables
├── configs/
│   └── pipeline_config.yaml         # YAML config file overriding any variable
├── generate_dataset.py              # Main CLI entrypoint
├── src/
│   ├── tts_client.py                # Vertex AI & Gemini 3.8 Flash TTS client + SHA-256 disk cache
│   ├── acoustics.py                 # RIR convolution, island extraction, wind gusts, polyphase 8kHz
│   ├── scenarios.py                 # Scenario/Turn dataclasses + 5 multi-variation showcase presets
│   └── composer.py                  # 4-stem mixer, ground-truth builder & dataset writer
├── assets/
│   └── noise_stems/                 # 42 bundled speech-free CC-BY-4.0 noise stems across 7 categories
├── examples/
│   └── custom_scenarios.json        # Example JSON template for custom customer call scripts
├── tests/
│   └── test_pipeline.py             # Automated unit & integration tests
└── requirements.txt
```

---

## Quick Start

### 1. Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Vertex AI or Gemini API Authentication

You can configure credentials in [`configs/pipeline_config.yaml`](configs/pipeline_config.yaml) or via environment variables:

**Option A — Google Cloud Vertex AI (Recommended for Enterprise):**
```bash
export USE_VERTEX_AI=true
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
# Either point to a Service Account JSON key file OR use `gcloud auth application-default login`:
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

**Option B — Gemini API Key:**
```bash
export USE_VERTEX_AI=false
export GEMINI_API_KEY="your-gemini-api-key"
```

### 3. Run the Pipeline

```bash
# Generate the 5 built-in multi-variation showcase calls:
python generate_dataset.py --config configs/pipeline_config.yaml --preset showcase

# Override the target conversation duration range (e.g., 35s to 60s, target average 45s):
python generate_dataset.py --config configs/pipeline_config.yaml --preset showcase \
  --min-duration 35 --max-duration 60 --target-avg-duration 45

# Dynamically generate N new multi-turn scenarios via LLM using SCENARIO_GENERATION_SYSTEM_PROMPT:
python generate_dataset.py --config configs/pipeline_config.yaml --generate-scenarios 3 \
  --min-duration 35 --max-duration 60

# Or run your own custom scenario scripts from JSON:
python generate_dataset.py --config configs/pipeline_config.yaml --scenarios-file examples/custom_scenarios.json

# Inspect all active configuration variables:
python generate_dataset.py --config configs/pipeline_config.yaml --dump-config
```

---

## Central Configuration: System Prompts & Conversation Duration Range

All variables are exposed in both [`config.py`](config.py) and [`configs/pipeline_config.yaml`](configs/pipeline_config.yaml):

1. **`system_prompts`**:
   - `scenario_generation_system_prompt`: System prompt instructing the default LLM (**`gemini-3.8-flash`**) how to author multi-turn telephony scripts across all 24 languages, prosodic variations, hardware quirks, and competing background speakers.
   - `tts_director_system_prompt` & `tts_prosody_style_instructions`: Expressive voice-acting instructions (`normal`, `short_micro`, `angry_shout`, `fast_panicked`, `slow_hesitant`, `laugh_chuckle`, `hesitation_stutter`) optionally prepended to `gemini-3.8-flash-tts` requests when `enable_tts_style_prompting: true`.
2. **`conversation_duration`**:
   - `min_conversation_duration_s` (default: `35.0` seconds)
   - `max_conversation_duration_s` (default: `60.0` seconds)
   - `target_avg_conversation_duration_s` (default: `45.0` seconds)
   - `pause_noise_burst_min_s` / `pause_noise_burst_max_s` (`2.4s` – `3.6s`)
   - Every generated call is automatically padded or trimmed so its total duration stays within `[min_conversation_duration_s, max_conversation_duration_s]` and records `target_duration_range_s` in `ground_truth/<id>.json` and `data_summary.md`.

---

## Bundled Example VAD Test Dataset (`output_dataset/`) & Ground-Truth Timestamps

This repository includes **5 ready-to-use multi-variation test conversations (`215.0 seconds / 3.58 minutes` total)** in [`output_dataset/`](output_dataset/) at both **16 kHz wideband** ([`output_dataset/audio_16k/`](output_dataset/audio_16k/)) and **8 kHz telephony** ([`output_dataset/audio_8k/`](output_dataset/audio_8k/)) along with exact sub-10ms ground-truth JSON annotations showing **`start_audio_sec`** and **`end_audio_sec`** for every `voice` (`should_trigger: true`), `noise` (`should_trigger: false`), and `background_voice` (`should_trigger: false`) segment:

- **Summary & Per-Call Segment Timelines**: [`output_dataset/data_summary.md`](output_dataset/data_summary.md)
- **Scenarios with Turn & Segment Timestamps (`start_audio_sec` / `end_audio_sec`)**: [`output_dataset/scenarios_with_timestamps.json`](output_dataset/scenarios_with_timestamps.json) (and [`examples/custom_scenarios.json`](examples/custom_scenarios.json))
- **Per-Call Ground-Truth JSONs**:
  - [`output_dataset/ground_truth/example_01_hindi_micro_bot_echo_backchannels.json`](output_dataset/ground_truth/example_01_hindi_micro_bot_echo_backchannels.json) (`49.6s`, Hindi)
  - [`output_dataset/ground_truth/example_02_english_emotion_clipping_packet_loss.json`](output_dataset/ground_truth/example_02_english_emotion_clipping_packet_loss.json) (`46.4s`, Indian English)
  - [`output_dataset/ground_truth/example_03_hinglish_wind_soft_laugh_cough.json`](output_dataset/ground_truth/example_03_hinglish_wind_soft_laugh_cough.json) (`43.2s`, Hinglish)
  - [`output_dataset/ground_truth/example_04_hindi_home_tv_kids_music.json`](output_dataset/ground_truth/example_04_hindi_home_tv_kids_music.json) (`38.7s`, Hindi)
  - [`output_dataset/ground_truth/example_05_tamil_multilingual_variations.json`](output_dataset/ground_truth/example_05_tamil_multilingual_variations.json) (`37.0s`, Tamil)

---

## Supported Variations Matrix

| Variation Dimension | Config / Turn Key | Supported Values |
| :--- | :--- | :--- |
| **Primary Speech Mode** | `voice_mode` | `clean`, `overlap_noise`, `overlap_bgv_and_noise` |
| **Prosody & Emotion Style** | `prosody_style` | `normal`, `short_micro` (`1.65x`, `0.15–0.45s`), `angry_shout`, `fast_panicked` (`1.20x`), `slow_hesitant` (`0.85x`), `laugh_chuckle`, `hesitation_stutter` (`>350ms` internal pauses) |
| **Caller Loudness** | `primary_gain_db` | Any dB offset (e.g. `-4.5 dB` whispered/soft speech to `+5.5 dB` shouted speech) |
| **Hardware & Channel Quirks** | `channel_quirk` | `none`, `mic_clipping` (nonlinear `tanh` saturation), `packet_loss` (`42ms` RTP VoIP dropouts) |
| **Background Voice Acoustics** | `pause_bgv_acoustics` | `far_field`, `near_field_loud`, `tv_radio_playback`, `cocktail_dual_talker`, `bot_echo_leak`, `bystander_backchannel`, `bystander_laugh_cough` |
| **Background Noise Categories** | `pause_noise_cats` / `overlap_noise_cats` | `vehicle_traffic`, `animal`, `baby_child`, `human_non_speech`, `phone_signal_alarm`, `appliance_machine`, `singing_music`, `wind_mic_rub` |
| **24 Supported Languages** | `language` | Hindi, Indian English, Hinglish, Bengali, Telugu, Marathi, Tamil, Urdu, Gujarati, Kannada, Malayalam, Odia, Punjabi, Assamese, Maithili, Nepali, Santali, Kashmiri, Sindhi, Manipuri, Bhojpuri, Awadhi, Chhattisgarhi, Magahi, Mizo |
