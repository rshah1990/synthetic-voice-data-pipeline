"""Scenario & Turn Schemas plus Built-In Multi-Variation Presets for the Synthetic Voice Pipeline.

Covers all 6 orthogonal dimensions of real-world telephony & conversational audio variations:
  1. Primary User Voice Modes (`clean`, `overlap_noise`, `overlap_bgv_and_noise`, soft/whispered `-4.5 dB` to shouted `+5.5 dB`)
  2. Conversational Prosody & Speaking-Rate Styles (`normal`, `short_micro`, `angry_shout`, `fast_panicked`, `slow_hesitant`, `laugh_chuckle`, `hesitation_stutter`)
  3. Paralinguistic & Micro-Utterance Events (`0.15s–0.50s` 1-word replies, mid-turn laughs, >350ms hesitation pauses, close-mic coughs/throat-clears, bystander backchannels)
  4. Background Competing Voice Acoustic Modes (`far_field`, `near_field_loud`, `tv_radio_playback`, `cocktail_dual_talker`, `bot_echo_leak`, `bystander_backchannel`, `bystander_laugh_cough`) across SIR `-3 dB` to `+16 dB`
  5. Microphone & Telephony Hardware Quirks (`mic_clipping` nonlinear `tanh` saturation, `packet_loss` 42ms RTP VoIP frame dropouts, `wind_mic_rub` low-frequency outdoor wind buffeting)
  6. Multi-Noise Bursts across all 8 Categories (`vehicle_traffic`, `animal`, `baby_child`, `human_non_speech`, `phone_signal_alarm`, `appliance_machine`, `singing_music`, `wind_mic_rub`)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path
from typing import Any


@dataclass
class CallTurnSpec:
    text: str
    voice_mode: str = "clean"  # "clean" | "overlap_noise" | "overlap_bgv_and_noise"
    overlap_noise_cats: list[str] = field(default_factory=list)
    overlap_snr_db: float = 8.0
    overlap_bgv_text: str = ""
    overlap_bgv_voice: str = "Fenrir"
    overlap_sir_db: float = 10.0
    primary_gain_db: float = 0.0  # e.g. -4.5 dB for soft/whispered speech, +5.5 dB for angry shout
    # "normal" | "short_micro" | "angry_shout" | "fast_panicked" | "slow_hesitant" | "laugh_chuckle" | "hesitation_stutter"
    prosody_style: str = "normal"
    # "none" | "mic_clipping" | "packet_loss"
    channel_quirk: str = "none"
    # Listening window after this user turn (when primary caller is silent, should_trigger=False)
    pause_noise_cats: list[str] = field(default_factory=list)
    pause_noise_snr_db: float = 6.0
    pause_bgv_text: str = ""
    pause_bgv_voice: str = "Aoede"
    pause_bgv_sir_db: float = 10.0
    pause_bgv_with_noise_cats: list[str] = field(default_factory=list)
    # "far_field" | "near_field_loud" | "tv_radio_playback" | "cocktail_dual_talker"
    # | "bot_echo_leak" | "bystander_backchannel" | "bystander_laugh_cough"
    pause_bgv_acoustics: str = "far_field"
    pause_bgv2_text: str = ""
    pause_bgv2_voice: str = "Orus"


@dataclass
class ScenarioSpec:
    cid: str
    title: str
    pilot_category: str
    language: str
    primary_voice: str
    ambient_categories: list[str]
    ambient_level_db: float
    rt60_s: float
    turns: list[CallTurnSpec]
    failure_focus: str = "Multi-noise bursts + competing background speakers"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScenarioSpec":
        valid_turn_keys = {f.name for f in fields(CallTurnSpec)}
        turns = [
            CallTurnSpec(**{k: v for k, v in t.items() if k in valid_turn_keys})
            for t in d.get("turns", [])
        ]
        return cls(
            cid=d["cid"],
            title=d.get("title", d["cid"]),
            pilot_category=d.get("pilot_category", "multi_hazard"),
            language=d.get("language", "Hindi"),
            primary_voice=d.get("primary_voice", "Kore"),
            ambient_categories=list(d.get("ambient_categories", ["appliance_machine"])),
            ambient_level_db=float(d.get("ambient_level_db", -20.0)),
            rt60_s=float(d.get("rt60_s", 0.50)),
            turns=turns,
            failure_focus=d.get("failure_focus", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_scenarios_from_json(path: str | Path) -> list[ScenarioSpec]:
    """Load custom customer scenarios from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_list = data.get("scenarios", data) if isinstance(data, dict) else data
    return [ScenarioSpec.from_dict(item) for item in raw_list]


def generate_scenarios_via_llm(
    cfg: Any,
    num_scenarios: int = 3,
    languages: list[str] | None = None,
    domain_context: str = "Retail & Corporate Banking Customer Support",
) -> list[ScenarioSpec]:
    """Dynamically generate realistic multi-turn `ScenarioSpec` scripts using the configured
    `scenario_generation_system_prompt` and target duration range (`min_conversation_duration_s`
    to `max_conversation_duration_s`, avg `target_avg_conversation_duration_s`)."""
    import re
    from google.genai import types
    from src.tts_client import create_genai_client

    client, _ = create_genai_client(cfg)
    langs = languages or ["Hindi", "Indian English", "Hinglish"]
    min_d = cfg.min_conversation_duration_s
    max_d = cfg.max_conversation_duration_s
    avg_d = cfg.target_avg_conversation_duration_s
    approx_turns = max(3, min(16, int(round(avg_d / 11.0))))

    user_prompt = f"""Generate {num_scenarios} realistic multi-turn voice conversation scenarios for domain: {domain_context}.
Target languages across scenarios: {', '.join(langs)}.
Target rendered audio duration range per conversation: {min_d:.1f}s to {max_d:.1f}s (average ~{avg_d:.1f}s, approximately {approx_turns} caller turns per conversation).

Return STRICTLY a valid JSON object with key "scenarios" containing a list of scenario objects matching this exact schema:
{{
  "scenarios": [
    {{
      "cid": "gen_01_...",
      "title": "...",
      "pilot_category": "vehicle_traffic",
      "language": "Hindi",
      "primary_voice": "Kore",
      "ambient_categories": ["vehicle_traffic", "appliance_machine"],
      "ambient_level_db": -20.0,
      "rt60_s": 0.48,
      "failure_focus": "...",
      "turns": [
        {{
          "text": "...",
          "voice_mode": "clean | overlap_noise | overlap_bgv_and_noise",
          "prosody_style": "normal | short_micro | angry_shout | fast_panicked | slow_hesitant | laugh_chuckle | hesitation_stutter",
          "channel_quirk": "none | mic_clipping | packet_loss",
          "primary_gain_db": 0.0,
          "overlap_noise_cats": ["vehicle_traffic"],
          "overlap_snr_db": 6.0,
          "overlap_bgv_text": "",
          "overlap_bgv_voice": "Fenrir",
          "overlap_sir_db": 10.0,
          "pause_noise_cats": ["vehicle_traffic", "phone_signal_alarm"],
          "pause_noise_snr_db": 4.0,
          "pause_bgv_text": "...",
          "pause_bgv_voice": "Aoede",
          "pause_bgv_sir_db": 10.0,
          "pause_bgv_acoustics": "far_field | near_field_loud | tv_radio_playback | cocktail_dual_talker | bot_echo_leak | bystander_backchannel | bystander_laugh_cough",
          "pause_bgv_with_noise_cats": ["appliance_machine"]
        }}
      ]
    }}
  ]
}}"""

    combined_prompt = f"{cfg.scenario_generation_system_prompt.strip()}\n\n{user_prompt}"
    models_to_try = [cfg.scenario_generator_model_id] + [
        m for m in cfg.fallback_scenario_models if m != cfg.scenario_generator_model_id
    ]
    last_err: Exception | None = None
    for model_id in models_to_try:
        try:
            resp = client.models.generate_content(
                model=model_id,
                contents=combined_prompt,
                config=types.GenerateContentConfig(temperature=0.4),
            )
            text = (resp.text or "").strip()
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not m:
                raise ValueError(f"No JSON object found in response from {model_id}")
            parsed = json.loads(m.group(0))
            raw_list = parsed.get("scenarios", parsed) if isinstance(parsed, dict) else parsed
            return [ScenarioSpec.from_dict(item) for item in raw_list]
        except Exception as err:
            last_err = err
            continue
    raise RuntimeError(f"Failed to generate scenarios via LLM: {last_err}")


def get_core_27_scenarios() -> list[ScenarioSpec]:
    """Return the 27-call core 24-Indian-language synthetic call suite (`synth_01` through `synth_27`, ~2–3 mins each)."""
    from src.synthetic_core10_scenarios import get_core_10_scenarios
    from src.synthetic_indian_scenarios import get_extra_indian_scenarios

    scenarios: list[ScenarioSpec] = []
    scenarios.extend(get_core_10_scenarios(ScenarioSpec, CallTurnSpec))
    scenarios.extend(get_extra_indian_scenarios(ScenarioSpec, CallTurnSpec))
    return scenarios


def get_edge_10_scenarios() -> list[ScenarioSpec]:
    """Return the 10-call conversational, paralinguistic, emotional & hardware edge-case suite (`synth_28` through `synth_37`, ~2+ mins each)."""
    from src.synthetic_edge_variation_scenarios import get_edge_variation_scenarios

    return list(get_edge_variation_scenarios(ScenarioSpec, CallTurnSpec))


def get_full_37_scenarios() -> list[ScenarioSpec]:
    """Return all 37 synthetic scenarios (`synth_01`..`synth_27` core 24-language suite + `synth_28`..`synth_37` edge-variation suite)."""
    return get_core_27_scenarios() + get_edge_10_scenarios()


def get_test_showcase_scenarios() -> list[ScenarioSpec]:
    """Default preset returning all 37 multi-variation synthetic calls (`synth_01` through `synth_37`)."""
    return get_full_37_scenarios()


def get_quick_5_scenarios() -> list[ScenarioSpec]:
    """5 Short Showcase Scenarios (~35–50s each) for quick smoke testing."""
    return [
        # -------------------------------------------------------------------
        # Example 1: Hindi Speakerphone Verification — Micro-Utterances, Bot Echo & Bystander Backchannels
        # -------------------------------------------------------------------
        ScenarioSpec(
            cid="example_01_hindi_micro_bot_echo_backchannels",
            title="Hindi Speakerphone Call — Micro-Utterances ('Haan Jee', 'Hmm'), AI Bot Echo Leak & Bystander Backchannels",
            pilot_category="background_voice",
            language="Hindi",
            primary_voice="Kore",
            ambient_categories=["appliance_machine", "vehicle_traffic"],
            ambient_level_db=-19.0,
            rt60_s=0.48,
            failure_focus="Short micro-utterances (0.15-0.45s 'हाँ जी', 'हूँ', 'बिल्कुल') + speakerphone AI bot echo + bystander backchannels + hesitation pauses",
            turns=[
                CallTurnSpec(
                    text="नमस्ते सर, मैं विकास शर्मा बोल रहा हूँ, मुझे अपने क्रेडिट कार्ड का इंटरनेशनल ट्रांजेक्शन ऑन करवाना था।",
                    voice_mode="clean",
                    pause_noise_cats=["appliance_machine", "phone_signal_alarm"],
                    pause_noise_snr_db=5.0,
                    pause_bgv_text="नमस्ते विकास जी, क्या आप अपने रजिस्टर्ड मोबाइल नंबर से बात कर रहे हैं? कृपया हाँ या ना में बताएं।",
                    pause_bgv_voice="Aoede",
                    pause_bgv_sir_db=16.0,
                    pause_bgv_acoustics="bot_echo_leak",
                    pause_bgv_with_noise_cats=["appliance_machine"],
                ),
                CallTurnSpec(
                    text="हाँ जी।",
                    voice_mode="overlap_noise",
                    prosody_style="short_micro",
                    overlap_noise_cats=["appliance_machine", "vehicle_traffic"],
                    overlap_snr_db=6.0,
                    pause_noise_cats=["appliance_machine", "human_non_speech"],
                    pause_noise_snr_db=4.0,
                    pause_bgv_text="अच्छा भाई!",
                    pause_bgv_voice="Fenrir",
                    pause_bgv_sir_db=1.0,
                    pause_bgv_acoustics="bystander_backchannel",
                    pause_bgv_with_noise_cats=["appliance_machine"],
                ),
                CallTurnSpec(
                    text="हूँ।",
                    voice_mode="clean",
                    prosody_style="short_micro",
                    pause_noise_cats=["appliance_machine", "vehicle_traffic"],
                    pause_noise_snr_db=4.0,
                    pause_bgv_text="क्या आपके कार्ड के आखिरी चार अंक सात दो शून्य चार हैं?",
                    pause_bgv_voice="Aoede",
                    pause_bgv_sir_db=15.0,
                    pause_bgv_acoustics="bot_echo_leak",
                ),
                CallTurnSpec(
                    text="अम्म... रुकिए सर... जन्म तिथि... हाँ, पंद्रह अगस्त उन्नीस सौ नब्बे है।",
                    voice_mode="overlap_noise",
                    prosody_style="hesitation_stutter",
                    overlap_noise_cats=["appliance_machine", "vehicle_traffic"],
                    overlap_snr_db=6.0,
                    pause_noise_cats=["vehicle_traffic", "phone_signal_alarm"],
                    pause_noise_snr_db=3.0,
                    pause_bgv_text="हाँ हाँ, सही है!",
                    pause_bgv_voice="Puck",
                    pause_bgv_sir_db=0.0,
                    pause_bgv_acoustics="bystander_backchannel",
                    pause_bgv_with_noise_cats=["vehicle_traffic"],
                ),
            ],
        ),

        # -------------------------------------------------------------------
        # Example 2: Indian English Escalation — Angry Shouted Clipping, Fast Panicked, Packet Loss & Dual-Talker Babble
        # -------------------------------------------------------------------
        ScenarioSpec(
            cid="example_02_english_emotion_clipping_packet_loss",
            title="Indian English Escalation — Angry Shouted Mic Clipping, Fast Panicked Speech, VoIP Packet Loss & Dual-Talker Babble",
            pilot_category="phone_signal_alarm",
            language="Indian English",
            primary_voice="Orus",
            ambient_categories=["appliance_machine", "human_non_speech"],
            ambient_level_db=-20.0,
            rt60_s=0.52,
            failure_focus="Angry shouted speech (+5.5 dB + tanh mic clipping) + fast panicked rate (1.20x) + 42ms RTP VoIP packet loss + 2-speaker cocktail babble",
            turns=[
                CallTurnSpec(
                    text="Why has my current account wire transfer been blocked for three hours without a single notification alert?!",
                    voice_mode="overlap_noise",
                    prosody_style="angry_shout",
                    channel_quirk="mic_clipping",
                    primary_gain_db=5.5,
                    overlap_noise_cats=["phone_signal_alarm", "appliance_machine"],
                    overlap_snr_db=5.0,
                    pause_noise_cats=["phone_signal_alarm", "appliance_machine", "human_non_speech"],
                    pause_noise_snr_db=3.0,
                    pause_bgv_text="Please check the invoice for the Pune warehouse shipment, the GST number on page two needs correction.",
                    pause_bgv_voice="Zephyr",
                    pause_bgv2_text="Yeah I already spoke to the vendor, they are dispatching the spare parts by evening courier.",
                    pause_bgv2_voice="Alnilam",
                    pause_bgv_sir_db=4.0,
                    pause_bgv_acoustics="cocktail_dual_talker",
                    pause_bgv_with_noise_cats=["appliance_machine"],
                ),
                CallTurnSpec(
                    text="Please hurry up and check batch zero zero eight nine right now before the clearing house window closes at four P M!",
                    voice_mode="overlap_bgv_and_noise",
                    prosody_style="fast_panicked",
                    overlap_noise_cats=["appliance_machine", "phone_signal_alarm"],
                    overlap_snr_db=6.0,
                    overlap_bgv_text="Send the updated spreadsheet to accounts before lunch so they can release the vendor payment.",
                    overlap_bgv_voice="Alnilam",
                    overlap_sir_db=8.0,
                    pause_noise_cats=["appliance_machine", "phone_signal_alarm"],
                    pause_noise_snr_db=4.0,
                ),
                CallTurnSpec(
                    text="Okay, I am reading out the beneficiary account number, it is nine one four zero two zero zero eight five five one.",
                    voice_mode="overlap_noise",
                    prosody_style="normal",
                    channel_quirk="packet_loss",
                    overlap_noise_cats=["appliance_machine", "human_non_speech"],
                    overlap_snr_db=6.0,
                    pause_bgv_text="Can someone pick up line two? The distributor from Bengaluru is waiting on hold!",
                    pause_bgv_voice="Zephyr",
                    pause_bgv_sir_db=-2.0,
                    pause_bgv_acoustics="near_field_loud",
                    pause_bgv_with_noise_cats=["phone_signal_alarm"],
                ),
            ],
        ),

        # -------------------------------------------------------------------
        # Example 3: Hinglish Outdoor Commute — Wind Buffeting, Soft Elderly Speech, Paralinguistic Laughs & Coughs
        # -------------------------------------------------------------------
        ScenarioSpec(
            cid="example_03_hinglish_wind_soft_laugh_cough",
            title="Hinglish Outdoor Commute — Wind Buffeting, Soft Elderly Speech (-4 dB), Mid-Turn Laughs & Bystander Coughs",
            pilot_category="vehicle_traffic",
            language="Hinglish",
            primary_voice="Puck",
            ambient_categories=["vehicle_traffic", "animal"],
            ambient_level_db=-19.0,
            rt60_s=0.45,
            failure_focus="Outdoor wind buffeting (22-210 Hz gusts) + soft slow elderly speech (0.85x, -4.0 dB) + mid-turn user laugh + close-mic bystander cough/throat-clear",
            turns=[
                CallTurnSpec(
                    text="Beta, mera pension account statement abhi tak email par nahi aaya hai, zara dheere se check karke batao na.",
                    voice_mode="overlap_noise",
                    prosody_style="slow_hesitant",
                    primary_gain_db=-4.0,
                    overlap_noise_cats=["wind_mic_rub", "vehicle_traffic"],
                    overlap_snr_db=6.0,
                    pause_noise_cats=["wind_mic_rub", "vehicle_traffic", "animal"],
                    pause_noise_snr_db=2.0,
                    pause_bgv_text="अरे खांसी आ रही है... उहूँ उहूँ... थोड़ा पानी देना भाई!",
                    pause_bgv_voice="Fenrir",
                    pause_bgv_sir_db=2.0,
                    pause_bgv_acoustics="bystander_laugh_cough",
                    pause_bgv_with_noise_cats=["wind_mic_rub"],
                ),
                CallTurnSpec(
                    text="हाहाहा, अरे वाह! मुझे लगा बैंक ब्रांच जाना पड़ेगा, पर आपने तो फोन पर ही मेरा काम आसान कर दिया!",
                    voice_mode="overlap_noise",
                    prosody_style="laugh_chuckle",
                    overlap_noise_cats=["vehicle_traffic", "animal"],
                    overlap_snr_db=7.0,
                    pause_noise_cats=["vehicle_traffic", "phone_signal_alarm", "animal"],
                    pause_noise_snr_db=4.0,
                    pause_bgv_text="चाय गरम चाय! समोसा वड़ा पाव गरम है भाई साहब!",
                    pause_bgv_voice="Orus",
                    pause_bgv_sir_db=6.0,
                    pause_bgv_acoustics="far_field",
                    pause_bgv_with_noise_cats=["vehicle_traffic"],
                ),
                CallTurnSpec(
                    text="ठीक है जी, बहुत बहुत शुक्रिया, मैं अभी व्हाट्सएप पर रसीद देख लेता हूँ।",
                    voice_mode="clean",
                    pause_noise_cats=["wind_mic_rub", "vehicle_traffic"],
                    pause_noise_snr_db=4.0,
                ),
            ],
        ),

        # -------------------------------------------------------------------
        # Example 4: Busy Indian Home — TV Broadcast Playback, Kids Crying, Music & Far-Field Family Chatter
        # -------------------------------------------------------------------
        ScenarioSpec(
            cid="example_04_hindi_home_tv_kids_music",
            title="Busy Indian Home — TV News Broadcast Playback, Baby Crying, Music & Family Chatter",
            pilot_category="baby_child",
            language="Hindi",
            primary_voice="Charon",
            ambient_categories=["appliance_machine", "singing_music"],
            ambient_level_db=-20.0,
            rt60_s=0.62,
            failure_focus="TV/Radio broadcast playback saturation + baby crying + music + overlapping family speech",
            turns=[
                CallTurnSpec(
                    text="हेलो मैम, मैंने पिछले हफ्ते क्रेडिट कार्ड अपग्रेड के लिए अप्लाई किया था, उसका स्टेटस अभी तक पेंडिंग दिखा रहा है।",
                    voice_mode="clean",
                    pause_noise_cats=["appliance_machine", "baby_child", "singing_music"],
                    pause_noise_snr_db=4.0,
                    pause_bgv_text="अरे रिमोट कहाँ रख दिया? न्यूज़ चैनल लगाओ ज़रा पांच मिनट के लिए।",
                    pause_bgv_voice="Aoede",
                    pause_bgv_sir_db=8.0,
                    pause_bgv_acoustics="tv_radio_playback",
                    pause_bgv_with_noise_cats=["appliance_machine", "singing_music"],
                ),
                CallTurnSpec(
                    text="हाँ वीडियो केवाईसी तो मैंने बुधवार को ही पूरी कर ली थी, एजेंट ने मेरा आधार और पैन कार्ड दोनों कैमरा पर देखा था।",
                    voice_mode="overlap_bgv_and_noise",
                    overlap_noise_cats=["baby_child", "appliance_machine", "singing_music"],
                    overlap_snr_db=6.0,
                    overlap_bgv_text="मम्मी मेरा स्कूल बैग कहाँ रखा है? मुझे अपनी ड्राइंग बुक नहीं मिल रही है!",
                    overlap_bgv_voice="Leda",
                    overlap_sir_db=9.0,
                    pause_noise_cats=["baby_child", "appliance_machine", "phone_signal_alarm"],
                    pause_noise_snr_db=3.0,
                    pause_bgv_text="दरवाज़े पर घंटी बजी है, ज़रा देखो कूरियर वाला आया होगा, पार्सल ले लो।",
                    pause_bgv_voice="Callirrhoe",
                    pause_bgv_sir_db=6.0,
                    pause_bgv_acoustics="far_field",
                    pause_bgv_with_noise_cats=["phone_signal_alarm", "baby_child"],
                ),
            ],
        ),

        # -------------------------------------------------------------------
        # Example 5: Multilingual Regional (Tamil) — All Acoustic & Conversational Variations Combined
        # -------------------------------------------------------------------
        ScenarioSpec(
            cid="example_05_tamil_multilingual_variations",
            title="Tamil Customer Support Call — Micro-Utterances ('Aamaa', 'Sari'), Competing Bystanders & Multi-Noise",
            pilot_category="multi_hazard",
            language="Tamil",
            primary_voice="Leda",
            ambient_categories=["vehicle_traffic", "appliance_machine"],
            ambient_level_db=-20.0,
            rt60_s=0.50,
            failure_focus="Regional Indian language (Tamil) + short micro-utterances ('ஆமா', 'சரி') + bot echo + competing talker",
            turns=[
                CallTurnSpec(
                    text="வணக்கம் சார், என் வங்கி கணக்கில் இருந்து இன்று காலை இரண்டு முறை இஎம்ஐ பணம் பிடித்தம் செய்யப்பட்டுள்ளது, கொஞ்சம் சரிபார்த்து சொல்லுங்கள்.",
                    voice_mode="overlap_noise",
                    overlap_noise_cats=["vehicle_traffic", "appliance_machine"],
                    overlap_snr_db=6.0,
                    pause_noise_cats=["vehicle_traffic", "phone_signal_alarm"],
                    pause_noise_snr_db=4.0,
                    pause_bgv_text="வணக்கம் மேடம், உங்கள் பதிவு செய்யப்பட்ட மொபைல் எண்ணிலிருந்து தான் பேசுகிறீர்களா?",
                    pause_bgv_voice="Orus",
                    pause_bgv_sir_db=15.0,
                    pause_bgv_acoustics="bot_echo_leak",
                ),
                CallTurnSpec(
                    text="ஆமா.",
                    voice_mode="clean",
                    prosody_style="short_micro",
                    pause_noise_cats=["vehicle_traffic", "animal"],
                    pause_noise_snr_db=4.0,
                    pause_bgv_text="அண்ணா, இந்த பஸ் தாம்பரம் போகுமா? சீக்கிரம் ஏறுங்க சிக்னல் விழப்போகுது!",
                    pause_bgv_voice="Fenrir",
                    pause_bgv_sir_db=4.0,
                    pause_bgv_acoustics="near_field_loud",
                    pause_bgv_with_noise_cats=["vehicle_traffic"],
                ),
                CallTurnSpec(
                    text="சரி, ரொம்ப நன்றி சார், நான் இப்போதே எஸ்எம்எஸ் எண்ணை குறித்துக் கொள்கிறேன்.",
                    voice_mode="overlap_noise",
                    overlap_noise_cats=["vehicle_traffic", "phone_signal_alarm"],
                    overlap_snr_db=7.0,
                    pause_noise_cats=["vehicle_traffic", "human_non_speech"],
                    pause_noise_snr_db=5.0,
                ),
            ],
        ),
    ]
