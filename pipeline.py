"""
Multilingual Indian ASR Intelligence Pipeline
=============================================
A multi-agent system for speech transcription, quality analysis,
and WER benchmarking across Indian languages using Whisper + AutoGen.

Agents:
  - TranscriberAgent   : Transcribes audio via Whisper (small/medium/large)
  - QualityAgent       : Computes WER, CER, and flags low-confidence segments
  - TranslatorAgent    : Translates transcript to English (or any target lang)
  - ReportAgent        : Generates structured JSON + markdown quality report

Supported Languages: Hindi, Bengali, Marathi, Gujarati, Tamil, Telugu, Kannada, English

Author: Rohit Raj Singh
Domain: ASR / NLP / Voice AI
"""

import os
import json
import time
import datetime
import argparse
import warnings
warnings.filterwarnings("ignore")

# ── Third-party ────────────────────────────────────────────────────────────────
try:
    import whisper
except ImportError:
    raise ImportError("Run: pip install openai-whisper")

try:
    import autogen
except ImportError:
    raise ImportError("Run: pip install pyautogen")

from utils.wer_calculator import compute_wer, compute_cer
from utils.language_config import SUPPORTED_LANGUAGES, get_language_meta
from utils.report_builder import build_markdown_report, build_json_report


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Whisper Transcription (TranscriberAgent tool)
# ══════════════════════════════════════════════════════════════════════════════

def transcribe_audio(audio_path: str, model_size: str = "small", language: str = None) -> dict:
    """
    Transcribe an audio file using OpenAI Whisper.

    Args:
        audio_path  : Path to .wav / .mp3 / .m4a file
        model_size  : Whisper model — tiny | small | medium | large
        language    : ISO 639-1 code (e.g. 'hi', 'ta', 'bn'). None = auto-detect.

    Returns:
        dict with keys: text, language, segments, duration, model_used
    """
    if not os.path.exists(audio_path):
        return {"error": f"File not found: {audio_path}"}

    print(f"\n[TranscriberAgent] Loading Whisper ({model_size}) ...")
    model = whisper.load_model(model_size)

    decode_opts = {"verbose": False}
    if language:
        decode_opts["language"] = language

    start = time.time()
    result = model.transcribe(audio_path, **decode_opts)
    elapsed = round(time.time() - start, 2)

    segments = []
    for seg in result["segments"]:
        segments.append({
            "id": seg["id"],
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
            "confidence": round(1 - seg.get("no_speech_prob", 0), 3),  # proxy confidence
        })

    detected_lang = result.get("language", "unknown")
    lang_meta = get_language_meta(detected_lang)

    return {
        "audio_file": os.path.basename(audio_path),
        "model_used": model_size,
        "detected_language": detected_lang,
        "language_name": lang_meta.get("name", detected_lang),
        "script": lang_meta.get("script", "Unknown"),
        "full_text": result["text"].strip(),
        "segments": segments,
        "total_segments": len(segments),
        "transcription_time_sec": elapsed,
        "timestamp": datetime.datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Quality Analysis (QualityAgent tool)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_quality(transcription: dict, reference_text: str = None) -> dict:
    """
    Compute WER/CER and identify low-confidence segments.

    Args:
        transcription  : Output dict from transcribe_audio()
        reference_text : Ground-truth text (optional). If None, only
                         confidence-based analysis is performed.

    Returns:
        dict with quality metrics and flagged segments
    """
    print("\n[QualityAgent] Running quality analysis ...")

    segments = transcription.get("segments", [])
    low_conf_threshold = 0.75

    flagged = [
        seg for seg in segments
        if seg["confidence"] < low_conf_threshold
    ]

    avg_confidence = (
        round(sum(s["confidence"] for s in segments) / len(segments), 3)
        if segments else 0.0
    )

    quality_report = {
        "avg_segment_confidence": avg_confidence,
        "total_segments": len(segments),
        "flagged_low_confidence": len(flagged),
        "flagged_segments": flagged,
        "quality_grade": _grade_confidence(avg_confidence),
    }

    # WER / CER only if reference provided
    if reference_text and reference_text.strip():
        hypothesis = transcription.get("full_text", "")
        wer = compute_wer(reference_text, hypothesis)
        cer = compute_cer(reference_text, hypothesis)
        quality_report["wer"] = wer
        quality_report["cer"] = cer
        quality_report["wer_grade"] = _grade_wer(wer)
    else:
        quality_report["wer"] = None
        quality_report["cer"] = None
        quality_report["note"] = "No reference text provided — WER/CER skipped."

    return quality_report


def _grade_confidence(score: float) -> str:
    if score >= 0.90:
        return "Excellent"
    elif score >= 0.80:
        return "Good"
    elif score >= 0.65:
        return "Acceptable"
    else:
        return "Poor — review recommended"


def _grade_wer(wer: float) -> str:
    if wer <= 0.05:
        return "Production-Ready (≤5%)"
    elif wer <= 0.15:
        return "Acceptable (≤15%)"
    elif wer <= 0.30:
        return "Needs Improvement"
    else:
        return "Critical — retrain/re-collect data"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Translation (TranslatorAgent tool)
# ══════════════════════════════════════════════════════════════════════════════

def translate_transcript(transcription: dict, target_language: str = "en") -> dict:
    """
    Uses Whisper's built-in translate task to convert speech to English,
    or wraps the detected text for downstream LLM translation.

    Note: Whisper's translate task converts directly from audio to English.
    For other target languages, an LLM call is recommended.

    Args:
        transcription  : Output from transcribe_audio()
        target_language: Target ISO code. 'en' uses Whisper native translate.

    Returns:
        dict with translated_text and metadata
    """
    print(f"\n[TranslatorAgent] Translating to '{target_language}' ...")

    source_lang = transcription.get("detected_language", "unknown")

    # If already English, skip
    if source_lang == "en" and target_language == "en":
        return {
            "source_language": source_lang,
            "target_language": target_language,
            "translated_text": transcription.get("full_text", ""),
            "note": "Source is already English — no translation needed.",
        }

    # Whisper native translate (to English only)
    if target_language == "en":
        audio_path = transcription.get("audio_file", "")
        if os.path.exists(audio_path):
            model = whisper.load_model("small")
            result = model.transcribe(audio_path, task="translate")
            translated = result["text"].strip()
        else:
            translated = "[Audio file not available for re-translation]"
    else:
        # Placeholder for LLM-based translation (e.g. GPT-4, Claude, etc.)
        translated = (
            f"[LLM translation to '{target_language}' — plug in your LLM client here]\n"
            f"Source text: {transcription.get('full_text', '')}"
        )

    return {
        "source_language": source_lang,
        "target_language": target_language,
        "translated_text": translated,
        "note": "Translation via Whisper native task" if target_language == "en" else "LLM translation stub",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Report Generation (ReportAgent tool)
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(transcription: dict, quality: dict, translation: dict,
                    output_dir: str = "outputs") -> dict:
    """
    Combines all agent outputs into JSON + Markdown reports.

    Args:
        transcription : Output from transcribe_audio()
        quality       : Output from analyze_quality()
        translation   : Output from translate_transcript()
        output_dir    : Directory to save reports

    Returns:
        dict with paths to saved files
    """
    print("\n[ReportAgent] Generating reports ...")
    os.makedirs(output_dir, exist_ok=True)

    payload = {
        "transcription": transcription,
        "quality_analysis": quality,
        "translation": translation,
        "pipeline_version": "1.0.0",
    }

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"report_{ts}.json")
    md_path = os.path.join(output_dir, f"report_{ts}.md")

    build_json_report(payload, json_path)
    build_markdown_report(payload, md_path)

    print(f"\n[ReportAgent] ✅ Reports saved:")
    print(f"  JSON → {json_path}")
    print(f"  MD   → {md_path}")

    return {"json_report": json_path, "markdown_report": md_path}


# ══════════════════════════════════════════════════════════════════════════════
#  AUTOGEN MULTI-AGENT ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def run_asr_agent_pipeline(
    audio_path: str,
    reference_text: str = None,
    language: str = None,
    target_language: str = "en",
    model_size: str = "small",
    output_dir: str = "outputs",
    openai_api_key: str = None,
):
    """
    Orchestrates all four agents using AutoGen's conversational framework.
    Falls back to direct function execution if no OpenAI key is provided.
    """

    print("\n" + "═" * 60)
    print("  Multilingual Indian ASR Intelligence Pipeline")
    print("  Built with Whisper + AutoGen")
    print("═" * 60)

    # ── Run pipeline steps ───────────────────────────────────────────────────
    transcription = transcribe_audio(audio_path, model_size, language)
    if "error" in transcription:
        print(f"\n❌ Error: {transcription['error']}")
        return

    quality = analyze_quality(transcription, reference_text)
    translation = translate_transcript(transcription, target_language)
    report_paths = generate_report(transcription, quality, translation, output_dir)

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  PIPELINE SUMMARY")
    print("─" * 60)
    print(f"  Audio File      : {transcription['audio_file']}")
    print(f"  Language        : {transcription['language_name']} ({transcription['detected_language']})")
    print(f"  Script          : {transcription['script']}")
    print(f"  Segments        : {transcription['total_segments']}")
    print(f"  Avg Confidence  : {quality['avg_segment_confidence']} → {quality['quality_grade']}")
    if quality.get("wer") is not None:
        print(f"  WER             : {quality['wer']:.2%} → {quality['wer_grade']}")
        print(f"  CER             : {quality['cer']:.2%}")
    print(f"  Transcription   : {transcription['full_text'][:120]}...")
    print(f"  Translation     : {translation['translated_text'][:120]}...")
    print("─" * 60)
    print(f"\n  Reports → {output_dir}/")
    print("═" * 60 + "\n")

    # ── AutoGen agent conversation (LLM-powered analysis summary) ─────────────
    if openai_api_key:
        _run_autogen_summary(transcription, quality, translation, openai_api_key)

    return {
        "transcription": transcription,
        "quality": quality,
        "translation": translation,
        "reports": report_paths,
    }


def _run_autogen_summary(transcription, quality, translation, api_key):
    """
    Spins up a lightweight 2-agent AutoGen conversation that reviews
    the ASR output and provides actionable data-quality recommendations.
    """
    print("\n[AutoGen] Starting agent conversation for quality review ...\n")

    config_list = [{"model": "gpt-4o", "api_key": api_key}]

    context = f"""
    ASR Pipeline Results:
    - Language: {transcription['language_name']} ({transcription['detected_language']})
    - Total Segments: {transcription['total_segments']}
    - Avg Confidence: {quality['avg_segment_confidence']} ({quality['quality_grade']})
    - Flagged Segments: {quality['flagged_low_confidence']}
    - WER: {quality.get('wer', 'N/A')}
    - CER: {quality.get('cer', 'N/A')}
    - Transcript (first 300 chars): {transcription['full_text'][:300]}
    - Translation (first 300 chars): {translation['translated_text'][:300]}
    """

    asr_quality_agent = autogen.AssistantAgent(
        name="ASR_Quality_Analyst",
        system_message="""You are a senior ASR data quality engineer specializing in 
        Indian languages. Given transcription metrics, you identify data quality issues,
        suggest improvements to the dataset pipeline, and provide actionable 
        recommendations to improve WER and model accuracy. Be concise and specific.""",
        llm_config={"config_list": config_list, "timeout": 60},
    )

    coordinator = autogen.UserProxyAgent(
        name="Pipeline_Coordinator",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=2,
        is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
        code_execution_config=False,
    )

    coordinator.initiate_chat(
        asr_quality_agent,
        message=f"""
        Review this ASR pipeline output for an Indian language audio file.
        Provide: (1) quality assessment, (2) top 3 actionable improvements,
        (3) dataset recommendations for this language.
        Then say TERMINATE.

        {context}
        """,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multilingual Indian ASR Intelligence Pipeline"
    )
    parser.add_argument("audio", help="Path to audio file (.wav / .mp3 / .m4a)")
    parser.add_argument("--language", "-l", default=None,
                        help="Source language ISO code (e.g. hi, ta, bn). Auto-detect if omitted.")
    parser.add_argument("--target", "-t", default="en",
                        help="Target translation language (default: en)")
    parser.add_argument("--model", "-m", default="small",
                        choices=["tiny", "small", "medium", "large"],
                        help="Whisper model size (default: small)")
    parser.add_argument("--reference", "-r", default=None,
                        help="Reference transcript text for WER calculation")
    parser.add_argument("--output", "-o", default="outputs",
                        help="Output directory for reports (default: outputs/)")
    parser.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY"),
                        help="OpenAI API key for AutoGen agent conversation")

    args = parser.parse_args()

    run_asr_agent_pipeline(
        audio_path=args.audio,
        reference_text=args.reference,
        language=args.language,
        target_language=args.target,
        model_size=args.model,
        output_dir=args.output,
        openai_api_key=args.openai_key,
    )
