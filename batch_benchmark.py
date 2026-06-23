"""
Batch WER Benchmarking Tool
============================
Runs Whisper transcription + WER computation across a folder of audio files
with matching reference transcripts. Outputs a consolidated benchmark report.

This mirrors real production QA workflows — exactly what Rohit did at
PinnacleWorks processing 10,000+ hours of multilingual speech data.

Usage:
    python batch_benchmark.py --audio-dir samples/ --refs-file refs.json
    python batch_benchmark.py --audio-dir samples/ --model medium --language hi
"""

import os
import json
import argparse
import datetime
from pathlib import Path

from utils.wer_calculator import compute_wer, compute_cer, compute_batch_wer
from utils.language_config import recommend_model, get_language_meta

try:
    import whisper
except ImportError:
    raise ImportError("Run: pip install openai-whisper")


SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}


def load_references(refs_path: str) -> dict:
    """
    Load reference transcripts from a JSON file.

    Expected format:
    {
      "audio_001.wav": "reference transcript here",
      "audio_002.mp3": "another reference transcript",
      ...
    }
    """
    if not refs_path or not os.path.exists(refs_path):
        print("[BatchBenchmark] No reference file provided — WER will be skipped.")
        return {}

    with open(refs_path, encoding="utf-8") as f:
        return json.load(f)


def run_batch_benchmark(
    audio_dir: str,
    refs_file: str = None,
    model_size: str = None,
    language: str = None,
    output_dir: str = "outputs",
) -> dict:
    """
    Transcribe all audio files in a directory and compute aggregate WER metrics.

    Args:
        audio_dir  : Directory containing audio files
        refs_file  : JSON file mapping filename → reference transcript
        model_size : Whisper model (auto-selected per language if None)
        language   : ISO language code (auto-detect if None)
        output_dir : Where to save benchmark report

    Returns:
        Benchmark summary dict
    """
    audio_path = Path(audio_dir)
    if not audio_path.exists():
        print(f"[BatchBenchmark] ❌ Directory not found: {audio_dir}")
        return {}

    audio_files = sorted([
        f for f in audio_path.iterdir()
        if f.suffix.lower() in SUPPORTED_AUDIO_EXTS
    ])

    if not audio_files:
        print(f"[BatchBenchmark] No audio files found in {audio_dir}")
        return {}

    references = load_references(refs_file)

    # Auto-select model if not specified
    if model_size is None:
        model_size = recommend_model(language) if language else "small"

    print(f"\n{'═'*60}")
    print(f"  Batch ASR Benchmark")
    print(f"  Files: {len(audio_files)} | Model: {model_size} | Language: {language or 'auto'}")
    print(f"{'═'*60}\n")

    model = whisper.load_model(model_size)
    results = []
    wer_pairs = []

    for i, audio_file in enumerate(audio_files):
        print(f"[{i+1}/{len(audio_files)}] Processing: {audio_file.name}")

        decode_opts = {}
        if language:
            decode_opts["language"] = language

        try:
            result = model.transcribe(str(audio_file), verbose=False, **decode_opts)
            hypothesis = result["text"].strip()
            detected_lang = result.get("language", "unknown")
            lang_meta = get_language_meta(detected_lang)

            ref_text = references.get(audio_file.name, "")

            entry = {
                "file": audio_file.name,
                "detected_language": detected_lang,
                "language_name": lang_meta.get("name", detected_lang),
                "hypothesis": hypothesis,
                "reference": ref_text,
                "has_reference": bool(ref_text),
            }

            if ref_text:
                wer = compute_wer(ref_text, hypothesis)
                cer = compute_cer(ref_text, hypothesis)
                entry["wer"] = wer
                entry["cer"] = cer
                wer_pairs.append({
                    "id": audio_file.name,
                    "reference": ref_text,
                    "hypothesis": hypothesis,
                })
            else:
                entry["wer"] = None
                entry["cer"] = None

            results.append(entry)

        except Exception as e:
            print(f"  ⚠️  Error processing {audio_file.name}: {e}")
            results.append({"file": audio_file.name, "error": str(e)})

    # Aggregate stats
    wer_values = [r["wer"] for r in results if r.get("wer") is not None]
    cer_values = [r["cer"] for r in results if r.get("cer") is not None]

    aggregate = {
        "total_files": len(audio_files),
        "successful": len([r for r in results if "error" not in r]),
        "with_reference": len(wer_values),
        "avg_wer": round(sum(wer_values) / len(wer_values), 4) if wer_values else None,
        "avg_cer": round(sum(cer_values) / len(cer_values), 4) if cer_values else None,
        "min_wer": min(wer_values) if wer_values else None,
        "max_wer": max(wer_values) if wer_values else None,
        "model_used": model_size,
        "language": language or "auto-detected",
    }

    # Save report
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "benchmark_timestamp": ts,
        "aggregate": aggregate,
        "per_file_results": results,
    }

    json_path = os.path.join(output_dir, f"benchmark_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\n{'─'*60}")
    print("  BENCHMARK SUMMARY")
    print(f"{'─'*60}")
    print(f"  Files Processed : {aggregate['successful']} / {aggregate['total_files']}")
    print(f"  With Reference  : {aggregate['with_reference']}")
    if aggregate["avg_wer"] is not None:
        print(f"  Avg WER         : {aggregate['avg_wer']:.2%}")
        print(f"  Avg CER         : {aggregate['avg_cer']:.2%}")
        print(f"  WER Range       : {aggregate['min_wer']:.2%} – {aggregate['max_wer']:.2%}")
    print(f"  Report Saved    : {json_path}")
    print(f"{'═'*60}\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch ASR WER Benchmarking Tool")
    parser.add_argument("--audio-dir", required=True, help="Directory with audio files")
    parser.add_argument("--refs-file", default=None,
                        help="JSON file: {filename: reference_text}")
    parser.add_argument("--model", default=None,
                        choices=["tiny", "small", "medium", "large"],
                        help="Whisper model size (auto if omitted)")
    parser.add_argument("--language", default=None,
                        help="Language ISO code (auto-detect if omitted)")
    parser.add_argument("--output", default="outputs",
                        help="Output directory for benchmark report")

    args = parser.parse_args()

    run_batch_benchmark(
        audio_dir=args.audio_dir,
        refs_file=args.refs_file,
        model_size=args.model,
        language=args.language,
        output_dir=args.output,
    )
