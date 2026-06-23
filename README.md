# 🎙️ Multilingual Indian ASR Intelligence Pipeline

> **Multi-agent speech transcription, quality analysis, and WER benchmarking across Indian languages — built with OpenAI Whisper + Microsoft AutoGen**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![Whisper](https://img.shields.io/badge/ASR-OpenAI%20Whisper-orange)](https://github.com/openai/whisper)
[![AutoGen](https://img.shields.io/badge/Agents-AutoGen%200.2-green)](https://github.com/microsoft/autogen)
[![Languages](https://img.shields.io/badge/Languages-8%20Supported-purple)](./utils/language_config.py)

---

## 📌 What Makes This Different

Most Whisper demos stop at transcription.  
This pipeline goes further — it **benchmarks quality**, **flags production risk**, and **generates structured reports** across 12+ Indian languages — reflecting real-world ASR data engineering workflows.

It was built around the same challenges faced when pushing ASR accuracy from **50% → 92%** and reducing WER from **30% → 5%** in a production multilingual Voice AI system.

---

## 🏗️ Architecture — 4-Agent Pipeline

```
Audio File
    │
    ▼
┌─────────────────────┐
│  TranscriberAgent   │  → Whisper (tiny/small/medium/large)
│  Language auto-detect│    + Segment-level confidence scores
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   QualityAgent      │  → WER / CER computation (no external dep)
│   Confidence Scorer  │    + Low-confidence segment flagging
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  TranslatorAgent    │  → Whisper native translate (→ English)
│  Multi-lingual      │    + LLM translation stub for other targets
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   ReportAgent       │  → JSON + GitHub Markdown report
│   Structured Output  │    + Actionable quality recommendations
└─────────────────────┘
         │
         ▼ (optional, requires OpenAI API key)
┌─────────────────────┐
│  AutoGen Agents     │  → ASR_Quality_Analyst + Pipeline_Coordinator
│  LLM-Powered Review │    2-agent conversation for improvement advice
└─────────────────────┘
```

---

## 🌏 Supported Languages

| Language | Code | Script | Challenge | Whisper Support |
|----------|------|--------|-----------|-----------------|
| Hindi | `hi` | Devanagari | Medium | ⭐⭐⭐⭐ |
| Bengali | `bn` | Bengali | Medium | ⭐⭐⭐⭐ |
| Marathi | `mr` | Devanagari | Medium | ⭐⭐⭐ |
| Gujarati | `gu` | Gujarati | Medium | ⭐⭐⭐ |
| Tamil | `ta` | Tamil | High | ⭐⭐⭐ |
| Telugu | `te` | Telugu | High | ⭐⭐⭐ |
| Kannada | `kn` | Kannada | High | ⭐⭐⭐ |
| English | `en` | Latin | Low | ⭐⭐⭐⭐⭐ |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
# System: ffmpeg is required by Whisper
sudo apt install ffmpeg         # Ubuntu/Debian
brew install ffmpeg             # macOS

# Python
pip install -r requirements.txt
```

### 2. Transcribe + Analyze a single file

```bash
python pipeline.py path/to/audio.wav
```

**With language hint and reference for WER:**
```bash
python pipeline.py path/to/audio.wav \
  --language hi \
  --reference "नमस्ते मेरा नाम रोहित है" \
  --model medium
```

**With AutoGen LLM analysis (requires OpenAI key):**
```bash
export OPENAI_API_KEY="sk-..."
python pipeline.py path/to/audio.wav --language hi
```

### 3. Batch benchmark a folder

```bash
python batch_benchmark.py \
  --audio-dir samples/ \
  --refs-file samples/sample_refs.json \
  --language hi \
  --model small
```

---

## 📂 Project Structure

```
multilingual-asr-agent/
├── pipeline.py              # Main 4-agent orchestration
├── batch_benchmark.py       # Batch WER benchmarking tool
├── requirements.txt
├── utils/
│   ├── wer_calculator.py    # WER/CER — no external dep, handles Unicode
│   ├── language_config.py   # Metadata for 12+ Indian languages
│   └── report_builder.py    # JSON + Markdown report generator
├── samples/
│   └── sample_refs.json     # Reference transcript template
└── outputs/                 # Generated reports (auto-created)
```

---

## 📊 Sample Output

```
═══════════════════════════════════════════════════════════
  Multilingual Indian ASR Intelligence Pipeline
  Built with Whisper + AutoGen
═══════════════════════════════════════════════════════════

[TranscriberAgent] Loading Whisper (small) ...
[QualityAgent] Running quality analysis ...
[TranslatorAgent] Translating to 'en' ...
[ReportAgent] Generating reports ...

──────────────────────────────────────────────────────────
  PIPELINE SUMMARY
──────────────────────────────────────────────────────────
  Audio File      : sample_hindi_01.wav
  Language        : Hindi (hi)
  Script          : Devanagari
  Segments        : 12
  Avg Confidence  : 0.921 → Excellent
  WER             : 4.35% → Production-Ready (≤5%)
  CER             : 2.10%
──────────────────────────────────────────────────────────
```

---

## 🔧 Key Technical Features

| Feature | Detail |
|---------|--------|
| **WER/CER** | Custom DP implementation — handles Unicode, Devanagari, Tamil, etc. without jiwer |
| **Language Auto-detect** | Whisper's built-in language detection + metadata enrichment |
| **Model Auto-selection** | Recommends tiny/small/medium/large based on language challenge tier |
| **Batch Processing** | Folder-level benchmarking with aggregate + per-file stats |
| **AutoGen Agents** | 2-agent LLM review: ASR Analyst + Pipeline Coordinator |
| **Structured Reports** | JSON (machine-readable) + Markdown (GitHub-friendly) |

---

## 💡 Extending This Project

- **Add your own LLM** in `translate_transcript()` for non-English targets (GPT-4, Claude, Gemini)
- **Swap Whisper** for a fine-tuned model (e.g. AI4Bharat IndicWav2Vec) in `transcribe_audio()`
- **Add VAD pre-processing** (Silero VAD) to improve segment quality before transcription
- **Connect to a database** — log all benchmark runs to MySQL/PostgreSQL for trend analysis

---

## 📚 Related Work & References

- [OpenAI Whisper](https://github.com/openai/whisper) — Base ASR model
- [Microsoft AutoGen](https://github.com/microsoft/autogen) — Multi-agent framework
- [AI4Bharat IndicSUPERB](https://github.com/AI4Bharat/IndicSUPERB) — Indian language ASR benchmarks
- [Microsoft SAPI](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/) — Production ASR reference

---

## 👤 Author

**Rohit Raj Singh** — Data Engineer & Team Lead  
Specializing in ASR/TTS/NLP systems and multilingual Voice AI pipelines.  
📧 rohitrajsingh200@gmail.com | [LinkedIn](https://linkedin.com/in/rohit-raj-singh)

---

*Built to reflect real production experience: 10,000+ hours of multilingual speech data, 10+ Indian languages, 99%+ pipeline reliability.*
