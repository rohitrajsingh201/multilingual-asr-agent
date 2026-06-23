"""
Indian Language Configuration
==============================
Metadata for 8 supported languages: Hindi, Bengali, Marathi, Gujarati,
Tamil, Telugu, Kannada, and English.
Includes script family, ISO codes, region, and ASR challenge level —
useful for dataset strategy and model selection.
"""

SUPPORTED_LANGUAGES = {
    # ── Indian Languages ──────────────────────────────────────────────────────
    "hi": {
        "name": "Hindi",
        "native_name": "हिन्दी",
        "script": "Devanagari",
        "script_family": "Brahmic",
        "region": "North India",
        "speakers_million": 600,
        "whisper_support": "strong",
        "asr_challenge": "medium",       # code-switching common
        "notes": "Most common Indian language; heavy code-mixing with English",
    },
    "bn": {
        "name": "Bengali",
        "native_name": "বাংলা",
        "script": "Bengali",
        "script_family": "Brahmic",
        "region": "East India / Bangladesh",
        "speakers_million": 230,
        "whisper_support": "good",
        "asr_challenge": "medium",
        "notes": "Second most spoken Indian language",
    },
    "mr": {
        "name": "Marathi",
        "native_name": "मराठी",
        "script": "Devanagari",
        "script_family": "Brahmic",
        "region": "West India",
        "speakers_million": 95,
        "whisper_support": "moderate",
        "asr_challenge": "medium",
        "notes": "Shares script with Hindi; distinct phonology",
    },
    "gu": {
        "name": "Gujarati",
        "native_name": "ગુજરાતી",
        "script": "Gujarati",
        "script_family": "Brahmic",
        "region": "West India",
        "speakers_million": 60,
        "whisper_support": "moderate",
        "asr_challenge": "medium",
        "notes": "Good diaspora dataset availability",
    },
    "ta": {
        "name": "Tamil",
        "native_name": "தமிழ்",
        "script": "Tamil",
        "script_family": "Brahmic",
        "region": "South India / Sri Lanka",
        "speakers_million": 80,
        "whisper_support": "good",
        "asr_challenge": "high",         # agglutinative morphology
        "notes": "Classical language; agglutinative — WER tends to be higher",
    },
    "te": {
        "name": "Telugu",
        "native_name": "తెలుగు",
        "script": "Telugu",
        "script_family": "Brahmic",
        "region": "South India",
        "speakers_million": 95,
        "whisper_support": "moderate",
        "asr_challenge": "high",
        "notes": "Agglutinative; limited public ASR datasets",
    },
    "kn": {
        "name": "Kannada",
        "native_name": "ಕನ್ನಡ",
        "script": "Kannada",
        "script_family": "Brahmic",
        "region": "South India",
        "speakers_million": 60,
        "whisper_support": "moderate",
        "asr_challenge": "high",
        "notes": "Agglutinative with complex sandhi rules",
    },
    # ── International ─────────────────────────────────────────────────────────
    "en": {
        "name": "English",
        "native_name": "English",
        "script": "Latin",
        "script_family": "Latin",
        "region": "Global",
        "speakers_million": 1500,
        "whisper_support": "excellent",
        "asr_challenge": "low",
        "notes": "Whisper primary training language",
    },
}


def get_language_meta(iso_code: str) -> dict:
    """
    Retrieve language metadata by ISO 639-1 code.

    Args:
        iso_code : Two-letter language code (e.g. 'hi', 'ta', 'en')

    Returns:
        Metadata dict, or a minimal dict if language not in registry.
    """
    return SUPPORTED_LANGUAGES.get(
        iso_code,
        {
            "name": iso_code.upper(),
            "script": "Unknown",
            "asr_challenge": "unknown",
            "whisper_support": "unknown",
        },
    )


def list_indian_languages() -> list[dict]:
    """Return metadata for all Indian languages in the registry."""
    indian_codes = ["hi", "bn", "mr", "gu", "ta", "te", "kn"]
    return [
        {"code": code, **SUPPORTED_LANGUAGES[code]}
        for code in indian_codes
        if code in SUPPORTED_LANGUAGES
    ]


def get_challenge_tier(iso_code: str) -> str:
    """
    Returns ASR challenge tier for a language.
    Useful for auto-selecting Whisper model size:
      low/medium → small
      high       → medium
      very_high  → large
    """
    meta = get_language_meta(iso_code)
    return meta.get("asr_challenge", "unknown")


def recommend_model(iso_code: str) -> str:
    """Suggest a Whisper model size based on language challenge tier."""
    tier = get_challenge_tier(iso_code)
    return {
        "low": "small",
        "medium": "small",
        "high": "medium",
        "very_high": "large",
        "unknown": "medium",
    }.get(tier, "medium")
