"""
WER / CER Calculator
====================
Lightweight implementation of Word Error Rate and Character Error Rate
without requiring jiwer as a dependency (though jiwer is supported too).

WER = (S + D + I) / N
  S = substitutions, D = deletions, I = insertions, N = words in reference

CER = same formula but at character level
"""

import re
import unicodedata


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, normalize unicode (important for Devanagari etc.)"""
    text = unicodedata.normalize("NFC", text.lower().strip())
    # Remove punctuation but keep spaces and unicode word chars
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _edit_distance_tokens(ref_tokens: list, hyp_tokens: list) -> int:
    """Standard dynamic-programming edit distance."""
    r, h = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (h + 1) for _ in range(r + 1)]

    for i in range(r + 1):
        dp[i][0] = i
    for j in range(h + 1):
        dp[0][j] = j

    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[r][h]


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate between reference and hypothesis strings.

    Args:
        reference  : Ground-truth transcript
        hypothesis : ASR output transcript

    Returns:
        WER as a float (0.0 = perfect, 1.0 = all wrong)
    """
    ref_norm = _normalize(reference)
    hyp_norm = _normalize(hypothesis)

    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    distance = _edit_distance_tokens(ref_words, hyp_words)
    return round(distance / len(ref_words), 4)


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Compute Character Error Rate between reference and hypothesis strings.

    Args:
        reference  : Ground-truth transcript
        hypothesis : ASR output transcript

    Returns:
        CER as a float (0.0 = perfect, 1.0 = all wrong)
    """
    ref_norm = _normalize(reference)
    hyp_norm = _normalize(hypothesis)

    # Remove spaces for pure character-level comparison
    ref_chars = list(ref_norm.replace(" ", ""))
    hyp_chars = list(hyp_norm.replace(" ", ""))

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    distance = _edit_distance_tokens(ref_chars, hyp_chars)
    return round(distance / len(ref_chars), 4)


def compute_batch_wer(pairs: list[dict]) -> list[dict]:
    """
    Compute WER/CER for a batch of reference-hypothesis pairs.

    Args:
        pairs: List of dicts with keys 'reference', 'hypothesis', and optional 'id'

    Returns:
        List of dicts with 'id', 'wer', 'cer', 'reference', 'hypothesis'

    Example:
        pairs = [
            {"id": "utt_001", "reference": "नमस्ते दुनिया", "hypothesis": "नमस्ते दुनिया"},
            {"id": "utt_002", "reference": "hello world", "hypothesis": "hello word"},
        ]
    """
    results = []
    total_wer, total_cer = 0.0, 0.0

    for i, pair in enumerate(pairs):
        ref = pair.get("reference", "")
        hyp = pair.get("hypothesis", "")
        uid = pair.get("id", f"utt_{i+1:03d}")

        wer = compute_wer(ref, hyp)
        cer = compute_cer(ref, hyp)
        total_wer += wer
        total_cer += cer

        results.append({
            "id": uid,
            "reference": ref,
            "hypothesis": hyp,
            "wer": wer,
            "cer": cer,
        })

    if results:
        avg_wer = round(total_wer / len(results), 4)
        avg_cer = round(total_cer / len(results), 4)
        results.append({
            "id": "AGGREGATE",
            "wer": avg_wer,
            "cer": avg_cer,
            "note": f"Average over {len(results)-1} utterances",
        })

    return results
