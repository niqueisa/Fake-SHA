"""
SHAP explainability for XLM-RoBERTa text classification.

This module is intentionally model-path-agnostic because it reuses the runtime model
bundle from `inference.xlmr.loader`. Replacing the model artifacts path/config should
not require SHAP code changes.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from core.config import shap_max_evals, shap_max_words, shap_top_k
from inference.xlmr.loader import XLMRBundle
from schemas.models import ExplanationIndicator, ExplanationResult, ExplanationTopToken

_PUNCT_RE = re.compile(r"^\W+$")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")

_EXPLANATION_NOTE = (
    "SHAP identifies token contributions to the model prediction. "
    "It does not verify factual correctness."
)

# IMPORTANT: This dictionary is used only to group SHAP token contributions into
# interpretable UI indicators. It does NOT influence model prediction and does NOT
# perform factual verification/fact-checking.
_INDICATOR_KEYWORDS: dict[str, list[str]] = {
    "Linguistic Tone": [
        "angry", "anger", "fear", "fearful", "worried", "worry", "sad", "sadness",
        "happy", "emotional", "emotive", "alarming", "alarm", "threat", "threatening",
        "danger", "dangerous", "panic", "panicked", "controversial", "criticized",
        "criticize", "condemned", "condemn", "praised", "praise", "mocked", "mock",
        "insulted", "insult", "accused", "accuse", "blamed", "blame", "outrage",
        "outraged", "furious", "shame", "shameful", "disappointed", "disappointing",
        "concern", "concerned", "warning", "warned", "warn",
        "galit", "nagagalit", "kinagalit", "takot", "natakot", "nakakatakot",
        "pangamba", "nangangamba", "kabado", "nakakabahala", "banta", "nagbabanta",
        "delikado", "panganib", "mapanganib", "gulat", "nagulat", "nakakagulat",
        "malungkot", "lungkot", "masaya", "emosyonal", "umiyak", "iyak",
        "kinondena", "kondena", "pinuna", "batikos", "binatikos", "puna",
        "pinuri", "papuri", "inis", "naiinis", "napahiya", "kahihiyan",
        "sinisi", "sisi", "nakakainis", "nakakabigla", "babala", "nagbabala",
    ],
    "Claim Certainty": [
        "confirmed", "confirm", "proven", "proved", "proof", "definitely",
        "certainly", "surely", "guaranteed", "guarantee", "undeniable",
        "undoubtedly", "always", "never", "must", "will", "cannot", "clearly",
        "obviously", "true", "false", "fake", "real", "claim", "claimed",
        "claims", "alleged", "allegedly", "reportedly", "supposedly",
        "rumored", "rumour", "rumor", "possible", "possibly", "may", "might",
        "could", "likely", "unlikely", "said to be", "believed", "according",
        "kumpirmado", "kinumpirma", "patunay", "napatunayan", "pinatunayan",
        "sigurado", "tiyak", "tiyak na", "talaga", "totoo", "hindi totoo",
        "peke", "huwad", "di umano", "umano", "diumano", "sinasabing",
        "sabi", "sinabi", "ayon", "ayon sa", "pinaniniwalaan", "maaaring",
        "posible", "malamang", "hindi maaari", "dapat", "hindi dapat",
        "walang duda", "klaro", "malinaw", "inaangkin", "pahayag",
        "balitang", "usap usapan", "kumakalat",
    ],
    "Presence of Evidence-related Language": [
        "evidence", "proof", "data", "record", "records", "document",
        "documents", "report", "reports", "study", "studies", "research",
        "investigation", "investigated", "findings", "analysis", "survey",
        "statistics", "statistical", "source", "sources", "statement",
        "official statement", "press release", "certificate", "court record",
        "medical record", "police report", "audit", "verified", "verification",
        "fact check", "factcheck", "fact checked", "based on", "according to",
        "ebidensya", "patunay", "datos", "rekord", "tala", "dokumento",
        "ulat", "pag aaral", "saliksik", "imbestigasyon", "sinisiyasat",
        "natuklasan", "resulta", "pagsusuri", "sarvey", "estadistika",
        "pinagmulan", "sanggunian", "pahayag", "opisyal na pahayag",
        "sertipiko", "rekord ng korte", "ulat ng pulis",
        "beripikado", "beripikasyon", "batay sa",
        "ayon sa ulat", "base sa", "basehan", "katibayan",
    ],
    "Textual Source Attribution Mentions": [
        "said", "says", "stated", "announced", "according", "according to",
        "reported by", "spokesperson", "official", "officials", "agency",
        "department", "government", "president", "senator", "mayor",
        "governor", "police", "authority", "authorities", "expert", "experts",
        "researcher", "researchers", "scientist", "scientists", "journalist",
        "news agency", "court", "supreme court", "congress", "senate",
        "house", "doh", "deped", "dilg", "comelec", "pnp", "nbi", "doj",
        "pna", "philippine news agency", "vera files", "rappler", "gma",
        "abs cbn", "cnn philippines", "inquirer", "philstar", "manila bulletin",
        "sinabi", "ayon", "ayon kay", "ayon sa", "pahayag", "ipinahayag",
        "iniulat", "ulat", "anunsyo", "inanunsyo", "tagapagsalita",
        "opisyal", "ahensya", "kagawaran", "gobyerno", "pangulo",
        "senador", "alkalde", "gobernador", "pulis", "awtoridad",
        "eksperto", "mananaliksik", "mamamahayag", "hukuman", "korte",
        "kongreso", "senado", "kamara", "barangay", "lgu", "lokal na pamahalaan",
    ],
    "Sensationalism": [
        "viral", "shocking", "shock", "exposed", "expose", "unbelievable",
        "breaking", "must see", "must watch", "secret", "scandal", "leaked",
        "leak", "watch", "share", "click", "click here", "urgent", "alert",
        "warning", "bombshell", "explosive", "truth revealed", "revealed",
        "finally revealed", "hidden truth", "banned", "censored", "you wont believe",
        "wow", "amazing", "miracle", "instant", "destroyed", "humiliated",
        "caught", "caught on camera", "exclusive", "latest", "trending", "omg",
        "kumalat", "kalat", "nakakagulat", "gulat", "ibinunyag",
        "binunyag", "pasabog", "eskandalo", "sekreto", "lihim",
        "kumalat na video", "panoorin", "i share",
        "pindutin", "alerto", "grabe",
        "di kapani paniwala", "hindi kapani paniwala", "malupit", "matindi",
        "wasak", "pinahiya", "nahuli", "huli sa camera", "eksklusibo",
        "pinakabago", "mainit na balita", "abangan",
        "alam niyo ba", "hindi mo aakalain", "ikakagulat mo",
    ],
}

_INDICATOR_PRIORITY: list[str] = [
    "Textual Source Attribution Mentions",
    "Presence of Evidence-related Language",
    "Sensationalism",
    "Claim Certainty",
    "Linguistic Tone",
]

_INDICATOR_SUMMARY: dict[str, str] = {
    "Linguistic Tone": (
        "These highlighted words suggest emotionally loaded tone that influenced the prediction."
    ),
    "Claim Certainty": (
        "These words reflect certainty or hedging cues that influenced the prediction."
    ),
    "Presence of Evidence-related Language": (
        "These words indicate evidence-related framing that influenced the prediction."
    ),
    "Textual Source Attribution Mentions": (
        "These words mention sources/attribution patterns that influenced the prediction."
    ),
    "Sensationalism": (
        "These highlighted words suggest attention-grabbing or sensational wording that influenced the prediction."
    ),
}


@dataclass
class _TokenContribution:
    text: str
    score: float


def _truncate_text_by_words(text: str, max_words: int | None = None) -> str:
    max_words = max_words or shap_max_words()
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _normalize_token(raw: str) -> str:
    token = (raw or "").strip()
    if not token:
        return ""
    # SentencePiece/BPE markers commonly used by XLM-R tokenizers.
    token = token.replace("▁", " ").replace("Ġ", " ").strip()
    # Collapse whitespace from merged pieces.
    token = re.sub(r"\s+", " ", token)
    return token


def _normalize_match_text(raw: str) -> str:
    text = (raw or "").lower().strip()
    if not text:
        return ""
    text = (
        text.replace("’", "'")
        .replace("`", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("-", " ")
    )
    text = re.sub(r"[\"'`]", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


_NORMALIZED_KEYWORDS: dict[str, list[str]] = {
    name: sorted(
        {_normalize_match_text(k) for k in kws if _normalize_match_text(k)},
        key=len,
        reverse=True,
    )
    for name, kws in _INDICATOR_KEYWORDS.items()
}


def _is_useful_token(token: str) -> bool:
    if not token.strip():
        return False
    if _PUNCT_RE.match(token):
        return False
    return bool(_WORD_RE.search(token))


def _match_indicator(token_text: str) -> str | None:
    # Phrase-aware matching with deterministic priority for overlapping keywords.
    low = _normalize_match_text(token_text)
    padded = f" {low} "
    for indicator_name in _INDICATOR_PRIORITY:
        for kw in _NORMALIZED_KEYWORDS.get(indicator_name, []):
            if low == kw or f" {kw} " in padded:
                return indicator_name
    return None


def _predict_proba_fn(bundle: XLMRBundle):
    """
    Return a prediction callable for SHAP:
    input list[str] -> ndarray[n_samples, n_labels] of softmax probabilities.
    """
    import torch

    tokenizer = bundle.tokenizer
    model = bundle.model
    device = bundle.device

    def _predict(texts) -> np.ndarray:
        # SHAP may pass numpy/object arrays; normalize to plain list[str] for tokenizer.
        if isinstance(texts, np.ndarray):
            batch = texts.reshape(-1).tolist()
        elif isinstance(texts, (list, tuple)):
            batch = list(texts)
        else:
            batch = [texts]
        batch = [x if isinstance(x, str) else str(x) for x in batch]

        encoded = tokenizer(
            batch,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits
            probs = torch.softmax(logits / 10.0, dim=-1)
        return probs.detach().cpu().numpy()

    return _predict


@lru_cache(maxsize=1)
def _get_shap_explainer() -> Any:
    import shap
    from inference.xlmr.loader import load_bundle

    bundle = load_bundle()
    predict_fn = _predict_proba_fn(bundle)
    masker = shap.maskers.Text(bundle.tokenizer)
    return shap.Explainer(predict_fn, masker)


def _extract_class_values(values: Any, predicted_class_index: int) -> np.ndarray:
    arr = np.asarray(values)
    # Common shape for text classifier explanations: [samples, tokens, classes].
    if arr.ndim == 3:
        return arr[0, :, predicted_class_index]
    # Fallback shape: [tokens, classes]
    if arr.ndim == 2 and arr.shape[1] > predicted_class_index:
        return arr[:, predicted_class_index]
    # Last fallback: flat contributions already selected for one class.
    return arr.reshape(-1)


def build_shap_explanation(
    text: str,
    predicted_class_index: int,
    *,
    max_evals: int | None = None,
) -> ExplanationResult:
    """
    Build SHAP explanation payload for one text sample.

    Raises exceptions if SHAP cannot run; caller should handle fallback behavior.
    """
    clipped = _truncate_text_by_words(text)
    if not clipped.strip():
        return ExplanationResult(note=_EXPLANATION_NOTE, top_tokens=[], indicators=[])

    explainer = _get_shap_explainer()
    # Keep evaluation budget bounded for API latency.
    shap_values = explainer([clipped], max_evals=max_evals or shap_max_evals())

    raw_tokens = list(np.asarray(shap_values.data[0]).tolist())
    raw_scores = _extract_class_values(shap_values.values, predicted_class_index)

    merged_scores: dict[str, float] = {}
    for raw_token, raw_score in zip(raw_tokens, raw_scores):
        token = _normalize_token(str(raw_token))
        if not _is_useful_token(token):
            continue
        merged_scores[token] = merged_scores.get(token, 0.0) + float(raw_score)

    ranked = sorted(
        (_TokenContribution(text=t, score=s) for t, s in merged_scores.items()),
        key=lambda x: abs(x.score),
        reverse=True,
    )[: shap_top_k()]

    top_tokens: list[ExplanationTopToken] = []
    grouped_abs_sum: dict[str, float] = {}
    grouped_tokens: dict[str, set[str]] = {}

    for item in ranked:
        indicator = _match_indicator(item.text)
        top_tokens.append(
            ExplanationTopToken(
                text=item.text,
                score=round(abs(item.score), 6),
                direction="supports_predicted_class" if item.score >= 0 else "opposes_predicted_class",
                indicator=indicator,
            )
        )
        if indicator:
            grouped_abs_sum[indicator] = grouped_abs_sum.get(indicator, 0.0) + abs(item.score)
            grouped_tokens.setdefault(indicator, set()).add(item.text)

    total_grouped = sum(grouped_abs_sum.values())
    indicators: list[ExplanationIndicator] = []
    if total_grouped > 0 and math.isfinite(total_grouped):
        for indicator_name, abs_sum in sorted(grouped_abs_sum.items(), key=lambda x: x[1], reverse=True):
            pct = (abs_sum / total_grouped) * 100.0
            indicators.append(
                ExplanationIndicator(
                    name=indicator_name,
                    contribution_percent=round(pct, 2),
                    tokens=sorted(grouped_tokens.get(indicator_name, set())),
                    summary=_INDICATOR_SUMMARY[indicator_name],
                )
            )

    return ExplanationResult(note=_EXPLANATION_NOTE, top_tokens=top_tokens, indicators=indicators)


def explanation_unavailable(note: str | None = None) -> ExplanationResult:
    """Fallback payload when SHAP is disabled, fails, or is unavailable."""
    return ExplanationResult(
        note=note or "SHAP explanation is currently unavailable. Classification result is still returned.",
        top_tokens=[],
        indicators=[],
    )


def demo_explanation_output() -> ExplanationResult:
    """
    Small local demo payload used by tests/docs without running SHAP.

    This mirrors the frontend shape (`top_tokens`, `indicators`) using manually
    constructed contributions.
    """
    sample_scores = [
        _TokenContribution(text="viral", score=0.213),
        _TokenContribution(text="shocking", score=0.172),
        _TokenContribution(text="reportedly", score=0.121),
        _TokenContribution(text="according to", score=0.099),
    ]

    top_tokens: list[ExplanationTopToken] = []
    grouped_abs_sum: dict[str, float] = {}
    grouped_tokens: dict[str, set[str]] = {}
    for item in sample_scores:
        indicator = _match_indicator(item.text)
        top_tokens.append(
            ExplanationTopToken(
                text=item.text,
                score=round(abs(item.score), 6),
                direction="supports_predicted_class",
                indicator=indicator,
            )
        )
        if indicator:
            grouped_abs_sum[indicator] = grouped_abs_sum.get(indicator, 0.0) + abs(item.score)
            grouped_tokens.setdefault(indicator, set()).add(item.text)

    total_grouped = sum(grouped_abs_sum.values()) or 1.0
    indicators = [
        ExplanationIndicator(
            name=name,
            contribution_percent=round((value / total_grouped) * 100.0, 2),
            tokens=sorted(grouped_tokens[name]),
            summary=_INDICATOR_SUMMARY[name],
        )
        for name, value in sorted(grouped_abs_sum.items(), key=lambda x: x[1], reverse=True)
    ]
    return ExplanationResult(note=_EXPLANATION_NOTE, top_tokens=top_tokens, indicators=indicators)
