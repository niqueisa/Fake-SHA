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

from core.config import shap_max_words, shap_top_k
from inference.xlmr.loader import XLMRBundle
from schemas.models import ExplanationIndicator, ExplanationResult, ExplanationTopToken

_PUNCT_RE = re.compile(r"^\W+$")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")

_EXPLANATION_NOTE = (
    "SHAP identifies token contributions to the model prediction. "
    "It does not verify factual correctness."
)

_INDICATOR_KEYWORDS: dict[str, list[str]] = {
    "Linguistic Tone": [
        "emotional", "alarming", "urgent", "angry", "shocking", "controversial",
        "criticized", "praised", "fear", "threat", "warned",
    ],
    "Claim Certainty": [
        "confirmed", "proven", "definitely", "always", "never", "guaranteed",
        "sure", "allegedly", "reportedly", "claimed", "supposed",
    ],
    "Presence of Evidence-related Language": [
        "evidence", "proof", "report", "study", "data", "investigation",
        "according", "records", "documents", "findings", "statement",
    ],
    "Textual Source Attribution Mentions": [
        "said", "according to", "spokesperson", "official", "agency", "department",
        "government", "police", "authority", "expert", "researchers",
    ],
    "Sensationalism": [
        "viral", "shocking", "exposed", "unbelievable", "breaking", "must see",
        "secret", "scandal", "leaked", "click", "watch", "share",
    ],
}

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


def _is_useful_token(token: str) -> bool:
    if not token.strip():
        return False
    if _PUNCT_RE.match(token):
        return False
    return bool(_WORD_RE.search(token))


def _match_indicator(token_text: str) -> str | None:
    low = token_text.lower()
    for indicator_name, keywords in _INDICATOR_KEYWORDS.items():
        for kw in keywords:
            if kw in low:
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


def build_shap_explanation(text: str, predicted_class_index: int) -> ExplanationResult:
    """
    Build SHAP explanation payload for one text sample.

    Raises exceptions if SHAP cannot run; caller should handle fallback behavior.
    """
    clipped = _truncate_text_by_words(text)
    if not clipped.strip():
        return ExplanationResult(note=_EXPLANATION_NOTE, top_tokens=[], indicators=[])

    explainer = _get_shap_explainer()
    # Keep evaluation budget bounded for API latency.
    shap_values = explainer([clipped], max_evals=256)

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
