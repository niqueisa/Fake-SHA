"""
XLM-RoBERTa inference: same AnalyzeResponse contract as SVM and RoBERTa.

Confidence is computed using temperature scaling over logits for realistic probabilities.
"""

from __future__ import annotations

import math

from core.config import (
    shap_enabled,
    xlmr_confidence_calibration_strength,
    xlmr_confidence_cap,
    xlmr_confidence_floor,
    xlmr_confidence_margin_weight,
    xlmr_inference_temperature,
)
from explainability.xlmr_shap import build_shap_explanation, explanation_unavailable
from .loader import load_bundle
from core.model_input import inference_input_text
from schemas.models import AnalyzeResponse
import torch


def _label_to_verdict(model, class_index: int) -> str:
    """Map predicted class id to API verdict using config.id2label when present."""
    id2label = getattr(model.config, "id2label", None)
    if isinstance(id2label, dict):
        raw = id2label.get(class_index) or id2label.get(str(class_index))
        if raw is not None:
            s = str(raw).upper()
            if "FAKE" in s or "FALSE" in s or s.strip() == "0":
                return "FAKE"
            if "REAL" in s or "TRUE" in s or s.strip() == "1":
                return "REAL"

    if int(getattr(model.config, "num_labels", 2)) == 2:
        return "REAL" if class_index == 1 else "FAKE"

    return "REAL"


def _calibrate_confidence(probs_vec: torch.Tensor, pred_idx: int) -> float:
    """
    Map softmax probabilities to a wider, user-facing confidence range.

    Uses top-2 margin and normalized entropy so borderline predictions land near
    ~55–70% and decisive ones near ~85–97%, instead of clustering around ~90%.
    """
    probs = probs_vec.detach().float().cpu()
    n_classes = max(2, int(probs.numel()))
    raw = float(probs[pred_idx].item())

    sorted_probs, _ = torch.sort(probs, descending=True)
    top1 = float(sorted_probs[0].item())
    top2 = float(sorted_probs[1].item()) if n_classes > 1 else 0.0
    margin = max(0.0, min(1.0, top1 - top2))
    # Soften very large margins so confidence does not saturate at one value.
    margin_signal = math.sqrt(margin)

    safe_probs = torch.clamp(probs, min=1e-12)
    entropy = float(-(safe_probs * torch.log(safe_probs)).sum().item())
    max_entropy = math.log(float(n_classes)) if n_classes > 1 else 1.0
    entropy_norm = entropy / max_entropy if max_entropy > 0 else 0.0
    certainty_entropy = max(0.0, min(1.0, 1.0 - entropy_norm))

    margin_weight = xlmr_confidence_margin_weight()
    certainty = (margin_weight * margin_signal) + ((1.0 - margin_weight) * certainty_entropy)

    conf_floor = xlmr_confidence_floor()
    conf_cap = xlmr_confidence_cap()
    target = conf_floor + ((conf_cap - conf_floor) * certainty)

    strength = xlmr_confidence_calibration_strength()
    calibrated = raw + (target - raw) * strength

    return max(conf_floor, min(conf_cap, calibrated))


def analyze_text(
    text: str,
    title: str = "",
    url: str = "",
    mode: str = "selection_only",
) -> AnalyzeResponse:
    """
    Run XLM-RoBERTa classification; returns verdict, confidence, summary, indicators, tokens.
    """

    combined = inference_input_text(text, title=title, url=url, mode=mode)

    if not combined.strip():
        return AnalyzeResponse(
            verdict="REAL",
            confidence=0.5,
            summary="No text provided for analysis.",
            indicators=[],
            tokens=[],
            explanation=None,
        )

    # Load model bundle (XLM-R)
    bundle = load_bundle()
    model = bundle.model

    probs_batch = predict_proba_texts([combined], bundle=bundle)
    probs_vec = probs_batch[0]

    # Temperature scaling (important for your overconfidence issue)
    pred_idx = int(torch.argmax(probs_vec).item())
    verdict = _label_to_verdict(model, pred_idx)
    confidence = _calibrate_confidence(probs_vec, pred_idx)

    # Legacy top-level `indicators` is kept for API compatibility only.
    # Real explainability indicators are returned in `explanation.indicators`.
    indicators = []

    summary = (
        f"Prediction based on XLM-RoBERTa "
        f"({getattr(model.config, 'model_type', 'transformer')})."
    )

    explanation = None
    if shap_enabled():
        try:
            explanation = build_shap_explanation(combined, predicted_class_index=pred_idx)
        except Exception:
            explanation = explanation_unavailable()

    return AnalyzeResponse(
        verdict=verdict,
        confidence=float(round(confidence, 4)),
        summary=summary,
        indicators=indicators,
        tokens=[],
        explanation=explanation,
    )


def predict_proba_texts(texts: list[str], bundle=None):
    """Predict class probabilities (softmax) for a batch of text inputs."""
    if bundle is None:
        bundle = load_bundle()
    tokenizer = bundle.tokenizer
    model = bundle.model
    device = bundle.device

    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=512,
        padding=True,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits

    temperature = xlmr_inference_temperature()
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    return probs