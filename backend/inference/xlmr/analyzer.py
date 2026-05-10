"""
XLM-RoBERTa inference: same AnalyzeResponse contract as SVM and RoBERTa.

Confidence is computed using temperature scaling over logits for realistic probabilities.
"""

from __future__ import annotations

from core.config import (
    shap_enabled,
    xlmr_confidence_calibration_strength,
    xlmr_confidence_margin_weight,
    xlmr_inference_temperature,
)
from explainability.xlmr_shap import build_shap_explanation, explanation_unavailable
from .loader import load_bundle
from .preprocess import build_model_input
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
    Calibrate confidence using margin + entropy certainty.

    This widens confidence range in a controlled way:
    - clearer predictions move slightly upward
    - uncertain predictions move slightly downward

    Certainty uses two signals:
    1) top-2 margin (how far winner is from runner-up),
    2) normalized entropy (how concentrated the full distribution is).
    Config weights control the blend and overall calibration strength.
    """
    probs = probs_vec.detach().float().cpu()
    n_classes = max(2, int(probs.numel()))
    base = 1.0 / float(n_classes)
    raw = float(probs[pred_idx].item())

    sorted_probs, _ = torch.sort(probs, descending=True)
    top1 = float(sorted_probs[0].item())
    top2 = float(sorted_probs[1].item()) if n_classes > 1 else 0.0
    margin = max(0.0, min(1.0, top1 - top2))

    safe_probs = torch.clamp(probs, min=1e-12)
    entropy = float(-(safe_probs * torch.log(safe_probs)).sum().item())
    max_entropy = float(torch.log(torch.tensor(float(n_classes))).item())
    entropy_norm = entropy / max_entropy if max_entropy > 0 else 0.0
    certainty_entropy = max(0.0, min(1.0, 1.0 - entropy_norm))

    margin_weight = xlmr_confidence_margin_weight()
    certainty = (margin_weight * margin) + ((1.0 - margin_weight) * certainty_entropy)
    target = base + ((1.0 - base) * certainty)

    strength = xlmr_confidence_calibration_strength()
    calibrated = raw + (target - raw) * strength

    # Keep confidence realistic and avoid exact 0/1.
    calibrated = max(base, min(0.99, calibrated))
    return calibrated


def analyze_text(text: str, title: str = "", url: str = "") -> AnalyzeResponse:
    """
    Run XLM-RoBERTa classification; returns verdict, confidence, summary, indicators, tokens.
    """

    combined = build_model_input(text, title=title, url=url)

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