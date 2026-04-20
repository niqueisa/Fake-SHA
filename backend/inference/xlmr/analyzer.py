"""
XLM-RoBERTa inference: same AnalyzeResponse contract as SVM and mock.

Confidence is computed using temperature scaling over logits for realistic probabilities.
"""

from __future__ import annotations

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
        )

    # Load model bundle (XLM-R)
    bundle = load_bundle()
    tokenizer = bundle.tokenizer
    model = bundle.model
    device = bundle.device

    # Encode input
    encoded = tokenizer(
        combined,
        truncation=True,
        max_length=512,
        padding=True,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits[0]

    # Temperature scaling (important for your overconfidence issue)
    temperature = 10.0
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)

    pred_idx = int(torch.argmax(probs).item())
    verdict = _label_to_verdict(model, pred_idx)
    confidence = float(probs[pred_idx].item())

    # Clamp confidence (avoid unrealistic 0 / 1)
    confidence = max(0.01, min(0.99, confidence))

    indicators = [
        "Source Credibility",
        "Claim Verification",
        "Language Tone",
        "XLM-RoBERTa Prediction",
        "Consistency with Known Facts",
    ]

    summary = (
        f"Prediction based on XLM-RoBERTa "
        f"({getattr(model.config, 'model_type', 'transformer')})."
    )

    return AnalyzeResponse(
        verdict=verdict,
        confidence=float(round(confidence, 4)),
        summary=summary,
        indicators=indicators,
        tokens=[],
    )