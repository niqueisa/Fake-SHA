"""
RoBERTa inference: same AnalyzeResponse contract as SVM and mock.

Forward pass uses load_bundle() so SHAP or similar tools can later use the
same model and tokenizer without changing the API surface.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

try:
    from core.config import ARTIFACTS_ROBERTA_DIR
except ModuleNotFoundError:
    from backend.core.config import ARTIFACTS_ROBERTA_DIR

from .loader import load_bundle
from .preprocess import build_model_input
from schemas.models import AnalyzeResponse


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


@lru_cache(maxsize=1)
def _load_temperature(model_dir: Path = ARTIFACTS_ROBERTA_DIR) -> float:
    """Load scalar temperature for logit scaling; fallback to 1.0 on any issue."""
    path = model_dir / "temperature.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        value = float(data.get("temperature", 1.0))
        if value <= 0:
            return 1.0
        return value
    except Exception:
        return 1.0


def analyze_text(text: str, title: str = "", url: str = "") -> AnalyzeResponse:
    """
    Run RoBERTa classification; returns verdict, confidence, summary, indicators, tokens.

    tokens is left empty until SHAP integration (schema uses TokenResult).
    """
    import torch

    combined = build_model_input(text, title=title, url=url)
    if not combined.strip():
        return AnalyzeResponse(
            verdict="REAL",
            confidence=0.5,
            summary="No text provided for analysis.",
            indicators=[],
            tokens=[],
        )

    bundle = load_bundle()
    tokenizer = bundle.tokenizer
    model = bundle.model
    device = bundle.device
    temperature = _load_temperature()

    encoded = tokenizer(
        combined,
        truncation=True,
        max_length=512,
        padding=True,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits

    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)[0]
    pred_idx = int(torch.argmax(probs).item())
    verdict = _label_to_verdict(model, pred_idx)
    confidence = float(probs[pred_idx].item())
    confidence = max(0.0, min(1.0, confidence))

    indicators = [
        "Source Credibility",
        "Claim Verification",
        "Language Tone",
        "RoBERTa Prediction",
        "Consistency with Known Facts",
    ]
    summary = f"Prediction based on RoBERTa ({getattr(model.config, 'model_type', 'transformer')})."

    return AnalyzeResponse(
        verdict=verdict,
        confidence=float(round(confidence, 4)),
        summary=summary,
        indicators=indicators,
        tokens=[],
    )
