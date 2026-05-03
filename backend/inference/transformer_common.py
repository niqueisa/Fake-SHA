"""
shared helpers for roberta and xlm-r sequence-classification inference.

confidence is ``softmax(logits / T)[predicted]`` with ``T=1`` by default, matching
standard cross-entropy training (no post-hoc flattening). optional ``T`` comes from
:func:`core.config.get_inference_temperature` or env; ``T>1`` lowers peak
probabilities (softer distribution).
"""

from __future__ import annotations

import logging
from typing import Any

from schemas.models import AnalyzeResponse

logger = logging.getLogger(__name__)


def verdict_from_class_index(model: Any, class_index: int) -> str:
    """map predicted class id to api verdict using config.id2label when present."""
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


def run_sequence_classification(
    tokenizer: Any,
    model: Any,
    device: Any,
    combined_text: str,
    *,
    max_length: int = 512,
    temperature: float = 1.0,
) -> tuple[str, float]:
    """
    single forward pass: ``probs = softmax(logits / temperature)`` on the last
    non-batch dimension (here ``logits`` is shape ``[num_labels]``, i.e. one row
    of the batch, same as ``softmax(..., dim=1)`` for a single row).

    returns (verdict, confidence) where confidence is the probability of the
    predicted class (not always the max prob if mapping is asymmetric—here it is
    the selected class's prob).
    """
    import torch

    encoded = tokenizer(
        combined_text,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    enc_len = int(encoded["input_ids"].shape[1])

    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        # [batch, num_labels] -> one document
        logits = outputs.logits[0]

    if temperature <= 0:
        raise ValueError(f"inference temperature must be > 0, got {temperature}")

    # unscaled reference (matches training: argmax on raw logits == argmax on softmax at T=1)
    probs_t1 = torch.softmax(logits, dim=-1)
    scaled = logits / temperature
    probs = torch.softmax(scaled, dim=-1)
    pred_idx = int(torch.argmax(probs).item())
    verdict = verdict_from_class_index(model, pred_idx)
    confidence = float(probs[pred_idx].item())
    confidence = max(0.01, min(0.99, confidence))

    if logger.isEnabledFor(logging.DEBUG):
        id2 = getattr(model.config, "id2label", None)
        logger.debug(
            "transformer infer: token_len=%d max_length=%d temperature=%s",
            enc_len,
            max_length,
            temperature,
        )
        logger.debug("transformer infer: raw_logits=%s", logits.detach().cpu().tolist())
        logger.debug(
            "transformer infer: softmax(logits) T=1 %s | softmax(logits/T) T=%s %s",
            probs_t1.detach().cpu().tolist(),
            temperature,
            probs.detach().cpu().tolist(),
        )
        logger.debug(
            "transformer infer: pred_class=%d id2label=%r verdict=%s conf=%.6f",
            pred_idx,
            id2,
            verdict,
            confidence,
        )

    return verdict, confidence


def empty_input_response() -> AnalyzeResponse:
    """standard response when title/url/body combine to nothing."""
    return AnalyzeResponse(
        verdict="REAL",
        confidence=0.5,
        summary="No text provided for analysis.",
        indicators=[],
        tokens=[],
    )


def build_transformer_analyze_response(
    *,
    verdict: str,
    confidence: float,
    model: Any,
    backend_label: str,
) -> AnalyzeResponse:
    """assemble analyze response for transformer backends."""
    model_type = getattr(model.config, "model_type", "transformer")
    indicators = [
        "Source Credibility",
        "Claim Verification",
        "Language Tone",
        f"{backend_label} Prediction",
        "Consistency with Known Facts",
    ]
    summary = f"Prediction based on {backend_label} ({model_type})."
    return AnalyzeResponse(
        verdict=verdict,
        confidence=float(round(confidence, 4)),
        summary=summary,
        indicators=indicators,
        tokens=[],
    )
