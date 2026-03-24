"""
RoBERTa inference: same AnalyzeResponse contract as SVM and mock.

Forward pass uses load_bundle() so SHAP or similar tools can later use the
same model and tokenizer without changing the API surface.
"""

from __future__ import annotations

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

    probs = torch.softmax(logits, dim=-1)[0]
    pred_idx = int(torch.argmax(probs).item())
    verdict = _label_to_verdict(model, pred_idx)
    confidence = float(probs[pred_idx].item())
    confidence = max(0.0, min(1.0, confidence))

    # Debug output for thesis integration:
    # - logits: raw model outputs (shape [2])
    # - probs: softmax probabilities (shape [2], sum ~= 1)
    # - pred_idx: chosen class index (0/1)
    # - confidence: probability of chosen class (0..1)
    try:
        logits_dbg = logits.detach().cpu().tolist()
        probs_dbg = probs.detach().cpu().tolist()
    except Exception:
        logits_dbg = "unavailable"
        probs_dbg = "unavailable"
    print(
        f"[FAKE-SHA][RoBERTa][confidence-debug] logits={logits_dbg}, probs={probs_dbg}, "
        f"pred_idx={pred_idx}, confidence={confidence}"
    )

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
        confidence=float(confidence),
        summary=summary,
        indicators=indicators,
        tokens=[],
    )


def roberta_smoke_test() -> AnalyzeResponse:
    """
    Lightweight internal test to verify:
    - artifacts load successfully
    - model forward pass runs
    - output format is valid
    """
    sample_text = "Breaking news: this story claims an incredible event happened."
    resp = analyze_text(sample_text, title="Smoke test", url="https://example.com")

    if resp.verdict not in ("FAKE", "REAL"):
        raise RuntimeError(f"Unexpected verdict from RoBERTa smoke test: {resp.verdict!r}")
    if not (0.0 <= resp.confidence <= 1.0):
        raise RuntimeError(f"Confidence out of range from RoBERTa smoke test: {resp.confidence}")

    return resp
