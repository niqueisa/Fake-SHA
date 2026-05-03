"""
xlm-roberta inference: same api response shape as svm and mock.

confidence is softmax on model logits (default T=1); see core.config for optional T.
"""

from __future__ import annotations

import logging

from core.config import get_inference_temperature, get_xlmr_model_source
from .loader import load_model
from .preprocess import build_model_input
from inference.transformer_common import (
    build_transformer_analyze_response,
    empty_input_response,
    run_sequence_classification,
)
from schemas.models import AnalyzeResponse

logger = logging.getLogger(__name__)


def analyze_text(text: str, title: str = "", url: str = "") -> AnalyzeResponse:
    """run xlm-roberta classification; returns verdict, confidence, summary, indicators, tokens."""
    combined = build_model_input(text, title=title, url=url)

    if not combined.strip():
        return empty_input_response()

    model_source = get_xlmr_model_source()
    temperature = get_inference_temperature(model_source, backend="xlmr")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "xlmr model_source=%r inference_temperature=%s",
            model_source,
            temperature,
        )
    bundle = load_model()
    verdict, confidence = run_sequence_classification(
        bundle.tokenizer,
        bundle.model,
        bundle.device,
        combined,
        temperature=temperature,
    )
    return build_transformer_analyze_response(
        verdict=verdict,
        confidence=confidence,
        model=bundle.model,
        backend_label="XLM-RoBERTa",
    )
