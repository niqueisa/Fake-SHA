"""
Load XLM-R tokenizer and sequence-classification head.

Source (first match wins):
  - ``FAKE_SHA_XLMR_HUB_ID`` — Hugging Face Hub, e.g. ``your-org/fake-sha-xlmr``
  - ``FAKE_SHA_XLMR_ARTIFACT_DIR`` / ``FAKE_SHA_XLMR_MODEL`` — local folder or Hub id
  - ``backend/artifacts/xlmr/`` — default local layout (save_pretrained)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.config import get_xlmr_model_ref
from inference.transformer_loader import (
    TransformerArtifactError,
    TransformerBundle,
    TransformerDependencyError,
    cached_transformer_loader,
)

# Backward-compatible exception names for main.py handlers
XLMRArtifactError = TransformerArtifactError
XLMRDependencyError = TransformerDependencyError

_load_cached = cached_transformer_loader(get_xlmr_model_ref, analyzer_name="XLM-R")


@dataclass
class XLMRBundle:
    """Holds tokenizer + model + compute device."""

    tokenizer: Any
    model: Any
    device: Any


def load_bundle() -> XLMRBundle:
    """Load and cache tokenizer + model once per process."""
    bundle: TransformerBundle = _load_cached()
    return XLMRBundle(
        tokenizer=bundle.tokenizer,
        model=bundle.model,
        device=bundle.device,
    )
