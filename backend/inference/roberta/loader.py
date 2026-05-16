"""
Load RoBERTa tokenizer and sequence-classification head.

Source (first match wins):
  - ``FAKE_SHA_ROBERTA_HUB_ID`` — Hugging Face Hub
  - ``FAKE_SHA_ROBERTA_ARTIFACT_DIR`` / ``FAKE_SHA_ROBERTA_MODEL`` — local or Hub id
  - ``backend/artifacts/roberta/`` — default local layout
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.config import get_roberta_model_ref
from inference.transformer_loader import (
    TransformerArtifactError,
    TransformerBundle,
    TransformerDependencyError,
    cached_transformer_loader,
)

RoBERTaArtifactError = TransformerArtifactError
RoBERTaDependencyError = TransformerDependencyError

_load_cached = cached_transformer_loader(get_roberta_model_ref, analyzer_name="RoBERTa")


@dataclass
class RoBERTaBundle:
    """Holds tokenizer + model + compute device."""

    tokenizer: Any
    model: Any
    device: Any


def load_bundle() -> RoBERTaBundle:
    """Load and cache tokenizer + model once per process."""
    bundle: TransformerBundle = _load_cached()
    return RoBERTaBundle(
        tokenizer=bundle.tokenizer,
        model=bundle.model,
        device=bundle.device,
    )
