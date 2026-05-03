"""
load tokenizer and sequence-classification head from env-configured source.

model source: ``FAKE_SHA_ROBERTA_MODEL`` (huggingface hub id or local path);
default: ``backend/artifacts/roberta`` (resolved under the backend package).

roberta bundle keeps tokenizer, model, and device together for explainability (e.g. shap).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.config import get_roberta_model_source

logger = logging.getLogger(__name__)


class RoBERTaArtifactError(RuntimeError):
    """raised when the model cannot be loaded for inference."""


class RoBERTaDependencyError(RuntimeError):
    """raised when torch/transformers are not installed."""


def _require_local_artifacts(model_dir: Path) -> None:
    """validate a local save_pretrained directory before load."""
    if not model_dir.is_dir():
        raise RoBERTaArtifactError(
            f"RoBERTa artifacts directory not found: {model_dir}. "
            "Set FAKE_SHA_ROBERTA_MODEL to a hub id or a path with config.json and weights."
        )
    if not (model_dir / "config.json").is_file():
        raise RoBERTaArtifactError(
            f"Missing config.json in {model_dir}. Export the trained model with save_pretrained()."
        )
    has_weights = (model_dir / "model.safetensors").is_file() or (model_dir / "pytorch_model.bin").is_file()
    if not has_weights:
        raise RoBERTaArtifactError(
            f"No model weights in {model_dir}. Expected model.safetensors or pytorch_model.bin."
        )


@dataclass
class RoBERTaBundle:
    """holds tokenizer + model + compute device (shap can reuse these references)."""

    tokenizer: Any
    model: Any
    device: Any


def _is_resolved_local_dir(model_source: str) -> bool:
    p = Path(model_source).expanduser()
    return p.is_dir()


@lru_cache(maxsize=1)
def load_model() -> RoBERTaBundle:
    """
    load and cache tokenizer + model once per process.

    model location comes from ``FAKE_SHA_ROBERTA_MODEL`` (see :mod:`core.config`).

    raises:
        roberta artifact error: missing local files or load failure.
        roberta dependency error: torch / transformers not installed.
    """
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        raise RoBERTaDependencyError(
            "RoBERTa inference requires torch and transformers. "
            "Install with: pip install torch transformers safetensors"
        ) from e

    model_source = get_roberta_model_source()

    if _is_resolved_local_dir(model_source):
        _require_local_artifacts(Path(model_source).expanduser().resolve())
        origin = "local directory"
    else:
        origin = "huggingface hub or remote path"

    logger.info("loading roberta model from %r (%s)", model_source, origin)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_source)
        model = AutoModelForSequenceClassification.from_pretrained(model_source)
    except Exception as e:
        raise RoBERTaArtifactError(
            f"Failed to load RoBERTa model from {model_source!r}: {e}"
        ) from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    return RoBERTaBundle(tokenizer=tokenizer, model=model, device=device)
