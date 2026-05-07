"""
Load tokenizer and sequence-classification head from artifacts/roberta.

Expected layout (Hugging Face save_pretrained):

- config.json, tokenizer files, and model.safetensors or pytorch_model.bin.

RoBERTaBundle keeps tokenizer, model, and compute device together so explainability code
(e.g. SHAP) can wrap the same objects used at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.config import ARTIFACTS_ROBERTA_DIR


class RoBERTaArtifactError(RuntimeError):
    """Raised when the artifact directory is missing files needed for inference."""


class RoBERTaDependencyError(RuntimeError):
    """Raised when torch/transformers are not installed."""


def _require_artifacts(model_dir: Path) -> None:
    if not model_dir.is_dir():
        raise RoBERTaArtifactError(
            f"RoBERTa artifacts directory not found: {model_dir}. "
            "Place a Hugging Face save_pretrained output under backend/artifacts/roberta/."
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
    """Holds tokenizer + model + compute device (SHAP can reuse these references)."""

    tokenizer: Any
    model: Any
    device: Any


@lru_cache(maxsize=1)
def load_bundle() -> RoBERTaBundle:
    """
    Load and cache tokenizer + model once per process.

    Raises:
        RoBERTaArtifactError: Missing or incomplete artifact tree.
        RoBERTaDependencyError: torch / transformers not installed.
    """
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        raise RoBERTaDependencyError(
            "RoBERTa inference requires torch and transformers. "
            "Install with: pip install torch transformers safetensors"
        ) from e

    model_dir = ARTIFACTS_ROBERTA_DIR
    _require_artifacts(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    model.to(device)
    return RoBERTaBundle(tokenizer=tokenizer, model=model, device=device)
