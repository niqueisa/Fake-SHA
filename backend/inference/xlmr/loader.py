"""
Load tokenizer and sequence-classification head from artifacts/xlmr.

Expected layout (Hugging Face save_pretrained):

- config.json, tokenizer files, and model.safetensors or pytorch_model.bin.

XLMRBundle keeps tokenizer, model, and compute device together so explainability code
(e.g. SHAP) can reuse the same objects used at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


class XLMRArtifactError(RuntimeError):
    """Raised when the artifact directory is missing files needed for inference."""


class XLMRDependencyError(RuntimeError):
    """Raised when torch/transformers are not installed."""


def _require_artifacts(model_dir: Path) -> None:
    if not model_dir.is_dir():
        raise XLMRArtifactError(
            f"XLM-R artifacts directory not found: {model_dir}. "
            "Place a Hugging Face save_pretrained output under backend/artifacts/xlmr/."
        )

    if not (model_dir / "config.json").is_file():
        raise XLMRArtifactError(
            f"Missing config.json in {model_dir}. Export the trained model with save_pretrained()."
        )

    has_weights = (
        (model_dir / "model.safetensors").is_file()
        or (model_dir / "pytorch_model.bin").is_file()
    )

    if not has_weights:
        raise XLMRArtifactError(
            f"No model weights in {model_dir}. Expected model.safetensors or pytorch_model.bin."
        )


@dataclass
class XLMRBundle:
    """Holds tokenizer + model + compute device."""

    tokenizer: Any
    model: Any
    device: Any


@lru_cache(maxsize=1)
def load_bundle() -> XLMRBundle:
    """
    Load and cache tokenizer + model once per process.

    Raises:
        XLMRArtifactError: Missing or incomplete artifact tree.
        XLMRDependencyError: torch / transformers not installed.
    """
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        raise XLMRDependencyError(
            "XLM-R inference requires torch and transformers. "
            "Install with: pip install torch transformers safetensors"
        ) from e

    # 🔥 IMPORTANT: new artifact path
    model_dir = Path(__file__).resolve().parent.parent.parent / "artifacts" / "xlmr"

    _require_artifacts(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

    model.eval()
    model.to(device)

    return XLMRBundle(tokenizer=tokenizer, model=model, device=device)