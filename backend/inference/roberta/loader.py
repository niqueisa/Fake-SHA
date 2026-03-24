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

    # Tokenizer files are required for AutoTokenizer.from_pretrained().
    # Some tokenizers store either a single tokenizer.json or a (vocab.json + merges.txt) pair.
    has_tokenizer_json = (model_dir / "tokenizer.json").is_file()
    has_vocab_merges = (model_dir / "vocab.json").is_file() and (model_dir / "merges.txt").is_file()
    has_tokenizer_cfg = (model_dir / "tokenizer_config.json").is_file()
    if not has_tokenizer_cfg or not (has_tokenizer_json or has_vocab_merges):
        raise RoBERTaArtifactError(
            "Missing tokenizer files in RoBERTa artifacts. Expected tokenizer_config.json plus "
            "tokenizer.json or (vocab.json + merges.txt). "
            f"Artifacts found under {model_dir} may be incomplete."
        )


def _ensure_label_mapping(model: Any) -> None:
    """
    Ensure config-based label mapping exists and matches thesis convention:
      0 -> FAKE
      1 -> REAL
    """
    try:
        id2label = getattr(model.config, "id2label", None)
        label2id = getattr(model.config, "label2id", None)

        ok_id2label = isinstance(id2label, dict) and (0 in id2label or "0" in id2label) and (1 in id2label or "1" in id2label)
        ok_label2id = isinstance(label2id, dict) and ("FAKE" in label2id) and ("REAL" in label2id)

        if ok_id2label and ok_label2id:
            return

        # Fall back to the known convention so analyze_text() verdict mapping is stable.
        model.config.id2label = {0: "FAKE", 1: "REAL"}
        model.config.label2id = {"FAKE": 0, "REAL": 1}
        # Keep num_labels consistent with the classifier head.
        model.config.num_labels = 2
    except Exception as e:
        # Do not prevent loading the model; inference can still work with argmax fallback.
        raise RoBERTaArtifactError(f"Failed to ensure label mapping in RoBERTa config: {e}") from e


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
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    except Exception as e:
        raise RoBERTaArtifactError(
            f"Failed to load RoBERTa tokenizer from {model_dir}: {e}"
        ) from e

    try:
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    except Exception as e:
        raise RoBERTaArtifactError(
            f"Failed to load RoBERTa model weights from {model_dir}: {e}"
        ) from e

    _ensure_label_mapping(model)
    model.eval()
    model.to(device)
    return RoBERTaBundle(tokenizer=tokenizer, model=model, device=device)
