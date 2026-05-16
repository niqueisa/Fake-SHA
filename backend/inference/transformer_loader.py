"""Shared Hugging Face ``from_pretrained`` loading for sequence classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

from core.model_artifacts import LocalArtifactError, prepare_pretrained_ref


class TransformerArtifactError(LocalArtifactError):
    """Raised when model artifacts cannot be loaded."""


class TransformerDependencyError(RuntimeError):
    """Raised when torch/transformers are not installed."""


@dataclass
class TransformerBundle:
    tokenizer: Any
    model: Any
    device: Any
    model_ref: str


def load_transformer_bundle(
    model_ref: str,
    *,
    analyzer_name: str,
    cache_key: str,
) -> TransformerBundle:
    """
    Load tokenizer + sequence classification model from Hub or local ``save_pretrained``.

    ``cache_key`` must be unique per analyzer so ``lru_cache`` wrappers do not collide.
    """
    del cache_key  # used by per-analyzer @lru_cache wrappers only
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        raise TransformerDependencyError(
            f"{analyzer_name} inference requires torch and transformers. "
            "Install with: pip install torch transformers safetensors"
        ) from e

    prepared = prepare_pretrained_ref(model_ref, analyzer_name=analyzer_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(prepared)
    model = AutoModelForSequenceClassification.from_pretrained(prepared)
    model.eval()
    model.to(device)

    return TransformerBundle(
        tokenizer=tokenizer,
        model=model,
        device=device,
        model_ref=prepared,
    )


def cached_transformer_loader(
    model_ref_fn: Callable[[], str],
    *,
    analyzer_name: str,
) -> Callable[[], TransformerBundle]:
    """Build a single-entry cached loader for one analyzer."""

    @lru_cache(maxsize=1)
    def _load() -> TransformerBundle:
        return load_transformer_bundle(
            model_ref_fn(),
            analyzer_name=analyzer_name,
            cache_key=analyzer_name,
        )

    return _load
