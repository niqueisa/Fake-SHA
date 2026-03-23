"""
Compose the string passed to the RoBERTa tokenizer.

Re-exports :func:`core.model_input.build_model_input` so training and inference
share one definition with SVM.
"""

from core.model_input import build_model_input

__all__ = ["build_model_input"]
