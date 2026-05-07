"""
Backend configuration (paths, analyzer selection).

Environment:
    FAKE_SHA_ANALYZER  - "svm" (default), "roberta", "xlmr", or "mock"
"""

from __future__ import annotations

import os
from pathlib import Path

# backend/ directory (parent of core/)
BACKEND_ROOT: Path = Path(__file__).resolve().parent.parent

# Persisted ML weights (not Pydantic schemas)
ARTIFACTS_SVM_DIR: Path = BACKEND_ROOT / "artifacts" / "svm"
ARTIFACTS_ROBERTA_DIR: Path = BACKEND_ROOT / "artifacts" / "roberta"

# 🔥 NEW: XLM-R artifacts directory
ARTIFACTS_XLMR_DIR: Path = BACKEND_ROOT / "artifacts" / "xlmr"

# Must match AnalyzeRequest.analyzer Literal and inference.factory branches.
VALID_ANALYZER_BACKENDS: frozenset[str] = frozenset(
    {"svm", "roberta", "xlmr", "mock"}
)


class UnknownAnalyzerBackendError(ValueError):
    """Raised when FAKE_SHA_ANALYZER (or a non-schema analyzer override) is not recognized."""


def get_analyzer_backend() -> str:
    """Which analyzer implementation to use for POST /analyze when the request omits ``analyzer``."""
    return os.environ.get("FAKE_SHA_ANALYZER", "svm").strip().lower()


def shap_enabled() -> bool:
    """Feature flag for SHAP explainability."""
    return os.environ.get("ENABLE_SHAP", "true").strip().lower() in {"1", "true", "yes", "on"}


def shap_max_words() -> int:
    """Max number of words to pass into SHAP for performance."""
    raw = os.environ.get("SHAP_MAX_WORDS", "400").strip()
    try:
        return max(50, min(1000, int(raw)))
    except ValueError:
        return 400


def shap_top_k() -> int:
    """Top-K tokens returned in explanation payload."""
    raw = os.environ.get("SHAP_TOP_K", "20").strip()
    try:
        return max(1, min(100, int(raw)))
    except ValueError:
        return 20


def shap_max_evals() -> int:
    """
    Max SHAP partition evaluations per request.
    Lower values are faster but less granular.
    """
    raw = os.environ.get("SHAP_MAX_EVALS", "96").strip()
    try:
        return max(16, min(1024, int(raw)))
    except ValueError:
        return 96