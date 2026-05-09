"""
Backend configuration (paths, analyzer selection).

Environment:
    FAKE_SHA_ANALYZER  - "svm" (default), "roberta", "xlmr", or "mock"
    FAKE_SHA_XLMR_ARTIFACT_DIR - optional override for XLM-R model directory
    FAKE_SHA_XLMR_MODEL        - alias for FAKE_SHA_XLMR_ARTIFACT_DIR (backward compatibility)
    FAKE_SHA_XLMR_TEMPERATURE  - softmax temperature for XLM-R logits (confidence + SHAP wrapper)
"""

from __future__ import annotations

import os
from pathlib import Path

# backend/ directory (parent of core/)
BACKEND_ROOT: Path = Path(__file__).resolve().parent.parent

# Persisted ML weights (not Pydantic schemas)
ARTIFACTS_SVM_DIR: Path = BACKEND_ROOT / "artifacts" / "svm"
ARTIFACTS_ROBERTA_DIR: Path = BACKEND_ROOT / "artifacts" / "roberta"

# Default XLM-R Hugging Face save_pretrained directory
ARTIFACTS_XLMR_DIR: Path = BACKEND_ROOT / "artifacts" / "xlmr"


def get_xlmr_artifacts_dir() -> Path:
    """Resolved directory for XLM-R weights and tokenizer (env override or default)."""
    raw = (
        os.environ.get("FAKE_SHA_XLMR_ARTIFACT_DIR")
        or os.environ.get("FAKE_SHA_XLMR_MODEL")
        or ""
    ).strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return ARTIFACTS_XLMR_DIR.resolve()


def cors_allow_origins() -> list[str]:
    """
    Comma-separated origins for CORS, or ``*`` for allow-all.

    Production example: ``https://your-extension-id.chromium.app`` or your HTTPS frontend URL.
    """
    raw = os.environ.get("FAKE_SHA_CORS_ORIGINS", "*").strip()
    if not raw or raw == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


def xlmr_inference_temperature() -> float:
    """
    Temperature T for softmax(logits / T) on XLM-R outputs.

    Used by both inference and SHAP wrapper so explanations align with displayed confidence.
    """
    raw = os.environ.get("FAKE_SHA_XLMR_TEMPERATURE", "10.0").strip()
    try:
        t = float(raw)
        return max(0.1, min(100.0, t))
    except ValueError:
        return 10.0

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


def shap_cache_enabled() -> bool:
    """Cache SHAP results in-memory to speed up repeated analyses."""
    return os.environ.get("SHAP_CACHE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


def shap_cache_maxsize() -> int:
    """Max number of distinct SHAP explanations to keep in memory."""
    raw = os.environ.get("SHAP_CACHE_MAXSIZE", "128").strip()
    try:
        return max(0, min(2048, int(raw)))
    except ValueError:
        return 128