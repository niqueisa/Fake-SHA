"""
Backend configuration (paths, analyzer selection).

Environment:
    FAKE_SHA_ANALYZER  - "svm" (default), "roberta", or "xlmr"
    FAKE_SHA_XLMR_HUB_ID       - Hugging Face repo id, e.g. your-org/fake-sha-xlmr
    FAKE_SHA_XLMR_ARTIFACT_DIR - local save_pretrained folder (optional)
    FAKE_SHA_XLMR_MODEL        - alias for ARTIFACT_DIR, or Hub id if org/name (backward compatible)
    FAKE_SHA_ROBERTA_HUB_ID / FAKE_SHA_ROBERTA_ARTIFACT_DIR / FAKE_SHA_ROBERTA_MODEL — same for RoBERTa
    FAKE_SHA_SVM_ARTIFACT_DIR  - local folder with svm pickles (SVM is not loaded from the Hub)
    FAKE_SHA_XLMR_TEMPERATURE  - softmax temperature for XLM-R logits (confidence + SHAP wrapper)
"""

from __future__ import annotations

import os
from pathlib import Path

from core.model_artifacts import resolve_pretrained_source

# backend/ directory (parent of core/)
BACKEND_ROOT: Path = Path(__file__).resolve().parent.parent

# Persisted ML weights (not Pydantic schemas)
ARTIFACTS_SVM_DIR: Path = BACKEND_ROOT / "artifacts" / "svm"
ARTIFACTS_ROBERTA_DIR: Path = BACKEND_ROOT / "artifacts" / "roberta"

# Default XLM-R Hugging Face save_pretrained directory
ARTIFACTS_XLMR_DIR: Path = BACKEND_ROOT / "artifacts" / "xlmr"


def get_xlmr_model_ref() -> str:
    """Hub repo id or local path for ``from_pretrained`` (XLM-R)."""
    return resolve_pretrained_source(
        hub_id=os.environ.get("FAKE_SHA_XLMR_HUB_ID"),
        artifact_dir=os.environ.get("FAKE_SHA_XLMR_ARTIFACT_DIR"),
        legacy_model_env=os.environ.get("FAKE_SHA_XLMR_MODEL"),
        default_local_dir=ARTIFACTS_XLMR_DIR,
    )


def get_roberta_model_ref() -> str:
    """Hub repo id or local path for ``from_pretrained`` (RoBERTa)."""
    return resolve_pretrained_source(
        hub_id=os.environ.get("FAKE_SHA_ROBERTA_HUB_ID"),
        artifact_dir=os.environ.get("FAKE_SHA_ROBERTA_ARTIFACT_DIR"),
        legacy_model_env=os.environ.get("FAKE_SHA_ROBERTA_MODEL"),
        default_local_dir=ARTIFACTS_ROBERTA_DIR,
    )


def get_svm_artifacts_dir() -> Path:
    """Local directory for SVM pickles (Hub not supported)."""
    raw = os.environ.get("FAKE_SHA_SVM_ARTIFACT_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return ARTIFACTS_SVM_DIR.resolve()


def get_xlmr_artifacts_dir() -> Path:
    """Backward-compatible local path helper; prefer ``get_xlmr_model_ref()``."""
    ref = get_xlmr_model_ref()
    return Path(ref).expanduser().resolve()


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
    raw = os.environ.get("FAKE_SHA_XLMR_TEMPERATURE", "6.0").strip()
    try:
        t = float(raw)
        return max(0.1, min(100.0, t))
    except ValueError:
        return 6.0


def xlmr_confidence_calibration_strength() -> float:
    """
    Blend strength for post-softmax confidence calibration.

    0.0 -> keep raw softmax confidence
    1.0 -> fully use calibrated confidence target
    """
    raw = os.environ.get("FAKE_SHA_XLMR_CONF_CAL_STRENGTH", "0.9").strip()
    try:
        value = float(raw)
        return max(0.0, min(1.0, value))
    except ValueError:
        return 0.9


def xlmr_confidence_margin_weight() -> float:
    """
    Weight of top-2 probability margin in confidence certainty score.
    """
    raw = os.environ.get("FAKE_SHA_XLMR_CONF_MARGIN_WEIGHT", "0.65").strip()
    try:
        value = float(raw)
        return max(0.0, min(1.0, value))
    except ValueError:
        return 0.65


def xlmr_confidence_floor() -> float:
    """Minimum displayed confidence (0–1) after calibration."""
    raw = os.environ.get("FAKE_SHA_XLMR_CONF_FLOOR", "0.52").strip()
    try:
        value = float(raw)
        return max(0.5, min(0.9, value))
    except ValueError:
        return 0.52


def xlmr_confidence_cap() -> float:
    """Maximum displayed confidence (0–1) after calibration."""
    raw = os.environ.get("FAKE_SHA_XLMR_CONF_CAP", "0.97").strip()
    try:
        value = float(raw)
        return max(0.6, min(0.99, value))
    except ValueError:
        return 0.97

# Must match AnalyzeRequest.analyzer Literal and inference.factory branches.
VALID_ANALYZER_BACKENDS: frozenset[str] = frozenset(
    {"svm", "roberta", "xlmr"}
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