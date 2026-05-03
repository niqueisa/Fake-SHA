"""
Backend configuration (paths, analyzer selection).

Environment:
    FAKE_SHA_ANALYZER              - "svm" (default), "roberta", "xlmr", or "mock"
    FAKE_SHA_ROBERTA_MODEL         - Hugging Face model id or path (default: backend/artifacts/roberta)
    FAKE_SHA_XLMR_MODEL            - Hugging Face model id or path (default: backend/artifacts/xlmr)
    FAKE_SHA_ROBERTA_TEMPERATURE   - optional softmax temperature for roberta confidence (default: 1.0)
    FAKE_SHA_XLMR_TEMPERATURE      - optional softmax temperature for xlm-r confidence (default: 1.0)
    FAKE_SHA_TRANSFORMER_TEMPERATURE - fallback if backend-specific temperature env is unset

softmax uses ``softmax(logits / T)``; ``T=1`` matches training-time cross-entropy (no extra scaling).
``T>1`` spreads probabilities (lower peak confidence); ``T<1`` sharpens them.
optional ``temperature.json`` in a local artifact dir: ``{"temperature": <float>}``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# backend/ directory (parent of core/)
BACKEND_ROOT: Path = Path(__file__).resolve().parent.parent

# Persisted ML weights (not Pydantic schemas)
ARTIFACTS_SVM_DIR: Path = BACKEND_ROOT / "artifacts" / "svm"
ARTIFACTS_ROBERTA_DIR: Path = BACKEND_ROOT / "artifacts" / "roberta"
ARTIFACTS_XLMR_DIR: Path = BACKEND_ROOT / "artifacts" / "xlmr"

# default paths match ``backend/artifacts/<name>`` under this repo's backend folder
_DEFAULT_ROBERTA_SOURCE: str = str(BACKEND_ROOT / "artifacts" / "roberta")
_DEFAULT_XLMR_SOURCE: str = str(BACKEND_ROOT / "artifacts" / "xlmr")


def get_roberta_model_source() -> str:
    """huggingface hub id or local directory for roberta inference."""
    return os.getenv("FAKE_SHA_ROBERTA_MODEL", _DEFAULT_ROBERTA_SOURCE).strip()


def get_xlmr_model_source() -> str:
    """huggingface hub id or local directory for xlm-r inference."""
    return os.getenv("FAKE_SHA_XLMR_MODEL", _DEFAULT_XLMR_SOURCE).strip()


def _parse_positive_float(name: str, raw: str) -> float:
    try:
        v = float(raw.strip())
    except ValueError as e:
        raise ValueError(f"{name} must be a positive float, got {raw!r}") from e
    if v <= 0:
        raise ValueError(f"{name} must be > 0, got {v}")
    return v


def _env_temperature(var: str) -> float | None:
    raw = os.environ.get(var)
    if raw is None or not str(raw).strip():
        return None
    return _parse_positive_float(var, raw)


def get_inference_temperature(model_source: str, *, backend: str) -> float:
    """
    softmax temperature for transformer confidence.

    precedence:
        1. ``FAKE_SHA_ROBERTA_TEMPERATURE`` or ``FAKE_SHA_XLMR_TEMPERATURE`` (by backend)
        2. ``FAKE_SHA_TRANSFORMER_TEMPERATURE``
        3. ``temperature.json`` next to a local artifact directory
        4. ``1.0`` (same as standard softmax on logits; matches training loss)
    """
    backend_key = backend.strip().lower()
    specific_env = {
        "roberta": "FAKE_SHA_ROBERTA_TEMPERATURE",
        "xlmr": "FAKE_SHA_XLMR_TEMPERATURE",
    }.get(backend_key)
    if specific_env:
        v = _env_temperature(specific_env)
        if v is not None:
            return v
    v = _env_temperature("FAKE_SHA_TRANSFORMER_TEMPERATURE")
    if v is not None:
        return v

    p = Path(model_source).expanduser()
    if p.is_dir():
        tpath = p / "temperature.json"
        if tpath.is_file():
            try:
                data = json.loads(tpath.read_text(encoding="utf-8"))
                t = float(data.get("temperature", 1.0))
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                return 1.0
            if t > 0:
                return t
    return 1.0


# Must match AnalyzeRequest.analyzer Literal and inference.factory branches.
VALID_ANALYZER_BACKENDS: frozenset[str] = frozenset(
    {"svm", "roberta", "xlmr", "mock"}
)


class UnknownAnalyzerBackendError(ValueError):
    """Raised when FAKE_SHA_ANALYZER (or a non-schema analyzer override) is not recognized."""


def get_analyzer_backend() -> str:
    """Which analyzer implementation to use for POST /analyze when the request omits ``analyzer``."""
    return os.environ.get("FAKE_SHA_ANALYZER", "svm").strip().lower()