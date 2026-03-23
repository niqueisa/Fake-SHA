"""
Backend configuration (paths, analyzer selection).

Environment:
    FAKE_SHA_ANALYZER  - "svm" (default), "roberta", or "mock"
"""

from __future__ import annotations

import os
from pathlib import Path

# backend/ directory (parent of core/)
BACKEND_ROOT: Path = Path(__file__).resolve().parent.parent

# Persisted ML weights (not Pydantic schemas)
ARTIFACTS_SVM_DIR: Path = BACKEND_ROOT / "artifacts" / "svm"
ARTIFACTS_ROBERTA_DIR: Path = BACKEND_ROOT / "artifacts" / "roberta"

# RoBERTa model source for inference.
# - Local dev: defaults to backend/artifacts/roberta (a directory with save_pretrained output).
# - Deployment: can be a Hugging Face repo id (e.g. niqueisa/fake-sha-roberta).
#
# NOTE: keep as a string because Hugging Face repo ids are not local paths.
ROBERTA_MODEL_SOURCE: str = os.environ.get(
    "FAKE_SHA_ROBERTA_MODEL",
    str(ARTIFACTS_ROBERTA_DIR),
).strip()

# Must match AnalyzeRequest.analyzer Literal and inference.factory branches.
VALID_ANALYZER_BACKENDS: frozenset[str] = frozenset({"svm", "roberta", "mock"})


class UnknownAnalyzerBackendError(ValueError):
    """Raised when FAKE_SHA_ANALYZER (or a non-schema analyzer override) is not recognized."""


def get_analyzer_backend() -> str:
    """Which analyzer implementation to use for POST /analyze when the request omits ``analyzer``."""
    return os.environ.get("FAKE_SHA_ANALYZER", "svm").strip().lower()
