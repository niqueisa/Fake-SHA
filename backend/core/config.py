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


def get_analyzer_backend() -> str:
    """Which analyzer implementation to use for POST /analyze."""
    return os.environ.get("FAKE_SHA_ANALYZER", "svm").strip().lower()
