"""
Selects the active analyzer implementation (SVM vs mock) without changing routes.
"""

from __future__ import annotations

from core.config import get_analyzer_backend
from schemas.models import AnalyzeResponse


def analyze_text(text: str, title: str = "", url: str = "") -> AnalyzeResponse:
    backend = get_analyzer_backend()
    if backend == "mock":
        from inference.mock.analyzer import analyze_text as _analyze

        return _analyze(text, title, url)

    from inference.svm.analyzer import analyze_text as _analyze

    return _analyze(text, title, url)
