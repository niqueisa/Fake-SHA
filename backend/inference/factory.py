"""
Selects the active analyzer implementation (SVM, RoBERTa, XLM-RoBERTa, or mock)
without changing routes.
"""

from __future__ import annotations

from core.config import (
    VALID_ANALYZER_BACKENDS,
    UnknownAnalyzerBackendError,
    get_analyzer_backend,
)
from schemas.models import AnalyzeResponse


def analyze_text(
    text: str,
    title: str = "",
    url: str = "",
    analyzer: str | None = None,
) -> AnalyzeResponse:
    """If ``analyzer`` is set (svm | roberta | xlmr | mock), it overrides FAKE_SHA_ANALYZER."""
    backend = analyzer.strip().lower() if analyzer else get_analyzer_backend()

    if backend not in VALID_ANALYZER_BACKENDS:
        allowed = ", ".join(sorted(VALID_ANALYZER_BACKENDS))
        raise UnknownAnalyzerBackendError(
            f"Unknown analyzer backend {backend!r}. Use one of: {allowed} "
            "(request field `analyzer` or environment variable FAKE_SHA_ANALYZER)."
        )

    if backend == "mock":
        from inference.mock.analyzer import analyze_text as _analyze
        return _analyze(text, title, url)

    if backend == "roberta":
        from inference.roberta.analyzer import analyze_text as _analyze
        return _analyze(text, title, url)

    if backend == "xlmr":
        from inference.xlmr.analyzer import analyze_text as _analyze
        return _analyze(text, title, url)

    from inference.svm.analyzer import analyze_text as _analyze
    return _analyze(text, title, url)