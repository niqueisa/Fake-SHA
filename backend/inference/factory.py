"""
Selects the active analyzer implementation (SVM, RoBERTa, or XLM-RoBERTa)
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
    mode: str = "selection_only",
    analyzer: str | None = None,
) -> AnalyzeResponse:
    """If ``analyzer`` is set (svm | roberta | xlmr), it overrides FAKE_SHA_ANALYZER."""
    # Request-level override takes priority so frontend can switch analyzers per call.
    backend = analyzer.strip().lower() if analyzer else get_analyzer_backend()

    if backend not in VALID_ANALYZER_BACKENDS:
        allowed = ", ".join(sorted(VALID_ANALYZER_BACKENDS))
        raise UnknownAnalyzerBackendError(
            f"Unknown analyzer backend {backend!r}. Use one of: {allowed} "
            "(request field `analyzer` or environment variable FAKE_SHA_ANALYZER)."
        )

    if backend == "roberta":
        # Lazy imports keep startup lightweight and avoid importing unused model stacks.
        from inference.roberta.analyzer import analyze_text as _analyze
        return _analyze(text, title, url, mode)

    if backend == "xlmr":
        from inference.xlmr.analyzer import analyze_text as _analyze
        return _analyze(text, title, url, mode)

    # Default fallback remains SVM for backwards compatibility.
    from inference.svm.analyzer import analyze_text as _analyze
    return _analyze(text, title, url, mode)