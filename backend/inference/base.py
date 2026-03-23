"""
Shared analyzer interface (for documentation and type hints).

New backends (e.g. RoBERTa) should expose the same callable shape as
`analyze_text(text, title="", url="", analyzer=None) -> AnalyzeResponse` (factory).
"""

from __future__ import annotations

from typing import Protocol

from schemas.models import AnalyzeResponse


class TextAnalyzer(Protocol):
    def __call__(self, text: str, title: str = "", url: str = "") -> AnalyzeResponse: ...
