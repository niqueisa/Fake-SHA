"""
FAKE-SHA Backend - Mock Analysis Logic

This module contains the keyword-based mock analyzer used until the real
ML/NLP model is ready. The interface is designed so that you can replace
this module with a real model without changing the API layer.

Usage:
    from mock_analyzer import analyze_text
    result = analyze_text(request.text, request.title, request.url)
"""

import random

from models import AnalyzeResponse, TokenResult

# -----------------------------------------------------------------------------
# Keywords that trigger a FAKE verdict (case-insensitive)
# Add or remove keywords here to adjust mock behavior.
# -----------------------------------------------------------------------------
FAKE_KEYWORDS = [
    "shocking",
    "secret",
    "viral",
    "unbelievable",
    "exposed",
    "must share",
]


def analyze_text(text: str, title: str = "", url: str = "") -> AnalyzeResponse:
    """
    Analyze the given text using simple keyword rules.

    Args:
        text: The article or selected text to analyze.
        title: Article title (used for context; not yet in mock logic).
        url: Source URL (used for context; not yet in mock logic).

    Returns:
        AnalyzeResponse with verdict, confidence, summary, indicators, and tokens.

    Note:
        This function is designed to be replaced by a real ML model.
        The signature and return type should remain compatible.
    """
    combined = f"{title} {text}".lower()
    found_keywords: list[str] = []

    for keyword in FAKE_KEYWORDS:
        if keyword in combined:
            found_keywords.append(keyword)

    is_fake = len(found_keywords) > 0

    if is_fake:
        return _build_fake_response(found_keywords, text)
    return _build_real_response(text)


def _build_fake_response(found_keywords: list[str], text: str) -> AnalyzeResponse:
    """Build a FAKE verdict response based on detected keywords."""
    # More keywords = higher confidence
    confidence = min(0.95, 0.75 + (len(found_keywords) * 0.04))

    # Build tokens from found keywords (prioritize by typical impact)
    high_impact = {"shocking", "exposed", "unbelievable"}
    medium_impact = {"viral", "secret", "must share"}

    tokens: list[TokenResult] = []
    for kw in found_keywords:
        impact = "high" if kw in high_impact else "medium"
        tokens.append(
            TokenResult(text=kw, impact=impact, label="fake_signal")
        )

    # Ensure we have at least one token; add a generic if needed
    if not tokens:
        tokens.append(
            TokenResult(text="sensational wording", impact="medium", label="fake_signal")
        )

    # Thesis-defined 5 indicators (order reflects mock relevance for FAKE verdict)
    indicators = [
        "Source Credibility",
        "Claim Verification",
        "Language Tone",
        "Sensational Wording",
        "Consistency with Known Facts",
    ]

    summary = (
        "The text contains sensational or unsupported wording that may indicate "
        "misleading or unverified content."
    )

    return AnalyzeResponse(
        verdict="FAKE",
        confidence=round(confidence, 2),
        summary=summary,
        indicators=indicators,
        tokens=tokens,
    )


def _build_real_response(text: str) -> AnalyzeResponse:
    """Build a REAL verdict response when no fake keywords are found."""
    # Vary confidence slightly for realism (e.g., 0.82–0.95)
    confidence = round(0.82 + random.uniform(0, 0.13), 2)

    # Extract a few representative words as "real_signal" tokens (simple heuristic)
    words = [w.strip() for w in text.split() if len(w.strip()) > 3][:5]
    tokens: list[TokenResult] = []
    for i, w in enumerate(words[:3]):  # Up to 3 tokens
        impact = "high" if i == 0 else ("medium" if i == 1 else "low")
        tokens.append(
            TokenResult(text=w, impact=impact, label="real_signal")
        )

    if not tokens:
        tokens.append(
            TokenResult(text="neutral phrasing", impact="medium", label="real_signal")
        )

    # Thesis-defined 5 indicators (same set for REAL verdict)
    indicators = [
        "Source Credibility",
        "Claim Verification",
        "Language Tone",
        "Sensational Wording",
        "Consistency with Known Facts",
    ]

    summary = (
        "The text appears to use neutral or factual language without obvious "
        "sensational or clickbait indicators."
    )

    return AnalyzeResponse(
        verdict="REAL",
        confidence=confidence,
        summary=summary,
        indicators=indicators,
        tokens=tokens,
    )
