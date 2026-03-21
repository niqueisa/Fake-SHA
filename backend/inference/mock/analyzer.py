"""
FAKE-SHA Backend - Mock Analysis Logic

Keyword-based analyzer for local testing. Enable with:
    FAKE_SHA_ANALYZER=mock
"""

import random

from schemas.models import AnalyzeResponse, TokenResult

FAKE_KEYWORDS = [
    "shocking",
    "secret",
    "viral",
    "unbelievable",
    "exposed",
    "must share",
]


def analyze_text(text: str, title: str = "", url: str = "") -> AnalyzeResponse:
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
    confidence = min(0.95, 0.75 + (len(found_keywords) * 0.04))

    high_impact = {"shocking", "exposed", "unbelievable"}
    medium_impact = {"viral", "secret", "must share"}

    tokens: list[TokenResult] = []
    for kw in found_keywords:
        impact = "high" if kw in high_impact else "medium"
        tokens.append(TokenResult(text=kw, impact=impact, label="fake_signal"))

    if not tokens:
        tokens.append(
            TokenResult(text="sensational wording", impact="medium", label="fake_signal")
        )

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
    confidence = round(0.82 + random.uniform(0, 0.13), 2)

    words = [w.strip() for w in text.split() if len(w.strip()) > 3][:5]
    tokens: list[TokenResult] = []
    for i, w in enumerate(words[:3]):
        impact = "high" if i == 0 else ("medium" if i == 1 else "low")
        tokens.append(TokenResult(text=w, impact=impact, label="real_signal"))

    if not tokens:
        tokens.append(
            TokenResult(text="neutral phrasing", impact="medium", label="real_signal")
        )

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
