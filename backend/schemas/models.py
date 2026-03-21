"""
FAKE-SHA Backend - Request and Response Models

These Pydantic models define the API contract for the analyze endpoint.
They ensure type safety and clear documentation for the frontend integration.
"""

from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# Request models (what the browser extension sends)
# -----------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    """Input payload for the /analyze endpoint."""

    text: str = Field(..., description="The article text or selected content to analyze")
    url: str = Field(..., description="Source URL of the article")
    title: str = Field(..., description="Article title")
    mode: str = Field(
        default="selection_only",
        description="Analysis mode: 'selection_only' or 'selection_fallback'",
    )


# -----------------------------------------------------------------------------
# Response models (what the backend returns)
# -----------------------------------------------------------------------------


class TokenResult(BaseModel):
    """A single token (word/phrase) with its impact on the verdict."""

    text: str = Field(..., description="The token text found in the content")
    impact: str = Field(..., description="Impact level: 'high', 'medium', or 'low'")
    label: str = Field(..., description="Token classification: 'fake_signal' or 'real_signal'")


class AnalyzeResponse(BaseModel):
    """Full analysis result returned by /analyze."""

    verdict: str = Field(..., description="Final verdict: 'FAKE' or 'REAL'")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score between 0 and 1")
    summary: str = Field(..., description="Brief human-readable explanation of the result")
    indicators: list[str] = Field(..., description="List of detected indicators")
    tokens: list[TokenResult] = Field(..., description="Key tokens contributing to the verdict")
