"""
FAKE-SHA Backend - Request and Response Models

These Pydantic models define the API contract for the analyze endpoint.
They ensure type safety and clear documentation for the frontend integration.
"""

from typing import Literal

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
    analyzer: Literal["svm", "roberta", "xlmr", "mock"] | None = Field(
        default=None,
        description=(
            "Which backend to use for this request. "
            "If omitted, uses environment variable FAKE_SHA_ANALYZER (default: svm)."
        ),
    )


# -----------------------------------------------------------------------------
# Response models (what the backend returns)
# -----------------------------------------------------------------------------


class TokenResult(BaseModel):
    """A single token (word/phrase) with its impact on the verdict."""

    text: str = Field(..., description="The token text found in the content")
    impact: str = Field(..., description="Impact level: 'high', 'medium', or 'low'")
    label: str = Field(..., description="Token classification: 'fake_signal' or 'real_signal'")


class ExplanationTopToken(BaseModel):
    """Top token contribution from SHAP for the predicted class."""

    text: str = Field(..., description="Token text after normalization and cleanup")
    score: float = Field(..., description="Absolute SHAP contribution score")
    direction: str = Field(..., description="Direction relative to predicted class contribution")
    indicator: str | None = Field(default=None, description="Mapped indicator name, if any")


class ExplanationIndicator(BaseModel):
    """Grouped contribution view used by frontend indicator visualizations."""

    name: str = Field(..., description="Indicator category name")
    contribution_percent: float = Field(
        ..., description="Relative grouped SHAP contribution percentage (not confidence)"
    )
    tokens: list[str] = Field(..., description="Matched tokens grouped under this indicator")
    summary: str = Field(..., description="Human-readable summary of this indicator's influence")


class ExplanationResult(BaseModel):
    """Optional SHAP explainability details attached to AnalyzeResponse."""

    note: str = Field(
        ...,
        description=(
            "SHAP explains feature contribution only and does not perform factual verification."
        ),
    )
    top_tokens: list[ExplanationTopToken] = Field(
        default_factory=list,
        description="Top contributing tokens for the predicted class",
    )
    indicators: list[ExplanationIndicator] = Field(
        default_factory=list,
        description="Detected indicators with contribution share percentages",
    )


class AnalyzeResponse(BaseModel):
    """Full analysis result returned by /analyze."""

    verdict: str = Field(..., description="Final verdict: 'FAKE' or 'REAL'")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score between 0 and 1")
    summary: str = Field(..., description="Brief human-readable explanation of the result")
    indicators: list[str] = Field(..., description="List of detected indicators")
    tokens: list[TokenResult] = Field(..., description="Key tokens contributing to the verdict")
    explanation: ExplanationResult | None = Field(
        default=None,
        description=(
            "Optional explainability payload with SHAP token contributions. "
            "When unavailable, this field may contain a fallback note."
        ),
    )
