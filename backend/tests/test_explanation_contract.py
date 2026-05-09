"""Contract tests for explainability payloads (no model / SHAP runtime required)."""

from schemas.models import ExplanationIndicator, ExplanationResult, ExplanationTopToken


def test_explanation_result_contract_minimal():
    """Ensures optional explanation JSON shape matches frontend expectations."""
    payload = ExplanationResult(
        note=(
            "SHAP identifies token contributions to the model prediction. "
            "It does not verify factual correctness."
        ),
        top_tokens=[
            ExplanationTopToken(
                text="viral",
                score=0.213,
                direction="supports_predicted_class",
                indicator="Sensationalism",
            ),
        ],
        indicators=[
            ExplanationIndicator(
                name="Sensationalism",
                contribution_percent=100.0,
                tokens=["viral"],
                summary="These highlighted words suggest attention-grabbing wording that influenced the prediction.",
            ),
        ],
    )
    assert "token contributions" in payload.note.lower()
    assert payload.top_tokens
    assert payload.indicators
    assert all(item.contribution_percent >= 0 for item in payload.indicators)


def test_explanation_unavailable_fallback():
    from explainability.xlmr_shap import explanation_unavailable

    payload = explanation_unavailable()
    assert "unavailable" in payload.note.lower()
    assert payload.top_tokens == []
    assert payload.indicators == []
