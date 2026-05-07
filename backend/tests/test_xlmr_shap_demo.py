from explainability.xlmr_shap import demo_explanation_output, explanation_unavailable


def test_demo_explanation_output_has_frontend_shape():
    payload = demo_explanation_output()
    assert "token contributions" in payload.note.lower()
    assert payload.top_tokens
    assert payload.indicators
    assert all(item.contribution_percent >= 0 for item in payload.indicators)


def test_explanation_unavailable_fallback():
    payload = explanation_unavailable()
    assert "unavailable" in payload.note.lower()
    assert payload.top_tokens == []
    assert payload.indicators == []
