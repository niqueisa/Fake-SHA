"""Tests for training vs inference text composition."""

from core.model_input import build_model_input, inference_input_text


def test_build_model_input_composes_metadata():
    combined = build_model_input(
        "Article body.",
        title="Headline",
        url="https://example.com/news",
    )
    assert "Headline" in combined
    assert "https://example.com/news" in combined
    assert "Article body." in combined


def test_inference_input_text_ignores_title_and_url():
    body = "Only this passage should be classified."
    assert (
        inference_input_text(
            body,
            title="Tab title must not affect verdict",
            url="https://news.example.com/story",
            mode="selection_only",
        )
        == body
    )


def test_inference_input_text_strips_whitespace():
    assert inference_input_text("  hello  ", title="T", url="U") == "hello"
