"""Tests for training.data_io label + article loading."""

from __future__ import annotations

import pandas as pd

from training.data_io import _prepare_classification_df, normalize_label


def test_prepare_label_and_article_only():
    df = pd.DataFrame(
        {
            "label": ["FAKE", "REAL"],
            "article": ["  First article.  ", "Second article."],
        }
    )
    texts, labels = _prepare_classification_df(
        df,
        source_name="test.csv",
        article_only=True,
        tfidf_preprocess=False,
    )
    assert texts == ["First article.", "Second article."]
    assert labels.tolist() == [0, 1]


def test_prepare_composes_title_and_url_when_present():
    df = pd.DataFrame(
        {
            "label": [0, 1],
            "article": ["Body one", "Body two"],
            "title": ["Headline", ""],
            "url": ["", "https://example.com"],
        }
    )
    texts, labels = _prepare_classification_df(
        df,
        source_name="test.csv",
        article_only=False,
        tfidf_preprocess=False,
    )
    assert "Headline" in texts[0]
    assert "Body one" in texts[0]
    assert "https://example.com" in texts[1]
    assert "Body two" in texts[1]
    assert labels.tolist() == [0, 1]


def test_normalize_label_variants():
    assert normalize_label("fake") == 0
    assert normalize_label("REAL") == 1
    assert normalize_label(1) == 1
