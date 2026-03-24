"""
Shared CSV loading for SVM and RoBERTa training.

Uses :func:`core.model_input.build_model_input` so both pipelines see the same
strings as ``POST /analyze`` (optional ``title`` / ``url`` + body).

- **SVM** — Applies TF-IDF-style lowercasing / whitespace normalization after composition.
- **RoBERTa / transformers** — Uses composed text only (strip empty rows); no lowercasing,
  matching :mod:`inference.roberta` tokenization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from core.model_input import build_model_input
except ModuleNotFoundError:
    from backend.core.model_input import build_model_input

LABEL_MAP = {
    "FAKE": 0,
    "REAL": 1,
    "0": 0,
    "1": 1,
}


def normalize_label(raw_label) -> int:
    """Convert dataset label into {0, 1}."""
    if pd.isna(raw_label):
        raise ValueError("Missing label encountered.")

    if isinstance(raw_label, (int, np.integer)):
        return int(raw_label)
    if isinstance(raw_label, (float, np.floating)):
        if raw_label in (0.0, 1.0):
            return int(raw_label)
        raise ValueError(f"Unexpected numeric label: {raw_label}")

    s = str(raw_label).strip().upper()
    if s in LABEL_MAP:
        return LABEL_MAP[s]

    raise ValueError(f"Unexpected label value: {raw_label}")


def preprocess_tfidf_style(text_series: pd.Series) -> pd.Series:
    """Lowercase and collapse whitespace (SVM / TF-IDF training only)."""
    s = text_series.str.lower()
    s = s.str.replace(r"\s+", " ", regex=True)
    s = s.str.strip()
    s = s.replace("", np.nan)
    return s


def _prepare_classification_df(
    df: pd.DataFrame,
    *,
    source_name: str,
    article_only: bool,
    tfidf_preprocess: bool,
) -> tuple[list[str], np.ndarray]:
    """Normalize labels and compose model input text from a dataframe."""
    if "article" in df.columns:
        text_col = "article"
    elif "text" in df.columns:
        text_col = "text"
    else:
        raise ValueError(
            f"Missing required text column ('article' or 'text') in {source_name}"
        )

    if "label" not in df.columns:
        raise ValueError(f"Missing required column 'label' in {source_name}")

    df = df.copy()
    df[text_col] = df[text_col].fillna("")
    df = df.dropna(subset=["label"]).copy()

    df[text_col] = df[text_col].astype(str)
    df["label"] = df["label"].apply(normalize_label)

    bodies = df[text_col]
    if article_only or "title" not in df.columns:
        titles = pd.Series([""] * len(df), index=df.index)
    else:
        titles = df["title"].fillna("").astype(str)
    if article_only or "url" not in df.columns:
        urls = pd.Series([""] * len(df), index=df.index)
    else:
        urls = df["url"].fillna("").astype(str)

    composed = pd.Series(
        [
            build_model_input(str(b), title=str(t), url=str(u))
            for b, t, u in zip(bodies, titles, urls)
        ],
        index=df.index,
        dtype=object,
    )
    if tfidf_preprocess:
        composed = preprocess_tfidf_style(composed)
    else:
        composed = composed.str.strip()
        composed = composed.replace("", np.nan)

    df = df.assign(_composed=composed)
    df = df.dropna(subset=["_composed"]).copy()

    texts = df["_composed"].tolist()
    labels = df["label"].to_numpy(dtype=np.int64)

    invalid_mask = (labels != 0) & (labels != 1)
    if invalid_mask.any():
        bad_values = df.loc[invalid_mask, "label"].head(10).tolist()
        raise ValueError(f"Found invalid labels in {source_name}. Examples: {bad_values}")

    return texts, labels


def load_classification_csv(
    csv_path: Path,
    *,
    article_only: bool = False,
    tfidf_preprocess: bool = False,
) -> tuple[list[str], np.ndarray]:
    """
    Load ``label`` + ``article`` or ``text``, optionally ``title`` / ``url``.

    Args:
        csv_path: Training, validation, or test CSV.
        article_only: If True, ignore ``title`` / ``url`` columns.
        tfidf_preprocess: If True, apply :func:`preprocess_tfidf_style` (SVM). If False,
            strip only and drop empty strings (RoBERTa / inference-aligned).

    Returns:
        (texts, labels) with labels in ``{0, 1}``.
    """
    df = pd.read_csv(csv_path)
    return _prepare_classification_df(
        df,
        source_name=str(csv_path),
        article_only=article_only,
        tfidf_preprocess=tfidf_preprocess,
    )


def load_classification_hf(
    dataset_name: str,
    *,
    split: str,
    article_only: bool = False,
    tfidf_preprocess: bool = False,
    revision: str | None = None,
) -> tuple[list[str], np.ndarray]:
    """
    Load a split from Hugging Face datasets and return (texts, labels).

    Expected columns: ``label`` plus ``article`` (or ``text``), with optional
    ``title`` and ``url``.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "The 'datasets' package is required for Hugging Face input. "
            "Install with: pip install datasets"
        ) from e

    ds = load_dataset(dataset_name, split=split, revision=revision)
    df = ds.to_pandas()
    source_name = f"{dataset_name}[{split}]" if revision is None else f"{dataset_name}[{split}]@{revision}"
    return _prepare_classification_df(
        df,
        source_name=source_name,
        article_only=article_only,
        tfidf_preprocess=tfidf_preprocess,
    )
