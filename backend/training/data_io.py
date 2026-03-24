"""
Shared CSV loading for SVM and RoBERTa training.

Current experiments use dataset columns ``label``, ``title``, and ``article``.
Composition is done via :func:`core.model_input.build_model_input` so training
and inference share the same text-construction rules. URL handling remains
optional/legacy and is excluded in current experiments to reduce source-based
shortcut learning.

- **SVM** — Applies TF-IDF-style lowercasing / whitespace normalization after composition.
- **RoBERTa / transformers** — Uses composed text only (strip empty rows); no lowercasing,
  matching :mod:`inference.roberta` tokenization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.model_input import build_model_input

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


def load_classification_csv(
    csv_path: Path,
    *,
    article_only: bool = False,
    tfidf_preprocess: bool = False,
) -> tuple[list[str], np.ndarray]:
    """
    Load classification CSV for FAKE-SHA training.

    Args:
        csv_path: Training, validation, or test CSV.
        article_only: If True, ignore ``title`` and use article only.
        tfidf_preprocess: If True, apply :func:`preprocess_tfidf_style` (SVM). If False,
            strip only and drop empty strings (RoBERTa / inference-aligned).

    Returns:
        (texts, labels) with labels in ``{0, 1}``.
    """
    df = pd.read_csv(csv_path)

    if "article" in df.columns:
        text_col = "article"
    elif "text" in df.columns:
        # Legacy compatibility for older datasets; current experiments use `article`.
        text_col = "text"
    else:
        raise ValueError(f"Missing required text column ('article') in {csv_path}")

    if "label" not in df.columns:
        raise ValueError(f"Missing required column 'label' in {csv_path}")

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
    # URL is intentionally excluded in the current experiment setup to avoid
    # source-based shortcut learning; keep empty-url behavior for compatibility.
    urls = pd.Series([""] * len(df), index=df.index)

    composed = pd.Series(
        [build_model_input(str(b), title=str(t), url=str(u)) for b, t, u in zip(bodies, titles, urls)],
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
        raise ValueError(f"Found invalid labels in {csv_path}. Examples: {bad_values}")

    return texts, labels
