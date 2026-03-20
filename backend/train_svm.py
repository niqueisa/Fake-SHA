"""
FAKE-SHA thesis: Train an SVM (LinearSVC) fake-news classifier.

This script is designed to be easy to integrate into the FastAPI backend by
saving two artifacts:
  - backend/models/svm_model.pkl
  - backend/models/tfidf_vectorizer.pkl

Assumed dataset schema (CSV):
  - text: string (article or selected text) OR article: string
  - label: either "FAKE"/"REAL" or 0/1

Preprocessing:
  - lowercase
  - normalize whitespace
  - drop rows with missing text/label
  - map labels: FAKE -> 0, REAL -> 1

Feature extraction:
  - TF-IDF (TfidfVectorizer), fitted on training split only
  - vectorizer reused to transform validation/test split
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import LinearSVC


LABEL_MAP = {
    "FAKE": 0,
    "REAL": 1,
    "0": 0,
    "1": 1,
}


def load_data(csv_path: Path) -> tuple[list[str], np.ndarray]:
    """
    Load a dataset CSV and apply required preprocessing + label normalization.

    Returns:
        (texts, labels) where labels are integers in {0, 1}
    """
    df = pd.read_csv(csv_path)

    # In this repository, the dataset columns are:
    #   label  (0=FAKE, 1=REAL)
    #   article (text)
    if "article" in df.columns:
        text_col = "article"
    elif "text" in df.columns:
        # Fallback if an alternative CSV schema is provided.
        text_col = "text"
    else:
        raise ValueError(f"Missing required text column ('article' or 'text') in {csv_path}")

    if "label" not in df.columns:
        raise ValueError(f"Missing required column 'label' in {csv_path}")

    # Handle missing values safely:
    # - fill missing article/text with empty string
    # - drop rows with missing labels (can't train/evaluate without them)
    df = df.copy()
    df[text_col] = df[text_col].fillna("")
    df = df.dropna(subset=["label"]).copy()

    # Convert to consistent types before further preprocessing.
    df[text_col] = df[text_col].astype(str)
    df["label"] = df["label"].apply(normalize_label)

    # Apply required text preprocessing (e.g., empty strings become NaN so we can
    # drop the corresponding label rows and keep alignment).
    df[text_col] = preprocess_text(df[text_col])
    df = df.dropna(subset=[text_col]).copy()

    # Return types are already list/ndarray; avoid calling `.tolist()` on lists.
    texts = df[text_col].tolist()
    labels = df["label"].to_numpy(dtype=np.int64)

    # After normalization, safety-check labels are valid 0/1.
    invalid_mask = (labels != 0) & (labels != 1)
    if invalid_mask.any():
        bad_values = df.loc[invalid_mask, "label"].head(10).tolist()
        raise ValueError(
            f"Found invalid labels in {csv_path}. Examples: {bad_values}"
        )

    return texts, labels


def preprocess_text(text_series: pd.Series) -> pd.Series:
    """
    Apply thesis-required text preprocessing:
      - lowercase
      - remove extra whitespace
      - strip leading/trailing whitespace
    """
    # Lowercase for case-insensitive modeling.
    s = text_series.str.lower()
    # Normalize whitespace (including tabs/newlines) to a single space.
    s = s.str.replace(r"\s+", " ", regex=True)
    # Remove leading/trailing whitespace.
    s = s.str.strip()
    # Replace empty strings with NaN so caller can drop rows while preserving
    # text/label alignment.
    s = s.replace("", np.nan)
    return s


def normalize_label(raw_label) -> int:
    """
    Convert dataset label into {0, 1}.

    Accepts:
      - "FAKE"/"REAL" (case-insensitive)
      - 0/1 (as numbers or strings)
    """
    if pd.isna(raw_label):
        raise ValueError("Missing label encountered.")

    # Handle numeric labels (sometimes CSVs load them as floats).
    if isinstance(raw_label, (int, np.integer)):
        return int(raw_label)
    if isinstance(raw_label, (float, np.floating)):
        # Expect only 0.0 or 1.0.
        if raw_label in (0.0, 1.0):
            return int(raw_label)
        raise ValueError(f"Unexpected numeric label: {raw_label}")

    # Handle string labels.
    s = str(raw_label).strip().upper()
    if s in LABEL_MAP:
        return LABEL_MAP[s]

    raise ValueError(f"Unexpected label value: {raw_label}")


def train_model(
    train_texts: list[str],
    train_labels: np.ndarray,
    *,
    max_features: int,
    ngram_range: tuple[int, int],
    min_df: int,
    max_df: float,
    C: float,
    class_weight: str | None,
) -> tuple[LinearSVC, TfidfVectorizer]:
    """
    Fit TF-IDF vectorizer on training data only, then train LinearSVC.
    """
    # Fit vectorizer ONLY on training texts.
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
    )
    X_train = vectorizer.fit_transform(train_texts)

    # LinearSVC expects labels as integers.
    # Increase max_iter to reduce the chance of premature convergence causing
    # a biased decision boundary.
    model = LinearSVC(C=C, class_weight=class_weight, max_iter=5000)
    model.fit(X_train, train_labels)

    return model, vectorizer


def evaluate_model(
    *,
    model: LinearSVC,
    vectorizer: TfidfVectorizer,
    texts: list[str],
    labels: np.ndarray,
    split_name: str,
    threshold: float | None = None,
) -> None:
    """
    Evaluate the model on a given split and print:
      - Accuracy
      - Precision, Recall, F1
      - Classification report
    """
    X = vectorizer.transform(texts)
    if threshold is None:
        preds = model.predict(X)
    else:
        # Custom threshold on the SVM decision function.
        # This helps when a default 0-threshold yields skewed predictions.
        scores = model.decision_function(X)
        preds = (scores >= threshold).astype(int)

    # Debugging output to verify training/evaluation consistency.
    # This is thesis-friendly logging that helps diagnose prediction skew.
    true_unique, true_counts = np.unique(labels, return_counts=True)
    pred_unique, pred_counts = np.unique(preds, return_counts=True)
    true_dist = {int(u): int(c) for u, c in zip(true_unique, true_counts)}
    pred_dist = {int(u): int(c) for u, c in zip(pred_unique, pred_counts)}

    accuracy = accuracy_score(labels, preds)
    # Report "positive class" metrics for FAKE (label 0).
    precision = precision_score(labels, preds, pos_label=0, zero_division=0)
    recall = recall_score(labels, preds, pos_label=0, zero_division=0)
    f1 = f1_score(labels, preds, pos_label=0, zero_division=0)

    print(f"\n=== {split_name} Evaluation ===")
    print(f"{split_name} true distribution: {true_dist} (expected balanced 0/1)")
    print(f"{split_name} pred distribution: {pred_dist} (skew indicates decision bias)")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            labels,
            preds,
            labels=[0, 1],
            target_names=["FAKE", "REAL"],
            zero_division=0,
        )
    )


def tune_decision_threshold(
    *,
    model: LinearSVC,
    vectorizer: TfidfVectorizer,
    texts: list[str],
    labels: np.ndarray,
    num_thresholds: int = 101,
) -> float:
    """
    Tune a threshold over `decision_function` outputs on the validation split.

    The threshold is chosen to maximize REAL F1 (label 1).
    """
    X = vectorizer.transform(texts)
    scores = model.decision_function(X)

    # Explore thresholds between min/max decision scores.
    t_min = float(scores.min())
    t_max = float(scores.max())
    thresholds = np.linspace(t_min, t_max, num_thresholds)

    best_t = 0.0
    best_f1_real = -1.0

    for t in thresholds:
        preds = (scores >= t).astype(int)
        f1_real = f1_score(labels, preds, pos_label=1, zero_division=0)
        if f1_real > best_f1_real:
            best_f1_real = f1_real
            best_t = float(t)

    print(
        f"\nTuned decision threshold for REAL (label=1): "
        f"threshold={best_t:.6f}, best_REAL_F1={best_f1_real:.4f}"
    )
    return best_t


def print_split_stats(labels: np.ndarray, split_name: str) -> None:
    """Print dataset size and class distribution for thesis logs."""
    print(f"\n{split_name} size: {len(labels)}")
    unique, counts = np.unique(labels, return_counts=True)
    # Build distribution without relying on `.tolist()` to be present.
    dist = {int(u): int(c) for u, c in zip(unique, counts)}
    fake_count = dist.get(0, 0)
    real_count = dist.get(1, 0)
    print(f"{split_name} distribution: FAKE={fake_count}, REAL={real_count}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dataset paths and TF-IDF hyperparameters."""
    backend_dir = Path(__file__).resolve().parent
    data_dir = backend_dir.parent / "data"
    # In this repository, the validation split is sometimes named `valid.csv`.
    val_csv_default = data_dir / "val.csv"
    if not val_csv_default.exists():
        val_csv_default = data_dir / "valid.csv"

    parser = argparse.ArgumentParser(
        description="Train a LinearSVC + TF-IDF SVM model for FAKE-SHA."
    )

    parser.add_argument(
        "--train-csv",
        type=Path,
        default=data_dir / "train.csv",
        help="Path to training CSV.",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=val_csv_default,
        help="Path to validation CSV.",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=data_dir / "test.csv",
        help="Path to test CSV.",
    )

    # TF-IDF parameters (configurable for thesis reproducibility).
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--ngram-min", type=int, default=1)
    # Using bigrams often improves linear TF-IDF SVM baselines for text classification.
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-df", type=float, default=1.0)

    # Model parameters.
    parser.add_argument("--C", type=float, default=1.0)
    # Balanced loss helps if the model decision boundary is biased toward one class.
    parser.add_argument(
        "--class-weight",
        type=str,
        default="balanced",
        choices=["balanced", "none"],
        help="Use class weights: 'balanced' (recommended) or 'none'.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Dataset loading + required thesis preprocessing.
    train_texts, train_labels = load_data(args.train_csv)
    val_texts, val_labels = load_data(args.val_csv)
    test_texts, test_labels = load_data(args.test_csv)

    # Logging: sizes + class distribution.
    print_split_stats(train_labels, "Train")
    print_split_stats(val_labels, "Validation")
    print_split_stats(test_labels, "Test")

    # Model training: TF-IDF fitted ONLY on training split.
    # Convert CLI value into what scikit-learn expects.
    class_weight = None if args.class_weight == "none" else "balanced"
    model, vectorizer = train_model(
        train_texts,
        train_labels,
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max),
        min_df=args.min_df,
        max_df=args.max_df,
        C=args.C,
        class_weight=class_weight,
    )
    print(
        "\nTF-IDF config: "
        f"max_features={args.max_features}, "
        f"ngram_range=({args.ngram_min},{args.ngram_max}), "
        f"min_df={args.min_df}, "
        f"max_df={args.max_df}, "
        f"learned_features={len(vectorizer.vocabulary_)}"
    )

    # Evaluation on validation and test splits separately.
    evaluate_model(
        model=model,
        vectorizer=vectorizer,
        texts=val_texts,
        labels=val_labels,
        split_name="Validation (default threshold)",
        threshold=None,
    )
    evaluate_model(
        model=model,
        vectorizer=vectorizer,
        texts=test_texts,
        labels=test_labels,
        split_name="Test (default threshold)",
        threshold=None,
    )

    # Baseline improvement: tune the decision threshold on validation.
    tuned_threshold = tune_decision_threshold(
        model=model,
        vectorizer=vectorizer,
        texts=val_texts,
        labels=val_labels,
    )

    # Re-evaluate using the tuned decision threshold.
    evaluate_model(
        model=model,
        vectorizer=vectorizer,
        texts=val_texts,
        labels=val_labels,
        split_name="Validation (tuned threshold)",
        threshold=tuned_threshold,
    )
    evaluate_model(
        model=model,
        vectorizer=vectorizer,
        texts=test_texts,
        labels=test_labels,
        split_name="Test (tuned threshold)",
        threshold=tuned_threshold,
    )

    # Save artifacts for later backend integration.
    backend_dir = Path(__file__).resolve().parent
    models_dir = backend_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    svm_path = models_dir / "svm_model.pkl"
    vect_path = models_dir / "tfidf_vectorizer.pkl"

    joblib.dump(model, svm_path)
    joblib.dump(vectorizer, vect_path)
    # Save tuned threshold for later backend integration.
    threshold_path = models_dir / "svm_decision_threshold.pkl"
    joblib.dump(tuned_threshold, threshold_path)

    print("\nArtifacts saved:")
    print(f"  SVM model:        {svm_path}")
    print(f"  TF-IDF vectorizer:{vect_path}")
    print(f"  Decision threshold:{threshold_path}")


if __name__ == "__main__":
    # Keep a deterministic seed for any operations that could be influenced by randomness.
    np.random.seed(42)
    main()

