"""
FAKE-SHA thesis: Train an SVM (LinearSVC) fake-news classifier.

This pipeline is aligned with the RoBERTa training plan:

- **Input text** — Uses :func:`core.model_input.build_model_input` (title, URL,
  body joined with blank lines; empty fields omitted), matching inference for
  both SVM and RoBERTa. Optional CSV columns ``title`` and ``url``; if absent,
  they are treated as empty (body-only), same as missing fields at runtime.
- **Labels** — ``0`` = FAKE, ``1`` = REAL (also accepts FAKE/REAL strings).
- **Splits** — Separate ``train.csv`` / ``valid.csv`` (or ``val.csv``) /
  ``test.csv``; TF-IDF is fit on **train** only; threshold is tuned on
  **validation**; final metrics on **test** for comparison with RoBERTa on the
  same held-out set.
- **Metrics** — Accuracy, precision/recall/F1 for FAKE (pos_label=0), plus
  macro F1; classification report on the test split.
- **Class imbalance** — ``--class-weight balanced`` (default) or ``none``.
- **Reproducibility** — ``--seed`` (default 42) for NumPy and LinearSVC.

Saves artifacts for the FastAPI backend under:

  - backend/artifacts/svm/svm_model.pkl
  - backend/artifacts/svm/tfidf_vectorizer.pkl
  - backend/artifacts/svm/svm_decision_threshold.pkl

Run from ``backend/``:

  python -m training.train_svm

Feature extraction: TF-IDF (TfidfVectorizer), fitted on training split only;
vectorizer reused for validation/test (same as a deployed service transform).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import LinearSVC

from training.data_io import load_classification_csv, load_classification_hf


def load_data(
    csv_path: Path,
    *,
    article_only: bool = False,
) -> tuple[list[str], np.ndarray]:
    """Load CSV with TF-IDF preprocessing after :func:`core.model_input.build_model_input`."""
    return load_classification_csv(csv_path, article_only=article_only, tfidf_preprocess=True)


def load_data_hf(
    dataset_name: str,
    split: str,
    *,
    article_only: bool = False,
    hf_revision: str | None = None,
) -> tuple[list[str], np.ndarray]:
    """Load Hugging Face dataset split with TF-IDF preprocessing."""
    return load_classification_hf(
        dataset_name,
        split=split,
        article_only=article_only,
        tfidf_preprocess=True,
        revision=hf_revision,
    )


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
    random_state: int,
) -> tuple[LinearSVC, TfidfVectorizer]:
    """Fit TF-IDF vectorizer on training data only, then train LinearSVC."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
    )
    X_train = vectorizer.fit_transform(train_texts)

    model = LinearSVC(
        C=C,
        class_weight=class_weight,
        max_iter=5000,
        random_state=random_state,
    )
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
    """Evaluate the model on a given split and print metrics."""
    X = vectorizer.transform(texts)
    if threshold is None:
        preds = model.predict(X)
    else:
        scores = model.decision_function(X)
        preds = (scores >= threshold).astype(int)

    true_unique, true_counts = np.unique(labels, return_counts=True)
    pred_unique, pred_counts = np.unique(preds, return_counts=True)
    true_dist = {int(u): int(c) for u, c in zip(true_unique, true_counts)}
    pred_dist = {int(u): int(c) for u, c in zip(pred_unique, pred_counts)}

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, pos_label=0, zero_division=0)
    recall = recall_score(labels, preds, pos_label=0, zero_division=0)
    f1 = f1_score(labels, preds, pos_label=0, zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)

    print(f"\n=== {split_name} Evaluation ===")
    print(f"{split_name} true distribution: {true_dist} (expected balanced 0/1)")
    print(f"{split_name} pred distribution: {pred_dist} (skew indicates decision bias)")
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"Precision:  {precision:.4f} (FAKE, pos_label=0)")
    print(f"Recall:     {recall:.4f}")
    print(f"F1 (FAKE):  {f1:.4f}")
    print(f"F1 (macro): {f1_macro:.4f}")

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
    """Tune a threshold over `decision_function` outputs on the validation split."""
    X = vectorizer.transform(texts)
    scores = model.decision_function(X)

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
    dist = {int(u): int(c) for u, c in zip(unique, counts)}
    fake_count = dist.get(0, 0)
    real_count = dist.get(1, 0)
    print(f"{split_name} distribution: FAKE={fake_count}, REAL={real_count}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dataset paths and TF-IDF hyperparameters."""
    # training/train_svm.py -> parent.parent == backend/
    backend_dir = Path(__file__).resolve().parent.parent
    data_dir = backend_dir.parent / "data"
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
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default=None,
        help="Hugging Face dataset ID (e.g. username/fake-sha). If set, CSV paths are ignored.",
    )
    parser.add_argument("--hf-train-split", type=str, default="train")
    parser.add_argument("--hf-val-split", type=str, default="validation")
    parser.add_argument("--hf-test-split", type=str, default="test")
    parser.add_argument(
        "--hf-revision",
        type=str,
        default=None,
        help="Optional Hugging Face dataset git revision (commit/tag/branch).",
    )

    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-df", type=float, default=1.0)

    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument(
        "--class-weight",
        type=str,
        default="balanced",
        choices=["balanced", "none"],
        help="Use class weights: 'balanced' (recommended) or 'none'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for NumPy and LinearSVC (thesis reproducibility).",
    )
    parser.add_argument(
        "--article-only",
        action="store_true",
        help="Ignore title/url columns and train on body text only (ablation / legacy CSVs).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    load_kw = {"article_only": args.article_only}
    if args.hf_dataset:
        train_texts, train_labels = load_data_hf(
            args.hf_dataset,
            args.hf_train_split,
            hf_revision=args.hf_revision,
            **load_kw,
        )
        val_texts, val_labels = load_data_hf(
            args.hf_dataset,
            args.hf_val_split,
            hf_revision=args.hf_revision,
            **load_kw,
        )
        test_texts, test_labels = load_data_hf(
            args.hf_dataset,
            args.hf_test_split,
            hf_revision=args.hf_revision,
            **load_kw,
        )
    else:
        train_texts, train_labels = load_data(args.train_csv, **load_kw)
        val_texts, val_labels = load_data(args.val_csv, **load_kw)
        test_texts, test_labels = load_data(args.test_csv, **load_kw)

    print("\n=== Run configuration (align with RoBERTa training notes) ===")
    print(f"seed={args.seed}, article_only={args.article_only}, class_weight={args.class_weight}")
    if args.hf_dataset:
        print(f"hf_dataset={args.hf_dataset}")
        print(f"hf_train_split={args.hf_train_split}")
        print(f"hf_val_split={args.hf_val_split}")
        print(f"hf_test_split={args.hf_test_split}")
        if args.hf_revision:
            print(f"hf_revision={args.hf_revision}")
    else:
        print(f"train_csv={args.train_csv}")
        print(f"val_csv={args.val_csv}")
        print(f"test_csv={args.test_csv}")

    print_split_stats(train_labels, "Train")
    print_split_stats(val_labels, "Validation")
    print_split_stats(test_labels, "Test")

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
        random_state=args.seed,
    )
    print(
        "\nTF-IDF config: "
        f"max_features={args.max_features}, "
        f"ngram_range=({args.ngram_min},{args.ngram_max}), "
        f"min_df={args.min_df}, "
        f"max_df={args.max_df}, "
        f"learned_features={len(vectorizer.vocabulary_)}"
    )

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

    tuned_threshold = tune_decision_threshold(
        model=model,
        vectorizer=vectorizer,
        texts=val_texts,
        labels=val_labels,
    )

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

    backend_dir = Path(__file__).resolve().parent.parent
    artifacts_dir = backend_dir / "artifacts" / "svm"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    svm_path = artifacts_dir / "svm_model.pkl"
    vect_path = artifacts_dir / "tfidf_vectorizer.pkl"
    threshold_path = artifacts_dir / "svm_decision_threshold.pkl"

    joblib.dump(model, svm_path)
    joblib.dump(vectorizer, vect_path)
    joblib.dump(tuned_threshold, threshold_path)

    print("\nArtifacts saved:")
    print(f"  SVM model:         {svm_path}")
    print(f"  TF-IDF vectorizer: {vect_path}")
    print(f"  Decision threshold:{threshold_path}")


if __name__ == "__main__":
    main()
