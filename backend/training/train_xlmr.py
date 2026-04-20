"""
FAKE-SHA thesis: Fine-tune an XLM-RoBERTa (or compatible) sequence classifier.

Aligned with :mod:`inference.roberta` / future :mod:`inference.xlmr`
and :mod:`training.train_svm`:

- **CSV loading** — :func:`training.data_io.load_classification_csv` with
  ``tfidf_preprocess=False`` (same strings as inference; optional ``title`` / ``url``).
- **Labels** — ``0`` = FAKE, ``1`` = REAL; saved model has ``id2label`` / ``label2id``
  so inference can map predictions correctly.
- **Splits** — Same default paths as ``train_svm`` (``train.csv``, ``valid.csv`` or
  ``val.csv``, ``test.csv`` under ``../data``). **Test** split is reported after
  training for direct comparison with SVM on identical documents.
- **Artifacts** — ``save_pretrained`` into ``backend/artifacts/xlmr/`` by default.

Run from ``backend/`` (GPU recommended):

  python -m training.train_xlmr

Or, if you keep this filename as train_roberta.py:

  python -m training.train_roberta
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    from core.config import ARTIFACTS_ROBERTA_DIR
except ModuleNotFoundError:
    from backend.core.config import ARTIFACTS_ROBERTA_DIR

try:
    from training.data_io import load_classification_csv, load_classification_hf
except ModuleNotFoundError:
    from backend.training.data_io import load_classification_csv, load_classification_hf


ID2LABEL = {0: "FAKE", 1: "REAL"}
LABEL2ID = {"FAKE": 0, "REAL": 1}


class ClassWeightedTrainer(Trainer):
    """Cross-entropy with optional per-class weights (imbalanced FAKE/REAL)."""

    def __init__(self, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device)
            )
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def _compute_metrics_builder():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "f1_fake": f1_score(
                labels,
                preds,
                pos_label=0,
                average="binary",
                zero_division=0,
            ),
            "precision_fake": precision_score(
                labels,
                preds,
                pos_label=0,
                zero_division=0,
            ),
            "recall_fake": recall_score(
                labels,
                preds,
                pos_label=0,
                zero_division=0,
            ),
        }

    return compute_metrics


def _tokenize_fn(batch: dict, tokenizer: PreTrainedTokenizerBase, max_length: int):
    return tokenizer(batch["text"], truncation=True, max_length=max_length)


def _print_split_report(name: str, labels: np.ndarray, preds: np.ndarray) -> None:
    print(f"\n=== {name} (detailed) ===")
    print(
        classification_report(
            labels,
            preds,
            labels=[0, 1],
            target_names=["FAKE", "REAL"],
            zero_division=0,
        )
    )


def parse_args() -> argparse.Namespace:
    backend_dir = Path(__file__).resolve().parent.parent
    data_dir = backend_dir.parent / "data"
    val_csv_default = data_dir / "val.csv"
    if not val_csv_default.exists():
        val_csv_default = data_dir / "valid.csv"

    default_output_dir = backend_dir / "artifacts" / "xlmr"

    parser = argparse.ArgumentParser(
        description="Fine-tune XLM-RoBERTa sequence classification for FAKE-SHA."
    )
    parser.add_argument("--train-csv", type=Path, default=data_dir / "train.csv")
    parser.add_argument("--val-csv", type=Path, default=val_csv_default)
    parser.add_argument("--test-csv", type=Path, default=data_dir / "test.csv")
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Hugging Face save_pretrained directory (default: backend/artifacts/xlmr).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="FacebookAI/xlm-roberta-base",
        help="Base checkpoint (recommended: FacebookAI/xlm-roberta-base).",
    )
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--class-weight",
        type=str,
        default="balanced",
        choices=["balanced", "none"],
        help="Use class-balanced loss weights from the training set.",
    )
    parser.add_argument(
        "--article-only",
        action="store_true",
        help="Ignore title/url columns (same flag as train_svm).",
    )
    parser.add_argument(
        "--save-strategy",
        type=str,
        default="epoch",
        choices=["epoch", "no"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    try:
        from datasets import Dataset
    except ImportError as e:
        raise SystemExit(
            "The 'datasets' package is required. Install with: pip install datasets"
        ) from e

    if args.hf_dataset:
        train_texts, train_labels = load_classification_hf(
            args.hf_dataset,
            split=args.hf_train_split,
            article_only=args.article_only,
            tfidf_preprocess=False,
            revision=args.hf_revision,
        )
        val_texts, val_labels = load_classification_hf(
            args.hf_dataset,
            split=args.hf_val_split,
            article_only=args.article_only,
            tfidf_preprocess=False,
            revision=args.hf_revision,
        )
        test_texts, test_labels = load_classification_hf(
            args.hf_dataset,
            split=args.hf_test_split,
            article_only=args.article_only,
            tfidf_preprocess=False,
            revision=args.hf_revision,
        )
    else:
        train_texts, train_labels = load_classification_csv(
            args.train_csv,
            article_only=args.article_only,
            tfidf_preprocess=False,
        )
        val_texts, val_labels = load_classification_csv(
            args.val_csv,
            article_only=args.article_only,
            tfidf_preprocess=False,
        )
        test_texts, test_labels = load_classification_csv(
            args.test_csv,
            article_only=args.article_only,
            tfidf_preprocess=False,
        )

    if set(np.unique(train_labels)) != {0, 1}:
        raise ValueError("Training set must include both classes: 0 (FAKE) and 1 (REAL).")

    print("\n=== XLM-RoBERTa fine-tuning configuration ===")
    print(f"model_name={args.model_name}")
    print(
        f"seed={args.seed}, article_only={args.article_only}, class_weight={args.class_weight}"
    )
    print(f"epochs={args.epochs}, lr={args.lr}, max_length={args.max_length}")

    if args.hf_dataset:
        print(f"hf_dataset={args.hf_dataset}")
        print(f"hf_train_split={args.hf_train_split} (n={len(train_labels)})")
        print(f"hf_val_split={args.hf_val_split} (n={len(val_labels)})")
        print(f"hf_test_split={args.hf_test_split} (n={len(test_labels)})")
        if args.hf_revision:
            print(f"hf_revision={args.hf_revision}")
    else:
        print(f"train_csv={args.train_csv} (n={len(train_labels)})")
        print(f"val_csv={args.val_csv} (n={len(val_labels)})")
        print(f"test_csv={args.test_csv} (n={len(test_labels)})")

    print(f"output_dir={args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        print(
            f"Resizing token embeddings from "
            f"{model.get_input_embeddings().num_embeddings} to {len(tokenizer)}"
        )
    model.resize_token_embeddings(len(tokenizer))

    def make_ds(texts: list[str], labels: np.ndarray) -> "Dataset":
        ds = Dataset.from_dict({"text": texts, "labels": labels.tolist()})
        return ds.map(
            lambda batch: _tokenize_fn(batch, tokenizer, args.max_length),
            batched=True,
            remove_columns=["text"],
        )

    train_ds = make_ds(train_texts, train_labels)
    val_ds = make_ds(val_texts, val_labels)
    test_ds = make_ds(test_texts, test_labels)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    class_weights: torch.Tensor | None = None
    if args.class_weight == "balanced":
        fake = int((train_labels == 0).sum())
        real = int((train_labels == 1).sum())
        n = len(train_labels)
        if fake and real:
            w0 = n / (2 * fake)
            w1 = n / (2 * real)
            class_weights = torch.tensor([w0, w1], dtype=torch.float32)
            print(
                f"Class weights (balanced): FAKE={w0:.4f}, REAL={w1:.4f} "
                f"(counts FAKE={fake}, REAL={real})"
            )

    use_fp16 = torch.cuda.is_available()

    training_kwargs = dict(
        output_dir=str(args.output_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        save_strategy=args.save_strategy,
        load_best_model_at_end=args.save_strategy == "epoch",
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        logging_steps=max(10, len(train_ds) // (args.train_batch_size * 10) or 1),
        seed=args.seed,
        fp16=use_fp16,
        save_total_limit=2,
        report_to="none",
    )

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in ta_params:
        training_kwargs["evaluation_strategy"] = "epoch"
    else:
        training_kwargs["eval_strategy"] = "epoch"

    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=_compute_metrics_builder(),
    )

    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer

    if class_weights is not None:
        trainer = ClassWeightedTrainer(class_weights=class_weights, **trainer_kwargs)
    else:
        trainer = Trainer(**trainer_kwargs)

    print("\n=== Training ===")
    trainer.train()

    if args.save_strategy == "epoch":
        print("\nBest checkpoint loaded (by validation f1_macro).")

    print("\n=== Validation (Trainer) ===")
    val_metrics = trainer.evaluate(val_ds)
    for k, v in val_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    print("\n=== Test split (same CSVs as SVM; compare metrics) ===")
    test_metrics = trainer.evaluate(test_ds)
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    test_preds = trainer.predict(test_ds)
    pred_ids = np.argmax(test_preds.predictions, axis=-1)
    _print_split_report("Test", test_labels, pred_ids)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving model + tokenizer to {args.output_dir}")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    print(
        "\nDone. Point your analyzer to this saved XLM-R model folder "
        "(or restart the backend after wiring inference)."
    )


if __name__ == "__main__":
    main()