"""
Temperature scaling for trained RoBERTa classifier confidence calibration.

Loads a trained model from backend/artifacts/roberta, runs inference on the
validation set (same text loading/tokenization style as train_roberta.py),
learns a scalar temperature with LBFGS + CrossEntropyLoss, and saves:

  backend/artifacts/roberta/temperature.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

try:
    from core.config import ARTIFACTS_ROBERTA_DIR
except ModuleNotFoundError:
    from backend.core.config import ARTIFACTS_ROBERTA_DIR

try:
    from training.data_io import load_classification_csv, load_classification_hf
except ModuleNotFoundError:
    from backend.training.data_io import load_classification_csv, load_classification_hf


def _tokenize_fn(batch: dict, tokenizer, max_length: int):
    # Keep tokenization behavior aligned with train_roberta.py.
    return tokenizer(batch["text"], truncation=True, max_length=max_length)


def parse_args() -> argparse.Namespace:
    backend_dir = Path(__file__).resolve().parent.parent
    data_dir = backend_dir.parent / "data"
    val_csv_default = data_dir / "val.csv"
    if not val_csv_default.exists():
        val_csv_default = data_dir / "valid.csv"

    parser = argparse.ArgumentParser(
        description="Calibrate RoBERTa confidence using temperature scaling on validation split."
    )
    parser.add_argument("--val-csv", type=Path, default=val_csv_default)
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default=None,
        help="Hugging Face dataset ID. If set, --val-csv is ignored.",
    )
    parser.add_argument("--hf-val-split", type=str, default="validation")
    parser.add_argument("--hf-revision", type=str, default=None)
    parser.add_argument("--article-only", action="store_true")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ARTIFACTS_ROBERTA_DIR,
        help="Directory containing trained RoBERTa save_pretrained artifacts.",
    )
    return parser.parse_args()


def _load_validation_texts_and_labels(args: argparse.Namespace) -> tuple[list[str], torch.Tensor]:
    if args.hf_dataset:
        val_texts, val_labels = load_classification_hf(
            args.hf_dataset,
            split=args.hf_val_split,
            article_only=args.article_only,
            tfidf_preprocess=False,
            revision=args.hf_revision,
        )
    else:
        val_texts, val_labels = load_classification_csv(
            args.val_csv,
            article_only=args.article_only,
            tfidf_preprocess=False,
        )
    return val_texts, torch.tensor(val_labels, dtype=torch.long)


def _collect_logits_and_labels(
    model,
    tokenizer,
    texts: list[str],
    labels: torch.Tensor,
    *,
    max_length: int,
    eval_batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    from datasets import Dataset

    ds = Dataset.from_dict({"text": texts, "labels": labels.tolist()})
    ds = ds.map(
        lambda batch: _tokenize_fn(batch, tokenizer, max_length),
        batched=True,
        remove_columns=["text"],
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(ds, batch_size=eval_batch_size, shuffle=False, collate_fn=collator)

    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = outputs.logits.detach().cpu()
            batch_labels = batch["labels"].detach().cpu()
            all_logits.append(logits)
            all_labels.append(batch_labels)

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def _fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    logits = logits.float()
    labels = labels.long()

    temperature = torch.nn.Parameter(torch.ones(1))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temperature], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        t = torch.clamp(temperature, min=1e-3)
        loss = criterion(logits / t, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    learned = float(torch.clamp(temperature.detach(), min=1e-3).item())
    return learned


def _save_temperature(model_dir: Path, temperature: float) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / "temperature.json"
    payload = {"temperature": float(temperature)}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    model_dir = args.model_dir
    if not model_dir.is_dir():
        raise SystemExit(f"Model directory not found: {model_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)

    val_texts, val_labels = _load_validation_texts_and_labels(args)
    logits, labels = _collect_logits_and_labels(
        model,
        tokenizer,
        val_texts,
        val_labels,
        max_length=args.max_length,
        eval_batch_size=args.eval_batch_size,
        device=device,
    )

    before_nll = torch.nn.functional.cross_entropy(logits, labels).item()
    temperature = _fit_temperature(logits, labels)
    after_nll = torch.nn.functional.cross_entropy(logits / temperature, labels).item()
    out_path = _save_temperature(model_dir, temperature)

    print("\n=== Temperature scaling complete ===")
    print(f"Validation samples: {len(labels)}")
    print(f"NLL before: {before_nll:.6f}")
    print(f"NLL after:  {after_nll:.6f}")
    print(f"Learned temperature: {temperature:.6f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
