# Archive: pre–re-split snapshot (2026-03-23)

This folder preserves the **datasets** and **SVM training outputs** from before a planned **80/10/10 re-split** (with balanced title/no-title across train/val/test).

| Subfolder | Contents |
|-----------|----------|
| `datasets/` | Copies of `data/train.csv`, `data/valid.csv`, and `data/test.csv` at archive time. |
| `svm_artifacts/` | Copies of `backend/artifacts/svm/` (LinearSVC model, TF–IDF vectorizer, decision threshold). |

**Active project paths** (`data/*.csv`, `backend/artifacts/svm/*.pkl`) were **not** removed; they still match this snapshot until you replace them after re-splitting and retraining.

To restore this SVM bundle into the API path, copy the three `.pkl` files from `svm_artifacts/` into `backend/artifacts/svm/` (overwriting the current files).
