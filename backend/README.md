# FAKE-SHA Backend

FastAPI backend for fake news detection. Inference lives under `inference/` (SVM + TF‑IDF, optional Hugging Face RoBERTa, and a keyword mock). Training scripts live under `training/` and are not imported by the API at runtime.

> **See [../README.md](../README.md)** for full project context (extension, structure, setup overview).

## Project Structure

```
backend/
├── main.py                 # FastAPI app, routes, CORS
├── core/                   # Config (paths, FAKE_SHA_ANALYZER)
├── schemas/                # Pydantic request/response models (API contract)
├── inference/              # Classifiers: svm/, roberta/, mock/, factory.py
├── explainability/         # Future: SHAP / attention (separate from inference)
├── storage/                # Supabase client + record_store
├── artifacts/              # Saved model weights (not Python code)
│   ├── svm/                # svm_model.pkl, tfidf_vectorizer.pkl, threshold
│   └── roberta/            # Hugging Face save_pretrained (config, weights, tokenizer)
├── training/               # train_svm.py (CLI; writes to artifacts/svm/)
├── sql/
│   └── analysis_records.sql
├── requirements.txt
├── .env.example
└── README.md               # This file
```

### File descriptions

| Path | Purpose |
|------|---------|
| `main.py` | FastAPI app: `GET /health`, `POST /analyze`. Delegates to `inference.factory.analyze_text`. |
| `schemas/models.py` | `AnalyzeRequest`, `AnalyzeResponse`, `TokenResult`. |
| `inference/factory.py` | Chooses analyzer via env `FAKE_SHA_ANALYZER` or request field `analyzer` (`svm` \| `roberta` \| `mock`). |
| `inference/svm/analyzer.py` | Loads artifacts from `artifacts/svm/`, runs LinearSVC + TF‑IDF. |
| `inference/roberta/` | Sequence classification via `transformers` + weights under `artifacts/roberta/`. |
| `inference/mock/analyzer.py` | Keyword-based mock (for local demos). |
| `storage/` | Supabase optional persistence. |
| `training/train_svm.py` | Train SVM; saves pickles under `artifacts/svm/`. |
| `explainability/` | Placeholder package for future SHAP hooks. |

## Environment

| Variable | Meaning |
|----------|---------|
| `SUPABASE_URL`, `SUPABASE_KEY` | Optional; if set, analyses are stored in Supabase. |
| `FAKE_SHA_ANALYZER` | `svm` (default), `roberta`, or `mock`. Invalid values return HTTP 400. |

## Setup

### 1. Virtual environment (recommended)

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. (Optional) Supabase

Copy `.env.example` to `.env` and set `SUPABASE_URL` / `SUPABASE_KEY`. The API runs without them.

### 4. Run the server

From the **`backend/`** directory (so imports resolve):

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Training (SVM)

From **`backend/`**:

```powershell
python -m training.train_svm
```

Artifacts are written to `artifacts/svm/` (create the folder automatically if missing).

## Testing

Health:

```powershell
Invoke-RestMethod -Uri http://localhost:8000/health
```

Analyze (PowerShell example):

```powershell
$body = @{ text = "..."; url = "https://example.com"; title = "..."; mode = "selection_only" } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/analyze -Method Post -Body $body -ContentType "application/json"
```

## Integration with the extension

The extension expects JSON with `verdict`, `confidence`, `summary`, `indicators`, `tokens` — defined by `schemas/models.py` and returned by `POST /analyze`.

If Supabase is not configured or insertions fail, `/analyze` still returns results; errors are logged only.
