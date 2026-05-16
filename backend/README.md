# FAKE-SHA Backend

FastAPI backend for fake news detection. Inference lives under `inference/` (SVM + TF‑IDF, optional Hugging Face RoBERTa/XLM‑R). Training scripts live under `training/` and are not imported by the API at runtime.

> **See [../README.md](../README.md)** for full project context (extension, structure, setup overview).

## Project Structure

```
backend/
├── main.py                 # FastAPI app, routes, CORS
├── core/                   # Config (paths, FAKE_SHA_ANALYZER)
├── schemas/                # Pydantic request/response models (API contract)
├── inference/              # Classifiers: svm/, roberta/, xlmr/, factory.py
├── explainability/         # SHAP helpers for XLM-R (`xlmr_shap.py`)
├── storage/                # Supabase client + record_store
├── artifacts/              # Saved model weights (not Python code)
│   ├── svm/                # svm_model.pkl, tfidf_vectorizer.pkl, threshold
│   ├── roberta/            # Hugging Face save_pretrained (config, weights, tokenizer)
│   └── xlmr/               # XLM-R save_pretrained (large files ignored by git; mount at deploy)
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
| `inference/factory.py` | Chooses analyzer via env `FAKE_SHA_ANALYZER` or request field `analyzer` (`svm` \| `roberta` \| `xlmr`). |
| `inference/svm/analyzer.py` | Loads artifacts from `artifacts/svm/`, runs LinearSVC + TF‑IDF. |
| `inference/roberta/` | Sequence classification via `transformers` + weights under `artifacts/roberta/`. |
| `inference/xlmr/` | XLM-R sequence classification + optional SHAP explanations. |
| `storage/` | Supabase optional persistence. |
| `training/train_svm.py` | Train SVM; saves pickles under `artifacts/svm/`. |
| `explainability/xlmr_shap.py` | SHAP text explainer + indicator grouping for `/analyze` optional `explanation`. |

## Environment

| Variable | Meaning |
|----------|---------|
| `SUPABASE_URL`, `SUPABASE_KEY` | Optional; if set, analyses are stored in Supabase. |
| `FAKE_SHA_ANALYZER` | `svm` (default), `roberta`, or `xlmr`. Invalid values return HTTP 400. |
| `FAKE_SHA_XLMR_HUB_ID` | Hugging Face repo id, e.g. `your-org/fake-sha-xlmr` (downloads via `transformers`). |
| `FAKE_SHA_XLMR_ARTIFACT_DIR` | Local `save_pretrained` folder (defaults to `backend/artifacts/xlmr` if unset). |
| `FAKE_SHA_XLMR_MODEL` | Alias for `ARTIFACT_DIR`, or Hub id if value looks like `org/name`. |
| `FAKE_SHA_ROBERTA_HUB_ID` / `FAKE_SHA_ROBERTA_ARTIFACT_DIR` | Same pattern for RoBERTa. |
| `FAKE_SHA_SVM_ARTIFACT_DIR` | Local folder with `svm_model.pkl` etc. (SVM does not load from the Hub). |
| `HF_TOKEN` | Optional; required for private Hugging Face model repos. |
| `FAKE_SHA_XLMR_TEMPERATURE` | Softmax temperature for XLM-R confidence and SHAP wrapper (default `3.5`). |
| `ENABLE_SHAP` | `true`/`false` — attach SHAP `explanation` on XLM-R responses when enabled. |
| `SHAP_MAX_WORDS`, `SHAP_TOP_K`, `SHAP_MAX_EVALS` | SHAP input budget / token cap / partition eval cap. |
| `FAKE_SHA_CORS_ORIGINS` | `*` or comma-separated origins; pinned origins enable `Access-Control-Allow-Credentials`. |

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

### Tests (optional)

```powershell
pip install -r requirements-dev.txt
pytest tests/
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
