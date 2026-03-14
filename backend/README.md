# FAKE-SHA Backend

FastAPI backend for fake news detection. Uses mock keyword-based analysis until the real ML/NLP model is ready.

> **See [../README.md](../README.md)** for full project context (extension, structure, setup overview).

## Project Structure

```
backend/
├── main.py           # FastAPI app, routes, CORS
├── models.py         # Request/response Pydantic models
├── mock_analyzer.py  # Mock analysis logic (replace with ML later)
├── supabase_client.py# Supabase client (optional; env-based)
├── record_store.py   # Saves analysis records to Supabase
├── requirements.txt  # Python dependencies
├── .env.example      # Template for environment variables
├── sql/
│   └── analysis_records.sql  # Table schema for Supabase
└── README.md         # This file
```

### File Descriptions

| File | Purpose |
|------|---------|
| `main.py` | FastAPI application. Defines `GET /health` and `POST /analyze`, configures CORS. Integrates Supabase storage when configured. |
| `models.py` | Pydantic models for `AnalyzeRequest` and `AnalyzeResponse`. Ensures type safety and clear API contract. |
| `mock_analyzer.py` | Keyword-based mock analyzer. Detects words like "shocking", "viral", "exposed" and returns FAKE; otherwise REAL. Designed to be swapped for a real model. |
| `supabase_client.py` | Lazy-loads Supabase client from env vars. Returns `None` if not configured so the backend runs without a database. |
| `record_store.py` | Saves analysis records to the `analysis_records` table. Never raises; logs failures instead. |
| `requirements.txt` | Dependencies: FastAPI, Uvicorn, Supabase client, python-dotenv. |

## Setup

### 1. Create a virtual environment (recommended)

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
```

> Use `venv` or `.venv`; both are in `.gitignore`.

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. (Optional) Configure Supabase for storing analysis records

The backend runs fine without Supabase. To store analyses in a PostgreSQL database:

1. Create a project at [supabase.com](https://supabase.com).
2. Run the table schema in the SQL Editor (see [SQL Schema](#supabase-sql-schema) below).
3. Copy `.env.example` to `.env` and fill in your credentials:

   ```powershell
   copy .env.example .env
   ```

   Edit `.env`:

   ```
   SUPABASE_URL=https://your-project-id.supabase.co
   SUPABASE_KEY=your-anon-or-service-role-key
   ```

   Get these from **Supabase Dashboard → Project Settings → API**.

4. The project `.gitignore` already excludes `.env`; do not commit it.

If Supabase is not configured or insertions fail, the `/analyze` endpoint still returns results. Errors are logged but do not break the API.

### 4. Run the server

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- `--reload`: Auto-restart on code changes (useful during development)
- `--host 0.0.0.0`: Accept connections from any interface (needed for browser extension)
- `--port 8000`: Matches the default backend URL in the FAKE-SHA extension settings

The API will be available at **http://localhost:8000**.

### Supabase SQL Schema

Run this in the Supabase SQL Editor to create the `analysis_records` table:

```sql
CREATE TABLE IF NOT EXISTS analysis_records (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    title           TEXT NOT NULL DEFAULT '',
    url             TEXT NOT NULL DEFAULT '',
    text            TEXT NOT NULL DEFAULT '',
    verdict         TEXT NOT NULL,
    confidence      DOUBLE PRECISION NOT NULL,
    summary         TEXT NOT NULL DEFAULT '',
    indicators      JSONB NOT NULL DEFAULT '[]',
    mode            TEXT NOT NULL DEFAULT 'selection_only',
    extraction_source TEXT
);
```

The schema is also in `sql/analysis_records.sql`.

## Testing

### Health check

**PowerShell:**
```powershell
Invoke-RestMethod -Uri http://localhost:8000/health
```

**curl (Windows/Linux/macOS):**
```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok","message":"FAKE-SHA backend is running"}
```

### Analyze (FAKE example)

Text containing sensational keywords returns a FAKE verdict:

**PowerShell:**
```powershell
$body = @{text="This shocking secret went viral and you wont believe what was exposed!"; url="https://example.com/news"; title="Must share news"; mode="selection_only"} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/analyze -Method Post -Body $body -ContentType "application/json"
```

**curl:**
```bash
curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" -d "{\"text\": \"This shocking secret went viral and you wont believe what was exposed!\", \"url\": \"https://example.com/news\", \"title\": \"Must share news\", \"mode\": \"selection_only\"}"
```

Expected response (shape):

```json
{
  "verdict": "FAKE",
  "confidence": 0.87,
  "summary": "The text contains sensational or unsupported wording...",
  "indicators": ["Sensational language detected", "Possible lack of source attribution", ...],
  "tokens": [
    {"text": "shocking", "impact": "high", "label": "fake_signal"},
    {"text": "secret", "impact": "medium", "label": "fake_signal"},
    ...
  ]
}
```

### Analyze (REAL example)

Text without sensational keywords returns a REAL verdict:

**PowerShell:**
```powershell
$body = @{text="The government announced new policies today. Officials confirmed the changes in a press conference."; url="https://example.com/news"; title="Policy Update"; mode="selection_only"} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/analyze -Method Post -Body $body -ContentType "application/json"
```

**curl:**
```bash
curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" -d "{\"text\": \"The government announced new policies today. Officials confirmed the changes in a press conference.\", \"url\": \"https://example.com/news\", \"title\": \"Policy Update\", \"mode\": \"selection_only\"}"
```

Expected response (shape):

```json
{
  "verdict": "REAL",
  "confidence": 0.89,
  "summary": "The text appears to use neutral or factual language...",
  "indicators": ["No sensational language detected", "Neutral or factual tone observed"],
  "tokens": [...]
}
```

## Integration with the Extension

1. Ensure the backend is running on `http://localhost:8000`.
2. In the FAKE-SHA extension settings, set **Backend URL** to `http://localhost:8000` (this is the default).
3. Update the extension's popup to call `POST /analyze` with the request body instead of using dummy data. The backend response format (`verdict`, `confidence`, `summary`, `indicators`, `tokens`) can be mapped to the frontend's expected format in the extension code.

## Replacing the Mock with a Real Model

1. Create a new module (e.g. `ml_analyzer.py`) that implements the same interface:
   - Function: `analyze_text(text: str, title: str = "", url: str = "") -> AnalyzeResponse`
2. In `main.py`, replace `from mock_analyzer import analyze_text` with your new analyzer.
3. The API contract remains unchanged; no changes to `models.py` or the routes are needed.
