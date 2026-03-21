"""
FAKE-SHA Backend - FastAPI Application

REST API for fake news detection. Analysis is delegated to `inference/`
(SVM now; RoBERTa later).

Endpoints:
    GET  /health  - Health check for monitoring and CORS preflight
    POST /analyze - Analyze article text and return verdict

Supabase: When SUPABASE_URL and SUPABASE_KEY are set, successful analyses
are stored in the analysis_records table. Database failures do not affect
the API response.
"""

from dotenv import load_dotenv

load_dotenv()  # Load .env file for local development (SUPABASE_URL, SUPABASE_KEY)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from inference.factory import analyze_text
from schemas.models import AnalyzeRequest
from storage.record_store import save_analysis_record

# -----------------------------------------------------------------------------
# Application setup
# -----------------------------------------------------------------------------
app = FastAPI(
    title="FAKE-SHA API",
    description="Backend API for fake news detection (thesis project)",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "message": "FAKE-SHA backend is running"}


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    """
    Analyze article text for potential fake news.

    Analyzer is selected via FAKE_SHA_ANALYZER (default: svm).
    When Supabase is configured, the analysis record is stored for later review.
    """
    result = analyze_text(
        text=request.text,
        title=request.title,
        url=request.url,
    )

    save_analysis_record(
        title=request.title,
        url=request.url,
        text=request.text,
        mode=request.mode,
        verdict=result.verdict,
        confidence=result.confidence,
        summary=result.summary,
        indicators=result.indicators,
        extraction_source=None,
    )

    return result.model_dump()
