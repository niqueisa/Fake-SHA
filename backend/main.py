"""
FAKE-SHA Backend - FastAPI Application

REST API for fake news detection. Analysis is delegated to `inference/`
(SVM, RoBERTa, or mock).

Endpoints:
    GET  /health  - Health check for monitoring and CORS preflight
    POST /analyze - Analyze article text and return verdict

Supabase: When SUPABASE_URL and SUPABASE_KEY are set, successful analyses
are stored in the analysis_records table. Database failures do not affect
the API response.
"""

from dotenv import load_dotenv

load_dotenv()  # Load .env file for local development (SUPABASE_URL, SUPABASE_KEY)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import UnknownAnalyzerBackendError
from inference.factory import analyze_text
from inference.roberta.loader import RoBERTaArtifactError, RoBERTaDependencyError
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


@app.exception_handler(RoBERTaArtifactError)
def roberta_artifacts_unavailable(_request: Request, exc: RoBERTaArtifactError) -> JSONResponse:
    """RoBERTa selected but model files are missing or incomplete under artifacts/roberta/."""
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(RoBERTaDependencyError)
def roberta_dependencies_missing(_request: Request, exc: RoBERTaDependencyError) -> JSONResponse:
    """torch/transformers not installed."""
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(UnknownAnalyzerBackendError)
def unknown_analyzer_backend(_request: Request, exc: UnknownAnalyzerBackendError) -> JSONResponse:
    """Invalid FAKE_SHA_ANALYZER or analyzer override (typos, unsupported values)."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


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

    Analyzer: optional request field ``analyzer`` (svm | roberta | mock), or
    environment variable FAKE_SHA_ANALYZER when ``analyzer`` is omitted (default: svm).
    When Supabase is configured, the analysis record is stored for later review.
    """
    result = analyze_text(
        text=request.text,
        title=request.title,
        url=request.url,
        analyzer=request.analyzer,
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
