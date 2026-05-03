"""
FAKE-SHA Backend - FastAPI Application

REST API for fake news detection. Analysis is delegated to `inference/`
(SVM, RoBERTa, XLM-RoBERTa, or mock).
"""

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import UnknownAnalyzerBackendError
from inference.factory import analyze_text

from inference.roberta.loader import RoBERTaArtifactError, RoBERTaDependencyError
from inference.xlmr.loader import XLMRArtifactError, XLMRDependencyError

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
# Exception handlers
# -----------------------------------------------------------------------------

# RoBERTa
@app.exception_handler(RoBERTaArtifactError)
def roberta_artifacts_unavailable(_request: Request, exc: RoBERTaArtifactError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(RoBERTaDependencyError)
def roberta_dependencies_missing(_request: Request, exc: RoBERTaDependencyError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(XLMRArtifactError)
def xlmr_artifacts_unavailable(_request: Request, exc: XLMRArtifactError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(XLMRDependencyError)
def xlmr_dependencies_missing(_request: Request, exc: XLMRDependencyError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


# Common
@app.exception_handler(UnknownAnalyzerBackendError)
def unknown_analyzer_backend(_request: Request, exc: UnknownAnalyzerBackendError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "message": "FAKE-SHA backend is running"}


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    """
    Analyzer: svm | roberta | xlmr | mock
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