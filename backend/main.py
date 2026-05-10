"""
FAKE-SHA Backend - FastAPI Application

REST API for fake news detection. Analysis is delegated to `inference/`
(SVM, RoBERTa, or XLM-RoBERTa).
"""

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import UnknownAnalyzerBackendError, cors_allow_origins
from inference.factory import analyze_text

# Import BOTH loaders
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

_origins = cors_allow_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    # Credentials + wildcard origin is invalid in browsers; disable credentials unless pinning origins.
    allow_credentials=bool(_origins != ["*"]),
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
    Analyzer: svm | roberta | xlmr
    """

    # Inference path (svm/roberta/xlmr) is selected by request.analyzer or env default.
    result = analyze_text(
        text=request.text,
        title=request.title,
        url=request.url,
        analyzer=request.analyzer,
    )

    # Persistence is best-effort and intentionally does not block API response.
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

    # Return normalized Pydantic payload consumed by extension popup/history UIs.
    return result.model_dump()