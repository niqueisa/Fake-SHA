"""
FAKE-SHA Backend - FastAPI Application

Simple REST API for fake news detection. Currently uses mock keyword-based
analysis; designed for easy replacement with a real ML/NLP model later.

Endpoints:
    GET  /health  - Health check for monitoring and CORS preflight
    POST /analyze - Analyze article text and return verdict
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from models import AnalyzeRequest
from mock_analyzer import analyze_text

# -----------------------------------------------------------------------------
# Application setup
# -----------------------------------------------------------------------------
app = FastAPI(
    title="FAKE-SHA API",
    description="Backend API for fake news detection (thesis project)",
    version="0.1.0",
)

# CORS: Allow browser extension (and localhost) to call the API.
# Extensions typically use chrome-extension:// or moz-extension:// origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Accept all origins for development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------


@app.get("/health")
def health():
    """
    Health check endpoint.

    Returns a simple success response. Useful for:
    - Verifying the backend is running
    - CORS preflight checks from the browser extension
    """
    return {"status": "ok", "message": "FAKE-SHA backend is running"}


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    """
    Analyze article text for potential fake news.

    Accepts text, url, title, and mode from the browser extension.
    Returns verdict (FAKE/REAL), confidence, summary, indicators, and tokens.

    The analysis logic lives in mock_analyzer.py and can be replaced
    with a real ML model without changing this route.
    """
    result = analyze_text(
        text=request.text,
        title=request.title,
        url=request.url,
    )
    return result.model_dump()
