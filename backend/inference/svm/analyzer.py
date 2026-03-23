"""
SVM + TF-IDF analyzer.

Artifacts expected in `backend/artifacts/svm/`:
- svm_model.pkl
- tfidf_vectorizer.pkl
- svm_decision_threshold.pkl
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import joblib

from core.config import ARTIFACTS_SVM_DIR
from core.model_input import build_model_input
from inference.svm.preprocess import preprocess_document
from schemas.models import AnalyzeResponse


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _load_artifacts(model_dir: Path):
    try:
        from sklearn.exceptions import InconsistentVersionWarning

        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    except Exception:
        pass

    svm_path = model_dir / "svm_model.pkl"
    vectorizer_path = model_dir / "tfidf_vectorizer.pkl"
    threshold_path = model_dir / "svm_decision_threshold.pkl"

    svm_model = joblib.load(svm_path)
    tfidf_vectorizer = joblib.load(vectorizer_path)
    decision_threshold = joblib.load(threshold_path)

    return svm_model, tfidf_vectorizer, float(decision_threshold)


_SVM_MODEL, _TFIDF_VECTORIZER, _DECISION_THRESHOLD = _load_artifacts(ARTIFACTS_SVM_DIR)


def analyze_text(text: str, title: str = "", url: str = "") -> AnalyzeResponse:
    """
    Analyze text using the persisted SVM + TF-IDF model.

    Uses the same title / URL / body composition as RoBERTa training
    (:func:`core.model_input.build_model_input`), then TF-IDF preprocessing.
    """
    combined = build_model_input(text, title=title, url=url)
    cleaned = preprocess_document(combined)
    X = _TFIDF_VECTORIZER.transform([cleaned])
    score = float(_SVM_MODEL.decision_function(X)[0])

    verdict = "REAL" if score >= _DECISION_THRESHOLD else "FAKE"
    confidence = _sigmoid(score - _DECISION_THRESHOLD)

    indicators = [
        "Source Credibility",
        "Claim Verification",
        "Language Tone",
        "SVM Prediction",
        "Consistency with Known Facts",
    ]

    summary = "Prediction based on SVM model."
    tokens = []

    return AnalyzeResponse(
        verdict=verdict,
        confidence=float(round(confidence, 4)),
        summary=summary,
        indicators=indicators,
        tokens=tokens,
    )
