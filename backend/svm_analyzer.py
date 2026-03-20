"""
FAKE-SHA Backend - SVM Analysis Logic

Replaces the keyword-based mock analyzer with a real SVM + TF-IDF pipeline.

Artifacts expected in `backend/models/`:
- svm_model.pkl
- tfidf_vectorizer.pkl
- svm_decision_threshold.pkl
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import joblib

from models import AnalyzeResponse


def _sigmoid(x: float) -> float:
    # Numerically-stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _load_artifacts():
    model_dir = Path(__file__).resolve().parent / "models"

    # Avoid noisy logs when unpickling across sklearn versions.
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


_SVM_MODEL, _TFIDF_VECTORIZER, _DECISION_THRESHOLD = _load_artifacts()


def analyze_text(text: str, title: str = "", url: str = "") -> AnalyzeResponse:
    """
    Analyze text using the persisted SVM + TF-IDF model.

    Args:
        text: Article or selected content to analyze.
        title: Article title (currently unused by the SVM pipeline).
        url: Source URL (currently unused by the SVM pipeline).
    """

    # Match training preprocessing: lowercase + strip.
    cleaned = (text or "").lower().strip()

    # TF-IDF expects an iterable of documents; we analyze one at a time.
    X = _TFIDF_VECTORIZER.transform([cleaned])

    # LinearSVC exposes decision_function (distance to separating hyperplane).
    score = float(_SVM_MODEL.decision_function(X)[0])

    # Apply decision threshold: score >= threshold => REAL, else FAKE.
    verdict = "REAL" if score >= _DECISION_THRESHOLD else "FAKE"

    # Convert score into a probability-like confidence:
    # center the sigmoid at the learned decision threshold (confidence ~ 0.5 at boundary).
    confidence = _sigmoid(score - _DECISION_THRESHOLD)

    indicators = [
        "Source Credibility",
        "Claim Verification",
        "Language Tone",
        "SVM Prediction",
        "Consistency with Known Facts",
    ]

    summary = "Prediction based on SVM model."
    tokens = []  # Placeholder; SHAP/RoBERTa integration can populate later.

    return AnalyzeResponse(
        verdict=verdict,
        confidence=float(round(confidence, 4)),
        summary=summary,
        indicators=indicators,
        tokens=tokens,
    )

