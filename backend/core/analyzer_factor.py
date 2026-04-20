import joblib
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

SVM_DIR = BASE_DIR / "artifacts" / "svm"
ROBERTA_DIR = BASE_DIR / "artifacts" / "roberta"
XLMR_DIR = BASE_DIR / "artifacts" / "xlmr"

# =========================
# LOAD MODELS (run once)
# =========================

# SVM
svm_vectorizer = joblib.load(SVM_DIR / "tfidf_vectorizer.pkl")
svm_model = joblib.load(SVM_DIR / "svm_model.pkl")

# RoBERTa
roberta_tokenizer = None
roberta_model = None
if ROBERTA_DIR.exists():
    roberta_tokenizer = AutoTokenizer.from_pretrained(str(ROBERTA_DIR))
    roberta_model = AutoModelForSequenceClassification.from_pretrained(str(ROBERTA_DIR))
    roberta_model.eval()

# XLM-R
xlmr_tokenizer = None
xlmr_model = None
if XLMR_DIR.exists():
    xlmr_tokenizer = AutoTokenizer.from_pretrained(str(XLMR_DIR))
    xlmr_model = AutoModelForSequenceClassification.from_pretrained(str(XLMR_DIR))
    xlmr_model.eval()


# =========================
# HELPERS
# =========================

def _label_from_prediction(pred: int) -> str:
    """
    Standard label mapping used across your thesis system:
    0 = FAKE
    1 = REAL
    """
    return "FAKE" if pred == 0 else "REAL"


# =========================
# ANALYZE FUNCTION
# =========================

def analyze_text(text: str, method: str = "xlmr"):
    """
    method: 'svm', 'roberta', or 'xlmr'
    """

    method = (method or "xlmr").strip().lower()

    if method == "svm":
        X = svm_vectorizer.transform([text])
        pred = int(svm_model.predict(X)[0])

    elif method == "roberta":
        if roberta_tokenizer is None or roberta_model is None:
            raise RuntimeError(
                f"RoBERTa artifacts not found in: {ROBERTA_DIR}"
            )

        inputs = roberta_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = roberta_model(**inputs)

        pred = int(torch.argmax(outputs.logits, dim=1).item())

    elif method == "xlmr":
        if xlmr_tokenizer is None or xlmr_model is None:
            raise RuntimeError(
                f"XLM-R artifacts not found in: {XLMR_DIR}"
            )

        inputs = xlmr_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = xlmr_model(**inputs)

        pred = int(torch.argmax(outputs.logits, dim=1).item())

    else:
        raise ValueError("method must be one of: 'svm', 'roberta', 'xlmr'")

    label = _label_from_prediction(pred)

    return {
        "prediction": label,
        "model": method
    }