import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# LOAD MODELS (run once)
# =========================

# SVM
svm_vectorizer = joblib.load("artifacts/svm/tfidf_vectorizer.pkl")
svm_model = joblib.load("artifacts/svm/svm_model.pkl")

# RoBERTa
roberta_path = "artifacts/roberta"
roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_path)
roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_path)


# =========================
# ANALYZE FUNCTION
# =========================

def analyze_text(text: str, method: str = "roberta"):
    """
    method: 'svm' or 'roberta'
    """

    if method == "svm":
        X = svm_vectorizer.transform([text])
        pred = svm_model.predict(X)[0]

    else:  # default roberta
        inputs = roberta_tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = roberta_model(**inputs)

        pred = torch.argmax(outputs.logits, dim=1).item()

    # Convert to label
    label = "FAKE" if pred == 1 else "REAL"

    return {
        "prediction": label,
        "model": method
    }