"""Runtime text preprocessing aligned with the deployed SVM service."""


def preprocess_document(text: str) -> str:
    """Same as current inference: lowercase + strip (see training for fuller offline prep)."""
    return (text or "").lower().strip()
