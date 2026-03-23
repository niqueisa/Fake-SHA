"""
Shared input string for all text classifiers (SVM, RoBERTa).

Matches what the browser sends: title, URL, and body combined the same way at
training and at inference so TF-IDF / transformer inputs stay aligned.
"""


def build_model_input(text: str, title: str = "", url: str = "") -> str:
    """Combine title, URL, and body; omit empty fields; join with blank lines."""
    parts: list[str] = []
    t = (title or "").strip()
    u = (url or "").strip()
    body = (text or "").strip()
    if t:
        parts.append(t)
    if u:
        parts.append(u)
    if body:
        parts.append(body)
    if not parts:
        return ""
    return "\n\n".join(parts)
