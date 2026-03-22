"""
Compose the string passed to the RoBERTa tokenizer.

Training should use the same title / URL / body composition so tokenization
matches the saved checkpoint.
"""


def build_model_input(text: str, title: str = "", url: str = "") -> str:
    """Combine title, URL, and body; omit empty fields."""
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
