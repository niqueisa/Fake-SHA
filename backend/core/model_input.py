"""
Shared input strings for text classifiers (SVM, RoBERTa, XLM-R).

- :func:`build_model_input` — training and batch jobs when CSV rows include
  title / URL metadata alongside article body.
- :func:`inference_input_text` — live ``/analyze`` requests: body / selection
  only; page title and URL are stored for history but excluded from prediction.
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


def inference_input_text(
    text: str,
    title: str = "",
    url: str = "",
    *,
    mode: str = "selection_only",
) -> str:
    """
    Text passed to classifiers and SHAP at inference time.

    Extension analysis always classifies the ``text`` field (selected passage or
    extracted page content). ``title`` and ``url`` are ignored so tab chrome does
    not influence verdicts or token highlights.
    """
    _ = title, url, mode  # reserved for API compatibility / future modes
    return (text or "").strip()
