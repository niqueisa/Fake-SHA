"""
Resolve where transformer models are loaded from (local folder or Hugging Face Hub).

Priority for XLM-R / RoBERTa:
  1. ``FAKE_SHA_*_HUB_ID`` — Hub repo id, e.g. ``your-org/fake-sha-xlmr``
  2. ``FAKE_SHA_*_ARTIFACT_DIR`` or legacy ``FAKE_SHA_*_MODEL`` — local path, or Hub id if
     the value looks like ``org/name`` and is not an existing directory
  3. Default under ``backend/artifacts/<analyzer>/``

SVM artifacts remain local-only (pickle + TF-IDF); see ``get_svm_artifacts_dir``.
"""

from __future__ import annotations

from pathlib import Path


class LocalArtifactError(RuntimeError):
    """Raised when a local artifact directory is missing required files."""


def is_probably_hub_id(ref: str) -> bool:
    """
    True when ``ref`` should be passed to ``transformers`` as a Hub repo id.

    Local paths (existing dirs, absolute paths, Windows drives) are excluded.
    """
    ref = (ref or "").strip()
    if not ref or "/" not in ref:
        return False

    path = Path(ref)
    if path.exists():
        return False
    if ref.startswith((".", "~")):
        return False
    if len(ref) >= 2 and ref[1] == ":":
        return False
    if ref.startswith(("/", "\\")):
        return False

    # org/model or org/model@revision
    head = ref.split("/")[0]
    if "@" in head:
        return True
    return bool(head and not path.is_dir())


def resolve_pretrained_source(
    *,
    hub_id: str | None,
    artifact_dir: str | None,
    legacy_model_env: str | None,
    default_local_dir: Path,
) -> str:
    """Return the ``from_pretrained`` identifier (Hub repo id or resolved local path)."""
    explicit_hub = (hub_id or "").strip()
    if explicit_hub:
        return explicit_hub

    for candidate in ((artifact_dir or "").strip(), (legacy_model_env or "").strip()):
        if not candidate:
            continue
        if is_probably_hub_id(candidate):
            return candidate
        return str(Path(candidate).expanduser().resolve())

    return str(default_local_dir.resolve())


def validate_local_pretrained_dir(model_dir: Path, *, analyzer_name: str) -> None:
    """Ensure a on-disk ``save_pretrained`` tree exists before loading locally."""
    if not model_dir.is_dir():
        raise LocalArtifactError(
            f"{analyzer_name} artifacts directory not found: {model_dir}. "
            f"Place a Hugging Face save_pretrained output there, or set "
            f"FAKE_SHA_{analyzer_name.upper()}_HUB_ID=org/model-name to load from the Hub."
        )

    if not (model_dir / "config.json").is_file():
        raise LocalArtifactError(
            f"Missing config.json in {model_dir}. Export the model with save_pretrained()."
        )

    has_weights = (model_dir / "model.safetensors").is_file() or (
        model_dir / "pytorch_model.bin"
    ).is_file()
    if not has_weights:
        raise LocalArtifactError(
            f"No model weights in {model_dir}. Expected model.safetensors or pytorch_model.bin."
        )


def prepare_pretrained_ref(model_ref: str, *, analyzer_name: str) -> str:
    """
    Validate local trees; pass Hub ids through unchanged.

    Returns ``model_ref`` suitable for ``AutoTokenizer.from_pretrained`` /
    ``AutoModelForSequenceClassification.from_pretrained``.
    """
    ref = (model_ref or "").strip()
    if not ref:
        raise LocalArtifactError(f"No {analyzer_name} model source configured.")

    if is_probably_hub_id(ref):
        return ref

    model_dir = Path(ref).expanduser().resolve()
    validate_local_pretrained_dir(model_dir, analyzer_name=analyzer_name)
    return str(model_dir)
