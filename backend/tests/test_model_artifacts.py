"""Tests for local vs Hugging Face Hub model source resolution."""

from pathlib import Path

from core.model_artifacts import is_probably_hub_id, resolve_pretrained_source


def test_resolve_hub_id_explicit():
    ref = resolve_pretrained_source(
        hub_id="org/fake-sha-xlmr",
        artifact_dir=None,
        legacy_model_env=None,
        default_local_dir=Path("/default/xlmr"),
    )
    assert ref == "org/fake-sha-xlmr"


def test_resolve_local_artifact_dir(tmp_path):
    model_dir = tmp_path / "xlmr"
    model_dir.mkdir()
    ref = resolve_pretrained_source(
        hub_id=None,
        artifact_dir=str(model_dir),
        legacy_model_env=None,
        default_local_dir=Path("/default/xlmr"),
    )
    assert Path(ref) == model_dir.resolve()


def test_resolve_legacy_env_as_hub_id():
    ref = resolve_pretrained_source(
        hub_id=None,
        artifact_dir=None,
        legacy_model_env="myteam/fake-news-xlmr",
        default_local_dir=Path("/default/xlmr"),
    )
    assert ref == "myteam/fake-news-xlmr"


def test_is_probably_hub_id():
    assert is_probably_hub_id("org/model")
    assert is_probably_hub_id("org/model@main")
    assert not is_probably_hub_id("C:\\models\\xlmr")
    assert not is_probably_hub_id("/var/models/xlmr")
