"""Model files in the Hugging Face cache, without network access."""

from __future__ import annotations

import json
from pathlib import Path

import huggingface_hub
import huggingface_hub.constants
import pytest

from nanorecon import models as M
from nanorecon.model_config import REPO_ID, REVISION
from nanorecon.types import NanoReconError

from ..helpers import RELEASE_CONFIG


@pytest.fixture
def hf_cache(tmp_path, monkeypatch) -> Path:
    cache = tmp_path / "hf-cache"
    cache.mkdir()
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(cache))
    return cache


def put_model(cache: Path, *, weights: bool = True) -> Path:
    """Lay out the model files the way the Hub client caches them."""
    snap = cache / f"models--{REPO_ID.replace('/', '--')}" / "snapshots" / REVISION
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "config.json").write_text(json.dumps(RELEASE_CONFIG))
    if weights:
        (snap / "flax_model.msgpack").write_bytes(b"\0" * 1000)
    return snap


def test_not_downloaded(hf_cache):
    with pytest.raises(NanoReconError, match="not downloaded") as info:
        M.load_local_model()
    assert "nanorecon pull" in info.value.hint
    put_model(hf_cache, weights=False)  # config alone (e.g. an interrupted pull) is not enough
    with pytest.raises(NanoReconError, match="not downloaded"):
        M.load_local_model()


def test_load_from_cache(hf_cache):
    snap = put_model(hf_cache)
    model = M.load_local_model()
    assert model.config_path == snap / "config.json" and model.weights_path == snap / "flax_model.msgpack"
    assert model.network.codebook_size == 65536


def test_pull_downloads_the_pinned_revision(hf_cache, monkeypatch):
    calls = []

    def fake_download(repo_id, filename, *, revision):
        calls.append((repo_id, filename, revision))
        return str(put_model(hf_cache) / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    model = M.pull()
    assert calls == [(REPO_ID, "config.json", REVISION), (REPO_ID, "flax_model.msgpack", REVISION)]
    assert model.weights_path.is_file()


def test_pull_failure_is_reported(monkeypatch):
    def offline(*a, **k):
        raise OSError("Connection refused\nmore detail")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    with pytest.raises(NanoReconError, match="cannot download config.json .*Connection refused$") as info:
        M.pull()
    assert "huggingface.co" in info.value.hint


def test_remote_status(monkeypatch):
    from types import SimpleNamespace

    from huggingface_hub.errors import RevisionNotFoundError

    class FakeApi:
        pinned = True

        def list_repo_refs(self, repo_id):
            return SimpleNamespace(branches=[SimpleNamespace(name="main", target_commit="f" * 40)])

        def model_info(self, repo_id, revision):
            if not FakeApi.pinned:
                raise RevisionNotFoundError.__new__(RevisionNotFoundError)  # its __init__ wants an HTTP response
            return SimpleNamespace(sha=revision)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    assert M.remote_status() == M.RemoteStatus("f" * 40, True)
    FakeApi.pinned = False
    assert M.remote_status() == M.RemoteStatus("f" * 40, False)
