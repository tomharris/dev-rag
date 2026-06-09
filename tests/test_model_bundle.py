import hashlib
import io
import tarfile
from pathlib import Path

import httpx
import pytest
import respx
import huggingface_hub.constants as hf_constants

from devrag.config import DevragConfig
from devrag.ingest import model_bundle


def test_resolve_fastembed_cache_dir_default():
    cfg = DevragConfig()
    expected = Path("~/.cache/devrag/fastembed").expanduser()
    assert model_bundle.resolve_fastembed_cache_dir(cfg) == expected


def test_resolve_fastembed_cache_dir_override(tmp_path):
    cfg = DevragConfig()
    cfg.sparse_embedding.cache_dir = str(tmp_path / "fe")
    assert model_bundle.resolve_fastembed_cache_dir(cfg) == tmp_path / "fe"


def test_bundle_target_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    cfg = DevragConfig()
    cfg.sparse_embedding.cache_dir = str(tmp_path / "fe")
    hf_dir, fe_dir = model_bundle.bundle_target_dirs(cfg)
    assert hf_dir == tmp_path / "hub"
    assert fe_dir == tmp_path / "fe"


def _hf_repo_dir(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def test_models_present_false_when_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    cfg = DevragConfig()
    cfg.sparse_embedding.cache_dir = str(tmp_path / "fe")
    assert model_bundle.models_present(cfg) is False


def test_models_present_true_when_both_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    cfg = DevragConfig()
    cfg.sparse_embedding.cache_dir = str(tmp_path / "fe")
    snap = tmp_path / "hub" / _hf_repo_dir(cfg.retrieval.reranker_model) / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    bm = tmp_path / "fe" / _hf_repo_dir(cfg.sparse_embedding.model)
    bm.mkdir(parents=True)
    (bm / "config.json").write_text("{}")
    assert model_bundle.models_present(cfg) is True


def _make_bundle_bytes() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in [
            ("hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2/snapshots/abc/config.json", b"{}"),
            ("fastembed/models--Qdrant--bm25/config.json", b"{}"),
        ]:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _make_evil_bundle_bytes() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = b"pwned"
        info = tarfile.TarInfo(name="../evil.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


@respx.mock
def test_download_bundle_fetches_verifies_and_extracts(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    cfg = DevragConfig()
    cfg.sparse_embedding.cache_dir = str(tmp_path / "fe")
    payload = _make_bundle_bytes()
    cfg.network.model_bundle_url = "https://example.test/bundle.tar.gz"
    cfg.network.model_bundle_sha256 = hashlib.sha256(payload).hexdigest()
    respx.get(cfg.network.model_bundle_url).mock(return_value=httpx.Response(200, content=payload))

    model_bundle.download_bundle(cfg)

    assert (tmp_path / "hub" / "models--cross-encoder--ms-marco-MiniLM-L-6-v2"
            / "snapshots" / "abc" / "config.json").exists()
    assert (tmp_path / "fe" / "models--Qdrant--bm25" / "config.json").exists()


@respx.mock
def test_download_bundle_checksum_mismatch_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    cfg = DevragConfig()
    cfg.sparse_embedding.cache_dir = str(tmp_path / "fe")
    cfg.network.model_bundle_url = "https://example.test/bundle.tar.gz"
    cfg.network.model_bundle_sha256 = "0" * 64
    respx.get(cfg.network.model_bundle_url).mock(return_value=httpx.Response(200, content=_make_bundle_bytes()))
    with pytest.raises(RuntimeError, match="checksum"):
        model_bundle.download_bundle(cfg)


@respx.mock
def test_download_bundle_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    cfg = DevragConfig()
    cfg.sparse_embedding.cache_dir = str(tmp_path / "fe")
    payload = _make_evil_bundle_bytes()
    cfg.network.model_bundle_url = "https://example.test/evil.tar.gz"
    cfg.network.model_bundle_sha256 = hashlib.sha256(payload).hexdigest()
    respx.get(cfg.network.model_bundle_url).mock(return_value=httpx.Response(200, content=payload))
    with pytest.raises(Exception):
        model_bundle.download_bundle(cfg)
    assert not (tmp_path / "evil.txt").exists()


@respx.mock
def test_download_bundle_skips_when_present_without_force(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    cfg = DevragConfig()
    cfg.sparse_embedding.cache_dir = str(tmp_path / "fe")
    snap = tmp_path / "hub" / "models--cross-encoder--ms-marco-MiniLM-L-6-v2" / "snapshots" / "abc"
    snap.mkdir(parents=True); (snap / "config.json").write_text("{}")
    bm = tmp_path / "fe" / "models--Qdrant--bm25"; bm.mkdir(parents=True); (bm / "x").write_text("{}")
    route = respx.get("https://example.test/bundle.tar.gz").mock(return_value=httpx.Response(200, content=b""))
    cfg.network.model_bundle_url = "https://example.test/bundle.tar.gz"
    model_bundle.download_bundle(cfg)
    assert route.call_count == 0
