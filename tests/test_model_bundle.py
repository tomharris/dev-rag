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
