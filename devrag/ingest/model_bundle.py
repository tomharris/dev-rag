"""Download and place DevRAG's HF models from a self-hosted bundle.

On networks that block huggingface.co, the reranker and BM25 models cannot be
downloaded from HF. This module fetches a pre-built bundle of both models from a
dev-rag GitHub release (reachable behind the corporate proxy) and unpacks it
into the HF hub cache and the FastEmbed cache, so the offline-first loaders find
them with no HF access.
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# Pinned per release. The release task (Task 9) creates the `models-v1` tag,
# uploads the bundle, and fills DEFAULT_BUNDLE_SHA256 with the printed digest.
DEFAULT_BUNDLE_URL = "https://github.com/tomharris/dev-rag/releases/download/models-v1/devrag-models.tar.gz"
DEFAULT_BUNDLE_SHA256 = ""


def resolve_fastembed_cache_dir(config) -> Path:
    """The persistent FastEmbed cache dir; default ~/.cache/devrag/fastembed."""
    raw = config.sparse_embedding.cache_dir
    if raw:
        return Path(raw).expanduser()
    return Path("~/.cache/devrag/fastembed").expanduser()


def bundle_target_dirs(config) -> tuple[Path, Path]:
    """(HF hub cache dir, FastEmbed cache dir) — the bundle's extraction targets."""
    from huggingface_hub import constants
    return Path(constants.HF_HUB_CACHE), resolve_fastembed_cache_dir(config)


def _hf_repo_dir(repo_id: str) -> str:
    """HF/FastEmbed on-disk cache dir name for a repo id (org/name)."""
    return "models--" + repo_id.replace("/", "--")


def _has_files(directory: Path) -> bool:
    return directory.is_dir() and any(p.is_file() for p in directory.rglob("*"))


def models_present(config) -> bool:
    """True when both the reranker and BM25 models are already on disk."""
    hf_dir, fe_dir = bundle_target_dirs(config)
    reranker_ok = _has_files(hf_dir / _hf_repo_dir(config.retrieval.reranker_model) / "snapshots")
    bm25_ok = _has_files(fe_dir / _hf_repo_dir(config.sparse_embedding.model))
    return reranker_ok and bm25_ok
