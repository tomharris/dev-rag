"""Tests for the all-repos refresh orchestrator (`devrag.ingest.refresh`).

The orchestrator is dependency-injected so these tests exercise the iteration,
missing-directory handling, and incremental/full branching without a real
embedder, vector store, or metadata DB.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from devrag.ingest.refresh import RefreshSummary, refresh_all_repos


class _Recorder:
    """Records (name, incremental) for each index call and (name) for removals."""

    def __init__(self) -> None:
        self.indexed: list[tuple[str, bool]] = []
        self.doc_indexed: list[tuple[str, bool]] = []
        self.removed: list[str] = []

    def index_repo(self, repo_dir: Path, repo_name: str, incremental: bool):
        self.indexed.append((repo_name, incremental))
        return f"code:{repo_name}"

    def index_repo_docs(self, repo_dir: Path, repo_name: str, incremental: bool):
        self.doc_indexed.append((repo_name, incremental))
        return f"docs:{repo_name}"

    def remove_repo(self, name: str) -> int:
        self.removed.append(name)
        return 7


def _existing(tmp_path: Path, name: str) -> tuple[str, str]:
    d = tmp_path / name
    d.mkdir()
    return (name, str(d))


def test_incremental_refresh_indexes_each_repo(tmp_path):
    rec = _Recorder()
    repos = [_existing(tmp_path, "alpha"), _existing(tmp_path, "beta")]

    summary = refresh_all_repos(
        repos,
        index_repo=rec.index_repo,
        index_repo_docs=rec.index_repo_docs,
        remove_repo=rec.remove_repo,
        full=False,
    )

    assert rec.indexed == [("alpha", True), ("beta", True)]
    assert rec.doc_indexed == [("alpha", True), ("beta", True)]
    assert rec.removed == []  # incremental never removes
    assert summary == RefreshSummary(refreshed=2, skipped=0, skipped_repos=[])


def test_missing_directory_is_skipped_not_fatal(tmp_path):
    rec = _Recorder()
    repos = [
        _existing(tmp_path, "alpha"),
        ("ghost", str(tmp_path / "does-not-exist")),
        _existing(tmp_path, "beta"),
    ]

    summary = refresh_all_repos(
        repos,
        index_repo=rec.index_repo,
        index_repo_docs=rec.index_repo_docs,
        remove_repo=rec.remove_repo,
        full=False,
    )

    # Healthy repos still refreshed; ghost neither indexed nor removed.
    assert rec.indexed == [("alpha", True), ("beta", True)]
    assert summary.refreshed == 2
    assert summary.skipped == 1
    assert summary.skipped_repos == [("ghost", str(tmp_path / "does-not-exist"))]


def test_full_removes_then_reindexes_non_incrementally(tmp_path):
    rec = _Recorder()
    repos = [_existing(tmp_path, "alpha")]

    summary = refresh_all_repos(
        repos,
        index_repo=rec.index_repo,
        index_repo_docs=rec.index_repo_docs,
        remove_repo=rec.remove_repo,
        full=True,
    )

    assert rec.removed == ["alpha"]
    assert rec.indexed == [("alpha", False)]
    assert rec.doc_indexed == [("alpha", False)]
    assert summary == RefreshSummary(refreshed=1, skipped=0, skipped_repos=[])


def test_full_does_not_remove_missing_repo(tmp_path):
    rec = _Recorder()
    repos = [("ghost", str(tmp_path / "nope"))]

    summary = refresh_all_repos(
        repos,
        index_repo=rec.index_repo,
        index_repo_docs=rec.index_repo_docs,
        remove_repo=rec.remove_repo,
        full=True,
    )

    assert rec.removed == []  # never touch a repo we can't re-index
    assert rec.indexed == []
    assert summary.skipped == 1


def test_docs_skipped_when_doc_indexer_absent(tmp_path):
    rec = _Recorder()
    repos = [_existing(tmp_path, "alpha")]

    summary = refresh_all_repos(
        repos,
        index_repo=rec.index_repo,
        index_repo_docs=None,
        remove_repo=rec.remove_repo,
        full=False,
    )

    assert rec.indexed == [("alpha", True)]
    assert rec.doc_indexed == []
    assert summary.refreshed == 1


def test_empty_registry_returns_zeroed_summary():
    rec = _Recorder()

    summary = refresh_all_repos(
        [],
        index_repo=rec.index_repo,
        index_repo_docs=rec.index_repo_docs,
        remove_repo=rec.remove_repo,
        full=False,
    )

    assert summary == RefreshSummary(refreshed=0, skipped=0, skipped_repos=[])
    assert rec.indexed == []
