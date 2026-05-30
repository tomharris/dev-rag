from pathlib import Path
from unittest.mock import MagicMock
import pytest
from devrag.ingest.doc_indexer import DocIndexer, chunk_document, split_markdown


def test_split_markdown_by_headings():
    text = "# Introduction\n\nThis is the intro.\n\n## Architecture\n\n### Components\n\nThe system has three parts.\n\n## Deployment\n\nDeploy with Docker.\n"
    sections = split_markdown(text)
    assert len(sections) >= 3
    paths = [s["section_path"] for s in sections]
    assert any("Introduction" in p for p in paths)
    assert any("Architecture" in p for p in paths)
    assert any("Deployment" in p for p in paths)


def test_split_markdown_preserves_hierarchy():
    text = "# Top\n\n## Middle\n\n### Bottom\n\nContent here.\n"
    sections = split_markdown(text)
    bottom = next(s for s in sections if "Bottom" in s["section_path"])
    assert bottom["section_path"] == "Top > Middle > Bottom"


def test_chunk_document_respects_max_tokens():
    long_text = "# Title\n\n" + ("This is a paragraph with enough words. " * 100 + "\n\n") * 5
    chunks = chunk_document(text=long_text, file_path="docs/long.md", max_tokens=100, overlap_tokens=10)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 100 * 4 + 200


def test_chunk_document_includes_metadata():
    text = "# API Guide\n\n## Authentication\n\nUse Bearer tokens.\n"
    chunks = chunk_document(text=text, file_path="docs/api.md", max_tokens=512, overlap_tokens=50)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.metadata["file_path"] == "docs/api.md"
        assert chunk.metadata["language"] == "markdown"
        assert "section_path" in chunk.metadata
        assert chunk.metadata["chunk_type"] == "document"


def test_chunk_document_plain_text():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.\n"
    chunks = chunk_document(text=text, file_path="notes.txt", max_tokens=512, overlap_tokens=50)
    assert len(chunks) >= 1
    assert chunks[0].metadata["language"] == "text"


def test_chunk_ids_deterministic():
    text = "# Hello\n\nWorld.\n"
    c1 = chunk_document(text=text, file_path="a.md", max_tokens=512, overlap_tokens=50)
    c2 = chunk_document(text=text, file_path="a.md", max_tokens=512, overlap_tokens=50)
    assert [c.id for c in c1] == [c.id for c in c2]


def test_doc_indexer_indexes_directory(tmp_dir, sparse_encoder):
    docs_dir = tmp_dir / "docs"
    docs_dir.mkdir()
    (docs_dir / "guide.md").write_text("# User Guide\n\nHow to use the app.\n\n## Setup\n\nInstall deps.\n")
    (docs_dir / "notes.txt").write_text("Some plain text notes.\n")
    (docs_dir / "image.png").write_bytes(b"\x89PNG")
    store = MagicMock()
    meta = MagicMock()
    meta.get_file_hash.return_value = None
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])
    indexer = DocIndexer(store, meta, embedder, sparse_encoder)
    stats = indexer.index_docs(docs_dir, glob_patterns=["**/*.md", "**/*.txt"])
    assert stats.files_scanned >= 2
    assert stats.files_indexed >= 2
    assert stats.chunks_created >= 2
    store.upsert.assert_called()
    embedder.embed.assert_called()


# ---------------------------------------------------------------------------
# index_repo_docs — per-repo doc indexing alongside code
# ---------------------------------------------------------------------------

import subprocess

from devrag.stores.metadata_db import MetadataDB


def _git_init(repo: Path) -> None:
    subprocess.run(["git", "init", str(repo)], capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=str(repo), capture_output=True)


@pytest.fixture
def repo_doc_deps(tmp_dir, vector_store, sparse_encoder):
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])
    return vector_store, meta, embedder, sparse_encoder


def test_index_repo_docs_indexes_and_tags_repo(tmp_dir, repo_doc_deps):
    store, meta, embedder, sparse_encoder = repo_doc_deps
    repo = tmp_dir / "repo"
    repo.mkdir()
    _git_init(repo)
    (repo / "README.md").write_text("# Project\n\nWhat it does.\n\n## Usage\n\nRun it.\n")
    (repo / "docs").mkdir()
    (repo / "docs" / "guide.md").write_text("# Guide\n\nHow to use.\n")
    (repo / "main.py").write_text("def hello():\n    return 'world'\n")  # not a doc — must be ignored

    indexer = DocIndexer(store, meta, embedder, sparse_encoder)
    stats = indexer.index_repo_docs(repo, repo_name="myrepo")

    assert stats.files_scanned == 2  # only the two .md files, not main.py
    assert stats.files_indexed == 2
    assert stats.chunks_created >= 2
    assert store.count("documents") >= 2

    # Doc chunks are tracked under the repo namespace, and carry the repo tag.
    readme_chunks = meta.get_chunks_for_file(str(repo / "README.md"), repo="myrepo")
    assert readme_chunks
    payload = store.get_by_ids("documents", readme_chunks[:1]).metadatas[0]
    assert payload["repo"] == "myrepo"
    assert payload["chunk_type"] == "document"


def test_index_repo_docs_incremental_skips_unchanged(tmp_dir, repo_doc_deps):
    store, meta, embedder, sparse_encoder = repo_doc_deps
    repo = tmp_dir / "repo"
    repo.mkdir()
    _git_init(repo)
    (repo / "README.md").write_text("# Project\n\nWhat it does.\n")
    indexer = DocIndexer(store, meta, embedder, sparse_encoder)
    s1 = indexer.index_repo_docs(repo, repo_name="myrepo")
    assert s1.files_indexed == 1
    embedder.embed.reset_mock()
    s2 = indexer.index_repo_docs(repo, repo_name="myrepo", incremental=True)
    assert s2.files_skipped == 1
    assert s2.files_indexed == 0
    embedder.embed.assert_not_called()


def test_index_repo_docs_removes_deleted_doc(tmp_dir, repo_doc_deps):
    store, meta, embedder, sparse_encoder = repo_doc_deps
    repo = tmp_dir / "repo"
    repo.mkdir()
    _git_init(repo)
    (repo / "README.md").write_text("# Project\n\nWhat it does.\n")
    (repo / "EXTRA.md").write_text("# Extra\n\nDelete me.\n")
    indexer = DocIndexer(store, meta, embedder, sparse_encoder)
    indexer.index_repo_docs(repo, repo_name="myrepo")
    extra_chunks = meta.get_chunks_for_file(str(repo / "EXTRA.md"), repo="myrepo")
    assert extra_chunks

    (repo / "EXTRA.md").unlink()
    stats = indexer.index_repo_docs(repo, repo_name="myrepo")
    assert stats.files_removed == 1
    assert meta.get_chunks_for_file(str(repo / "EXTRA.md"), repo="myrepo") == []
    # The deleted doc's chunks are gone from the documents collection.
    assert store.get_by_ids("documents", extra_chunks).ids == []


def test_index_repo_docs_isolates_failing_file(tmp_dir, repo_doc_deps):
    """One file that fails to embed is counted and skipped, not fatal — and the
    other files in the same repo still get indexed."""
    store, meta, embedder, sparse_encoder = repo_doc_deps
    repo = tmp_dir / "repo"
    repo.mkdir()
    _git_init(repo)
    (repo / "GOOD.md").write_text("# Good\n\nThis indexes fine.\n")
    (repo / "BAD.md").write_text("# Bad\n\nThis one explodes on embed.\n")

    def embed(texts):
        if any("explodes" in t for t in texts):
            raise RuntimeError("Ollama embed failed (400): input length exceeds context length")
        return [[0.1] * 768 for _ in texts]

    embedder.embed = MagicMock(side_effect=embed)
    indexer = DocIndexer(store, meta, embedder, sparse_encoder)
    stats = indexer.index_repo_docs(repo, repo_name="myrepo")

    assert stats.files_failed == 1
    assert stats.files_indexed == 1  # GOOD.md still got in
    assert meta.get_chunks_for_file(str(repo / "GOOD.md"), repo="myrepo")


def test_failed_file_is_retried_not_skipped(tmp_dir, repo_doc_deps):
    """A file whose embed fails must NOT have its hash persisted, so the next
    incremental run retries it instead of treating it as done."""
    store, meta, embedder, sparse_encoder = repo_doc_deps
    repo = tmp_dir / "repo"
    repo.mkdir()
    _git_init(repo)
    (repo / "FLAKY.md").write_text("# Flaky\n\nFails the first time only.\n")

    calls = {"n": 0}

    def embed(texts):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient embed failure")
        return [[0.1] * 768 for _ in texts]

    embedder.embed = MagicMock(side_effect=embed)
    indexer = DocIndexer(store, meta, embedder, sparse_encoder)

    s1 = indexer.index_repo_docs(repo, repo_name="myrepo")
    assert s1.files_failed == 1
    assert meta.get_file_hash(str(repo / "FLAKY.md"), repo="myrepo") is None

    # Second incremental run retries (hash was never stored) and succeeds.
    s2 = indexer.index_repo_docs(repo, repo_name="myrepo", incremental=True)
    assert s2.files_indexed == 1
    assert s2.files_skipped == 0
    assert meta.get_chunks_for_file(str(repo / "FLAKY.md"), repo="myrepo")
