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


def test_doc_indexer_indexes_directory(tmp_dir):
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
    indexer = DocIndexer(store, meta, embedder)
    stats = indexer.index_docs(docs_dir, glob_patterns=["**/*.md", "**/*.txt"])
    assert stats.files_scanned >= 2
    assert stats.files_indexed >= 2
    assert stats.chunks_created >= 2
    store.upsert.assert_called()
    embedder.embed.assert_called()
