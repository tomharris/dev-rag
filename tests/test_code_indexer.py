import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from devrag.ingest import code_indexer as code_indexer_mod
from devrag.ingest.code_indexer import (
    LANGUAGE_EXTENSIONS,
    CodeIndexer,
    extract_chunks_from_file,
)
from devrag.types import Chunk


def test_extract_chunks_from_python_file(sample_python_file):
    chunks = extract_chunks_from_file(sample_python_file, max_tokens=512)
    entity_names = [c.metadata["entity_name"] for c in chunks]
    assert "FileProcessor" in entity_names or any(
        c.metadata.get("parent_entity") == "FileProcessor" for c in chunks
    )
    assert "standalone_function" in entity_names

    func_chunk = next(c for c in chunks if c.metadata["entity_name"] == "standalone_function")
    assert func_chunk.metadata["language"] == "python"
    assert func_chunk.metadata["entity_type"] in ("function", "function_definition")
    assert "line_range" in func_chunk.metadata
    assert "def standalone_function" in func_chunk.text


def test_extract_chunks_from_typescript_file(sample_ts_file):
    chunks = extract_chunks_from_file(sample_ts_file, max_tokens=512)
    entity_names = [c.metadata["entity_name"] for c in chunks]
    assert "Server" in entity_names or "loadConfig" in entity_names
    assert any(c.metadata["language"] == "typescript" for c in chunks)


def test_extract_chunks_unsupported_language(tmp_dir):
    path = tmp_dir / "data.csv"
    path.write_text("a,b,c\n1,2,3\n")
    chunks = extract_chunks_from_file(path, max_tokens=512)
    assert chunks == []


def test_language_extensions_mapping():
    assert LANGUAGE_EXTENSIONS[".py"] == "python"
    assert LANGUAGE_EXTENSIONS[".ts"] == "typescript"
    assert LANGUAGE_EXTENSIONS[".js"] == "javascript"
    assert LANGUAGE_EXTENSIONS[".rs"] == "rust"
    assert LANGUAGE_EXTENSIONS[".go"] == "go"


def test_chunk_ids_are_deterministic(sample_python_file):
    chunks1 = extract_chunks_from_file(sample_python_file, max_tokens=512)
    chunks2 = extract_chunks_from_file(sample_python_file, max_tokens=512)
    ids1 = [c.id for c in chunks1]
    ids2 = [c.id for c in chunks2]
    assert ids1 == ids2


def test_chunk_text_includes_context(sample_python_file):
    chunks = extract_chunks_from_file(sample_python_file, max_tokens=512)
    method_chunks = [c for c in chunks if c.metadata.get("parent_entity") == "FileProcessor"]
    if method_chunks:
        assert any("FileProcessor" in c.text or "read_file" in c.text for c in method_chunks)


# --- Integration tests for CodeIndexer class ---

from devrag.ingest.code_indexer import CodeIndexer
from devrag.stores.chroma_store import ChromaStore
from devrag.stores.metadata_db import MetadataDB


@pytest.fixture
def indexer_deps(tmp_dir):
    store = ChromaStore(persist_dir=str(tmp_dir / "chroma"))
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])
    return store, meta, embedder


def test_code_indexer_indexes_repo(tmp_dir, indexer_deps):
    store, meta, embedder = indexer_deps
    repo = tmp_dir / "repo"
    repo.mkdir()
    import subprocess
    subprocess.run(["git", "init", str(repo)], capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=str(repo), capture_output=True)
    (repo / "main.py").write_text("def hello():\n    return 'world'\n")
    (repo / "utils.py").write_text("def add(a, b):\n    return a + b\n")
    subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), capture_output=True)
    indexer = CodeIndexer(store, meta, embedder)
    stats = indexer.index_repo(repo)
    assert stats.files_scanned >= 2
    assert stats.files_indexed >= 2
    assert stats.chunks_created >= 2
    assert store.count("code_chunks") >= 2


def test_code_indexer_incremental_skips_unchanged(tmp_dir, indexer_deps):
    store, meta, embedder = indexer_deps
    repo = tmp_dir / "repo"
    repo.mkdir()
    import subprocess
    subprocess.run(["git", "init", str(repo)], capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=str(repo), capture_output=True)
    (repo / "main.py").write_text("def hello():\n    return 'world'\n")
    subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), capture_output=True)
    indexer = CodeIndexer(store, meta, embedder)
    stats1 = indexer.index_repo(repo)
    assert stats1.files_indexed >= 1
    embedder.embed.reset_mock()
    stats2 = indexer.index_repo(repo, incremental=True)
    assert stats2.files_skipped >= 1
    assert stats2.files_indexed == 0
    embedder.embed.assert_not_called()


def test_extract_chunks_skips_empty_text_nodes(tmp_dir):
    """Nodes whose source text is empty/whitespace-only should be excluded."""
    code = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
    p = tmp_dir / "test.py"
    p.write_text(code)

    # Baseline: both functions produce chunks
    normal_chunks = extract_chunks_from_file(p, max_tokens=512)
    assert len(normal_chunks) == 2

    # Simulate an empty-text node by patching _node_to_text to return
    # whitespace for the first entity while leaving the rest unchanged.
    original_fn = code_indexer_mod._node_to_text
    call_count = 0

    def _fake_node_to_text(node, source):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "   "
        return original_fn(node, source)

    with patch.object(code_indexer_mod, "_node_to_text", side_effect=_fake_node_to_text):
        filtered_chunks = extract_chunks_from_file(p, max_tokens=512)

    assert len(filtered_chunks) == 1
    assert filtered_chunks[0].metadata["entity_name"] == "bar"


def test_whole_file_chunk_skips_empty_file(tmp_dir):
    """Empty files should produce no chunks."""
    p = tmp_dir / "empty.py"
    p.write_text("")
    chunks = extract_chunks_from_file(p, max_tokens=512)
    assert chunks == []


def test_whole_file_chunk_skips_whitespace_only_file(tmp_dir):
    """Files containing only whitespace should produce no chunks."""
    p = tmp_dir / "blank.py"
    p.write_text("   \n\n  \t  \n")
    chunks = extract_chunks_from_file(p, max_tokens=512)
    assert chunks == []


def test_code_indexer_skips_empty_file(tmp_dir, indexer_deps):
    """Empty files should not cause embedding or upsert errors."""
    store, meta, embedder = indexer_deps
    repo = tmp_dir / "repo"
    repo.mkdir()
    import subprocess
    subprocess.run(["git", "init", str(repo)], capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=str(repo), capture_output=True)
    (repo / "empty.py").write_text("")
    (repo / "real.py").write_text("def hello():\n    return 'world'\n")
    subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), capture_output=True)
    indexer = CodeIndexer(store, meta, embedder)
    stats = indexer.index_repo(repo)
    assert stats.files_indexed >= 1
    assert stats.chunks_created >= 1


def test_code_indexer_multi_repo_no_cross_deletion(tmp_dir, indexer_deps):
    """Indexing repo-b should not delete repo-a's data."""
    store, meta, embedder = indexer_deps
    import subprocess

    # Create repo-a
    repo_a = tmp_dir / "repo-a"
    repo_a.mkdir()
    subprocess.run(["git", "init", str(repo_a)], capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=str(repo_a), capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=str(repo_a), capture_output=True)
    (repo_a / "main.py").write_text("def hello_a():\n    return 'a'\n")
    subprocess.run(["git", "add", "."], cwd=str(repo_a), capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo_a), capture_output=True)

    # Create repo-b
    repo_b = tmp_dir / "repo-b"
    repo_b.mkdir()
    subprocess.run(["git", "init", str(repo_b)], capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=str(repo_b), capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=str(repo_b), capture_output=True)
    (repo_b / "app.py").write_text("def hello_b():\n    return 'b'\n")
    subprocess.run(["git", "add", "."], cwd=str(repo_b), capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo_b), capture_output=True)

    indexer = CodeIndexer(store, meta, embedder)
    stats_a = indexer.index_repo(repo_a, repo_name="repo-a")
    assert stats_a.files_indexed >= 1
    count_after_a = store.count("code_chunks")

    stats_b = indexer.index_repo(repo_b, repo_name="repo-b")
    assert stats_b.files_indexed >= 1
    assert stats_b.files_removed == 0  # Must not remove repo-a's files

    # Both repos' chunks should be present
    assert store.count("code_chunks") >= count_after_a + stats_b.chunks_created

    # Repo registry should have both
    repos = meta.get_all_repos()
    repo_names = {r[0] for r in repos}
    assert "repo-a" in repo_names
    assert "repo-b" in repo_names


def test_code_indexer_detects_removed_files(tmp_dir, indexer_deps):
    store, meta, embedder = indexer_deps
    repo = tmp_dir / "repo"
    repo.mkdir()
    import subprocess
    subprocess.run(["git", "init", str(repo)], capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=str(repo), capture_output=True)
    (repo / "main.py").write_text("def hello():\n    return 'world'\n")
    (repo / "old.py").write_text("def old():\n    pass\n")
    subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), capture_output=True)
    indexer = CodeIndexer(store, meta, embedder)
    indexer.index_repo(repo)
    initial_count = store.count("code_chunks")
    (repo / "old.py").unlink()
    subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "commit", "-m", "remove old"], cwd=str(repo), capture_output=True)
    stats = indexer.index_repo(repo)
    assert stats.files_removed >= 1
    assert store.count("code_chunks") < initial_count
