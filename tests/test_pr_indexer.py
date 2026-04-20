import hashlib
from unittest.mock import MagicMock, patch
import pytest
from devrag.ingest.pr_indexer import PRIndexer, chunk_pr


def _make_pr(number=1, title="Add auth", body="Adds authentication", state="closed",
             author="alice", labels=None, merged_at="2026-03-02T12:00:00Z",
             updated_at="2026-03-02T12:00:00Z", draft=False):
    return {"number": number, "title": title, "body": body, "state": state,
            "user": {"login": author}, "labels": [{"name": l} for l in (labels or [])],
            "merged_at": merged_at, "updated_at": updated_at, "draft": draft}


def _make_file(filename="src/auth.py", status="added", patch="@@ -0,0 +1,3 @@\n+def auth():\n+    pass"):
    return {"filename": filename, "status": status, "patch": patch}


def _make_comment(body="Looks good", user="bob", path="src/auth.py", line=10):
    return {"body": body, "user": {"login": user}, "path": path, "line": line,
            "created_at": "2026-03-02T10:00:00Z"}


def test_chunk_pr_creates_description_chunk():
    pr = _make_pr()
    chunks = chunk_pr(pr, [], [], repo="acme/backend")
    desc_chunks = [c for c in chunks if c.metadata["chunk_type"] == "description"]
    assert len(desc_chunks) == 1
    assert "Add auth" in desc_chunks[0].text
    assert "Adds authentication" in desc_chunks[0].text
    assert desc_chunks[0].metadata["pr_number"] == 1
    assert desc_chunks[0].metadata["pr_author"] == "alice"
    assert desc_chunks[0].metadata["repo"] == "acme/backend"


def test_chunk_pr_creates_diff_chunks():
    chunks = chunk_pr(_make_pr(), [_make_file()], [], repo="acme/backend")
    diff_chunks = [c for c in chunks if c.metadata["chunk_type"] == "diff"]
    assert len(diff_chunks) >= 1
    assert diff_chunks[0].metadata["file_path"] == "src/auth.py"
    assert "+def auth()" in diff_chunks[0].text


def test_chunk_pr_creates_comment_chunks():
    chunks = chunk_pr(_make_pr(), [], [_make_comment()], repo="acme/backend")
    comment_chunks = [c for c in chunks if c.metadata["chunk_type"] == "review_comment"]
    assert len(comment_chunks) == 1
    assert "Looks good" in comment_chunks[0].text
    assert comment_chunks[0].metadata["reviewer"] == "bob"
    assert comment_chunks[0].metadata["file_path"] == "src/auth.py"


def test_chunk_pr_metadata_fields():
    chunks = chunk_pr(_make_pr(labels=["security", "breaking-change"]), [_make_file()], [], repo="acme/backend")
    for chunk in chunks:
        assert chunk.metadata["repo"] == "acme/backend"
        assert chunk.metadata["pr_number"] == 1
        assert chunk.metadata["pr_title"] == "Add auth"
        assert chunk.metadata["pr_state"] == "closed"
        assert chunk.metadata["pr_author"] == "alice"


def test_chunk_pr_skips_empty_body():
    chunks = chunk_pr(_make_pr(body=None), [], [], repo="acme/backend")
    desc_chunks = [c for c in chunks if c.metadata["chunk_type"] == "description"]
    assert len(desc_chunks) == 1
    assert "Add auth" in desc_chunks[0].text


def test_chunk_ids_are_deterministic():
    pr = _make_pr()
    chunks1 = chunk_pr(pr, [_make_file()], [_make_comment()], repo="acme/backend")
    chunks2 = chunk_pr(pr, [_make_file()], [_make_comment()], repo="acme/backend")
    assert [c.id for c in chunks1] == [c.id for c in chunks2]


def test_pr_indexer_sync(tmp_dir, sparse_encoder):
    from devrag.stores.qdrant_store import QdrantStore
    from devrag.stores.metadata_db import MetadataDB
    store = QdrantStore(path=str(tmp_dir / "qdrant"), embedding_dim=768)
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])
    mock_github = MagicMock()
    mock_github.list_prs.return_value = [_make_pr(number=42, title="Fix bug")]
    mock_github.get_pr_files.return_value = [_make_file()]
    mock_github.get_pr_comments.return_value = [_make_comment()]
    indexer = PRIndexer(store, meta, embedder, sparse_encoder, mock_github)
    stats = indexer.sync("acme/backend", since_days=90)
    assert stats.prs_fetched == 1
    assert stats.prs_indexed == 1
    assert stats.chunks_created >= 3
    assert store.count("pr_diffs") >= 1
    assert store.count("pr_discussions") >= 1
    cursor = meta.get_pr_sync_cursor("acme/backend")
    assert cursor is not None


def test_pr_indexer_uses_cursor_when_since_days_none(tmp_dir, sparse_encoder):
    from devrag.stores.qdrant_store import QdrantStore
    from devrag.stores.metadata_db import MetadataDB
    store = QdrantStore(path=str(tmp_dir / "qdrant"), embedding_dim=768)
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    meta.set_pr_sync_cursor("acme/backend", "2026-03-15T10:00:00Z")
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])
    mock_github = MagicMock()
    mock_github.list_prs.return_value = []
    indexer = PRIndexer(store, meta, embedder, sparse_encoder, mock_github)

    indexer.sync("acme/backend", since_days=None)

    # list_prs should have been called with since = cursor
    call_kwargs = mock_github.list_prs.call_args.kwargs
    assert call_kwargs.get("since") == "2026-03-15T10:00:00Z"


def test_pr_indexer_since_days_overrides_cursor(tmp_dir, sparse_encoder):
    from devrag.stores.qdrant_store import QdrantStore
    from devrag.stores.metadata_db import MetadataDB
    store = QdrantStore(path=str(tmp_dir / "qdrant"), embedding_dim=768)
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    # Cursor is set but caller explicitly asks for a 180-day backfill.
    meta.set_pr_sync_cursor("acme/backend", "2026-03-15T10:00:00Z")
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])
    mock_github = MagicMock()
    mock_github.list_prs.return_value = []
    indexer = PRIndexer(store, meta, embedder, sparse_encoder, mock_github)

    indexer.sync("acme/backend", since_days=180)

    # list_prs should have been called with a since *older than the cursor* (~now - 180d),
    # not the cursor value.
    call_kwargs = mock_github.list_prs.call_args.kwargs
    since_passed = call_kwargs.get("since")
    assert since_passed is not None
    assert since_passed < "2026-03-15T10:00:00Z"
