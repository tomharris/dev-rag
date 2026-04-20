from unittest.mock import MagicMock
from devrag.ingest.issue_indexer import IssueIndexer, chunk_issue


def _make_issue(number=1, title="Login fails", body="Users can't log in", state="open",
                author="alice", labels=None, created_at="2026-03-01T12:00:00Z",
                updated_at="2026-03-02T12:00:00Z", pull_request=None):
    issue = {"number": number, "title": title, "body": body, "state": state,
             "user": {"login": author}, "labels": [{"name": l} for l in (labels or [])],
             "created_at": created_at, "updated_at": updated_at}
    if pull_request is not None:
        issue["pull_request"] = pull_request
    return issue


def _make_issue_comment(body="I can reproduce this", user="bob",
                        created_at="2026-03-02T10:00:00Z"):
    return {"body": body, "user": {"login": user}, "created_at": created_at}


def test_chunk_issue_creates_description_chunk():
    issue = _make_issue()
    chunks = chunk_issue(issue, [], repo="acme/backend")
    desc_chunks = [c for c in chunks if c.metadata["chunk_type"] == "description"]
    assert len(desc_chunks) == 1
    assert "Login fails" in desc_chunks[0].text
    assert "Users can't log in" in desc_chunks[0].text
    assert desc_chunks[0].metadata["issue_number"] == 1
    assert desc_chunks[0].metadata["issue_author"] == "alice"
    assert desc_chunks[0].metadata["repo"] == "acme/backend"


def test_chunk_issue_creates_comment_chunks():
    chunks = chunk_issue(_make_issue(), [_make_issue_comment()], repo="acme/backend")
    comment_chunks = [c for c in chunks if c.metadata["chunk_type"] == "comment"]
    assert len(comment_chunks) == 1
    assert "I can reproduce this" in comment_chunks[0].text
    assert comment_chunks[0].metadata["comment_author"] == "bob"


def test_chunk_issue_metadata_fields():
    chunks = chunk_issue(_make_issue(labels=["bug", "critical"]), [], repo="acme/backend")
    for chunk in chunks:
        assert chunk.metadata["repo"] == "acme/backend"
        assert chunk.metadata["issue_number"] == 1
        assert chunk.metadata["issue_title"] == "Login fails"
        assert chunk.metadata["issue_state"] == "open"
        assert chunk.metadata["issue_author"] == "alice"
        assert "bug" in chunk.metadata["issue_labels"]


def test_chunk_issue_empty_body():
    chunks = chunk_issue(_make_issue(body=None), [], repo="acme/backend")
    desc_chunks = [c for c in chunks if c.metadata["chunk_type"] == "description"]
    assert len(desc_chunks) == 1
    assert "Login fails" in desc_chunks[0].text


def test_chunk_ids_deterministic():
    issue = _make_issue()
    comments = [_make_issue_comment()]
    chunks1 = chunk_issue(issue, comments, repo="acme/backend")
    chunks2 = chunk_issue(issue, comments, repo="acme/backend")
    assert [c.id for c in chunks1] == [c.id for c in chunks2]


def test_chunk_issue_truncation():
    long_body = "x" * 10000
    chunks = chunk_issue(_make_issue(body=long_body), [], repo="acme/backend", max_tokens=512)
    desc_chunks = [c for c in chunks if c.metadata["chunk_type"] == "description"]
    assert len(desc_chunks[0].text) < 10000
    assert desc_chunks[0].text.endswith("# ... (truncated)")


def test_issue_indexer_sync(tmp_dir, sparse_encoder):
    from devrag.stores.qdrant_store import QdrantStore
    from devrag.stores.metadata_db import MetadataDB
    store = QdrantStore(path=str(tmp_dir / "qdrant"), embedding_dim=768)
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])
    mock_github = MagicMock()
    # Return one real issue and one PR (should be skipped)
    mock_github.list_issues.return_value = [
        _make_issue(number=10, title="Bug report"),
        _make_issue(number=99, title="Some PR", pull_request={"url": "https://..."}),
    ]
    mock_github.get_issue_comments.return_value = [_make_issue_comment()]
    indexer = IssueIndexer(store, meta, embedder, sparse_encoder, mock_github)
    stats = indexer.sync("acme/backend", since_days=90)
    assert stats.issues_fetched == 2
    assert stats.issues_indexed == 1
    assert stats.issues_skipped == 1
    assert stats.chunks_created >= 2  # 1 description + 1 comment
    assert store.count("issue_descriptions") >= 1
    assert store.count("issue_discussions") >= 1
    cursor = meta.get_issue_sync_cursor("acme/backend")
    assert cursor is not None


def test_issue_indexer_incremental_sync(tmp_dir, sparse_encoder):
    from devrag.stores.qdrant_store import QdrantStore
    from devrag.stores.metadata_db import MetadataDB
    store = QdrantStore(path=str(tmp_dir / "qdrant"), embedding_dim=768)
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])
    mock_github = MagicMock()
    mock_github.list_issues.return_value = [_make_issue(number=10, updated_at="2026-03-02T12:00:00Z")]
    mock_github.get_issue_comments.return_value = []
    indexer = IssueIndexer(store, meta, embedder, sparse_encoder, mock_github)

    # First sync
    indexer.sync("acme/backend", since_days=90)
    first_cursor = meta.get_issue_sync_cursor("acme/backend")

    # Second sync with same data — cursor should be passed as `since`
    mock_github.list_issues.return_value = []
    stats = indexer.sync("acme/backend", since_days=90)
    assert stats.issues_fetched == 0
    # Verify list_issues was called with the cursor as `since`
    call_kwargs = mock_github.list_issues.call_args
    assert call_kwargs.kwargs.get("since") == first_cursor or call_kwargs[1].get("since") == first_cursor


def test_issue_indexer_include_labels(tmp_dir, sparse_encoder):
    from devrag.stores.qdrant_store import QdrantStore
    from devrag.stores.metadata_db import MetadataDB
    store = QdrantStore(path=str(tmp_dir / "qdrant"), embedding_dim=768)
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])
    mock_github = MagicMock()
    mock_github.list_issues.return_value = [
        _make_issue(number=1, title="Bug", labels=["bug"]),
        _make_issue(number=2, title="Feature", labels=["enhancement"]),
        _make_issue(number=3, title="Both", labels=["bug", "enhancement"]),
    ]
    mock_github.get_issue_comments.return_value = []
    indexer = IssueIndexer(store, meta, embedder, sparse_encoder, mock_github, include_labels=["bug"])
    stats = indexer.sync("acme/backend", since_days=90)
    assert stats.issues_indexed == 2  # #1 (bug) and #3 (bug + enhancement)
    assert stats.issues_skipped == 1  # #2 (enhancement only)


def test_issue_indexer_exclude_labels(tmp_dir, sparse_encoder):
    from devrag.stores.qdrant_store import QdrantStore
    from devrag.stores.metadata_db import MetadataDB
    store = QdrantStore(path=str(tmp_dir / "qdrant"), embedding_dim=768)
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])
    mock_github = MagicMock()
    mock_github.list_issues.return_value = [
        _make_issue(number=1, title="Bug", labels=["bug"]),
        _make_issue(number=2, title="Wontfix", labels=["wontfix"]),
        _make_issue(number=3, title="Both", labels=["bug", "wontfix"]),
    ]
    mock_github.get_issue_comments.return_value = []
    indexer = IssueIndexer(store, meta, embedder, sparse_encoder, mock_github, exclude_labels=["wontfix"])
    stats = indexer.sync("acme/backend", since_days=90)
    assert stats.issues_indexed == 1  # #1 only
    assert stats.issues_skipped == 2  # #2 and #3


def test_issue_indexer_include_and_exclude_labels(tmp_dir, sparse_encoder):
    from devrag.stores.qdrant_store import QdrantStore
    from devrag.stores.metadata_db import MetadataDB
    store = QdrantStore(path=str(tmp_dir / "qdrant"), embedding_dim=768)
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])
    mock_github = MagicMock()
    mock_github.list_issues.return_value = [
        _make_issue(number=1, title="Bug", labels=["bug"]),
        _make_issue(number=2, title="Bug wontfix", labels=["bug", "wontfix"]),
        _make_issue(number=3, title="Feature", labels=["enhancement"]),
    ]
    mock_github.get_issue_comments.return_value = []
    indexer = IssueIndexer(store, meta, embedder, sparse_encoder, mock_github,
                           include_labels=["bug"], exclude_labels=["wontfix"])
    stats = indexer.sync("acme/backend", since_days=90)
    assert stats.issues_indexed == 1  # #1 only (has bug, no wontfix)
    assert stats.issues_skipped == 2  # #2 (excluded) and #3 (not included)
