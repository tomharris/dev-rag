from unittest.mock import MagicMock

from devrag.ingest.jira_indexer import (
    JiraIndexer,
    _inject_cursor_into_jql,
    _iso_to_jql_datetime,
    _make_cursor_key,
    chunk_jira_ticket,
)
from devrag.utils.jira_client import JiraClient

INSTANCE = "https://acme.atlassian.net"


def _make_jira_issue(key="DEV-123", summary="Login fails", description=None,
                     status="Open", issuetype="Bug", reporter="Alice",
                     assignee="Bob", priority="High", labels=None,
                     created="2026-03-01T12:00:00.000+0000",
                     updated="2026-03-02T12:00:00.000+0000",
                     comments=None):
    if description is None:
        description = {
            "type": "doc", "version": 1,
            "content": [{"type": "paragraph", "content": [
                {"type": "text", "text": "Users can't log in after password reset."},
            ]}],
        }
    return {
        "key": key,
        "fields": {
            "summary": summary,
            "description": description,
            "status": {"name": status},
            "issuetype": {"name": issuetype},
            "reporter": {"displayName": reporter},
            "assignee": {"displayName": assignee},
            "priority": {"name": priority},
            "labels": labels or [],
            "created": created,
            "updated": updated,
            "comment": {
                "comments": comments or [],
            },
        },
    }


def _make_jira_comment(body_text="I can reproduce this", author="Charlie",
                       created="2026-03-02T10:00:00.000+0000"):
    return {
        "author": {"displayName": author},
        "body": {
            "type": "doc", "version": 1,
            "content": [{"type": "paragraph", "content": [
                {"type": "text", "text": body_text},
            ]}],
        },
        "created": created,
    }


# --- ADF-to-text tests ---

def test_adf_to_text_simple_paragraph():
    adf = {"type": "doc", "version": 1, "content": [
        {"type": "paragraph", "content": [
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "world"},
        ]},
    ]}
    assert JiraClient.adf_to_text(adf) == "Hello world"


def test_adf_to_text_nested_structure():
    adf = {"type": "doc", "version": 1, "content": [
        {"type": "heading", "attrs": {"level": 2}, "content": [
            {"type": "text", "text": "Section Title"},
        ]},
        {"type": "paragraph", "content": [
            {"type": "text", "text": "Body text here."},
        ]},
        {"type": "bulletList", "content": [
            {"type": "listItem", "content": [
                {"type": "paragraph", "content": [
                    {"type": "text", "text": "Item one"},
                ]},
            ]},
            {"type": "listItem", "content": [
                {"type": "paragraph", "content": [
                    {"type": "text", "text": "Item two"},
                ]},
            ]},
        ]},
    ]}
    result = JiraClient.adf_to_text(adf)
    assert "Section Title" in result
    assert "Body text here." in result
    assert "Item one" in result
    assert "Item two" in result


def test_adf_to_text_empty_doc():
    assert JiraClient.adf_to_text(None) == ""
    assert JiraClient.adf_to_text({"type": "doc", "content": []}) == ""


def test_adf_to_text_plain_text_fallback():
    assert JiraClient.adf_to_text("Just a plain string") == "Just a plain string"


# --- Chunking tests ---

def test_chunk_jira_ticket_creates_description_chunk():
    issue = _make_jira_issue()
    chunks = chunk_jira_ticket(issue, INSTANCE)
    desc_chunks = [c for c in chunks if c.metadata["chunk_type"] == "description"]
    assert len(desc_chunks) == 1
    assert "Login fails" in desc_chunks[0].text
    assert "Users can't log in" in desc_chunks[0].text
    assert desc_chunks[0].metadata["ticket_key"] == "DEV-123"
    assert desc_chunks[0].metadata["jira_instance"] == INSTANCE


def test_chunk_jira_ticket_creates_comment_chunks():
    issue = _make_jira_issue(comments=[_make_jira_comment()])
    chunks = chunk_jira_ticket(issue, INSTANCE)
    comment_chunks = [c for c in chunks if c.metadata["chunk_type"] == "comment"]
    assert len(comment_chunks) == 1
    assert "I can reproduce this" in comment_chunks[0].text
    assert comment_chunks[0].metadata["comment_author"] == "Charlie"


def test_chunk_jira_ticket_metadata_fields():
    issue = _make_jira_issue(labels=["bug", "critical"])
    chunks = chunk_jira_ticket(issue, INSTANCE)
    for chunk in chunks:
        assert chunk.metadata["jira_instance"] == INSTANCE
        assert chunk.metadata["ticket_key"] == "DEV-123"
        assert chunk.metadata["ticket_summary"] == "Login fails"
        assert chunk.metadata["ticket_status"] == "Open"
        assert chunk.metadata["ticket_type"] == "Bug"
        assert chunk.metadata["reporter"] == "Alice"
        assert chunk.metadata["assignee"] == "Bob"
        assert "bug" in chunk.metadata["labels"]


def test_chunk_jira_ticket_empty_description():
    issue = _make_jira_issue(description=None)
    chunks = chunk_jira_ticket(issue, INSTANCE)
    desc_chunks = [c for c in chunks if c.metadata["chunk_type"] == "description"]
    assert len(desc_chunks) == 1
    assert "Login fails" in desc_chunks[0].text


def test_chunk_ids_deterministic():
    issue = _make_jira_issue(comments=[_make_jira_comment()])
    chunks1 = chunk_jira_ticket(issue, INSTANCE)
    chunks2 = chunk_jira_ticket(issue, INSTANCE)
    assert [c.id for c in chunks1] == [c.id for c in chunks2]


def test_chunk_jira_ticket_truncation():
    long_desc = {"type": "doc", "version": 1, "content": [
        {"type": "paragraph", "content": [
            {"type": "text", "text": "x" * 10000},
        ]},
    ]}
    issue = _make_jira_issue(description=long_desc)
    chunks = chunk_jira_ticket(issue, INSTANCE, max_tokens=512)
    desc_chunks = [c for c in chunks if c.metadata["chunk_type"] == "description"]
    assert len(desc_chunks[0].text) < 10000
    assert desc_chunks[0].text.endswith("# ... (truncated)")


# --- Cursor and JQL tests ---

def test_make_cursor_key():
    key1 = _make_cursor_key(INSTANCE, "project = DEV")
    key2 = _make_cursor_key(INSTANCE, "project = DEV")
    key3 = _make_cursor_key(INSTANCE, "project = OPS")
    assert key1 == key2
    assert key1 != key3
    assert len(key1) == 16


def test_jql_cursor_injection():
    jql = "project = DEV AND type = Bug"
    result = _inject_cursor_into_jql(jql, "2026/04/01 00:00")
    assert 'AND updated >= "2026/04/01 00:00"' in result
    assert result.startswith("project = DEV AND type = Bug")


def test_jql_cursor_injection_with_order_by():
    jql = "project = DEV ORDER BY created DESC"
    result = _inject_cursor_into_jql(jql, "2026/04/01 00:00")
    assert 'AND updated >= "2026/04/01 00:00"' in result
    assert "ORDER BY created DESC" in result
    # Cursor should come before ORDER BY
    cursor_pos = result.index("updated >=")
    order_pos = result.index("ORDER BY")
    assert cursor_pos < order_pos


# --- Integration tests ---

def test_jira_indexer_sync(tmp_dir, sparse_encoder):
    from devrag.stores.qdrant_store import QdrantStore
    from devrag.stores.metadata_db import MetadataDB
    store = QdrantStore(path=str(tmp_dir / "qdrant"), embedding_dim=768)
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])
    mock_jira = MagicMock(spec=JiraClient)
    mock_jira.search_issues.return_value = iter([
        _make_jira_issue(key="DEV-1", comments=[_make_jira_comment()]),
    ])
    indexer = JiraIndexer(store, meta, embedder, sparse_encoder, mock_jira)
    stats = indexer.sync(INSTANCE, "project = DEV", since_days=90)
    assert stats.tickets_fetched == 1
    assert stats.tickets_indexed == 1
    assert stats.chunks_created >= 2  # 1 description + 1 comment
    assert store.count("jira_descriptions") >= 1
    assert store.count("jira_discussions") >= 1
    cursor_key = _make_cursor_key(INSTANCE, "project = DEV")
    assert meta.get_jira_sync_cursor(cursor_key) is not None


def test_jira_indexer_incremental_sync(tmp_dir, sparse_encoder):
    from devrag.stores.qdrant_store import QdrantStore
    from devrag.stores.metadata_db import MetadataDB
    store = QdrantStore(path=str(tmp_dir / "qdrant"), embedding_dim=768)
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])
    mock_jira = MagicMock(spec=JiraClient)
    mock_jira.search_issues.return_value = iter([_make_jira_issue(key="DEV-1")])
    indexer = JiraIndexer(store, meta, embedder, sparse_encoder, mock_jira)

    # First sync
    indexer.sync(INSTANCE, "project = DEV", since_days=90)
    cursor_key = _make_cursor_key(INSTANCE, "project = DEV")
    first_cursor = meta.get_jira_sync_cursor(cursor_key)
    assert first_cursor is not None

    # Second sync — JQL should include cursor date
    mock_jira.search_issues.return_value = iter([])
    indexer.sync(INSTANCE, "project = DEV", since_days=90)
    call_args = mock_jira.search_issues.call_args
    effective_jql = call_args[0][0] if call_args[0] else call_args[1]["jql"]
    assert "updated >=" in effective_jql


def test_iso_to_jql_datetime_plus_offset():
    # Jira API typically returns offset without colon, e.g. '+0000'
    assert _iso_to_jql_datetime("2026-03-02T12:00:00.000+0000") == "2026/03/02 12:00"


def test_iso_to_jql_datetime_z_suffix():
    assert _iso_to_jql_datetime("2026-03-02T12:00:00Z") == "2026/03/02 12:00"


def test_iso_to_jql_datetime_with_colon_offset():
    assert _iso_to_jql_datetime("2026-03-02T12:00:00+00:00") == "2026/03/02 12:00"


def test_jira_indexer_recovers_from_stale_iso_cursor(tmp_dir, sparse_encoder):
    """If a previous (buggy) run stored an ISO-formatted cursor, the next sync should
    defensively normalize it rather than injecting invalid JQL."""
    from devrag.stores.qdrant_store import QdrantStore
    from devrag.stores.metadata_db import MetadataDB
    store = QdrantStore(path=str(tmp_dir / "qdrant"), embedding_dim=768)
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])
    mock_jira = MagicMock(spec=JiraClient)
    mock_jira.search_issues.return_value = iter([])
    indexer = JiraIndexer(store, meta, embedder, sparse_encoder, mock_jira)

    cursor_key = _make_cursor_key(INSTANCE, "project = DEV")
    meta.set_jira_sync_cursor(cursor_key, "2026-03-02T12:00:00.000+0000")

    indexer.sync(INSTANCE, "project = DEV", since_days=90)
    effective_jql = mock_jira.search_issues.call_args[0][0]
    assert 'updated >= "2026/03/02 12:00"' in effective_jql


def test_jira_indexer_stored_cursor_is_valid_jql_format(tmp_dir, sparse_encoder):
    """Regression: cursor must be stored in JQL datetime format, not raw ISO from Jira API.
    If this fails, the next sync's JQL clause will be malformed and incremental sync won't work."""
    from devrag.stores.qdrant_store import QdrantStore
    from devrag.stores.metadata_db import MetadataDB
    store = QdrantStore(path=str(tmp_dir / "qdrant"), embedding_dim=768)
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])
    mock_jira = MagicMock(spec=JiraClient)
    mock_jira.search_issues.return_value = iter([
        _make_jira_issue(key="DEV-1", updated="2026-03-02T12:00:00.000+0000"),
    ])
    indexer = JiraIndexer(store, meta, embedder, sparse_encoder, mock_jira)
    indexer.sync(INSTANCE, "project = DEV", since_days=90)

    cursor = meta.get_jira_sync_cursor(_make_cursor_key(INSTANCE, "project = DEV"))
    # JQL format is 'YYYY/MM/DD HH:MM' — not ISO
    assert cursor == "2026/03/02 12:00"

    # Second sync should inject a valid-looking JQL cursor clause
    mock_jira.search_issues.return_value = iter([])
    indexer.sync(INSTANCE, "project = DEV", since_days=90)
    effective_jql = mock_jira.search_issues.call_args[0][0]
    assert 'updated >= "2026/03/02 12:00"' in effective_jql
    # Must NOT contain ISO markers
    assert "T" not in effective_jql.split("updated >=")[1].split('"')[1]
