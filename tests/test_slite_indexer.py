from unittest.mock import MagicMock

import httpx

from devrag.ingest.slite_indexer import (
    SliteIndexer,
    _make_chunk_id,
    _truncate_text,
    chunk_slite_page,
)
from devrag.utils.slite_client import SliteClient


def _make_note(note_id="page-1", title="Getting Started", content=None,
               url="https://app.slite.com/p/page-1",
               last_edited_at="2026-04-01T12:00:00Z",
               updated_at="2026-06-30T12:00:00Z"):
    if content is None:
        content = "# Introduction\n\nWelcome to the team.\n\n## Setup\n\nRun `make install` to get started."
    return {
        "id": note_id,
        "title": title,
        "url": url,
        # updatedAt is a volatile popularity timestamp; lastEditedAt is the
        # real content-edit time that drives incremental sync.
        "updatedAt": updated_at,
        "lastEditedAt": last_edited_at,
        "content": content,
    }


# --- Chunking tests ---

def test_chunk_slite_page_creates_section_chunks():
    note = _make_note()
    chunks = chunk_slite_page(note)
    assert len(chunks) >= 2  # Introduction + Setup sections
    section_paths = [c.metadata["section_path"] for c in chunks]
    assert any("Introduction" in p for p in section_paths)
    assert any("Setup" in p for p in section_paths)


def test_chunk_slite_page_metadata():
    note = _make_note()
    chunks = chunk_slite_page(note)
    for chunk in chunks:
        assert chunk.metadata["page_id"] == "page-1"
        assert chunk.metadata["page_title"] == "Getting Started"
        assert chunk.metadata["page_url"] == "https://app.slite.com/p/page-1"
        assert chunk.metadata["chunk_type"] == "slite_page"
        # updated_at payload reflects the content-edit time (lastEditedAt),
        # not the volatile updatedAt popularity timestamp.
        assert chunk.metadata["updated_at"] == "2026-04-01T12:00:00Z"


def test_chunk_slite_page_empty_content():
    note = _make_note(content="")
    assert chunk_slite_page(note) == []
    note2 = _make_note(content="   ")
    assert chunk_slite_page(note2) == []


def test_chunk_slite_page_no_headings():
    note = _make_note(content="Just a plain paragraph with no headings.")
    chunks = chunk_slite_page(note)
    assert len(chunks) == 1
    assert chunks[0].metadata["section_path"] == "Document"


def test_chunk_ids_deterministic():
    note = _make_note()
    chunks1 = chunk_slite_page(note)
    chunks2 = chunk_slite_page(note)
    assert [c.id for c in chunks1] == [c.id for c in chunks2]


def test_chunk_slite_page_long_section():
    long_content = "# Big Section\n\n" + "word " * 1000
    note = _make_note(content=long_content)
    chunks = chunk_slite_page(note, max_tokens=128)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["chunk_type"] == "slite_page"


def test_truncate_text_short_text_unchanged():
    text = "Short text that fits."
    result = _truncate_text(text, max_tokens=512)
    assert result == text


def test_truncate_text_long_text_truncated():
    # 512 tokens * 4 chars/token = 2048 chars max
    long_text = "x" * 3000
    result = _truncate_text(long_text, max_tokens=512)
    assert len(result) < len(long_text)
    assert result.endswith("\n... (truncated)")
    assert len(result) == 2048 + len("\n... (truncated)")


# --- Make chunk ID tests ---

def test_make_chunk_id_deterministic():
    id1 = _make_chunk_id("page-1", "Intro", 0)
    id2 = _make_chunk_id("page-1", "Intro", 0)
    assert id1 == id2
    assert len(id1) == 16


def test_make_chunk_id_varies():
    id1 = _make_chunk_id("page-1", "Intro", 0)
    id2 = _make_chunk_id("page-2", "Intro", 0)
    id3 = _make_chunk_id("page-1", "Setup", 0)
    assert id1 != id2
    assert id1 != id3


# --- Indexer sync tests ---

def test_slite_indexer_sync(tmp_dir, sparse_encoder):
    from devrag.stores.qdrant_store import QdrantStore
    from devrag.stores.metadata_db import MetadataDB
    store = QdrantStore(path=str(tmp_dir / "qdrant"), embedding_dim=768)
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])

    mock_slite = MagicMock(spec=SliteClient)
    mock_slite.list_notes.return_value = iter([
        {"id": "page-1", "title": "Onboarding", "url": "https://app.slite.com/p/page-1",
         "lastEditedAt": "2026-04-01T12:00:00Z", "updatedAt": "2026-06-30T12:00:00Z"},
    ])
    mock_slite.get_note.return_value = {
        "id": "page-1", "title": "Onboarding",
        "url": "https://app.slite.com/p/page-1",
        "content": "# Onboarding\n\nWelcome aboard.\n\n## Day 1\n\nSet up your laptop.",
    }

    indexer = SliteIndexer(store, meta, embedder, sparse_encoder, mock_slite)
    stats = indexer.sync(since_days=90)
    assert stats.pages_fetched == 1
    assert stats.pages_indexed == 1
    assert stats.chunks_created >= 2
    assert store.count("slite_pages") >= 2
    # Cursor tracks the content-edit time, not the popularity timestamp.
    assert meta.get_slite_sync_cursor("default") == "2026-04-01T12:00:00Z"


def test_slite_indexer_skips_empty_pages(tmp_dir, sparse_encoder):
    from devrag.stores.qdrant_store import QdrantStore
    from devrag.stores.metadata_db import MetadataDB
    store = QdrantStore(path=str(tmp_dir / "qdrant"), embedding_dim=768)
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])

    mock_slite = MagicMock(spec=SliteClient)
    mock_slite.list_notes.return_value = iter([
        {"id": "page-empty", "title": "Empty", "url": "...",
         "lastEditedAt": "2026-04-01T12:00:00Z", "updatedAt": "2026-06-30T12:00:00Z"},
    ])
    mock_slite.get_note.return_value = {"id": "page-empty", "title": "Empty", "content": ""}

    indexer = SliteIndexer(store, meta, embedder, sparse_encoder, mock_slite)
    stats = indexer.sync(since_days=90)
    assert stats.pages_fetched == 1
    assert stats.pages_skipped == 1
    assert stats.pages_indexed == 0


def test_slite_indexer_incremental_sync(tmp_dir, sparse_encoder):
    from devrag.stores.qdrant_store import QdrantStore
    from devrag.stores.metadata_db import MetadataDB
    store = QdrantStore(path=str(tmp_dir / "qdrant"), embedding_dim=768)
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])

    mock_slite = MagicMock(spec=SliteClient)
    mock_slite.get_note.return_value = {
        "id": "page-1", "title": "Guide", "content": "# Guide\n\nSome content.",
    }

    indexer = SliteIndexer(store, meta, embedder, sparse_encoder, mock_slite)

    def _note(last_edited, updated="2026-06-30T12:00:00Z"):
        return {"id": "page-1", "title": "Guide", "url": "...",
                "lastEditedAt": last_edited, "updatedAt": updated}

    # First sync — indexes the page, sets cursor to lastEditedAt
    mock_slite.list_notes.return_value = iter([_note("2026-04-01T12:00:00Z")])
    stats1 = indexer.sync(since_days=90)
    assert stats1.pages_indexed == 1
    assert meta.get_slite_sync_cursor("default") == "2026-04-01T12:00:00Z"

    # Second sync — same lastEditedAt (content unchanged); skip client-side
    mock_slite.list_notes.return_value = iter([_note("2026-04-01T12:00:00Z")])
    stats2 = indexer.sync(since_days=90)
    assert stats2.pages_indexed == 0
    assert stats2.pages_skipped == 1
    assert stats2.pages_fetched == 0
    # We no longer narrow the fetch by sinceDaysAgo — list_notes is called with
    # only the channel filter.
    call_kwargs = mock_slite.list_notes.call_args.kwargs
    assert "since_days_ago" not in call_kwargs

    # Third sync — content edited (newer lastEditedAt); re-index and advance cursor
    mock_slite.list_notes.return_value = iter([_note("2026-04-10T12:00:00Z")])
    stats3 = indexer.sync(since_days=90)
    assert stats3.pages_indexed == 1
    assert stats3.pages_skipped == 0
    assert meta.get_slite_sync_cursor("default") == "2026-04-10T12:00:00Z"


def test_slite_indexer_ignores_updatedat_churn(tmp_dir, sparse_encoder):
    """Regression: a note whose updatedAt is bumped to 'now' but whose
    lastEditedAt is unchanged must be skipped, not re-indexed."""
    from devrag.stores.qdrant_store import QdrantStore
    from devrag.stores.metadata_db import MetadataDB
    store = QdrantStore(path=str(tmp_dir / "qdrant"), embedding_dim=768)
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])

    mock_slite = MagicMock(spec=SliteClient)
    mock_slite.get_note.return_value = {
        "id": "page-1", "title": "Guide", "content": "# Guide\n\nSome content.",
    }

    indexer = SliteIndexer(store, meta, embedder, sparse_encoder, mock_slite)

    # First sync sets the cursor from lastEditedAt.
    mock_slite.list_notes.return_value = iter([
        {"id": "page-1", "title": "Guide", "url": "...",
         "lastEditedAt": "2026-01-01T00:00:00Z", "updatedAt": "2026-05-01T00:00:00Z"},
    ])
    indexer.sync(since_days=90)
    assert meta.get_slite_sync_cursor("default") == "2026-01-01T00:00:00Z"

    # Second sync: updatedAt jumps far forward (popularity churn) but the
    # content edit time is unchanged — must be skipped.
    mock_slite.list_notes.return_value = iter([
        {"id": "page-1", "title": "Guide", "url": "...",
         "lastEditedAt": "2026-01-01T00:00:00Z", "updatedAt": "2026-07-02T00:00:00Z"},
    ])
    stats = indexer.sync(since_days=90)
    assert stats.pages_indexed == 0
    assert stats.pages_skipped == 1
    assert stats.pages_fetched == 0
    # Cursor stays put — no phantom advancement from updatedAt.
    assert meta.get_slite_sync_cursor("default") == "2026-01-01T00:00:00Z"


def _429():
    return httpx.HTTPStatusError(
        "rate limited",
        request=httpx.Request("GET", "https://api.slite.com/v1/notes/x"),
        response=httpx.Response(429),
    )


def test_slite_indexer_saves_progress_on_mid_sweep_429(tmp_dir, sparse_encoder):
    """A 429 partway through the sweep must not discard progress: pages indexed
    before the failure stay indexed and the cursor advances to the last success."""
    from devrag.stores.qdrant_store import QdrantStore
    from devrag.stores.metadata_db import MetadataDB
    store = QdrantStore(path=str(tmp_dir / "qdrant"), embedding_dim=768)
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])

    mock_slite = MagicMock(spec=SliteClient)
    mock_slite.list_notes.return_value = iter([
        _make_note("p1", last_edited_at="2026-04-01T12:00:00Z"),
        _make_note("p2", last_edited_at="2026-04-02T12:00:00Z"),
        _make_note("p3", last_edited_at="2026-04-03T12:00:00Z"),
    ])
    # First two fetches succeed; the third 429s (survived the client's retries).
    mock_slite.get_note.side_effect = [
        {"id": "p1", "content": "# One\n\nBody."},
        {"id": "p2", "content": "# Two\n\nBody."},
        _429(),
    ]

    indexer = SliteIndexer(store, meta, embedder, sparse_encoder, mock_slite)
    stats = indexer.sync(since_days=90)  # must not raise

    assert stats.pages_indexed == 2
    # Cursor sits at the last successfully indexed note, not the failed third.
    assert meta.get_slite_sync_cursor("default") == "2026-04-02T12:00:00Z"


def test_slite_indexer_cursor_tie_safe_on_failure(tmp_dir, sparse_encoder):
    """Two notes share a lastEditedAt; if the second fails, the cursor must NOT
    advance to the shared timestamp (else the `<= cursor` filter drops it)."""
    from devrag.stores.qdrant_store import QdrantStore
    from devrag.stores.metadata_db import MetadataDB
    store = QdrantStore(path=str(tmp_dir / "qdrant"), embedding_dim=768)
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])

    mock_slite = MagicMock(spec=SliteClient)
    mock_slite.list_notes.return_value = iter([
        _make_note("p1", last_edited_at="2026-04-01T12:00:00Z"),
        _make_note("p2", last_edited_at="2026-04-01T12:00:00Z"),
    ])
    mock_slite.get_note.side_effect = [
        {"id": "p1", "content": "# One\n\nBody."},
        _429(),
    ]

    indexer = SliteIndexer(store, meta, embedder, sparse_encoder, mock_slite)
    stats = indexer.sync(since_days=90)

    assert stats.pages_indexed == 1
    # No cursor was committed — the tied second note is still pending.
    assert meta.get_slite_sync_cursor("default") is None


def test_slite_indexer_channel_filtering(tmp_dir, sparse_encoder):
    from devrag.stores.qdrant_store import QdrantStore
    from devrag.stores.metadata_db import MetadataDB
    store = QdrantStore(path=str(tmp_dir / "qdrant"), embedding_dim=768)
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])

    mock_slite = MagicMock(spec=SliteClient)
    mock_slite.list_notes.return_value = iter([])

    indexer = SliteIndexer(store, meta, embedder, sparse_encoder, mock_slite,
                           channel_ids=["ch-1", "ch-2"])
    indexer.sync(since_days=90)
    call_args = mock_slite.list_notes.call_args
    assert call_args[1]["channel_ids"] == ["ch-1", "ch-2"]
