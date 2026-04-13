from unittest.mock import MagicMock

from devrag.ingest.slite_indexer import (
    SliteIndexer,
    _cursor_to_days_ago,
    _make_chunk_id,
    _truncate_text,
    chunk_slite_page,
)
from devrag.utils.slite_client import SliteClient


def _make_note(note_id="page-1", title="Getting Started", content=None,
               url="https://app.slite.com/p/page-1",
               updated_at="2026-04-01T12:00:00Z"):
    if content is None:
        content = "# Introduction\n\nWelcome to the team.\n\n## Setup\n\nRun `make install` to get started."
    return {
        "id": note_id,
        "title": title,
        "url": url,
        "updatedAt": updated_at,
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


# --- Cursor conversion tests ---

def test_cursor_to_days_ago_recent():
    from datetime import datetime, timezone, timedelta
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    result = _cursor_to_days_ago(yesterday)
    assert result >= 1
    assert result <= 3  # 1 day + 1 buffer


def test_cursor_to_days_ago_old():
    result = _cursor_to_days_ago("2026-01-01T00:00:00Z")
    assert result > 90  # More than 90 days ago from April 2026


def test_cursor_to_days_ago_minimum_is_one():
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    result = _cursor_to_days_ago(now)
    assert result >= 1


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

def test_slite_indexer_sync(tmp_dir):
    from devrag.stores.chroma_store import ChromaStore
    from devrag.stores.metadata_db import MetadataDB
    store = ChromaStore(persist_dir=str(tmp_dir / "chroma"))
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])

    mock_slite = MagicMock(spec=SliteClient)
    mock_slite.list_notes.return_value = iter([
        {"id": "page-1", "title": "Onboarding", "url": "https://app.slite.com/p/page-1",
         "updatedAt": "2026-04-01T12:00:00Z"},
    ])
    mock_slite.get_note.return_value = {
        "id": "page-1", "title": "Onboarding",
        "url": "https://app.slite.com/p/page-1",
        "content": "# Onboarding\n\nWelcome aboard.\n\n## Day 1\n\nSet up your laptop.",
    }

    indexer = SliteIndexer(store, meta, embedder, mock_slite)
    stats = indexer.sync(since_days=90)
    assert stats.pages_fetched == 1
    assert stats.pages_indexed == 1
    assert stats.chunks_created >= 2
    assert store.count("slite_pages") >= 2
    assert meta.get_slite_sync_cursor("default") == "2026-04-01T12:00:00Z"


def test_slite_indexer_skips_empty_pages(tmp_dir):
    from devrag.stores.chroma_store import ChromaStore
    from devrag.stores.metadata_db import MetadataDB
    store = ChromaStore(persist_dir=str(tmp_dir / "chroma"))
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])

    mock_slite = MagicMock(spec=SliteClient)
    mock_slite.list_notes.return_value = iter([
        {"id": "page-empty", "title": "Empty", "url": "...", "updatedAt": "2026-04-01T12:00:00Z"},
    ])
    mock_slite.get_note.return_value = {"id": "page-empty", "title": "Empty", "content": ""}

    indexer = SliteIndexer(store, meta, embedder, mock_slite)
    stats = indexer.sync(since_days=90)
    assert stats.pages_fetched == 1
    assert stats.pages_skipped == 1
    assert stats.pages_indexed == 0


def test_slite_indexer_incremental_sync(tmp_dir):
    from devrag.stores.chroma_store import ChromaStore
    from devrag.stores.metadata_db import MetadataDB
    store = ChromaStore(persist_dir=str(tmp_dir / "chroma"))
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])

    mock_slite = MagicMock(spec=SliteClient)
    mock_slite.list_notes.return_value = iter([
        {"id": "page-1", "title": "Guide", "url": "...", "updatedAt": "2026-04-01T12:00:00Z"},
    ])
    mock_slite.get_note.return_value = {
        "id": "page-1", "title": "Guide", "content": "# Guide\n\nSome content.",
    }

    indexer = SliteIndexer(store, meta, embedder, mock_slite)

    # First sync
    indexer.sync(since_days=90)
    assert meta.get_slite_sync_cursor("default") is not None

    # Second sync — should use cursor
    mock_slite.list_notes.return_value = iter([])
    indexer.sync(since_days=90)
    call_args = mock_slite.list_notes.call_args
    assert call_args[1].get("since_days_ago") is not None
    # The since_days_ago should be derived from cursor, not the default 90


def test_slite_indexer_channel_filtering(tmp_dir):
    from devrag.stores.chroma_store import ChromaStore
    from devrag.stores.metadata_db import MetadataDB
    store = ChromaStore(persist_dir=str(tmp_dir / "chroma"))
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])

    mock_slite = MagicMock(spec=SliteClient)
    mock_slite.list_notes.return_value = iter([])

    indexer = SliteIndexer(store, meta, embedder, mock_slite,
                           channel_ids=["ch-1", "ch-2"])
    indexer.sync(since_days=90)
    call_args = mock_slite.list_notes.call_args
    assert call_args[1]["channel_ids"] == ["ch-1", "ch-2"]
