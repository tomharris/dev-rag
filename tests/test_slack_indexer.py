from unittest.mock import MagicMock

from devrag.ingest.slack_indexer import (
    SlackIndexer,
    _group_windows,
    _make_chunk_id,
    _resolve_mentions,
    chunk_slack_channel,
)
from devrag.utils.slack_client import SlackClient

USER_MAP = {"U1": "Alice", "U2": "Bob", "U3": "Carol"}


# --- Helper tests ---

def test_make_chunk_id_deterministic_and_varies():
    a = _make_chunk_id("C1", "100.0")
    assert a == _make_chunk_id("C1", "100.0")
    assert len(a) == 16
    assert a != _make_chunk_id("C1", "200.0")
    assert a != _make_chunk_id("C2", "100.0")


def test_resolve_mentions():
    text = "hey <@U1> can you review <@U2>'s PR"
    assert _resolve_mentions(text, USER_MAP) == "hey @Alice can you review @Bob's PR"
    # Unknown user IDs fall back to the raw id
    assert _resolve_mentions("ping <@U9>", USER_MAP) == "ping @U9"


def test_group_windows_splits_on_gap():
    msgs = [
        {"ts": "1000.0"}, {"ts": "1060.0"},  # 60s apart — same window
        {"ts": "5000.0"},                    # ~65 min later — new window
    ]
    windows = _group_windows(msgs, gap_seconds=1800)
    assert len(windows) == 2
    assert [m["ts"] for m in windows[0]] == ["1000.0", "1060.0"]
    assert [m["ts"] for m in windows[1]] == ["5000.0"]


def test_group_windows_empty():
    assert _group_windows([], gap_seconds=1800) == []


# --- Chunking tests ---

def _channel(cid="C1", name="general"):
    return {"id": cid, "name": name, "is_member": True}


def test_chunk_thread_root_becomes_thread_chunk():
    messages = [
        {"ts": "100.0", "text": "How do we deploy?", "user": "U1",
         "thread_ts": "100.0", "reply_count": 1},
    ]
    replies_map = {"100.0": [
        {"ts": "100.0", "text": "How do we deploy?", "user": "U1", "thread_ts": "100.0"},
        {"ts": "150.0", "text": "Use the deploy script", "user": "U2", "thread_ts": "100.0"},
    ]}
    chunks = chunk_slack_channel(_channel(), messages, replies_map, USER_MAP)
    thread_chunks = [c for c in chunks if c.metadata["chunk_type"] == "slack_thread"]
    assert len(thread_chunks) == 1
    c = thread_chunks[0]
    assert "@Alice" in c.text and "@Bob" in c.text
    assert "deploy script" in c.text
    assert c.metadata["channel_id"] == "C1"
    assert c.metadata["channel_name"] == "general"
    assert c.metadata["thread_ts"] == "100.0"


def test_chunk_standalone_messages_become_window():
    messages = [
        {"ts": "100.0", "text": "morning", "user": "U1"},
        {"ts": "130.0", "text": "morning all", "user": "U2"},
    ]
    chunks = chunk_slack_channel(_channel(), messages, {}, USER_MAP)
    window_chunks = [c for c in chunks if c.metadata["chunk_type"] == "slack_window"]
    assert len(window_chunks) == 1
    assert "@Alice" in window_chunks[0].text
    assert "@Bob" in window_chunks[0].text


def test_chunk_mixes_threads_and_windows():
    messages = [
        {"ts": "100.0", "text": "chit chat", "user": "U1"},
        {"ts": "200.0", "text": "real question", "user": "U2",
         "thread_ts": "200.0", "reply_count": 1},
    ]
    replies_map = {"200.0": [
        {"ts": "200.0", "text": "real question", "user": "U2", "thread_ts": "200.0"},
        {"ts": "250.0", "text": "answer", "user": "U3", "thread_ts": "200.0"},
    ]}
    chunks = chunk_slack_channel(_channel(), messages, replies_map, USER_MAP)
    types = sorted(c.metadata["chunk_type"] for c in chunks)
    assert types == ["slack_thread", "slack_window"]


def test_chunk_ids_deterministic():
    messages = [{"ts": "100.0", "text": "hi", "user": "U1"}]
    c1 = chunk_slack_channel(_channel(), messages, {}, USER_MAP)
    c2 = chunk_slack_channel(_channel(), messages, {}, USER_MAP)
    assert [c.id for c in c1] == [c.id for c in c2]


# --- Indexer sync tests ---

def _mock_client(channels, history, replies=None, members=None):
    client = MagicMock(spec=SlackClient)
    client.list_conversations.return_value = iter(channels)
    client.conversations_history.return_value = iter(history)
    client.conversations_replies.return_value = replies or []
    client.users_list.return_value = iter(members or [
        {"id": "U1", "name": "alice", "profile": {"display_name": "Alice"}},
        {"id": "U2", "name": "bob", "profile": {"display_name": "Bob"}},
    ])
    return client


def test_slack_indexer_sync(tmp_dir, sparse_encoder):
    from devrag.stores.qdrant_store import QdrantStore
    from devrag.stores.metadata_db import MetadataDB
    store = QdrantStore(path=str(tmp_dir / "qdrant"), embedding_dim=768)
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])

    client = _mock_client(
        channels=[_channel()],
        history=[{"ts": "100.0", "text": "hello team", "user": "U1"}],
    )
    indexer = SlackIndexer(store, meta, embedder, sparse_encoder, client)
    stats = indexer.sync(since_days=90)
    assert stats.channels_scanned == 1
    assert stats.chunks_created >= 1
    assert store.count("slack_messages") >= 1
    assert meta.get_slack_sync_cursor("C1") == "100.0"


def test_slack_indexer_incremental_skips_old(tmp_dir, sparse_encoder):
    from devrag.stores.qdrant_store import QdrantStore
    from devrag.stores.metadata_db import MetadataDB
    store = QdrantStore(path=str(tmp_dir / "qdrant"), embedding_dim=768)
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])

    client = _mock_client(
        channels=[_channel()],
        history=[{"ts": "100.0", "text": "hello", "user": "U1"}],
    )
    indexer = SlackIndexer(store, meta, embedder, sparse_encoder, client)
    indexer.sync(since_days=90)
    assert meta.get_slack_sync_cursor("C1") == "100.0"

    # Second sync: history call should pass oldest=cursor
    client.list_conversations.return_value = iter([_channel()])
    client.conversations_history.return_value = iter([])
    indexer.sync(since_days=90)
    call = client.conversations_history.call_args
    assert call.kwargs.get("oldest") == "100.0" or call[1].get("oldest") == "100.0"


def test_slack_indexer_allowlist(tmp_dir, sparse_encoder):
    from devrag.stores.qdrant_store import QdrantStore
    from devrag.stores.metadata_db import MetadataDB
    store = QdrantStore(path=str(tmp_dir / "qdrant"), embedding_dim=768)
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])

    client = _mock_client(
        channels=[_channel("C1", "general"), _channel("C2", "random")],
        history=[{"ts": "100.0", "text": "hi", "user": "U1"}],
    )
    indexer = SlackIndexer(store, meta, embedder, sparse_encoder, client, channel_ids=["C2"])
    indexer.sync(since_days=90)
    # Only the allowlisted channel should have been fetched
    fetched = [c.kwargs.get("channel") or c.args[0] for c in client.conversations_history.call_args_list]
    assert fetched == ["C2"]
    assert meta.get_slack_sync_cursor("C2") == "100.0"
    assert meta.get_slack_sync_cursor("C1") is None


def test_slack_indexer_idempotent(tmp_dir, sparse_encoder):
    from devrag.stores.qdrant_store import QdrantStore
    from devrag.stores.metadata_db import MetadataDB
    store = QdrantStore(path=str(tmp_dir / "qdrant"), embedding_dim=768)
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])

    history = [{"ts": "100.0", "text": "hello", "user": "U1"}]
    client = _mock_client(channels=[_channel()], history=history)
    indexer = SlackIndexer(store, meta, embedder, sparse_encoder, client)
    indexer.sync(since_days=90)
    count_after_first = store.count("slack_messages")

    # Re-sync the SAME message (force re-process by clearing cursor) — must not duplicate
    meta.set_slack_sync_cursor("C1", "0")
    client.list_conversations.return_value = iter([_channel()])
    client.conversations_history.return_value = iter(history)
    indexer.sync(since_days=90)
    assert store.count("slack_messages") == count_after_first
