import json
from unittest.mock import MagicMock

from devrag.ingest.session_indexer import (
    SessionsIndexer,
    _assistant_text,
    _make_chunk_id,
    _user_prompt_text,
    chunk_session_file,
)


def _write_jsonl(path, entries):
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _user(text, ts="2026-04-10T12:00:00Z", cwd="/proj", branch="main", session="sess-a"):
    return {
        "type": "user",
        "timestamp": ts,
        "cwd": cwd,
        "gitBranch": branch,
        "sessionId": session,
        "message": {"role": "user", "content": text},
    }


def _assistant_text_turn(text, ts="2026-04-10T12:00:01Z", cwd="/proj", branch="main", session="sess-a"):
    return {
        "type": "assistant",
        "timestamp": ts,
        "cwd": cwd,
        "gitBranch": branch,
        "sessionId": session,
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


def _assistant_tool(name, inp, ts="2026-04-10T12:00:02Z", session="sess-a"):
    return {
        "type": "assistant",
        "timestamp": ts,
        "sessionId": session,
        "message": {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "x", "name": name, "input": inp}],
        },
    }


def _tool_result_user(result, ts="2026-04-10T12:00:03Z", session="sess-a"):
    return {
        "type": "user",
        "timestamp": ts,
        "sessionId": session,
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "x", "content": result}],
        },
    }


# --- Extraction helpers ---

def test_user_prompt_text_string():
    msg = {"content": "Hello Claude"}
    assert _user_prompt_text(msg) == "Hello Claude"


def test_user_prompt_text_strips_system_reminder():
    msg = {"content": "Real question<system-reminder>ignore me</system-reminder>"}
    assert _user_prompt_text(msg) == "Real question"


def test_user_prompt_text_none_for_tool_result():
    msg = {"content": [{"type": "tool_result", "tool_use_id": "x", "content": "ok"}]}
    assert _user_prompt_text(msg) is None


def test_user_prompt_text_text_blocks():
    msg = {"content": [{"type": "text", "text": "slash command body"}]}
    assert _user_prompt_text(msg) == "slash command body"


def test_assistant_text_drops_thinking_keeps_text_summarizes_tool_use():
    msg = {
        "content": [
            {"type": "thinking", "thinking": "long private reasoning", "signature": "..."},
            {"type": "text", "text": "Here is my reply."},
            {"type": "tool_use", "name": "Read", "input": {"file_path": "/a/b.py"}},
        ]
    }
    out = _assistant_text(msg)
    assert "long private reasoning" not in out
    assert "Here is my reply." in out
    assert "[tool:Read(" in out
    assert "file_path=" in out


# --- Chunking ---

def test_chunk_session_file_pairs_user_and_assistant(tmp_dir, sparse_encoder):
    path = tmp_dir / "sess-a.jsonl"
    _write_jsonl(path, [
        _user("How do I run tests?"),
        _assistant_text_turn("Run pytest."),
        _user("And with coverage?"),
        _assistant_text_turn("Use pytest --cov."),
    ])
    chunks = chunk_session_file(path)
    assert len(chunks) == 2
    assert "How do I run tests?" in chunks[0].text
    assert "Run pytest." in chunks[0].text
    assert "And with coverage?" in chunks[1].text
    assert chunks[0].metadata["chunk_type"] == "session_exchange"
    assert chunks[0].metadata["session_id"] == "sess-a"
    assert chunks[0].metadata["turn_index"] == 0
    assert chunks[1].metadata["turn_index"] == 1


def test_chunk_session_file_spans_tool_calls_in_one_exchange(tmp_dir, sparse_encoder):
    path = tmp_dir / "sess-b.jsonl"
    _write_jsonl(path, [
        _user("Read foo.py", session="sess-b"),
        _assistant_tool("Read", {"file_path": "/foo.py"}, session="sess-b"),
        _tool_result_user("contents", session="sess-b"),
        _assistant_text_turn("Here's what's in the file.", session="sess-b"),
    ])
    chunks = chunk_session_file(path)
    assert len(chunks) == 1
    body = chunks[0].text
    assert "Read foo.py" in body
    assert "[tool:Read(" in body
    assert "Here's what's in the file." in body


def test_chunk_session_file_skips_non_user_assistant_entries(tmp_dir, sparse_encoder):
    path = tmp_dir / "sess-c.jsonl"
    _write_jsonl(path, [
        {"type": "file-history-snapshot", "snapshot": {}},
        {"type": "custom-title", "customTitle": "x", "sessionId": "sess-c"},
        _user("hi", session="sess-c"),
        _assistant_text_turn("hello", session="sess-c"),
        {"type": "attachment", "attachment": {}},
    ])
    chunks = chunk_session_file(path)
    assert len(chunks) == 1


def test_chunk_session_file_deterministic_ids(tmp_dir, sparse_encoder):
    path = tmp_dir / "sess-d.jsonl"
    _write_jsonl(path, [
        _user("q", session="sess-d"),
        _assistant_text_turn("a", session="sess-d"),
    ])
    c1 = chunk_session_file(path)
    c2 = chunk_session_file(path)
    assert [c.id for c in c1] == [c.id for c in c2]


def test_make_chunk_id_varies_by_session_and_turn():
    a = _make_chunk_id("s1", 0)
    b = _make_chunk_id("s1", 1)
    c = _make_chunk_id("s2", 0)
    assert len({a, b, c}) == 3
    assert len(a) == 16


def test_chunk_session_file_truncates_long_body(tmp_dir, sparse_encoder):
    path = tmp_dir / "long.jsonl"
    _write_jsonl(path, [
        _user("q", session="long"),
        _assistant_text_turn("x " * 2000, session="long"),
    ])
    chunks = chunk_session_file(path, max_tokens=64)
    assert len(chunks) == 1
    assert chunks[0].text.endswith("... (truncated)")


# --- Indexer end-to-end ---

def test_sessions_indexer_indexes_and_sets_cursor(tmp_dir, sparse_encoder):
    from devrag.stores.qdrant_store import QdrantStore
    from devrag.stores.metadata_db import MetadataDB

    logs_dir = tmp_dir / "projects" / "-proj"
    logs_dir.mkdir(parents=True)
    _write_jsonl(logs_dir / "aaa.jsonl", [
        _user("q1", session="aaa"),
        _assistant_text_turn("a1", session="aaa"),
    ])

    store = QdrantStore(path=str(tmp_dir / "qdrant"), embedding_dim=768)
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])

    indexer = SessionsIndexer(store, meta, embedder, sparse_encoder, tmp_dir / "projects", chunk_max_tokens=512)
    stats = indexer.sync(since_days=30)

    assert stats.files_scanned == 1
    assert stats.files_indexed == 1
    assert stats.sessions_indexed == 1
    assert stats.chunks_created == 1
    assert store.count("session_logs") == 1
    assert meta.get_session_sync_cursor("default") is not None


def test_sessions_indexer_incremental_skip(tmp_dir, sparse_encoder):
    from devrag.stores.qdrant_store import QdrantStore
    from devrag.stores.metadata_db import MetadataDB

    logs_dir = tmp_dir / "projects" / "-proj"
    logs_dir.mkdir(parents=True)
    path = logs_dir / "aaa.jsonl"
    _write_jsonl(path, [
        _user("q1", session="aaa"),
        _assistant_text_turn("a1", session="aaa"),
    ])

    store = QdrantStore(path=str(tmp_dir / "qdrant"), embedding_dim=768)
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])
    indexer = SessionsIndexer(store, meta, embedder, sparse_encoder, tmp_dir / "projects")

    indexer.sync(since_days=30)
    stats2 = indexer.sync()  # incremental from cursor
    assert stats2.files_scanned == 0
    assert stats2.files_indexed == 0
