from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from devrag.ingest.doc_indexer import CHARS_PER_TOKEN
from devrag.types import Chunk, SessionSyncStats

logger = logging.getLogger(__name__)

CURSOR_KEY = "default"
COLLECTION = "session_logs"

_SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
_LOCAL_CMD_RE = re.compile(r"<local-command-[^>]+>.*?</local-command-[^>]+>", re.DOTALL)


def _make_chunk_id(session_id: str, turn_idx: int) -> str:
    return hashlib.sha256(f"session:{session_id}:{turn_idx}".encode()).hexdigest()[:16]


def _strip_noise(text: str) -> str:
    text = _SYSTEM_REMINDER_RE.sub("", text)
    text = _LOCAL_CMD_RE.sub("", text)
    return text.strip()


def _user_prompt_text(message: dict) -> str | None:
    """Real user prompt text, or None for tool-result / empty continuations."""
    content = message.get("content")
    if isinstance(content, str):
        return _strip_noise(content) or None
    if isinstance(content, list):
        if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            return None
        parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        return _strip_noise("\n".join(parts)) or None
    return None


def _assistant_text(message: dict) -> str:
    """Assistant text from `text` blocks; short summary for `tool_use`; thinking dropped."""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for b in content:
        if not isinstance(b, dict):
            continue
        btype = b.get("type")
        if btype == "text":
            parts.append(b.get("text", ""))
        elif btype == "tool_use":
            name = b.get("name", "?")
            tool_input = b.get("input", {})
            if isinstance(tool_input, dict):
                hints = []
                for k, v in list(tool_input.items())[:3]:
                    if isinstance(v, str):
                        hints.append(f"{k}={v[:60]!r}")
                    else:
                        hints.append(f"{k}=...")
                summary = ", ".join(hints)
            else:
                summary = ""
            parts.append(f"[tool:{name}({summary})]")
    return "\n\n".join(p for p in parts if p).strip()


def chunk_session_file(path: Path, max_tokens: int = 512) -> list[Chunk]:
    """Parse a Claude Code JSONL session file into user/assistant exchange chunks."""
    max_chars = max_tokens * CHARS_PER_TOKEN
    project_dir = path.parent.name
    session_id = path.stem

    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Cannot read session %s: %s", path, exc)
        return []

    entries: list[dict] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    chunks: list[Chunk] = []
    turn_idx = 0
    pending_user: str | None = None
    pending_ts = ""
    pending_cwd = ""
    pending_branch = ""
    asst_parts: list[str] = []

    def flush() -> None:
        nonlocal turn_idx, pending_user, asst_parts
        if pending_user is None:
            return
        assistant = "\n\n".join(p for p in asst_parts if p).strip()
        if assistant:
            body = f"User: {pending_user}\n\nAssistant: {assistant}"
            if len(body) > max_chars:
                body = body[:max_chars] + "\n... (truncated)"
            chunks.append(Chunk(
                id=_make_chunk_id(session_id, turn_idx),
                text=body,
                metadata={
                    "session_id": session_id,
                    "project_dir": project_dir,
                    "turn_index": turn_idx,
                    "timestamp": pending_ts,
                    "cwd": pending_cwd,
                    "git_branch": pending_branch,
                    "chunk_type": "session_exchange",
                },
            ))
            turn_idx += 1
        pending_user = None
        asst_parts = []

    for entry in entries:
        etype = entry.get("type")
        if etype not in ("user", "assistant"):
            continue
        message = entry.get("message") or {}
        if etype == "user":
            prompt = _user_prompt_text(message)
            if prompt is None:
                continue
            flush()
            pending_user = prompt
            pending_ts = entry.get("timestamp", "")
            pending_cwd = entry.get("cwd", "")
            pending_branch = entry.get("gitBranch", "")
        else:  # assistant
            if pending_user is None:
                continue
            text = _assistant_text(message)
            if text:
                asst_parts.append(text)
    flush()
    return chunks


class SessionsIndexer:
    def __init__(
        self,
        vector_store,
        metadata_db,
        embedder,
        sparse_encoder,
        logs_dir: Path,
        chunk_max_tokens: int = 512,
        backfill_days: int = 60,
    ) -> None:
        self.vector_store = vector_store
        self.metadata_db = metadata_db
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder
        self.logs_dir = logs_dir
        self.chunk_max_tokens = chunk_max_tokens
        self.backfill_days = backfill_days

    def _discover_files(self, cutoff_mtime: float) -> list[Path]:
        if not self.logs_dir.exists():
            return []
        return [
            p for p in self.logs_dir.rglob("*.jsonl")
            if p.is_file() and p.stat().st_mtime > cutoff_mtime
        ]

    def sync(self, since_days: int | None = None) -> SessionSyncStats:
        stats = SessionSyncStats()
        cursor = self.metadata_db.get_session_sync_cursor(CURSOR_KEY)
        now_ts = datetime.now(timezone.utc).timestamp()
        if since_days is not None:
            cutoff = now_ts - since_days * 86400
        elif cursor:
            cutoff = float(cursor)
        else:
            cutoff = now_ts - self.backfill_days * 86400

        latest_mtime = cutoff
        files = self._discover_files(cutoff)
        stats.files_scanned = len(files)
        seen_sessions: set[str] = set()

        for path in files:
            mtime = path.stat().st_mtime
            if mtime > latest_mtime:
                latest_mtime = mtime

            chunks = chunk_session_file(path, max_tokens=self.chunk_max_tokens)
            if not chunks:
                stats.files_skipped += 1
                continue

            session_id = str(chunks[0].metadata["session_id"])
            seen_sessions.add(session_id)

            # Drop stale chunks from previous runs where turn_index may have shifted.
            old_ids = self.metadata_db.get_chunks_for_session(CURSOR_KEY, session_id)
            current_ids = {c.id for c in chunks}
            stale_ids = [cid for cid in old_ids if cid not in current_ids]
            if stale_ids:
                self.vector_store.delete(COLLECTION, stale_ids)
                self.metadata_db.delete_session_chunk_sources(stale_ids)

            texts = [c.text for c in chunks]
            embeddings = self.embedder.embed(texts)
            sparse_embeddings = self.sparse_encoder.encode(texts)
            self.vector_store.upsert(
                collection=COLLECTION,
                ids=[c.id for c in chunks],
                embeddings=embeddings,
                documents=texts,
                metadatas=[c.metadata for c in chunks],
                sparse_embeddings=sparse_embeddings,
            )
            for chunk in chunks:
                self.metadata_db.set_session_chunk_source(chunk.id, CURSOR_KEY, session_id)

            stats.files_indexed += 1
            stats.chunks_created += len(chunks)

        stats.sessions_indexed = len(seen_sessions)
        if stats.files_scanned:
            self.metadata_db.set_session_sync_cursor(CURSOR_KEY, str(latest_mtime))
        return stats
