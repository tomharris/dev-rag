from __future__ import annotations

import hashlib
import logging
import re

from devrag.ingest.doc_indexer import CHARS_PER_TOKEN
from devrag.types import Chunk, SlackSyncStats
from devrag.utils.slack_client import SlackAuthError, SlackClient

logger = logging.getLogger(__name__)

COLLECTION = "slack_messages"

_MENTION_RE = re.compile(r"<@(U[A-Z0-9]+)>")
# Subtypes that carry no conversational value (joins/leaves/topic changes, etc.).
_NOISE_SUBTYPES = {
    "channel_join", "channel_leave", "channel_topic", "channel_purpose",
    "channel_name", "channel_archive", "channel_unarchive", "bot_add", "bot_remove",
}


def _make_chunk_id(channel_id: str, key: str) -> str:
    """Deterministic 16-char id from channel + thread/window key (idempotent re-sync)."""
    return hashlib.sha256(f"slack:{channel_id}:{key}".encode()).hexdigest()[:16]


def _resolve_mentions(text: str, user_map: dict[str, str]) -> str:
    """Replace ``<@U123>`` mention tokens with ``@DisplayName`` for readable embeddings."""
    return _MENTION_RE.sub(lambda m: "@" + user_map.get(m.group(1), m.group(1)), text)


def _display_name(member: dict) -> str:
    profile = member.get("profile", {}) or {}
    return profile.get("display_name") or member.get("real_name") or member.get("name") or member["id"]


def _message_line(msg: dict, user_map: dict[str, str]) -> str | None:
    """Render one message as ``@author: text``, or None if it carries no content."""
    if msg.get("subtype") in _NOISE_SUBTYPES:
        return None
    text = (msg.get("text") or "").strip()
    if not text:
        return None
    author = user_map.get(msg.get("user", ""), msg.get("user", "unknown"))
    return f"@{author}: {_resolve_mentions(text, user_map)}"


def _truncate(text: str, max_tokens: int) -> str:
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) > max_chars:
        return text[:max_chars] + "\n... (truncated)"
    return text


def _participants(messages: list[dict], user_map: dict[str, str]) -> list[str]:
    seen: list[str] = []
    for m in messages:
        name = user_map.get(m.get("user", ""), m.get("user", ""))
        if name and name not in seen:
            seen.append(name)
    return seen


def _group_windows(
    messages: list[dict], gap_seconds: int, max_messages: int = 50
) -> list[list[dict]]:
    """Group time-ordered non-threaded messages into conversation windows.

    A window is a run of messages with no large quiet gap between them. Start a
    new window whenever the time since the previous message exceeds
    ``gap_seconds``. ``messages`` arrives sorted oldest-first; each carries a
    Slack ``ts`` (a unix-epoch float encoded as a string, e.g. ``"1700000000.000100"``).

    ``max_messages`` also caps a window's size so a continuously-active channel
    (no quiet gaps) doesn't collapse into one enormous chunk that later gets
    truncated — splitting on count keeps each window embeddable in full.

    Returns a list of windows, each a non-empty list of message dicts in order.
    """
    windows: list[list[dict]] = []
    current: list[dict] = []
    prev_ts: float | None = None

    for msg in messages:
        ts = float(msg.get("ts", "0"))
        gap_break = prev_ts is not None and (ts - prev_ts) > gap_seconds
        size_break = len(current) >= max_messages
        if current and (gap_break or size_break):
            windows.append(current)
            current = []
        current.append(msg)
        prev_ts = ts

    if current:
        windows.append(current)
    return windows


def chunk_slack_channel(
    channel: dict,
    messages: list[dict],
    replies_map: dict[str, list[dict]],
    user_map: dict[str, str],
    gap_minutes: int = 30,
    max_tokens: int = 512,
) -> list[Chunk]:
    """Turn one channel's history into hybrid thread + time-window chunks.

    ``messages`` is the channel's top-level history; thread replies live in
    ``replies_map`` keyed by the root ts. Thread roots (``reply_count > 0``)
    render as one ``slack_thread`` chunk; the remaining standalone messages are
    grouped by time gap into ``slack_window`` chunks.
    """
    channel_id = channel["id"]
    channel_name = channel.get("name", channel_id)
    ordered = sorted(messages, key=lambda m: float(m.get("ts", "0")))

    chunks: list[Chunk] = []
    standalone: list[dict] = []

    for msg in ordered:
        ts = msg.get("ts", "")
        if msg.get("reply_count", 0) > 0:
            thread_msgs = replies_map.get(ts) or [msg]
            lines = [ln for m in thread_msgs if (ln := _message_line(m, user_map))]
            if not lines:
                continue
            body = f"[#{channel_name}] thread:\n" + "\n".join(lines)
            chunks.append(Chunk(
                id=_make_chunk_id(channel_id, ts),
                text=_truncate(body, max_tokens),
                metadata={
                    "channel_id": channel_id,
                    "channel_name": channel_name,
                    "chunk_type": "slack_thread",
                    "thread_ts": ts,
                    "ts": ts,
                    "participants": _participants(thread_msgs, user_map),
                    "permalink_path": f"archives/{channel_id}/p{ts.replace('.', '')}",
                },
            ))
        else:
            standalone.append(msg)

    for window in _group_windows(standalone, gap_seconds=gap_minutes * 60):
        lines = [ln for m in window if (ln := _message_line(m, user_map))]
        if not lines:
            continue
        start_ts = window[0].get("ts", "")
        body = f"[#{channel_name}] conversation:\n" + "\n".join(lines)
        chunks.append(Chunk(
            id=_make_chunk_id(channel_id, start_ts),
            text=_truncate(body, max_tokens),
            metadata={
                "channel_id": channel_id,
                "channel_name": channel_name,
                "chunk_type": "slack_window",
                "ts": start_ts,
                "participants": _participants(window, user_map),
                "permalink_path": f"archives/{channel_id}/p{start_ts.replace('.', '')}",
            },
        ))

    return chunks


class SlackIndexer:
    def __init__(
        self,
        vector_store,
        metadata_db,
        embedder,
        sparse_encoder,
        slack_client: SlackClient,
        chunk_max_tokens: int = 512,
        chunk_overlap_tokens: int = 50,
        channel_ids: list[str] | None = None,
        gap_minutes: int = 30,
    ) -> None:
        self.vector_store = vector_store
        self.metadata_db = metadata_db
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder
        self.slack = slack_client
        self.chunk_max_tokens = chunk_max_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.channel_ids = channel_ids or []
        self.gap_minutes = gap_minutes

    def _build_user_map(self) -> dict[str, str]:
        return {m["id"]: _display_name(m) for m in self.slack.users_list()}

    def _select_channels(self) -> list[dict]:
        allow = set(self.channel_ids)
        selected = []
        for ch in self.slack.list_conversations():
            if allow:
                if ch.get("id") in allow:
                    selected.append(ch)
            elif ch.get("is_member", False):
                selected.append(ch)
        return selected

    def sync(self, since_days: int = 90) -> SlackSyncStats:
        stats = SlackSyncStats()
        user_map = self._build_user_map()
        channels = self._select_channels()

        for channel in channels:
            channel_id = channel["id"]
            stats.channels_scanned += 1
            cursor = self.metadata_db.get_slack_sync_cursor(channel_id)

            try:
                history = list(self.slack.conversations_history(channel_id, oldest=cursor))
            except SlackAuthError:
                raise
            except Exception as exc:  # noqa: BLE001 - skip a bad channel, keep going
                logger.warning("Slack history failed for %s: %s", channel_id, exc)
                stats.channels_errored += 1
                continue

            if not history:
                stats.channels_skipped += 1
                continue

            stats.messages_fetched += len(history)

            replies_map: dict[str, list[dict]] = {}
            for msg in history:
                if msg.get("reply_count", 0) > 0:
                    ts = msg.get("ts", "")
                    replies_map[ts] = self.slack.conversations_replies(channel_id, ts)

            chunks = chunk_slack_channel(
                channel, history, replies_map, user_map,
                gap_minutes=self.gap_minutes, max_tokens=self.chunk_max_tokens,
            )
            if not chunks:
                stats.channels_skipped += 1
                continue

            # Idempotent reconciliation: drop chunks from prior runs whose keys no
            # longer exist (e.g. a window whose boundary shifted), then upsert.
            old_ids = self.metadata_db.get_chunks_for_slack_channel(channel_id)
            current_ids = {c.id for c in chunks}
            stale_ids = [cid for cid in old_ids if cid not in current_ids]
            if stale_ids:
                self.vector_store.delete(COLLECTION, stale_ids)
                self.metadata_db.delete_slack_chunk_sources(stale_ids)

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
                wait=False,
            )
            for chunk in chunks:
                self.metadata_db.set_slack_chunk_source(chunk.id, channel_id)

            stats.threads_indexed += sum(1 for c in chunks if c.metadata["chunk_type"] == "slack_thread")
            stats.windows_indexed += sum(1 for c in chunks if c.metadata["chunk_type"] == "slack_window")
            stats.chunks_created += len(chunks)

            newest_ts = max(history, key=lambda m: float(m.get("ts", "0"))).get("ts", "")
            if newest_ts:
                self.metadata_db.set_slack_sync_cursor(channel_id, newest_ts)

        return stats
