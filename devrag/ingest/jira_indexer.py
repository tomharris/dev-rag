from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timedelta, timezone

from devrag.types import Chunk, JiraSyncStats
from devrag.utils.jira_client import JiraClient

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4

_SEARCH_FIELDS = [
    "summary", "description", "status", "issuetype",
    "reporter", "assignee", "priority", "labels",
    "comment", "created", "updated",
]


def _make_jira_chunk_id(instance: str, key: str, chunk_type: str, index: int) -> str:
    raw = f"{instance}:{key}:{chunk_type}:{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _make_cursor_key(instance_url: str, jql: str) -> str:
    raw = f"{instance_url}:{jql}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _jira_base_metadata(issue: dict, instance_url: str) -> dict:
    fields = issue.get("fields", {})
    labels = ",".join(fields.get("labels", []))
    return {
        "jira_instance": instance_url,
        "ticket_key": issue["key"],
        "ticket_summary": fields.get("summary", ""),
        "ticket_status": (fields.get("status") or {}).get("name", ""),
        "ticket_type": (fields.get("issuetype") or {}).get("name", ""),
        "reporter": (fields.get("reporter") or {}).get("displayName", ""),
        "assignee": (fields.get("assignee") or {}).get("displayName", ""),
        "priority": (fields.get("priority") or {}).get("name", ""),
        "labels": labels,
        "created_at": fields.get("created", ""),
        "updated_at": fields.get("updated", ""),
    }


def _truncate_text(text: str, max_tokens: int) -> str:
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) > max_chars:
        logger.debug("Truncating Jira chunk from %d to %d chars", len(text), max_chars)
        return text[:max_chars] + "\n# ... (truncated)"
    return text


def _inject_cursor_into_jql(jql: str, cursor_date: str) -> str:
    """Append 'AND updated >= ...' to JQL, inserting before ORDER BY if present."""
    cursor_clause = f' AND updated >= "{cursor_date}"'
    order_match = re.search(r"\bORDER\s+BY\b", jql, re.IGNORECASE)
    if order_match:
        return jql[:order_match.start()] + cursor_clause + " " + jql[order_match.start():]
    return jql + cursor_clause


def chunk_jira_ticket(issue: dict, instance_url: str, max_tokens: int = 512) -> list[Chunk]:
    chunks: list[Chunk] = []
    base_meta = _jira_base_metadata(issue, instance_url)
    fields = issue.get("fields", {})
    key = issue["key"]

    # Description chunk
    summary = fields.get("summary") or ""
    description_adf = fields.get("description")
    description_text = JiraClient.adf_to_text(description_adf)
    desc_text = f"[{instance_url} {key}] {summary}\n\n{description_text}".strip()
    if not desc_text:
        desc_text = f"Jira ticket {key}"
    desc_text = _truncate_text(desc_text, max_tokens)
    chunks.append(Chunk(
        id=_make_jira_chunk_id(instance_url, key, "description", 0),
        text=desc_text,
        metadata={**base_meta, "chunk_type": "description"},
    ))

    # Comment chunks
    comment_data = fields.get("comment", {})
    comments = comment_data.get("comments", []) if isinstance(comment_data, dict) else []
    for i, comment in enumerate(comments):
        author = (comment.get("author") or {}).get("displayName", "unknown")
        body_adf = comment.get("body")
        body_text = JiraClient.adf_to_text(body_adf)
        comment_text = f"Comment by {author} on {key}:\n\n{body_text}"
        comment_text = _truncate_text(comment_text, max_tokens)
        chunks.append(Chunk(
            id=_make_jira_chunk_id(instance_url, key, "comment", i),
            text=comment_text,
            metadata={**base_meta, "chunk_type": "comment",
                      "comment_author": author,
                      "comment_created_at": comment.get("created", "")},
        ))

    return chunks


class JiraIndexer:
    def __init__(self, vector_store, metadata_db, embedder, jira_client: JiraClient,
                 chunk_max_tokens: int = 512) -> None:
        self.vector_store = vector_store
        self.metadata_db = metadata_db
        self.embedder = embedder
        self.jira = jira_client
        self.chunk_max_tokens = chunk_max_tokens

    def sync(self, instance_url: str, jql: str, since_days: int = 90) -> JiraSyncStats:
        stats = JiraSyncStats()
        cursor_key = _make_cursor_key(instance_url, jql)
        cursor = self.metadata_db.get_jira_sync_cursor(cursor_key)

        if cursor:
            cursor_date = cursor
        else:
            since_dt = datetime.now(timezone.utc) - timedelta(days=since_days)
            cursor_date = since_dt.strftime("%Y/%m/%d %H:%M")

        effective_jql = _inject_cursor_into_jql(jql, cursor_date)
        latest_updated = cursor_date

        for issue in self.jira.search_issues(effective_jql, fields=_SEARCH_FIELDS):
            stats.tickets_fetched += 1
            key = issue["key"]
            updated_at = issue.get("fields", {}).get("updated", "")

            self.metadata_db.delete_chunks_for_jira_ticket(instance_url, key)
            chunks = chunk_jira_ticket(issue, instance_url, max_tokens=self.chunk_max_tokens)
            if not chunks:
                stats.tickets_indexed += 1
                continue

            texts = [c.text for c in chunks]
            embeddings = self.embedder.embed(texts)
            embed_map = {c.id: embeddings[i] for i, c in enumerate(chunks)}

            desc_chunks = [c for c in chunks if c.metadata["chunk_type"] == "description"]
            discussion_chunks = [c for c in chunks if c.metadata["chunk_type"] == "comment"]

            if desc_chunks:
                self.vector_store.upsert(
                    collection="jira_descriptions", ids=[c.id for c in desc_chunks],
                    embeddings=[embed_map[c.id] for c in desc_chunks],
                    documents=[c.text for c in desc_chunks],
                    metadatas=[c.metadata for c in desc_chunks],
                )
            if discussion_chunks:
                self.vector_store.upsert(
                    collection="jira_discussions", ids=[c.id for c in discussion_chunks],
                    embeddings=[embed_map[c.id] for c in discussion_chunks],
                    documents=[c.text for c in discussion_chunks],
                    metadatas=[c.metadata for c in discussion_chunks],
                )

            for chunk in chunks:
                self.metadata_db.set_jira_chunk_source(chunk.id, instance_url, key)
                self.metadata_db.upsert_fts(chunk.id, chunk.text)

            # Track latest updated timestamp for cursor
            if updated_at and updated_at > latest_updated:
                latest_updated = updated_at

            stats.tickets_indexed += 1
            stats.chunks_created += len(chunks)

        # Store cursor as Jira datetime format for next sync
        self.metadata_db.set_jira_sync_cursor(cursor_key, latest_updated)
        return stats
