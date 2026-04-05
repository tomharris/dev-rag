from __future__ import annotations
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from devrag.types import Chunk, IssueSyncStats
from devrag.utils.github import GitHubClient

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4


def _make_issue_chunk_id(repo: str, issue_number: int, chunk_type: str, index: int) -> str:
    raw = f"{repo}:issue{issue_number}:{chunk_type}:{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _issue_base_metadata(issue: dict, repo: str) -> dict:
    labels = ",".join(l["name"] for l in issue.get("labels", []))
    return {
        "repo": repo,
        "issue_number": issue["number"],
        "issue_title": issue["title"],
        "issue_state": issue["state"],
        "issue_author": issue["user"]["login"],
        "issue_labels": labels,
        "created_at": issue.get("created_at", ""),
        "updated_at": issue.get("updated_at", ""),
    }


def _truncate_text(text: str, max_tokens: int) -> str:
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) > max_chars:
        logger.debug("Truncating issue chunk from %d to %d chars", len(text), max_chars)
        return text[:max_chars] + "\n# ... (truncated)"
    return text


def chunk_issue(issue: dict, comments: list[dict], repo: str, max_tokens: int = 512) -> list[Chunk]:
    chunks: list[Chunk] = []
    base_meta = _issue_base_metadata(issue, repo)

    # Description chunk
    title = issue["title"] or ""
    body = issue.get("body") or ""
    desc_text = f"[{repo} #{issue['number']}] {title}\n\n{body}".strip()
    if not desc_text:
        desc_text = f"Issue #{issue['number']}"
    desc_text = _truncate_text(desc_text, max_tokens)
    chunks.append(Chunk(
        id=_make_issue_chunk_id(repo, issue["number"], "description", 0),
        text=desc_text,
        metadata={**base_meta, "chunk_type": "description"},
    ))

    # Comment chunks
    for i, comment in enumerate(comments):
        author = comment.get("user", {}).get("login", "unknown") if comment.get("user") else "unknown"
        comment_text = f"Comment by {author} on #{issue['number']}:\n\n{comment['body']}"
        comment_text = _truncate_text(comment_text, max_tokens)
        chunks.append(Chunk(
            id=_make_issue_chunk_id(repo, issue["number"], "comment", i),
            text=comment_text,
            metadata={**base_meta, "chunk_type": "comment",
                      "comment_author": author,
                      "comment_created_at": comment.get("created_at", "")},
        ))

    return chunks


class IssueIndexer:
    def __init__(self, vector_store, metadata_db, embedder, github_client,
                 chunk_max_tokens: int = 512,
                 include_labels: list[str] | None = None,
                 exclude_labels: list[str] | None = None) -> None:
        self.vector_store = vector_store
        self.metadata_db = metadata_db
        self.embedder = embedder
        self.github = github_client
        self.chunk_max_tokens = chunk_max_tokens
        self.include_labels = set(include_labels) if include_labels else set()
        self.exclude_labels = set(exclude_labels) if exclude_labels else set()

    def sync(self, repo: str, since_days: int = 90) -> IssueSyncStats:
        stats = IssueSyncStats()
        cursor = self.metadata_db.get_issue_sync_cursor(repo)
        if cursor:
            since_date = cursor
        else:
            since_dt = datetime.now(timezone.utc) - timedelta(days=since_days)
            since_date = since_dt.isoformat()

        issues = self.github.list_issues(repo, state="all", sort="updated", since=since_date)
        stats.issues_fetched = len(issues)
        latest_updated = since_date

        for issue in issues:
            # Skip pull requests (GitHub Issues API returns PRs too)
            if "pull_request" in issue:
                stats.issues_skipped += 1
                continue

            # Label filtering
            issue_labels = {l["name"] for l in issue.get("labels", [])}
            if self.include_labels and not issue_labels & self.include_labels:
                stats.issues_skipped += 1
                continue
            if self.exclude_labels and issue_labels & self.exclude_labels:
                stats.issues_skipped += 1
                continue

            updated_at = issue.get("updated_at", "")
            if updated_at and updated_at < since_date:
                stats.issues_skipped += 1
                continue
            if updated_at and updated_at > latest_updated:
                latest_updated = updated_at

            self.metadata_db.delete_chunks_for_issue(repo, issue["number"])
            comments = self.github.get_issue_comments(repo, issue["number"])
            chunks = chunk_issue(issue, comments, repo=repo, max_tokens=self.chunk_max_tokens)
            if not chunks:
                stats.issues_indexed += 1
                continue

            texts = [c.text for c in chunks]
            embeddings = self.embedder.embed(texts)
            embed_map = {c.id: embeddings[i] for i, c in enumerate(chunks)}

            desc_chunks = [c for c in chunks if c.metadata["chunk_type"] == "description"]
            discussion_chunks = [c for c in chunks if c.metadata["chunk_type"] == "comment"]

            if desc_chunks:
                self.vector_store.upsert(
                    collection="issue_descriptions", ids=[c.id for c in desc_chunks],
                    embeddings=[embed_map[c.id] for c in desc_chunks],
                    documents=[c.text for c in desc_chunks],
                    metadatas=[c.metadata for c in desc_chunks],
                )
            if discussion_chunks:
                self.vector_store.upsert(
                    collection="issue_discussions", ids=[c.id for c in discussion_chunks],
                    embeddings=[embed_map[c.id] for c in discussion_chunks],
                    documents=[c.text for c in discussion_chunks],
                    metadatas=[c.metadata for c in discussion_chunks],
                )

            for chunk in chunks:
                self.metadata_db.set_issue_chunk_source(chunk.id, repo, issue["number"])
                self.metadata_db.upsert_fts(chunk.id, chunk.text)

            stats.issues_indexed += 1
            stats.chunks_created += len(chunks)

        self.metadata_db.set_issue_sync_cursor(repo, latest_updated)
        return stats
