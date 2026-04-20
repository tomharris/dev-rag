from __future__ import annotations
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from devrag.types import Chunk, PRSyncStats
from devrag.utils.github import GitHubClient, parse_diff_hunks

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4


def _make_pr_chunk_id(repo: str, pr_number: int, chunk_type: str, index: int) -> str:
    raw = f"{repo}:pr{pr_number}:{chunk_type}:{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _pr_base_metadata(pr: dict, repo: str) -> dict:
    labels = ",".join(l["name"] for l in pr.get("labels", []))
    return {"repo": repo, "pr_number": pr["number"], "pr_title": pr["title"],
            "pr_state": pr["state"], "pr_author": pr["user"]["login"],
            "pr_labels": labels, "merged_at": pr.get("merged_at") or ""}


def _truncate_text(text: str, max_tokens: int) -> str:
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) > max_chars:
        logger.debug("Truncating PR chunk from %d to %d chars", len(text), max_chars)
        return text[:max_chars] + "\n# ... (truncated)"
    return text


def chunk_pr(pr: dict, files: list[dict], comments: list[dict], repo: str, max_tokens: int = 512) -> list[Chunk]:
    chunks: list[Chunk] = []
    base_meta = _pr_base_metadata(pr, repo)

    # Description chunk
    title = pr["title"] or ""
    body = pr.get("body") or ""
    desc_text = f"# PR #{pr['number']}: {title}\n\n{body}".strip()
    if not desc_text:
        desc_text = f"PR #{pr['number']}"
    desc_text = _truncate_text(desc_text, max_tokens)
    chunks.append(Chunk(
        id=_make_pr_chunk_id(repo, pr["number"], "description", 0),
        text=desc_text,
        metadata={**base_meta, "chunk_type": "description", "file_path": ""},
    ))

    # Diff chunks
    diff_index = 0
    for file_info in files:
        patch = file_info.get("patch")
        if not patch:
            continue
        filename = file_info["filename"]
        hunks = parse_diff_hunks(patch, filename)
        for hunk in hunks:
            diff_text = f"File: {filename} ({file_info.get('status', 'modified')})\n{hunk['content']}"
            diff_text = _truncate_text(diff_text, max_tokens)
            chunks.append(Chunk(
                id=_make_pr_chunk_id(repo, pr["number"], "diff", diff_index),
                text=diff_text,
                metadata={**base_meta, "chunk_type": "diff", "file_path": filename},
            ))
            diff_index += 1

    # Review comment chunks
    for i, comment in enumerate(comments):
        reviewer = comment.get("user", {}).get("login", "unknown") if comment.get("user") else "unknown"
        comment_path = comment.get("path", "")
        comment_text = f"Review comment by {reviewer} on {comment_path}:\n{comment['body']}"
        comment_text = _truncate_text(comment_text, max_tokens)
        chunks.append(Chunk(
            id=_make_pr_chunk_id(repo, pr["number"], "review_comment", i),
            text=comment_text,
            metadata={**base_meta, "chunk_type": "review_comment", "file_path": comment_path, "reviewer": reviewer},
        ))

    return chunks


class PRIndexer:
    def __init__(self, vector_store, metadata_db, embedder, sparse_encoder, github_client, chunk_max_tokens: int = 512) -> None:
        self.vector_store = vector_store
        self.metadata_db = metadata_db
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder
        self.github = github_client
        self.chunk_max_tokens = chunk_max_tokens

    def sync(self, repo: str, since_days: int | None = None) -> PRSyncStats:
        stats = PRSyncStats()
        if since_days is not None:
            since_date = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat()
        else:
            cursor = self.metadata_db.get_pr_sync_cursor(repo)
            if cursor:
                since_date = cursor
            else:
                since_date = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()

        prs = self.github.list_prs(repo, state="all", sort="updated", since=since_date)
        stats.prs_fetched = len(prs)
        latest_updated = since_date

        for pr in prs:
            updated_at = pr.get("updated_at", "")
            if updated_at and updated_at < since_date:
                stats.prs_skipped += 1
                continue
            if updated_at and updated_at > latest_updated:
                latest_updated = updated_at

            self.metadata_db.delete_chunks_for_pr(repo, pr["number"])
            files = self.github.get_pr_files(repo, pr["number"])
            comments = self.github.get_pr_comments(repo, pr["number"])
            chunks = chunk_pr(pr, files, comments, repo=repo, max_tokens=self.chunk_max_tokens)
            if not chunks:
                stats.prs_indexed += 1
                continue

            texts = [c.text for c in chunks]
            embeddings = self.embedder.embed(texts)
            sparse_embeddings = self.sparse_encoder.encode(texts)
            embed_map = {c.id: embeddings[i] for i, c in enumerate(chunks)}
            sparse_map = {c.id: sparse_embeddings[i] for i, c in enumerate(chunks)}

            diff_chunks = [c for c in chunks if c.metadata["chunk_type"] in ("diff", "description")]
            discussion_chunks = [c for c in chunks if c.metadata["chunk_type"] == "review_comment"]

            if diff_chunks:
                self.vector_store.upsert(
                    collection="pr_diffs", ids=[c.id for c in diff_chunks],
                    embeddings=[embed_map[c.id] for c in diff_chunks],
                    documents=[c.text for c in diff_chunks],
                    metadatas=[c.metadata for c in diff_chunks],
                    sparse_embeddings=[sparse_map[c.id] for c in diff_chunks],
                )
            if discussion_chunks:
                self.vector_store.upsert(
                    collection="pr_discussions", ids=[c.id for c in discussion_chunks],
                    embeddings=[embed_map[c.id] for c in discussion_chunks],
                    documents=[c.text for c in discussion_chunks],
                    metadatas=[c.metadata for c in discussion_chunks],
                    sparse_embeddings=[sparse_map[c.id] for c in discussion_chunks],
                )

            for chunk in chunks:
                self.metadata_db.set_pr_chunk_source(chunk.id, repo, pr["number"])

            stats.prs_indexed += 1
            stats.chunks_created += len(chunks)

        self.metadata_db.set_pr_sync_cursor(repo, latest_updated)
        return stats
