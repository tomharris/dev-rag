from __future__ import annotations

import hashlib
import logging

import httpx

from devrag.ingest.doc_indexer import split_markdown, CHARS_PER_TOKEN
from devrag.types import Chunk, SliteSyncStats
from devrag.utils.slite_client import SliteClient

logger = logging.getLogger(__name__)


def _make_chunk_id(page_id: str, section_path: str, index: int) -> str:
    raw = f"slite:{page_id}:{section_path}:{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _truncate_text(text: str, max_tokens: int) -> str:
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) > max_chars:
        logger.debug("Truncating Slite chunk from %d to %d chars", len(text), max_chars)
        return text[:max_chars] + "\n... (truncated)"
    return text


def chunk_slite_page(
    note: dict,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    """Chunk a Slite page using section-aware markdown splitting."""
    page_id = note["id"]
    title = note.get("title", "")
    url = note.get("url", "")
    updated_at = note.get("lastEditedAt", "")
    content = note.get("content", "")
    if not content or not content.strip():
        return []

    sections = split_markdown(content)
    if not sections:
        return []

    max_chars = max_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN
    chunks: list[Chunk] = []
    chunk_index = 0

    for section in sections:
        sec_content = section["content"]
        section_path = section["section_path"]
        meta = {
            "page_id": page_id,
            "page_title": title,
            "page_url": url,
            "updated_at": updated_at,
            "section_path": section_path,
            "chunk_type": "slite_page",
            "entity_name": section_path.split(" > ")[-1] if " > " in section_path else section_path,
        }

        if len(sec_content) <= max_chars:
            chunks.append(Chunk(
                id=_make_chunk_id(page_id, section_path, chunk_index),
                text=sec_content,
                metadata=meta,
            ))
            chunk_index += 1
        else:
            paragraphs = sec_content.split("\n\n")
            current_text = ""
            for para in paragraphs:
                if len(para) > max_chars:
                    if current_text.strip():
                        chunks.append(Chunk(
                            id=_make_chunk_id(page_id, section_path, chunk_index),
                            text=current_text.strip(),
                            metadata=meta.copy(),
                        ))
                        chunk_index += 1
                        current_text = ""
                    words = para.split(" ")
                    word_buf = ""
                    for word in words:
                        if len(word_buf) + len(word) + 1 > max_chars and word_buf:
                            chunks.append(Chunk(
                                id=_make_chunk_id(page_id, section_path, chunk_index),
                                text=word_buf.strip(),
                                metadata=meta.copy(),
                            ))
                            chunk_index += 1
                            word_buf = word_buf[-overlap_chars:] + " " + word if overlap_chars else word
                        else:
                            word_buf = word_buf + " " + word if word_buf else word
                    if word_buf.strip():
                        current_text = word_buf
                elif len(current_text) + len(para) + 2 > max_chars and current_text:
                    chunks.append(Chunk(
                        id=_make_chunk_id(page_id, section_path, chunk_index),
                        text=current_text.strip(),
                        metadata=meta.copy(),
                    ))
                    chunk_index += 1
                    current_text = current_text[-overlap_chars:] + "\n\n" + para if overlap_chars else para
                else:
                    current_text = current_text + "\n\n" + para if current_text else para
            if current_text.strip():
                chunks.append(Chunk(
                    id=_make_chunk_id(page_id, section_path, chunk_index),
                    text=current_text.strip(),
                    metadata=meta.copy(),
                ))
                chunk_index += 1
    return chunks


class SliteIndexer:
    def __init__(
        self,
        vector_store,
        metadata_db,
        embedder,
        sparse_encoder,
        slite_client: SliteClient,
        chunk_max_tokens: int = 512,
        chunk_overlap_tokens: int = 50,
        channel_ids: list[str] | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.metadata_db = metadata_db
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder
        self.slite = slite_client
        self.chunk_max_tokens = chunk_max_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.channel_ids = channel_ids or []

    def sync(self, since_days: int = 90) -> SliteSyncStats:
        """Incrementally sync Slite pages.

        Incrementality is keyed on each note's ``lastEditedAt`` (the actual
        content-edit time), NOT ``updatedAt`` — the latter is a volatile
        metadata/popularity timestamp that Slite bumps for essentially every
        note, so using it re-indexes the whole workspace on every run.

        The full note list is paginated and filtered client-side on
        ``lastEditedAt``. ``since_days`` is accepted for CLI/MCP signature
        compatibility but no longer narrows the fetch (the API's
        ``sinceDaysAgo`` is a popularity window, not an edit filter, so it
        could silently drop edited-but-unpopular notes).
        """
        stats = SliteSyncStats()
        cursor_key = "default"

        cursor = self.metadata_db.get_slite_sync_cursor(cursor_key)

        latest_edited: str | None = None

        for note in self.slite.list_notes(
            channel_ids=self.channel_ids or None,
        ):
            note_id = note["id"]
            last_edited = note.get("lastEditedAt", "")

            # Skip only when we have a reliable edit time that predates the
            # cursor. A missing lastEditedAt falls through and re-indexes so a
            # field gap can never silently drop an update.
            if cursor and last_edited and last_edited <= cursor:
                stats.pages_skipped += 1
                continue

            stats.pages_fetched += 1

            if last_edited and (latest_edited is None or last_edited > latest_edited):
                latest_edited = last_edited

            try:
                full_note = self.slite.get_note(note_id, fmt="md")
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code >= 500:
                    logger.warning("Slite API %d for note %s, skipping", exc.response.status_code, note_id)
                    stats.pages_errored += 1
                    continue
                raise
            except httpx.TimeoutException:
                logger.warning("Slite API timeout for note %s, skipping", note_id)
                stats.pages_errored += 1
                continue
            content = full_note.get("content", "")
            if not content or not content.strip():
                stats.pages_skipped += 1
                continue

            old_chunk_ids = self.metadata_db.get_chunks_for_slite_page(cursor_key, note_id)
            if old_chunk_ids:
                self.vector_store.delete("slite_pages", old_chunk_ids)
                self.metadata_db.delete_chunks_for_slite_page(cursor_key, note_id)

            full_note["title"] = full_note.get("title", note.get("title", ""))
            full_note["url"] = full_note.get("url", note.get("url", ""))
            full_note["lastEditedAt"] = last_edited

            chunks = chunk_slite_page(
                full_note,
                max_tokens=self.chunk_max_tokens,
                overlap_tokens=self.chunk_overlap_tokens,
            )
            if not chunks:
                stats.pages_skipped += 1
                continue

            texts = [_truncate_text(c.text, self.chunk_max_tokens) for c in chunks]
            embeddings = self.embedder.embed(texts)
            sparse_embeddings = self.sparse_encoder.encode(texts)
            self.vector_store.upsert(
                collection="slite_pages",
                ids=[c.id for c in chunks],
                embeddings=embeddings,
                documents=texts,
                metadatas=[c.metadata for c in chunks],
                sparse_embeddings=sparse_embeddings,
                wait=False,
            )
            for chunk in chunks:
                self.metadata_db.set_slite_chunk_source(chunk.id, cursor_key, note_id)

            stats.pages_indexed += 1
            stats.chunks_created += len(chunks)

        if latest_edited:
            self.metadata_db.set_slite_sync_cursor(cursor_key, latest_edited)

        return stats
