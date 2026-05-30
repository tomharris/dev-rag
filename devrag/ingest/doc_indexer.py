from __future__ import annotations
import hashlib
import logging
import re
from pathlib import Path
from devrag.types import Chunk, DocIndexStats

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4
DOC_EXTENSIONS: dict[str, str] = {
    ".md": "markdown", ".mdx": "markdown", ".txt": "text",
    ".rst": "rst", ".html": "html", ".adoc": "asciidoc",
}


def split_markdown(text: str) -> list[dict]:
    lines = text.split("\n")
    sections: list[dict] = []
    heading_stack: list[str] = []
    current_content: list[str] = []
    current_level = 0

    def flush_section():
        content = "\n".join(current_content).strip()
        if content:
            path = " > ".join(heading_stack) if heading_stack else "Document"
            sections.append({"section_path": path, "content": content, "level": current_level})

    for line in lines:
        heading_match = re.match(r"^(#{1,4})\s+(.+)$", line)
        if heading_match:
            flush_section()
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            while len(heading_stack) >= level:
                heading_stack.pop()
            heading_stack.append(title)
            current_level = level
            current_content = []
        else:
            current_content.append(line)
    flush_section()
    return sections


def _split_plain_text(text: str) -> list[dict]:
    paragraphs = re.split(r"\n\s*\n", text)
    sections = []
    for i, para in enumerate(paragraphs):
        para = para.strip()
        if para:
            sections.append({"section_path": f"Paragraph {i + 1}", "content": para, "level": 0})
    return sections


def _make_doc_chunk_id(file_path: str, section_path: str, index: int) -> str:
    raw = f"doc:{file_path}:{section_path}:{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def chunk_document(text: str, file_path: str, max_tokens: int = 512, overlap_tokens: int = 50) -> list[Chunk]:
    ext = Path(file_path).suffix.lower()
    language = DOC_EXTENSIONS.get(ext, "text")
    if language == "markdown":
        sections = split_markdown(text)
    else:
        sections = _split_plain_text(text)
    if not sections:
        return []

    max_chars = max_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN
    chunks: list[Chunk] = []
    chunk_index = 0

    for section in sections:
        content = section["content"]
        section_path = section["section_path"]
        meta = {"file_path": file_path, "language": language, "section_path": section_path,
                "chunk_type": "document",
                "entity_name": section_path.split(" > ")[-1] if " > " in section_path else section_path}

        if len(content) <= max_chars:
            chunks.append(Chunk(id=_make_doc_chunk_id(file_path, section_path, chunk_index),
                                text=content, metadata=meta))
            chunk_index += 1
        else:
            paragraphs = content.split("\n\n")
            current_text = ""
            for para in paragraphs:
                # If a single paragraph exceeds max_chars, split it by words
                if len(para) > max_chars:
                    # Flush current accumulated text first
                    if current_text.strip():
                        chunks.append(Chunk(id=_make_doc_chunk_id(file_path, section_path, chunk_index),
                                            text=current_text.strip(), metadata=meta.copy()))
                        chunk_index += 1
                        current_text = ""
                    # Split oversized paragraph by words
                    words = para.split(" ")
                    word_buf = ""
                    for word in words:
                        if len(word_buf) + len(word) + 1 > max_chars and word_buf:
                            chunks.append(Chunk(id=_make_doc_chunk_id(file_path, section_path, chunk_index),
                                                text=word_buf.strip(), metadata=meta.copy()))
                            chunk_index += 1
                            word_buf = word_buf[-overlap_chars:] + " " + word if overlap_chars else word
                        else:
                            word_buf = word_buf + " " + word if word_buf else word
                    if word_buf.strip():
                        current_text = word_buf
                elif len(current_text) + len(para) + 2 > max_chars and current_text:
                    chunks.append(Chunk(id=_make_doc_chunk_id(file_path, section_path, chunk_index),
                                        text=current_text.strip(), metadata=meta.copy()))
                    chunk_index += 1
                    current_text = current_text[-overlap_chars:] + "\n\n" + para if overlap_chars else para
                else:
                    current_text = current_text + "\n\n" + para if current_text else para
            if current_text.strip():
                chunks.append(Chunk(id=_make_doc_chunk_id(file_path, section_path, chunk_index),
                                    text=current_text.strip(), metadata=meta.copy()))
                chunk_index += 1
    return chunks


class DocIndexer:
    def __init__(self, vector_store, metadata_db, embedder, sparse_encoder, config=None) -> None:
        from devrag.config import DevragConfig
        self.vector_store = vector_store
        self.metadata_db = metadata_db
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder
        if config is None:
            config = DevragConfig()
        self.doc_config = config.documents

    def index_docs(self, docs_path: Path, glob_patterns: list[str] | None = None, incremental: bool = True) -> DocIndexStats:
        """Index a standalone directory of documents (repo-agnostic, glob-discovered)."""
        stats = DocIndexStats()
        if glob_patterns is None:
            glob_patterns = self.doc_config.glob_patterns
        files: list[Path] = []
        for pattern in glob_patterns:
            files.extend(docs_path.glob(pattern))
        seen: set[Path] = set()
        unique_files: list[Path] = []
        for f in files:
            resolved = f.resolve()
            if resolved not in seen and resolved.suffix.lower() in DOC_EXTENSIONS:
                seen.add(resolved)
                unique_files.append(f)
        stats.files_scanned = len(unique_files)

        for file_path in unique_files:
            self._safe_index_doc_file(file_path, repo="", incremental=incremental, stats=stats)
        return stats

    def index_repo_docs(
        self,
        repo_path: Path,
        repo_name: str,
        incremental: bool = True,
        exclude_patterns: list[str] | None = None,
    ) -> DocIndexStats:
        """Index a code repo's docs into the ``documents`` collection, tagged with *repo_name*.

        Uses the same gitignore/.devragignore-aware discovery as ``CodeIndexer`` and
        applies the full per-repo lifecycle: incremental skip, repo-scoped removal of
        deleted docs, and a ``repo`` tag on every chunk. Doc rows share the
        ``(repo, file_path)`` namespace in MetadataDB with code rows; removal here is
        scoped to ``DOC_EXTENSIONS`` so it never touches the repo's code files.
        """
        from devrag.ingest.code_indexer import _DEFAULT_EXCLUDE
        from devrag.utils.git import discover_files

        stats = DocIndexStats()
        exclude = list(exclude_patterns or []) + _DEFAULT_EXCLUDE
        files = discover_files(repo_path, exclude_patterns=exclude)
        doc_files = [f for f in files if f.suffix.lower() in DOC_EXTENSIONS]
        stats.files_scanned = len(doc_files)

        current_paths = {str(f) for f in doc_files}

        # Detect removed docs — scoped to this repo and to doc extensions only, so the
        # repo's code files (sharing the same repo namespace) are never deleted here.
        previously_indexed = set(self.metadata_db.get_indexed_files_for_repo(repo_name))
        removed = {
            p for p in previously_indexed if Path(p).suffix.lower() in DOC_EXTENSIONS
        } - current_paths
        for removed_path in removed:
            old_chunk_ids = self.metadata_db.get_chunks_for_file(removed_path, repo=repo_name)
            if old_chunk_ids:
                self.vector_store.delete("documents", old_chunk_ids)
            self.metadata_db.remove_file(removed_path, repo=repo_name)
            stats.files_removed += 1

        for file_path in doc_files:
            self._safe_index_doc_file(file_path, repo=repo_name, incremental=incremental, stats=stats)
        return stats

    def _safe_index_doc_file(self, file_path: Path, repo: str, incremental: bool, stats: DocIndexStats) -> None:
        """Index one doc file, isolating failures so one bad file can't abort the run.

        A single oversized/unembeddable file (e.g. an embed 400) is logged and
        counted in ``stats.files_failed`` rather than propagating — important for
        ``index refresh``, where an exception would skip every later repo. Because
        ``_index_doc_file`` persists the file hash only after a successful upsert,
        a failed file is retried (not silently skipped) on the next run.
        """
        try:
            self._index_doc_file(file_path, repo=repo, incremental=incremental, stats=stats)
        except Exception as exc:
            stats.files_failed += 1
            logger.warning("Failed to index doc %s: %s", file_path, exc)

    def _index_doc_file(self, file_path: Path, repo: str, incremental: bool, stats: DocIndexStats) -> None:
        """Index a single doc file into ``documents``, updating *stats* in place.

        Shared by ``index_docs`` (repo="") and ``index_repo_docs`` (repo=<name>).
        """
        rel_path = str(file_path)
        content_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if incremental:
            stored_hash = self.metadata_db.get_file_hash(rel_path, repo=repo)
            if stored_hash == content_hash:
                stats.files_skipped += 1
                return

        # Drop old chunks for this file first (handles re-indexing of changed files).
        old_chunk_ids = self.metadata_db.get_chunks_for_file(rel_path, repo=repo)
        if old_chunk_ids:
            self.vector_store.delete("documents", old_chunk_ids)
            self.metadata_db.remove_file(rel_path, repo=repo)

        text = file_path.read_text(errors="replace")
        chunks = chunk_document(text=text, file_path=rel_path,
                                 max_tokens=self.doc_config.chunk_max_tokens,
                                 overlap_tokens=self.doc_config.chunk_overlap_tokens)
        if not chunks:
            # Record the hash so an empty file is skipped next run.
            self.metadata_db.set_file_hash(rel_path, content_hash, repo=repo)
            stats.files_indexed += 1
            return
        if repo:
            for chunk in chunks:
                chunk.metadata["repo"] = repo
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts)
        sparse_embeddings = self.sparse_encoder.encode(texts)
        self.vector_store.upsert(collection="documents", ids=[c.id for c in chunks],
                                  embeddings=embeddings, documents=texts,
                                  metadatas=[c.metadata for c in chunks],
                                  sparse_embeddings=sparse_embeddings, wait=False)
        for chunk in chunks:
            self.metadata_db.set_chunk_source(chunk.id, rel_path, 0, 0, repo=repo)
        # Persist the hash only after a successful upsert, so a file that fails
        # to embed is retried on the next run rather than marked done.
        self.metadata_db.set_file_hash(rel_path, content_hash, repo=repo)
        stats.files_indexed += 1
        stats.chunks_created += len(chunks)
