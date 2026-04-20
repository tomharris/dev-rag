from __future__ import annotations
import hashlib
import re
from pathlib import Path
from devrag.types import Chunk, DocIndexStats

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
            rel_path = str(file_path)
            content_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            if incremental:
                stored_hash = self.metadata_db.get_file_hash(rel_path)
                if stored_hash == content_hash:
                    continue
            self.metadata_db.set_file_hash(rel_path, content_hash)
            old_chunk_ids = self.metadata_db.get_chunks_for_file(rel_path)
            if old_chunk_ids:
                self.vector_store.delete("documents", old_chunk_ids)
            text = file_path.read_text(errors="replace")
            chunks = chunk_document(text=text, file_path=rel_path,
                                     max_tokens=self.doc_config.chunk_max_tokens,
                                     overlap_tokens=self.doc_config.chunk_overlap_tokens)
            if not chunks:
                stats.files_indexed += 1
                continue
            texts = [c.text for c in chunks]
            embeddings = self.embedder.embed(texts)
            sparse_embeddings = self.sparse_encoder.encode(texts)
            self.vector_store.upsert(collection="documents", ids=[c.id for c in chunks],
                                      embeddings=embeddings, documents=texts,
                                      metadatas=[c.metadata for c in chunks],
                                      sparse_embeddings=sparse_embeddings)
            for chunk in chunks:
                self.metadata_db.set_chunk_source(chunk.id, rel_path, 0, 0)
            stats.files_indexed += 1
            stats.chunks_created += len(chunks)
        return stats
