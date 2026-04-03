# DevRAG Phase 3: Document Search + CLI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add internal documents (Markdown, text, RST, HTML) as a third searchable knowledge source, add `.devragignore` support, and build a standalone CLI for use outside Claude Code.

**Architecture:** Document indexer splits files by headings → paragraphs → sentences with configurable chunk size and overlap. New `documents` ChromaDB collection. Query router gains "docs" scope and document-intent patterns. CLI built with typer wraps the same core library as the MCP server. `.devragignore` extends code indexing exclusions.

**Tech Stack:** typer (CLI), existing ChromaDB + SQLite + Ollama stack.

---

## File Structure

```
devrag/
├── pyproject.toml              # MODIFY: add typer, console_scripts entry
├── devrag/
│   ├── config.py               # MODIFY: add DocumentsConfig
│   ├── types.py                # MODIFY: add DocIndexStats
│   ├── cli.py                  # CREATE: typer CLI
│   ├── mcp_server.py           # MODIFY: add index_docs tool, docs scope
│   │
│   ├── ingest/
│   │   └── doc_indexer.py      # CREATE: document chunking and indexing
│   │
│   ├── retrieve/
│   │   └── query_router.py     # MODIFY: add docs patterns + scope
│   │
│   └── utils/
│       ├── git.py              # MODIFY: .devragignore support
│       └── formatters.py       # MODIFY: doc result formatting + doc index stats
│
└── tests/
    ├── test_doc_indexer.py     # CREATE
    ├── test_cli.py             # CREATE
    ├── test_query_router.py    # MODIFY: docs routing tests
    ├── test_git.py             # MODIFY: .devragignore tests
    └── test_formatters.py      # MODIFY: doc formatting tests
```

---

## Task 1: Config + Types + Dependencies

**Files:**
- Modify: `pyproject.toml`
- Modify: `devrag/config.py`
- Modify: `devrag/types.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_default_config_has_documents_section():
    config = DevragConfig()
    assert "**/*.md" in config.documents.glob_patterns
    assert "**/*.txt" in config.documents.glob_patterns
    assert config.documents.chunk_max_tokens == 512
    assert config.documents.chunk_overlap_tokens == 50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_default_config_has_documents_section -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 3: Add DocumentsConfig and DocIndexStats**

Add to `devrag/config.py` after `PrsConfig`:

```python
@dataclass
class DocumentsConfig:
    glob_patterns: list[str] = field(default_factory=lambda: [
        "**/*.md", "**/*.mdx", "**/*.txt", "**/*.rst", "**/*.html", "**/*.adoc",
    ])
    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 50
```

Add `documents` to `DevragConfig`:

```python
@dataclass
class DevragConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    code: CodeConfig = field(default_factory=CodeConfig)
    prs: PrsConfig = field(default_factory=PrsConfig)
    documents: DocumentsConfig = field(default_factory=DocumentsConfig)
```

Add to `devrag/types.py`:

```python
@dataclass
class DocIndexStats:
    """Statistics from a document indexing run."""
    files_scanned: int = 0
    files_indexed: int = 0
    chunks_created: int = 0
```

- [ ] **Step 4: Add typer to pyproject.toml**

Add `"typer>=0.12.0"` to `dependencies` in `pyproject.toml`. Add console script entry:

```toml
[project.scripts]
devrag = "devrag.cli:app"
```

- [ ] **Step 5: Run `uv sync` and verify tests pass**

Run: `uv sync --all-extras && uv run pytest tests/test_config.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml devrag/config.py devrag/types.py tests/test_config.py
git commit -m "feat: add DocumentsConfig, DocIndexStats, and typer dependency"
```

---

## Task 2: Document Indexer

**Files:**
- Create: `devrag/ingest/doc_indexer.py`
- Create: `tests/test_doc_indexer.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_doc_indexer.py
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from devrag.ingest.doc_indexer import DocIndexer, chunk_document, split_markdown


def test_split_markdown_by_headings():
    text = """# Introduction

This is the intro.

## Architecture

### Components

The system has three parts.

## Deployment

Deploy with Docker.
"""
    sections = split_markdown(text)
    assert len(sections) >= 3
    # Each section should have a section_path
    paths = [s["section_path"] for s in sections]
    assert any("Introduction" in p for p in paths)
    assert any("Architecture" in p for p in paths)
    assert any("Deployment" in p for p in paths)


def test_split_markdown_preserves_hierarchy():
    text = """# Top

## Middle

### Bottom

Content here.
"""
    sections = split_markdown(text)
    bottom = next(s for s in sections if "Bottom" in s["section_path"])
    assert bottom["section_path"] == "Top > Middle > Bottom"


def test_chunk_document_respects_max_tokens():
    # Create a long document
    long_text = "# Title\n\n" + ("This is a paragraph with enough words. " * 100 + "\n\n") * 5
    chunks = chunk_document(
        text=long_text,
        file_path="docs/long.md",
        max_tokens=100,
        overlap_tokens=10,
    )
    # Should create multiple chunks
    assert len(chunks) > 1
    # Each chunk text should be under max_tokens * 4 chars (approximate)
    for chunk in chunks:
        assert len(chunk.text) <= 100 * 4 + 200  # some tolerance for headers


def test_chunk_document_includes_metadata():
    text = "# API Guide\n\n## Authentication\n\nUse Bearer tokens.\n"
    chunks = chunk_document(text=text, file_path="docs/api.md", max_tokens=512, overlap_tokens=50)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.metadata["file_path"] == "docs/api.md"
        assert chunk.metadata["language"] == "markdown"
        assert "section_path" in chunk.metadata
        assert chunk.metadata["chunk_type"] == "document"


def test_chunk_document_plain_text():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.\n"
    chunks = chunk_document(text=text, file_path="notes.txt", max_tokens=512, overlap_tokens=50)
    assert len(chunks) >= 1
    assert chunks[0].metadata["language"] == "text"


def test_chunk_ids_deterministic():
    text = "# Hello\n\nWorld.\n"
    c1 = chunk_document(text=text, file_path="a.md", max_tokens=512, overlap_tokens=50)
    c2 = chunk_document(text=text, file_path="a.md", max_tokens=512, overlap_tokens=50)
    assert [c.id for c in c1] == [c.id for c in c2]


def test_doc_indexer_indexes_directory(tmp_dir):
    docs_dir = tmp_dir / "docs"
    docs_dir.mkdir()
    (docs_dir / "guide.md").write_text("# User Guide\n\nHow to use the app.\n\n## Setup\n\nInstall deps.\n")
    (docs_dir / "notes.txt").write_text("Some plain text notes.\n")
    (docs_dir / "image.png").write_bytes(b"\x89PNG")  # should be skipped

    store = MagicMock()
    meta = MagicMock()
    meta.get_file_hash.return_value = None  # not previously indexed
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])

    indexer = DocIndexer(store, meta, embedder)
    stats = indexer.index_docs(docs_dir, glob_patterns=["**/*.md", "**/*.txt"])

    assert stats.files_scanned >= 2
    assert stats.files_indexed >= 2
    assert stats.chunks_created >= 2
    store.upsert.assert_called()
    embedder.embed.assert_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_doc_indexer.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write document indexer implementation**

```python
# devrag/ingest/doc_indexer.py
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from devrag.types import Chunk, DocIndexStats

CHARS_PER_TOKEN = 4

# Map file extensions to language identifiers
DOC_EXTENSIONS: dict[str, str] = {
    ".md": "markdown",
    ".mdx": "markdown",
    ".txt": "text",
    ".rst": "rst",
    ".html": "html",
    ".adoc": "asciidoc",
}


def split_markdown(text: str) -> list[dict]:
    """Split markdown text into sections based on heading hierarchy.

    Returns list of dicts with keys: section_path, content, level.
    """
    lines = text.split("\n")
    sections: list[dict] = []
    heading_stack: list[str] = []  # tracks current heading hierarchy
    current_content: list[str] = []
    current_level = 0

    def flush_section():
        content = "\n".join(current_content).strip()
        if content:
            path = " > ".join(heading_stack) if heading_stack else "Document"
            sections.append({
                "section_path": path,
                "content": content,
                "level": current_level,
            })

    for line in lines:
        heading_match = re.match(r"^(#{1,4})\s+(.+)$", line)
        if heading_match:
            flush_section()
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()

            # Adjust heading stack to current level
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
    """Split plain text by double-newline paragraphs."""
    paragraphs = re.split(r"\n\s*\n", text)
    sections = []
    for i, para in enumerate(paragraphs):
        para = para.strip()
        if para:
            sections.append({
                "section_path": f"Paragraph {i + 1}",
                "content": para,
                "level": 0,
            })
    return sections


def _make_doc_chunk_id(file_path: str, section_path: str, index: int) -> str:
    raw = f"doc:{file_path}:{section_path}:{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def chunk_document(
    text: str,
    file_path: str,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    """Split a document into chunks with metadata."""
    ext = Path(file_path).suffix.lower()
    language = DOC_EXTENSIONS.get(ext, "text")

    # Split into sections
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

        if len(content) <= max_chars:
            # Section fits in one chunk
            chunks.append(Chunk(
                id=_make_doc_chunk_id(file_path, section_path, chunk_index),
                text=content,
                metadata={
                    "file_path": file_path,
                    "language": language,
                    "section_path": section_path,
                    "chunk_type": "document",
                    "entity_name": section_path.split(" > ")[-1] if " > " in section_path else section_path,
                },
            ))
            chunk_index += 1
        else:
            # Split large section into overlapping chunks by paragraph/sentence
            paragraphs = content.split("\n\n")
            current_text = ""

            for para in paragraphs:
                if len(current_text) + len(para) + 2 > max_chars and current_text:
                    chunks.append(Chunk(
                        id=_make_doc_chunk_id(file_path, section_path, chunk_index),
                        text=current_text.strip(),
                        metadata={
                            "file_path": file_path,
                            "language": language,
                            "section_path": section_path,
                            "chunk_type": "document",
                            "entity_name": section_path.split(" > ")[-1] if " > " in section_path else section_path,
                        },
                    ))
                    chunk_index += 1
                    # Keep overlap from end of previous chunk
                    current_text = current_text[-overlap_chars:] + "\n\n" + para if overlap_chars else para
                else:
                    current_text = current_text + "\n\n" + para if current_text else para

            if current_text.strip():
                chunks.append(Chunk(
                    id=_make_doc_chunk_id(file_path, section_path, chunk_index),
                    text=current_text.strip(),
                    metadata={
                        "file_path": file_path,
                        "language": language,
                        "section_path": section_path,
                        "chunk_type": "document",
                        "entity_name": section_path.split(" > ")[-1] if " > " in section_path else section_path,
                    },
                ))
                chunk_index += 1

    return chunks


class DocIndexer:
    """Indexes document files by chunking and storing embeddings."""

    def __init__(self, vector_store, metadata_db, embedder, config=None) -> None:
        from devrag.config import DevragConfig
        self.vector_store = vector_store
        self.metadata_db = metadata_db
        self.embedder = embedder
        if config is None:
            config = DevragConfig()
        self.doc_config = config.documents

    def index_docs(
        self,
        docs_path: Path,
        glob_patterns: list[str] | None = None,
        incremental: bool = True,
    ) -> DocIndexStats:
        """Index documents in a directory."""
        stats = DocIndexStats()
        if glob_patterns is None:
            glob_patterns = self.doc_config.glob_patterns

        # Discover files matching glob patterns
        files: list[Path] = []
        for pattern in glob_patterns:
            files.extend(docs_path.glob(pattern))

        # Deduplicate and filter to supported extensions
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

            # Remove old chunks
            old_chunk_ids = self.metadata_db.get_chunks_for_file(rel_path)
            if old_chunk_ids:
                self.vector_store.delete("documents", old_chunk_ids)
                self.metadata_db.delete_fts(old_chunk_ids)

            # Read and chunk
            text = file_path.read_text(errors="replace")
            chunks = chunk_document(
                text=text,
                file_path=rel_path,
                max_tokens=self.doc_config.chunk_max_tokens,
                overlap_tokens=self.doc_config.chunk_overlap_tokens,
            )

            if not chunks:
                stats.files_indexed += 1
                continue

            # Embed
            texts = [c.text for c in chunks]
            embeddings = self.embedder.embed(texts)

            # Store in vector DB
            self.vector_store.upsert(
                collection="documents",
                ids=[c.id for c in chunks],
                embeddings=embeddings,
                documents=texts,
                metadatas=[c.metadata for c in chunks],
            )

            # Store metadata and FTS
            for chunk in chunks:
                self.metadata_db.set_chunk_source(chunk.id, rel_path, 0, 0)
                self.metadata_db.upsert_fts(chunk.id, chunk.text)

            stats.files_indexed += 1
            stats.chunks_created += len(chunks)

        return stats
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_doc_indexer.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add devrag/ingest/doc_indexer.py tests/test_doc_indexer.py
git commit -m "feat: document indexer with section-aware markdown and text chunking"
```

---

## Task 3: Query Router — Docs Support

**Files:**
- Modify: `devrag/retrieve/query_router.py`
- Modify: `tests/test_query_router.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_query_router.py`:

```python
def test_docs_query_policy():
    router = QueryRouter()
    collections = router.route("what is our API versioning policy")
    assert "documents" in collections


def test_docs_query_architecture():
    router = QueryRouter()
    collections = router.route("describe the system architecture")
    assert "documents" in collections


def test_docs_query_spec():
    router = QueryRouter()
    collections = router.route("what does the design spec say about caching")
    assert "documents" in collections


def test_docs_only_scope():
    router = QueryRouter()
    assert router.route("anything", scope="docs") == ["documents"]


def test_ambiguous_query_includes_docs():
    router = QueryRouter()
    collections = router.route("tell me about authentication")
    assert "documents" in collections
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_query_router.py::test_docs_query_policy -v`
Expected: FAIL — `"documents"` not in collections

- [ ] **Step 3: Update query router**

Update `devrag/retrieve/query_router.py`:

```python
from __future__ import annotations
import re

ALL_COLLECTIONS = ["code_chunks", "pr_diffs", "pr_discussions", "documents"]
CODE_COLLECTIONS = ["code_chunks"]
PR_COLLECTIONS = ["pr_diffs", "pr_discussions"]
DOC_COLLECTIONS = ["documents"]

_PR_PATTERNS = [
    r"\bwhy\s+did\s+we\b", r"\bwhy\s+was\b", r"\bwhy\s+were\b",
    r"\bwhen\s+did\s+we\b", r"\bwho\s+changed\b", r"\bwho\s+added\b", r"\bwho\s+removed\b",
    r"\bswitch(?:ed)?\s+(?:from|to)\b", r"\bmigrat(?:e|ed|ion)\b",
    r"\bchange(?:d|s)?\s+(?:the|to|from)\b", r"\bremov(?:e|ed)\b.*\bwhy\b",
    r"\bwhy\b.*\bremov(?:e|ed)\b", r"\bintroduc(?:e|ed)\b", r"\brevert(?:ed)?\b", r"\bdeprecated?\b",
]

_DOC_PATTERNS = [
    r"\bpolicy\b", r"\bpolicies\b", r"\bspec(?:ification)?\b", r"\bdesign\s+doc\b",
    r"\barchitecture\b", r"\bdiagram\b", r"\bprocess\b", r"\bprocedure\b",
    r"\bguideline\b", r"\bstandard\b", r"\bconvention\b", r"\bdocument(?:ation)?\b",
    r"\bplaybook\b", r"\brunbook\b", r"\bonboarding\b", r"\btutorial\b",
    r"\bdescribe\s+the\b", r"\bwhat\s+does\s+the\s+(?:spec|doc|guide)\b",
]

_CODE_PATTERNS = [
    r"\bhow\s+does\b", r"\bhow\s+do\b", r"\bhow\s+is\b", r"\bwhat\s+does\b",
    r"\bimplement(?:s|ed|ation)?\b", r"\bdefin(?:e|ed|ition)\b",
]

_USAGE_PATTERNS = [
    r"\bwhere\s+is\b", r"\bwhere\s+are\b", r"\bwho\s+uses\b",
    r"\busage\s+of\b", r"\bcall(?:s|ed)\s+(?:from|by|in)\b",
]


class QueryRouter:
    def route(self, query: str, scope: str = "all") -> list[str]:
        if scope == "code":
            return CODE_COLLECTIONS
        if scope == "prs":
            return PR_COLLECTIONS
        if scope == "docs":
            return DOC_COLLECTIONS
        q = query.lower()
        for pattern in _PR_PATTERNS:
            if re.search(pattern, q):
                return PR_COLLECTIONS
        for pattern in _DOC_PATTERNS:
            if re.search(pattern, q):
                return DOC_COLLECTIONS
        for pattern in _USAGE_PATTERNS:
            if re.search(pattern, q):
                return ["code_chunks", "pr_diffs"]
        for pattern in _CODE_PATTERNS:
            if re.search(pattern, q):
                return CODE_COLLECTIONS
        return ALL_COLLECTIONS
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `uv run pytest tests/test_query_router.py -v`
Expected: All 13 tests PASS (8 existing + 5 new).

- [ ] **Step 5: Commit**

```bash
git add devrag/retrieve/query_router.py tests/test_query_router.py
git commit -m "feat: query router with document intent patterns and docs scope"
```

---

## Task 4: .devragignore Support

**Files:**
- Modify: `devrag/utils/git.py`
- Modify: `tests/test_git.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_git.py`:

```python
def test_discover_files_respects_devragignore(git_repo):
    # Create .devragignore
    (git_repo / ".devragignore").write_text("*.md\nsrc/data.min.js\n")
    import subprocess
    subprocess.run(["git", "add", ".devragignore"], cwd=str(git_repo), capture_output=True)
    subprocess.run(["git", "commit", "-m", "add devragignore"], cwd=str(git_repo), capture_output=True)

    files = discover_files(git_repo, exclude_patterns=[])
    rel_paths = {str(f.relative_to(git_repo)) for f in files}
    assert "src/main.py" in rel_paths
    assert "src/utils.py" in rel_paths
    # .devragignore excludes *.md and src/data.min.js
    assert "README.md" not in rel_paths
    assert "src/data.min.js" not in rel_paths
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_git.py::test_discover_files_respects_devragignore -v`
Expected: FAIL — assertion error (README.md still present)

- [ ] **Step 3: Update discover_files to read .devragignore**

Update `devrag/utils/git.py` — add `.devragignore` parsing at the beginning of `discover_files`:

```python
def discover_files(
    repo_path: Path,
    exclude_patterns: list[str],
) -> list[Path]:
    if not repo_path.exists():
        return []

    # Read .devragignore patterns
    devragignore = repo_path / ".devragignore"
    extra_excludes: list[str] = []
    if devragignore.exists():
        for line in devragignore.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                extra_excludes.append(line)

    all_excludes = list(exclude_patterns) + extra_excludes

    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=True,
        )
        rel_paths = [p for p in result.stdout.strip().split("\n") if p]
    except (subprocess.CalledProcessError, FileNotFoundError):
        rel_paths = [
            str(f.relative_to(repo_path))
            for f in repo_path.rglob("*")
            if f.is_file()
        ]

    filtered: list[Path] = []
    for rel in rel_paths:
        if any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(Path(rel).name, pat)
               for pat in all_excludes):
            continue
        full = repo_path / rel
        if full.is_file():
            filtered.append(full)

    return filtered
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `uv run pytest tests/test_git.py -v`
Expected: All 4 tests PASS (3 existing + 1 new).

- [ ] **Step 5: Commit**

```bash
git add devrag/utils/git.py tests/test_git.py
git commit -m "feat: .devragignore support for additional exclusion patterns"
```

---

## Task 5: Formatter Updates + MCP Server

**Files:**
- Modify: `devrag/utils/formatters.py`
- Modify: `tests/test_formatters.py`
- Modify: `devrag/mcp_server.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_formatters.py`:

```python
def test_format_search_results_with_document():
    results = [
        SearchResult(
            chunk_id="doc1",
            text="Use Bearer tokens for all API requests.",
            score=0.9,
            metadata={
                "file_path": "docs/api.md",
                "chunk_type": "document",
                "section_path": "API Guide > Authentication",
                "entity_name": "Authentication",
            },
        ),
    ]
    output = format_search_results(results)
    assert "API Guide > Authentication" in output or "Authentication" in output
    assert "docs/api.md" in output
    assert "Bearer tokens" in output


def test_format_doc_index_stats():
    from devrag.types import DocIndexStats
    stats = DocIndexStats(files_scanned=10, files_indexed=8, chunks_created=42)
    output = format_doc_index_stats(stats)
    assert "10" in output
    assert "8" in output
    assert "42" in output
```

Update import in `tests/test_formatters.py`:
```python
from devrag.utils.formatters import format_search_results, format_index_stats, format_pr_sync_stats, format_doc_index_stats
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_formatters.py::test_format_doc_index_stats -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Update formatters**

Add document handling to `format_search_results` in `devrag/utils/formatters.py`. In the `if chunk_type` check, add `"document"` as a distinct case:

After the PR block (line ~31) and before the `else` block, add:

```python
        elif chunk_type == "document":
            file_path = r.metadata.get("file_path", "unknown")
            section_path = r.metadata.get("section_path", "")
            entity_name = r.metadata.get("entity_name", section_path)
            lines.append(f"### {i}. [{entity_name}] {file_path}")
            if section_path:
                lines.append(f"*Section: {section_path}*")
            lines.append("```")
            text_lines = r.text.strip().split("\n")
            preview = "\n".join(text_lines[:10])
            if len(text_lines) > 10:
                preview += "\n# ... (truncated)"
            lines.append(preview)
            lines.append("```")
            lines.append("")
```

Add `format_doc_index_stats`:

```python
def format_doc_index_stats(stats: DocIndexStats) -> str:
    return f"Scanned {stats.files_scanned} files. Indexed {stats.files_indexed} files ({stats.chunks_created} chunks)."
```

Update import at top of formatters.py:
```python
from devrag.types import DocIndexStats, IndexStats, PRSyncStats, SearchResult
```

- [ ] **Step 4: Update MCP server — add index_docs tool and docs scope**

Add imports to `devrag/mcp_server.py`:
```python
from devrag.ingest.doc_indexer import DocIndexer
from devrag.utils.formatters import format_doc_index_stats
```

Add `index_docs` tool:
```python
@mcp.tool
def index_docs(path: str, glob: str = "**/*.md") -> str:
    """Index a directory of documents for search.

    Supports Markdown, text, RST, HTML, and AsciiDoc files.
    Splits documents by section headings for precise retrieval.
    """
    docs_path = Path(path).resolve()
    if not docs_path.exists():
        return f"Error: path '{path}' does not exist."

    glob_patterns = [g.strip() for g in glob.split(",")]
    indexer = DocIndexer(
        vector_store=_get_vector_store(),
        metadata_db=_get_metadata_db(),
        embedder=_get_embedder(),
        config=_get_config(),
    )
    stats = indexer.index_docs(docs_path, glob_patterns=glob_patterns)
    return format_doc_index_stats(stats)
```

Update `status` tool to include document count:
```python
@mcp.tool
def status() -> str:
    """Show indexing status: files, code chunks, PRs, and documents."""
    store = _get_vector_store()
    meta = _get_metadata_db()
    chunk_count = store.count("code_chunks")
    pr_diff_count = store.count("pr_diffs")
    pr_disc_count = store.count("pr_discussions")
    doc_count = store.count("documents")
    indexed_files = meta.get_all_indexed_files()
    lines = [
        f"Indexed files: {len(indexed_files)}",
        f"Code chunks: {chunk_count}",
        f"PR diff chunks: {pr_diff_count}",
        f"PR discussion chunks: {pr_disc_count}",
        f"Document chunks: {doc_count}",
    ]
    return "\n".join(lines)
```

- [ ] **Step 5: Verify MCP server imports cleanly**

Run: `uv run python -c "from devrag.mcp_server import mcp; print(f'Server: {mcp.name}')"`
Expected: `Server: DevRAG`

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_formatters.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add devrag/utils/formatters.py tests/test_formatters.py devrag/mcp_server.py
git commit -m "feat: document formatting, index_docs MCP tool, and docs scope"
```

---

## Task 6: Standalone CLI

**Files:**
- Create: `devrag/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli.py
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from devrag.cli import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "search" in result.stdout.lower() or "Search" in result.stdout


def test_cli_search_help():
    result = runner.invoke(app, ["search", "--help"])
    assert result.exit_code == 0
    assert "query" in result.stdout.lower()
    assert "scope" in result.stdout.lower()


@patch("devrag.cli._get_search_components")
def test_cli_search(mock_get):
    mock_hybrid = MagicMock()
    mock_hybrid.search.return_value = []
    mock_reranker = MagicMock()
    mock_get.return_value = (mock_hybrid, mock_reranker, MagicMock())
    result = runner.invoke(app, ["search", "how does auth work"])
    assert result.exit_code == 0


def test_cli_status_help():
    result = runner.invoke(app, ["status", "--help"])
    assert result.exit_code == 0


def test_cli_index_repo_help():
    result = runner.invoke(app, ["index", "repo", "--help"])
    assert result.exit_code == 0
    assert "path" in result.stdout.lower()


def test_cli_index_docs_help():
    result = runner.invoke(app, ["index", "docs", "--help"])
    assert result.exit_code == 0


def test_cli_index_prs_help():
    result = runner.invoke(app, ["index", "prs", "--help"])
    assert result.exit_code == 0


def test_cli_config_help():
    result = runner.invoke(app, ["config", "--help"])
    assert result.exit_code == 0


def test_cli_serve_help():
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write CLI implementation**

```python
# devrag/cli.py
from __future__ import annotations

import os
from pathlib import Path

import typer

app = typer.Typer(name="devrag", help="Local RAG system for developer teams.")
index_app = typer.Typer(help="Index code, docs, or PRs.")
config_app = typer.Typer(help="Manage configuration.")
app.add_typer(index_app, name="index")
app.add_typer(config_app, name="config")


def _get_search_components():
    """Lazy-initialize search components. Separated for testability."""
    from devrag.config import load_config
    from devrag.ingest.embedder import OllamaEmbedder
    from devrag.retrieve.hybrid_search import HybridSearch
    from devrag.retrieve.query_router import QueryRouter
    from devrag.retrieve.reranker import Reranker
    from devrag.stores.chroma_store import ChromaStore
    from devrag.stores.metadata_db import MetadataDB

    config = load_config(project_dir=Path.cwd())
    persist_dir = Path(config.vector_store.persist_dir).expanduser()
    persist_dir.mkdir(parents=True, exist_ok=True)
    store = ChromaStore(persist_dir=str(persist_dir))
    db_dir = Path("~/.local/share/devrag").expanduser()
    db_dir.mkdir(parents=True, exist_ok=True)
    meta = MetadataDB(str(db_dir / "metadata.db"))
    embedder = OllamaEmbedder(
        model=config.embedding.model,
        ollama_url=config.embedding.ollama_url,
        batch_size=config.embedding.batch_size,
    )
    hybrid = HybridSearch(store, meta, embedder)
    reranker = Reranker(model_name=config.retrieval.reranker_model) if config.retrieval.rerank else None
    return hybrid, reranker, config


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query text"),
    scope: str = typer.Option("all", help="Scope: all, code, prs, docs"),
    top_k: int = typer.Option(5, help="Number of results"),
):
    """Search code, PRs, and docs."""
    from devrag.retrieve.query_router import QueryRouter
    from devrag.utils.formatters import format_search_results

    hybrid, reranker, config = _get_search_components()
    router = QueryRouter()
    collections = router.route(query, scope=scope)
    candidates = hybrid.search(query, top_k=config.retrieval.top_k, collections=collections)

    if reranker and candidates:
        results = reranker.rerank(query, candidates, top_k=top_k)
    else:
        results = candidates[:top_k]

    typer.echo(format_search_results(results))


@index_app.command("repo")
def index_repo(
    path: str = typer.Argument(".", help="Path to repository"),
    full: bool = typer.Option(False, "--full", help="Full re-index (skip incremental)"),
):
    """Index a local code repository."""
    from devrag.config import load_config
    from devrag.ingest.code_indexer import CodeIndexer
    from devrag.ingest.embedder import OllamaEmbedder
    from devrag.stores.chroma_store import ChromaStore
    from devrag.stores.metadata_db import MetadataDB
    from devrag.utils.formatters import format_index_stats

    config = load_config(project_dir=Path.cwd())
    persist_dir = Path(config.vector_store.persist_dir).expanduser()
    persist_dir.mkdir(parents=True, exist_ok=True)
    store = ChromaStore(persist_dir=str(persist_dir))
    db_dir = Path("~/.local/share/devrag").expanduser()
    db_dir.mkdir(parents=True, exist_ok=True)
    meta = MetadataDB(str(db_dir / "metadata.db"))
    embedder = OllamaEmbedder(
        model=config.embedding.model,
        ollama_url=config.embedding.ollama_url,
        batch_size=config.embedding.batch_size,
    )
    indexer = CodeIndexer(store, meta, embedder, config)
    stats = indexer.index_repo(Path(path).resolve(), incremental=not full)
    typer.echo(format_index_stats(stats))


@index_app.command("docs")
def index_docs(
    path: str = typer.Argument(..., help="Path to docs directory"),
    glob: str = typer.Option("**/*.md", help="Glob pattern(s), comma-separated"),
):
    """Index a directory of documents."""
    from devrag.config import load_config
    from devrag.ingest.doc_indexer import DocIndexer
    from devrag.ingest.embedder import OllamaEmbedder
    from devrag.stores.chroma_store import ChromaStore
    from devrag.stores.metadata_db import MetadataDB
    from devrag.utils.formatters import format_doc_index_stats

    config = load_config(project_dir=Path.cwd())
    persist_dir = Path(config.vector_store.persist_dir).expanduser()
    persist_dir.mkdir(parents=True, exist_ok=True)
    store = ChromaStore(persist_dir=str(persist_dir))
    db_dir = Path("~/.local/share/devrag").expanduser()
    db_dir.mkdir(parents=True, exist_ok=True)
    meta = MetadataDB(str(db_dir / "metadata.db"))
    embedder = OllamaEmbedder(
        model=config.embedding.model,
        ollama_url=config.embedding.ollama_url,
        batch_size=config.embedding.batch_size,
    )
    glob_patterns = [g.strip() for g in glob.split(",")]
    indexer = DocIndexer(store, meta, embedder, config)
    stats = indexer.index_docs(Path(path).resolve(), glob_patterns=glob_patterns)
    typer.echo(format_doc_index_stats(stats))


@index_app.command("prs")
def index_prs(
    repo: str = typer.Argument(..., help="GitHub repo (owner/name)"),
    since: str = typer.Option("90d", help="Lookback period (e.g. 90d)"),
):
    """Sync GitHub PRs for a repository."""
    from devrag.config import load_config
    from devrag.ingest.embedder import OllamaEmbedder
    from devrag.ingest.pr_indexer import PRIndexer
    from devrag.stores.chroma_store import ChromaStore
    from devrag.stores.metadata_db import MetadataDB
    from devrag.utils.formatters import format_pr_sync_stats
    from devrag.utils.github import GitHubClient

    config = load_config(project_dir=Path.cwd())
    token = os.environ.get(config.prs.github_token_env)
    if not token:
        typer.echo(f"Error: {config.prs.github_token_env} environment variable not set.", err=True)
        raise typer.Exit(1)

    persist_dir = Path(config.vector_store.persist_dir).expanduser()
    persist_dir.mkdir(parents=True, exist_ok=True)
    store = ChromaStore(persist_dir=str(persist_dir))
    db_dir = Path("~/.local/share/devrag").expanduser()
    db_dir.mkdir(parents=True, exist_ok=True)
    meta = MetadataDB(str(db_dir / "metadata.db"))
    embedder = OllamaEmbedder(
        model=config.embedding.model,
        ollama_url=config.embedding.ollama_url,
        batch_size=config.embedding.batch_size,
    )

    # Parse "90d" → 90
    days = int(since.rstrip("d"))
    github = GitHubClient(token=token)
    indexer = PRIndexer(store, meta, embedder, github)
    stats = indexer.sync(repo, since_days=days)
    typer.echo(format_pr_sync_stats(stats))


@app.command()
def status():
    """Show indexing status."""
    from devrag.config import load_config
    from devrag.stores.chroma_store import ChromaStore
    from devrag.stores.metadata_db import MetadataDB

    config = load_config(project_dir=Path.cwd())
    persist_dir = Path(config.vector_store.persist_dir).expanduser()
    if not persist_dir.exists():
        typer.echo("No index found. Run 'devrag index repo .' first.")
        return
    store = ChromaStore(persist_dir=str(persist_dir))
    db_dir = Path("~/.local/share/devrag").expanduser()
    db_dir.mkdir(parents=True, exist_ok=True)
    meta = MetadataDB(str(db_dir / "metadata.db"))
    indexed_files = meta.get_all_indexed_files()
    lines = [
        f"Indexed files: {len(indexed_files)}",
        f"Code chunks: {store.count('code_chunks')}",
        f"PR diff chunks: {store.count('pr_diffs')}",
        f"PR discussion chunks: {store.count('pr_discussions')}",
        f"Document chunks: {store.count('documents')}",
    ]
    typer.echo("\n".join(lines))


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Config key (e.g. embedding.model)"),
    value: str = typer.Argument(..., help="Config value"),
):
    """Set a config value in project .devrag.yaml."""
    import yaml

    config_path = Path.cwd() / ".devrag.yaml"
    data: dict = {}
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

    # Navigate nested keys
    keys = key.split(".")
    current = data
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value

    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    typer.echo(f"Set {key} = {value}")


@config_app.command("get")
def config_get(
    key: str = typer.Argument(..., help="Config key (e.g. embedding.model)"),
):
    """Get a config value."""
    from devrag.config import load_config
    config = load_config(project_dir=Path.cwd())

    keys = key.split(".")
    current = config
    for k in keys:
        if hasattr(current, k):
            current = getattr(current, k)
        else:
            typer.echo(f"Unknown key: {key}", err=True)
            raise typer.Exit(1)
    typer.echo(f"{key} = {current}")


@app.command()
def serve():
    """Start the MCP server."""
    from devrag.mcp_server import mcp
    mcp.run()


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: All 9 tests PASS.

- [ ] **Step 5: Verify CLI runs**

Run: `uv run devrag --help`
Expected: Shows help with search, index, status, config, serve commands.

- [ ] **Step 6: Commit**

```bash
git add devrag/cli.py tests/test_cli.py
git commit -m "feat: standalone CLI with search, index, status, config, and serve commands"
```

---

## Spec Coverage Checklist

| Spec Section | Task(s) | Status |
|---|---|---|
| 3.1 Doc Indexer (recursive splitting, section-aware) | Task 2 | Covered |
| 3.1 Heading hierarchy as metadata | Task 2 | Covered |
| 3.1 Supported formats (.md, .mdx, .txt, .rst, .html, .adoc) | Task 2 | Covered |
| 3.1 New collection: documents | Task 2 | Covered |
| 3.2 Full Query Router (docs patterns) | Task 3 | Covered |
| 3.3 CLI: search, index repo/docs/prs, status, config, serve | Task 6 | Covered |
| 3.4 Config system (DocumentsConfig) | Task 1 | Covered |
| 3.5 .devragignore | Task 4 | Covered |
| MCP: index_docs tool | Task 5 | Covered |
| MCP: docs scope on search | Task 5 | Covered |
| Formatters: document results | Task 5 | Covered |
| Dependencies: typer | Task 1 | Covered |
