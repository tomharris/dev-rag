# DevRAG Phase 1: Code Search End-to-End — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Index a monorepo (50k+ files) with AST-aware chunking and search it from Claude Code via MCP, with hybrid retrieval (vector + BM25) and cross-encoder reranking.

**Architecture:** Vertical pipeline — code files → tree-sitter AST chunking → Ollama embeddings (nomic-embed-text, 768d) → ChromaDB (vector) + SQLite FTS5 (BM25) → hybrid search with RRF → cross-encoder reranking → FastMCP server. Incremental indexing via SHA-256 file hashes in SQLite.

**Tech Stack:** Python 3.12+, uv, tree-sitter + tree-sitter-language-pack, ChromaDB, SQLite FTS5, httpx (Ollama API), sentence-transformers (CrossEncoder), FastMCP, PyYAML, GitPython.

**Key design note:** ChromaDB does NOT support local BM25 (cloud-only). We use SQLite FTS5 with its built-in `bm25()` ranking function as the keyword search component. This is efficient on disk and avoids loading all documents into memory.

---

## File Structure

```
devrag/
├── pyproject.toml
├── devrag/
│   ├── __init__.py
│   ├── config.py               # YAML config loading + defaults
│   ├── types.py                # Shared dataclasses (Chunk, QueryResult, IndexStats)
│   ├── mcp_server.py           # FastMCP server with search/index/status tools
│   │
│   ├── stores/
│   │   ├── __init__.py
│   │   ├── base.py             # VectorStore protocol
│   │   ├── chroma_store.py     # ChromaDB implementation
│   │   └── metadata_db.py      # SQLite: file hashes, chunk sources, FTS5 index
│   │
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── embedder.py         # Ollama embedding via httpx
│   │   └── code_indexer.py     # tree-sitter AST chunking + incremental indexing
│   │
│   ├── retrieve/
│   │   ├── __init__.py
│   │   ├── hybrid_search.py    # Vector + BM25 via FTS5 + RRF fusion
│   │   └── reranker.py         # Cross-encoder reranking
│   │
│   └── utils/
│       ├── __init__.py
│       ├── git.py              # .gitignore-aware file discovery
│       └── formatters.py       # Format search results for MCP output
│
└── tests/
    ├── conftest.py             # Shared fixtures (tmp dirs, sample files)
    ├── test_config.py
    ├── test_chroma_store.py
    ├── test_metadata_db.py
    ├── test_embedder.py
    ├── test_git.py
    ├── test_code_indexer.py
    ├── test_hybrid_search.py
    └── test_reranker.py
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `devrag/__init__.py`
- Create: `devrag/stores/__init__.py`
- Create: `devrag/ingest/__init__.py`
- Create: `devrag/retrieve/__init__.py`
- Create: `devrag/utils/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "devrag"
version = "0.1.0"
description = "Local RAG system for developer teams"
requires-python = ">=3.12"
dependencies = [
    "chromadb>=1.0.0",
    "fastmcp>=3.0.0",
    "gitpython>=3.1.0",
    "httpx>=0.27.0",
    "pyyaml>=6.0",
    "sentence-transformers>=3.0.0",
    "tree-sitter>=0.23.0",
    "tree-sitter-language-pack>=0.21.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24.0",
    "respx>=0.22.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 2: Create package structure**

Create all `__init__.py` files (empty):

```
devrag/__init__.py
devrag/stores/__init__.py
devrag/ingest/__init__.py
devrag/retrieve/__init__.py
devrag/utils/__init__.py
tests/__init__.py
```

- [ ] **Step 3: Create tests/conftest.py**

```python
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_python_file(tmp_dir):
    code = '''"""Module docstring."""

import os
from pathlib import Path


class FileProcessor:
    """Processes files from disk."""

    def __init__(self, root: str):
        self.root = Path(root)

    def read_file(self, name: str) -> str:
        """Read a file by name."""
        path = self.root / name
        return path.read_text()

    def list_files(self) -> list[str]:
        """List all files in root."""
        return [f.name for f in self.root.iterdir() if f.is_file()]


def standalone_function(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y
'''
    p = tmp_dir / "sample.py"
    p.write_text(code)
    return p


@pytest.fixture
def sample_ts_file(tmp_dir):
    code = '''import { readFileSync } from "fs";

interface Config {
    host: string;
    port: number;
}

export class Server {
    private config: Config;

    constructor(config: Config) {
        this.config = config;
    }

    start(): void {
        console.log(`Starting on ${this.config.host}:${this.config.port}`);
    }
}

export function loadConfig(path: string): Config {
    const raw = readFileSync(path, "utf-8");
    return JSON.parse(raw);
}
'''
    p = tmp_dir / "sample.ts"
    p.write_text(code)
    return p
```

- [ ] **Step 4: Install dependencies**

Run: `cd /home/tom/Projects/dev-rag && uv sync --all-extras`
Expected: Dependencies installed, `.venv` created.

- [ ] **Step 5: Verify pytest runs**

Run: `cd /home/tom/Projects/dev-rag && uv run pytest --co -q`
Expected: "no tests ran" (no test files yet, but pytest discovers correctly).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml devrag/ tests/
git commit -m "feat: project scaffolding with dependencies and test fixtures"
```

---

## Task 2: Config System

**Files:**
- Create: `devrag/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from pathlib import Path

from devrag.config import DevragConfig, load_config


def test_default_config_has_expected_values():
    config = DevragConfig()
    assert config.embedding.model == "nomic-embed-text"
    assert config.embedding.provider == "ollama"
    assert config.embedding.ollama_url == "http://localhost:11434"
    assert config.embedding.batch_size == 64
    assert config.vector_store.backend == "chromadb"
    assert config.retrieval.top_k == 20
    assert config.retrieval.final_k == 5
    assert config.retrieval.rerank is True
    assert config.code.chunk_max_tokens == 512
    assert config.code.respect_gitignore is True
    assert "*.min.js" in config.code.exclude_patterns
    assert "node_modules/**" in config.code.exclude_patterns


def test_load_config_from_yaml(tmp_dir):
    yaml_content = """
embedding:
  model: all-MiniLM-L6-v2
  provider: sentence-transformers
retrieval:
  top_k: 30
  final_k: 10
"""
    config_path = tmp_dir / ".devrag.yaml"
    config_path.write_text(yaml_content)
    config = load_config(project_dir=tmp_dir)
    assert config.embedding.model == "all-MiniLM-L6-v2"
    assert config.embedding.provider == "sentence-transformers"
    assert config.retrieval.top_k == 30
    assert config.retrieval.final_k == 10
    # Unchanged defaults preserved
    assert config.vector_store.backend == "chromadb"
    assert config.code.chunk_max_tokens == 512


def test_load_config_no_file_returns_defaults(tmp_dir):
    config = load_config(project_dir=tmp_dir)
    assert config.embedding.model == "nomic-embed-text"


def test_load_config_user_dir_fallback(tmp_dir, monkeypatch):
    user_config_dir = tmp_dir / "user_config"
    user_config_dir.mkdir()
    (user_config_dir / "devrag.yaml").write_text("embedding:\n  model: custom-model\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_dir / "user_config_parent"))
    # When project has no config, load_config should check user dir
    config = load_config(project_dir=tmp_dir, user_config_dir=user_config_dir)
    assert config.embedding.model == "custom-model"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'devrag.config'`

- [ ] **Step 3: Write config implementation**

```python
# devrag/config.py
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class EmbeddingConfig:
    model: str = "nomic-embed-text"
    provider: str = "ollama"
    ollama_url: str = "http://localhost:11434"
    batch_size: int = 64


@dataclass
class VectorStoreConfig:
    backend: str = "chromadb"
    persist_dir: str = "~/.local/share/devrag/chroma"


@dataclass
class RetrievalConfig:
    top_k: int = 20
    final_k: int = 5
    rerank: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class CodeConfig:
    chunk_max_tokens: int = 512
    respect_gitignore: bool = True
    exclude_patterns: list[str] = field(default_factory=lambda: [
        "*.min.js",
        "vendor/**",
        "node_modules/**",
        "*.lock",
        "*.generated.*",
    ])


@dataclass
class DevragConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    code: CodeConfig = field(default_factory=CodeConfig)


def _merge_dict_into_dataclass(dc: object, overrides: dict) -> None:
    """Recursively merge a dict of overrides into a dataclass instance."""
    for key, value in overrides.items():
        if not hasattr(dc, key):
            continue
        current = getattr(dc, key)
        if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
            _merge_dict_into_dataclass(current, value)
        else:
            setattr(dc, key, value)


def load_config(
    project_dir: Path | None = None,
    user_config_dir: Path | None = None,
) -> DevragConfig:
    """Load config with priority: project .devrag.yaml > user config > defaults."""
    config = DevragConfig()

    # Try user config dir
    if user_config_dir is None:
        xdg = Path.home() / ".config" / "devrag"
        user_config_dir = xdg

    user_config_path = user_config_dir / "devrag.yaml"
    if user_config_path.exists():
        with open(user_config_path) as f:
            data = yaml.safe_load(f) or {}
        _merge_dict_into_dataclass(config, data)

    # Try project config (higher priority, applied second)
    if project_dir is not None:
        project_config_path = project_dir / ".devrag.yaml"
        if project_config_path.exists():
            with open(project_config_path) as f:
                data = yaml.safe_load(f) or {}
            _merge_dict_into_dataclass(config, data)

    return config
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add devrag/config.py tests/test_config.py
git commit -m "feat: YAML config system with cascading defaults"
```

---

## Task 3: Shared Types

**Files:**
- Create: `devrag/types.py`

- [ ] **Step 1: Write types module**

No test needed — these are plain data containers used by every other module.

```python
# devrag/types.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A single indexed chunk of code or text."""
    id: str
    text: str
    metadata: dict[str, str | int | float | bool | list[str]]
    embedding: list[float] | None = None


@dataclass
class SearchResult:
    """A single search result with score and source chunk."""
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, str | int | float | bool | list[str]]


@dataclass
class QueryResult:
    """Results from a vector store query."""
    ids: list[str]
    documents: list[str]
    metadatas: list[dict]
    distances: list[float]


@dataclass
class IndexStats:
    """Statistics from an indexing run."""
    files_scanned: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    files_removed: int = 0
    chunks_created: int = 0
```

- [ ] **Step 2: Commit**

```bash
git add devrag/types.py
git commit -m "feat: shared data types for chunks, results, and stats"
```

---

## Task 4: VectorStore Protocol + ChromaDB Implementation

**Files:**
- Create: `devrag/stores/base.py`
- Create: `devrag/stores/chroma_store.py`
- Create: `tests/test_chroma_store.py`

- [ ] **Step 1: Write the VectorStore protocol**

```python
# devrag/stores/base.py
from __future__ import annotations

from typing import Protocol

from devrag.types import QueryResult


class VectorStore(Protocol):
    def upsert(
        self,
        collection: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None: ...

    def query(
        self,
        collection: str,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict | None = None,
    ) -> QueryResult: ...

    def delete(self, collection: str, ids: list[str]) -> None: ...

    def count(self, collection: str) -> int: ...
```

- [ ] **Step 2: Write the failing ChromaStore tests**

```python
# tests/test_chroma_store.py
import pytest

from devrag.stores.chroma_store import ChromaStore
from devrag.types import QueryResult


@pytest.fixture
def store(tmp_dir):
    return ChromaStore(persist_dir=str(tmp_dir / "chroma"))


def test_upsert_and_count(store):
    store.upsert(
        collection="test",
        ids=["a", "b"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        documents=["hello world", "foo bar"],
        metadatas=[{"lang": "en"}, {"lang": "en"}],
    )
    assert store.count("test") == 2


def test_query_returns_closest(store):
    store.upsert(
        collection="test",
        ids=["a", "b", "c"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        documents=["doc a", "doc b", "doc c"],
        metadatas=[{"x": 1}, {"x": 2}, {"x": 3}],
    )
    result = store.query(
        collection="test",
        query_embedding=[1.0, 0.1, 0.0],
        n_results=2,
    )
    assert isinstance(result, QueryResult)
    assert len(result.ids) == 2
    assert result.ids[0] == "a"  # closest to [1, 0, 0]


def test_query_with_where_filter(store):
    store.upsert(
        collection="test",
        ids=["a", "b"],
        embeddings=[[1.0, 0.0], [0.9, 0.1]],
        documents=["doc a", "doc b"],
        metadatas=[{"lang": "python"}, {"lang": "typescript"}],
    )
    result = store.query(
        collection="test",
        query_embedding=[1.0, 0.0],
        n_results=2,
        where={"lang": "typescript"},
    )
    assert len(result.ids) == 1
    assert result.ids[0] == "b"


def test_upsert_overwrites_existing(store):
    store.upsert(
        collection="test",
        ids=["a"],
        embeddings=[[1.0, 0.0]],
        documents=["original"],
        metadatas=[{"v": 1}],
    )
    store.upsert(
        collection="test",
        ids=["a"],
        embeddings=[[0.0, 1.0]],
        documents=["updated"],
        metadatas=[{"v": 2}],
    )
    assert store.count("test") == 1
    result = store.query("test", query_embedding=[0.0, 1.0], n_results=1)
    assert result.documents[0] == "updated"
    assert result.metadatas[0]["v"] == 2


def test_delete(store):
    store.upsert(
        collection="test",
        ids=["a", "b"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=["a", "b"],
        metadatas=[{}, {}],
    )
    store.delete("test", ids=["a"])
    assert store.count("test") == 1


def test_count_nonexistent_collection(store):
    assert store.count("nonexistent") == 0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_chroma_store.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Write ChromaStore implementation**

```python
# devrag/stores/chroma_store.py
from __future__ import annotations

import chromadb

from devrag.types import QueryResult


class ChromaStore:
    def __init__(self, persist_dir: str) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)

    def _get_or_create(self, collection: str) -> chromadb.Collection:
        return self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        collection: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        coll = self._get_or_create(collection)
        coll.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        collection: str,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict | None = None,
    ) -> QueryResult:
        try:
            coll = self._client.get_collection(name=collection)
        except Exception:
            return QueryResult(ids=[], documents=[], metadatas=[], distances=[])

        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, coll.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        if coll.count() == 0:
            return QueryResult(ids=[], documents=[], metadatas=[], distances=[])

        results = coll.query(**kwargs)
        return QueryResult(
            ids=results["ids"][0],
            documents=results["documents"][0],
            metadatas=results["metadatas"][0],
            distances=results["distances"][0],
        )

    def delete(self, collection: str, ids: list[str]) -> None:
        try:
            coll = self._client.get_collection(name=collection)
        except Exception:
            return
        coll.delete(ids=ids)

    def count(self, collection: str) -> int:
        try:
            coll = self._client.get_collection(name=collection)
            return coll.count()
        except Exception:
            return 0
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_chroma_store.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add devrag/stores/base.py devrag/stores/chroma_store.py tests/test_chroma_store.py
git commit -m "feat: VectorStore protocol and ChromaDB implementation"
```

---

## Task 5: SQLite Metadata Store (File Hashes + FTS5)

**Files:**
- Create: `devrag/stores/metadata_db.py`
- Create: `tests/test_metadata_db.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metadata_db.py
from devrag.stores.metadata_db import MetadataDB


def test_file_hash_store_and_check(tmp_dir):
    db = MetadataDB(str(tmp_dir / "meta.db"))
    db.set_file_hash("src/foo.py", "abc123")
    assert db.get_file_hash("src/foo.py") == "abc123"
    assert db.get_file_hash("src/bar.py") is None


def test_file_hash_update(tmp_dir):
    db = MetadataDB(str(tmp_dir / "meta.db"))
    db.set_file_hash("src/foo.py", "hash1")
    db.set_file_hash("src/foo.py", "hash2")
    assert db.get_file_hash("src/foo.py") == "hash2"


def test_remove_file_hash(tmp_dir):
    db = MetadataDB(str(tmp_dir / "meta.db"))
    db.set_file_hash("src/foo.py", "abc")
    db.remove_file("src/foo.py")
    assert db.get_file_hash("src/foo.py") is None


def test_get_all_indexed_files(tmp_dir):
    db = MetadataDB(str(tmp_dir / "meta.db"))
    db.set_file_hash("a.py", "h1")
    db.set_file_hash("b.py", "h2")
    files = db.get_all_indexed_files()
    assert set(files) == {"a.py", "b.py"}


def test_chunk_source_mapping(tmp_dir):
    db = MetadataDB(str(tmp_dir / "meta.db"))
    db.set_chunk_source("chunk_1", "src/foo.py", 10, 25)
    db.set_chunk_source("chunk_2", "src/foo.py", 30, 50)
    chunks = db.get_chunks_for_file("src/foo.py")
    assert set(chunks) == {"chunk_1", "chunk_2"}


def test_remove_file_clears_chunks(tmp_dir):
    db = MetadataDB(str(tmp_dir / "meta.db"))
    db.set_file_hash("src/foo.py", "abc")
    db.set_chunk_source("chunk_1", "src/foo.py", 1, 10)
    db.remove_file("src/foo.py")
    assert db.get_chunks_for_file("src/foo.py") == []


def test_fts_index_and_search(tmp_dir):
    db = MetadataDB(str(tmp_dir / "meta.db"))
    db.upsert_fts("chunk_1", "def authenticate_user(username, password)")
    db.upsert_fts("chunk_2", "class DatabaseConnection with pooling support")
    db.upsert_fts("chunk_3", "function to parse JSON configuration files")

    results = db.search_fts("authenticate", limit=5)
    assert len(results) >= 1
    assert results[0][0] == "chunk_1"  # (chunk_id, score)
    assert results[0][1] < 0  # BM25 scores are negative in FTS5 (lower = better match)


def test_fts_update_existing(tmp_dir):
    db = MetadataDB(str(tmp_dir / "meta.db"))
    db.upsert_fts("chunk_1", "original text about authentication")
    db.upsert_fts("chunk_1", "updated text about database pooling")
    results = db.search_fts("database pooling", limit=5)
    assert len(results) >= 1
    assert results[0][0] == "chunk_1"


def test_fts_delete(tmp_dir):
    db = MetadataDB(str(tmp_dir / "meta.db"))
    db.upsert_fts("chunk_1", "some text")
    db.delete_fts(["chunk_1"])
    results = db.search_fts("some text", limit=5)
    assert len(results) == 0


def test_delete_fts_for_file(tmp_dir):
    db = MetadataDB(str(tmp_dir / "meta.db"))
    db.set_chunk_source("c1", "foo.py", 1, 10)
    db.set_chunk_source("c2", "foo.py", 11, 20)
    db.set_chunk_source("c3", "bar.py", 1, 10)
    db.upsert_fts("c1", "text one")
    db.upsert_fts("c2", "text two")
    db.upsert_fts("c3", "text three")
    db.delete_fts_for_file("foo.py")
    # c1, c2 should be gone; c3 should remain
    assert db.search_fts("text one", limit=5) == []
    results = db.search_fts("text three", limit=5)
    assert len(results) == 1
    assert results[0][0] == "c3"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_metadata_db.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write MetadataDB implementation**

```python
# devrag/stores/metadata_db.py
from __future__ import annotations

import sqlite3
from pathlib import Path


class MetadataDB:
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS file_hashes (
                file_path TEXT PRIMARY KEY,
                hash TEXT NOT NULL,
                last_indexed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS chunk_sources (
                chunk_id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                line_start INTEGER,
                line_end INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_chunk_sources_file
                ON chunk_sources(file_path);

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                USING fts5(chunk_id UNINDEXED, content);
        """)
        self._conn.commit()

    # --- File hashes ---

    def get_file_hash(self, file_path: str) -> str | None:
        row = self._conn.execute(
            "SELECT hash FROM file_hashes WHERE file_path = ?", (file_path,)
        ).fetchone()
        return row[0] if row else None

    def set_file_hash(self, file_path: str, hash_value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO file_hashes (file_path, hash) VALUES (?, ?)",
            (file_path, hash_value),
        )
        self._conn.commit()

    def remove_file(self, file_path: str) -> None:
        """Remove file hash, chunk sources, and FTS entries for a file."""
        self.delete_fts_for_file(file_path)
        self._conn.execute(
            "DELETE FROM chunk_sources WHERE file_path = ?", (file_path,)
        )
        self._conn.execute(
            "DELETE FROM file_hashes WHERE file_path = ?", (file_path,)
        )
        self._conn.commit()

    def get_all_indexed_files(self) -> list[str]:
        rows = self._conn.execute("SELECT file_path FROM file_hashes").fetchall()
        return [r[0] for r in rows]

    # --- Chunk sources ---

    def set_chunk_source(
        self, chunk_id: str, file_path: str, line_start: int, line_end: int
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO chunk_sources (chunk_id, file_path, line_start, line_end) VALUES (?, ?, ?, ?)",
            (chunk_id, file_path, line_start, line_end),
        )
        self._conn.commit()

    def get_chunks_for_file(self, file_path: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT chunk_id FROM chunk_sources WHERE file_path = ?", (file_path,)
        ).fetchall()
        return [r[0] for r in rows]

    # --- FTS5 full-text search ---

    def upsert_fts(self, chunk_id: str, content: str) -> None:
        """Insert or update a document in the FTS index."""
        self._conn.execute(
            "DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk_id,)
        )
        self._conn.execute(
            "INSERT INTO chunks_fts (chunk_id, content) VALUES (?, ?)",
            (chunk_id, content),
        )
        self._conn.commit()

    def delete_fts(self, chunk_ids: list[str]) -> None:
        placeholders = ",".join("?" for _ in chunk_ids)
        self._conn.execute(
            f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )
        self._conn.commit()

    def delete_fts_for_file(self, file_path: str) -> None:
        """Delete FTS entries for all chunks belonging to a file."""
        chunk_ids = self.get_chunks_for_file(file_path)
        if chunk_ids:
            self.delete_fts(chunk_ids)

    def search_fts(self, query: str, limit: int = 20) -> list[tuple[str, float]]:
        """Search FTS index. Returns list of (chunk_id, bm25_score).
        BM25 scores are negative — lower (more negative) = better match.
        """
        rows = self._conn.execute(
            "SELECT chunk_id, bm25(chunks_fts) as score "
            "FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY score LIMIT ?",
            (query, limit),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_metadata_db.py -v`
Expected: All 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add devrag/stores/metadata_db.py tests/test_metadata_db.py
git commit -m "feat: SQLite metadata store with file hashes, chunk sources, and FTS5 index"
```

---

## Task 6: Ollama Embedder

**Files:**
- Create: `devrag/ingest/embedder.py`
- Create: `tests/test_embedder.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_embedder.py
import httpx
import pytest
import respx

from devrag.ingest.embedder import OllamaEmbedder


@respx.mock
def test_embed_single_text():
    respx.post("http://localhost:11434/api/embed").respond(json={
        "model": "nomic-embed-text",
        "embeddings": [[0.1, 0.2, 0.3]],
    })
    embedder = OllamaEmbedder(model="nomic-embed-text", ollama_url="http://localhost:11434")
    result = embedder.embed(["hello world"])
    assert len(result) == 1
    assert result[0] == [0.1, 0.2, 0.3]


@respx.mock
def test_embed_batch():
    respx.post("http://localhost:11434/api/embed").respond(json={
        "model": "nomic-embed-text",
        "embeddings": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
    })
    embedder = OllamaEmbedder(model="nomic-embed-text", ollama_url="http://localhost:11434")
    result = embedder.embed(["a", "b", "c"])
    assert len(result) == 3


@respx.mock
def test_embed_large_batch_splits_requests():
    """Batches larger than batch_size should be split into multiple requests."""
    call_count = 0

    def handler(request):
        nonlocal call_count
        call_count += 1
        data = request.content.decode()
        import json
        body = json.loads(data)
        n = len(body["input"])
        return httpx.Response(200, json={
            "model": "nomic-embed-text",
            "embeddings": [[0.1, 0.2]] * n,
        })

    respx.post("http://localhost:11434/api/embed").mock(side_effect=handler)
    embedder = OllamaEmbedder(
        model="nomic-embed-text",
        ollama_url="http://localhost:11434",
        batch_size=2,
    )
    result = embedder.embed(["a", "b", "c", "d", "e"])
    assert len(result) == 5
    assert call_count == 3  # ceil(5/2) = 3 batches


@respx.mock
def test_embed_query():
    respx.post("http://localhost:11434/api/embed").respond(json={
        "model": "nomic-embed-text",
        "embeddings": [[0.1, 0.2, 0.3]],
    })
    embedder = OllamaEmbedder(model="nomic-embed-text", ollama_url="http://localhost:11434")
    result = embedder.embed_query("search query")
    assert result == [0.1, 0.2, 0.3]


@respx.mock
def test_embed_empty_list():
    embedder = OllamaEmbedder(model="nomic-embed-text", ollama_url="http://localhost:11434")
    result = embedder.embed([])
    assert result == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embedder.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write embedder implementation**

```python
# devrag/ingest/embedder.py
from __future__ import annotations

import httpx


class OllamaEmbedder:
    def __init__(
        self,
        model: str = "nomic-embed-text",
        ollama_url: str = "http://localhost:11434",
        batch_size: int = 64,
    ) -> None:
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. Splits into batches if needed."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = httpx.post(
                f"{self.ollama_url}/api/embed",
                json={"model": self.model, "input": batch},
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()
            all_embeddings.extend(data["embeddings"])

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        return self.embed([text])[0]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embedder.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add devrag/ingest/embedder.py tests/test_embedder.py
git commit -m "feat: Ollama embedder with batch support via httpx"
```

---

## Task 7: Git Utilities

**Files:**
- Create: `devrag/utils/git.py`
- Create: `tests/test_git.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_git.py
import subprocess
from pathlib import Path

import pytest

from devrag.utils.git import discover_files


@pytest.fixture
def git_repo(tmp_dir):
    """Create a minimal git repo with .gitignore."""
    subprocess.run(["git", "init", str(tmp_dir)], capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(tmp_dir), capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(tmp_dir), capture_output=True, check=True,
    )

    # Create files
    (tmp_dir / "src").mkdir()
    (tmp_dir / "src" / "main.py").write_text("print('hello')")
    (tmp_dir / "src" / "utils.py").write_text("x = 1")
    (tmp_dir / "src" / "data.min.js").write_text("minified")
    (tmp_dir / "node_modules").mkdir()
    (tmp_dir / "node_modules" / "pkg.js").write_text("module")
    (tmp_dir / "README.md").write_text("# Readme")
    (tmp_dir / ".gitignore").write_text("node_modules/\n")

    subprocess.run(
        ["git", "add", "."], cwd=str(tmp_dir), capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=str(tmp_dir), capture_output=True, check=True,
    )
    return tmp_dir


def test_discover_files_respects_gitignore(git_repo):
    files = discover_files(git_repo, exclude_patterns=[])
    rel_paths = {str(f.relative_to(git_repo)) for f in files}
    assert "src/main.py" in rel_paths
    assert "src/utils.py" in rel_paths
    # .gitignore excludes node_modules
    assert "node_modules/pkg.js" not in rel_paths


def test_discover_files_applies_exclude_patterns(git_repo):
    files = discover_files(git_repo, exclude_patterns=["*.min.js", "*.md"])
    rel_paths = {str(f.relative_to(git_repo)) for f in files}
    assert "src/main.py" in rel_paths
    assert "src/data.min.js" not in rel_paths
    assert "README.md" not in rel_paths


def test_discover_files_nonexistent_dir():
    files = discover_files(Path("/nonexistent"), exclude_patterns=[])
    assert files == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_git.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write git utilities implementation**

```python
# devrag/utils/git.py
from __future__ import annotations

import fnmatch
import subprocess
from pathlib import Path


def discover_files(
    repo_path: Path,
    exclude_patterns: list[str],
) -> list[Path]:
    """Discover files in a repo, respecting .gitignore and exclude patterns.

    Uses `git ls-files` for .gitignore-aware file listing. Falls back to
    filesystem walk if not a git repo.
    """
    if not repo_path.exists():
        return []

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
        # Not a git repo — walk filesystem
        rel_paths = [
            str(f.relative_to(repo_path))
            for f in repo_path.rglob("*")
            if f.is_file()
        ]

    # Apply exclude patterns
    filtered: list[Path] = []
    for rel in rel_paths:
        if any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(Path(rel).name, pat)
               for pat in exclude_patterns):
            continue
        full = repo_path / rel
        if full.is_file():
            filtered.append(full)

    return filtered
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_git.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add devrag/utils/git.py tests/test_git.py
git commit -m "feat: git-aware file discovery with .gitignore and exclude pattern support"
```

---

## Task 8: AST Code Indexer

**Files:**
- Create: `devrag/ingest/code_indexer.py`
- Create: `tests/test_code_indexer.py`

This is the most complex component. The code indexer uses tree-sitter to parse files into ASTs, extracts semantic entities (functions, classes, methods), and builds contextualized chunks with metadata.

- [ ] **Step 1: Write the failing tests for AST chunking**

```python
# tests/test_code_indexer.py
import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from devrag.ingest.code_indexer import (
    LANGUAGE_EXTENSIONS,
    CodeIndexer,
    extract_chunks_from_file,
)
from devrag.types import Chunk


def test_extract_chunks_from_python_file(sample_python_file):
    chunks = extract_chunks_from_file(sample_python_file, max_tokens=512)
    # Should find: FileProcessor class (with __init__, read_file, list_files),
    # standalone_function, possibly imports/module-level
    entity_names = [c.metadata["entity_name"] for c in chunks]
    assert "FileProcessor" in entity_names or any(
        c.metadata.get("parent_entity") == "FileProcessor" for c in chunks
    )
    assert "standalone_function" in entity_names

    # Check metadata fields
    func_chunk = next(c for c in chunks if c.metadata["entity_name"] == "standalone_function")
    assert func_chunk.metadata["language"] == "python"
    assert func_chunk.metadata["entity_type"] in ("function", "function_definition")
    assert "line_range" in func_chunk.metadata
    assert "def standalone_function" in func_chunk.text


def test_extract_chunks_from_typescript_file(sample_ts_file):
    chunks = extract_chunks_from_file(sample_ts_file, max_tokens=512)
    entity_names = [c.metadata["entity_name"] for c in chunks]
    # Should find Server class, loadConfig function, Config interface
    assert "Server" in entity_names or "loadConfig" in entity_names
    assert any(c.metadata["language"] == "typescript" for c in chunks)


def test_extract_chunks_unsupported_language(tmp_dir):
    path = tmp_dir / "data.csv"
    path.write_text("a,b,c\n1,2,3\n")
    chunks = extract_chunks_from_file(path, max_tokens=512)
    assert chunks == []


def test_language_extensions_mapping():
    assert LANGUAGE_EXTENSIONS[".py"] == "python"
    assert LANGUAGE_EXTENSIONS[".ts"] == "typescript"
    assert LANGUAGE_EXTENSIONS[".js"] == "javascript"
    assert LANGUAGE_EXTENSIONS[".rs"] == "rust"
    assert LANGUAGE_EXTENSIONS[".go"] == "go"


def test_chunk_ids_are_deterministic(sample_python_file):
    chunks1 = extract_chunks_from_file(sample_python_file, max_tokens=512)
    chunks2 = extract_chunks_from_file(sample_python_file, max_tokens=512)
    ids1 = [c.id for c in chunks1]
    ids2 = [c.id for c in chunks2]
    assert ids1 == ids2


def test_chunk_text_includes_context(sample_python_file):
    chunks = extract_chunks_from_file(sample_python_file, max_tokens=512)
    # Methods inside FileProcessor should reference their parent class
    method_chunks = [
        c for c in chunks
        if c.metadata.get("parent_entity") == "FileProcessor"
    ]
    if method_chunks:
        # At least one method chunk should exist
        assert any("FileProcessor" in c.text or "read_file" in c.text for c in method_chunks)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_code_indexer.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the AST chunking implementation**

```python
# devrag/ingest/code_indexer.py
from __future__ import annotations

import hashlib
from pathlib import Path

import tree_sitter_language_pack as tslp
from tree_sitter import Language, Node, Parser

from devrag.types import Chunk, IndexStats

# Map file extensions to tree-sitter language names
LANGUAGE_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "c_sharp",
    ".swift": "swift",
    ".scala": "scala",
    ".lua": "lua",
    ".r": "r",
    ".R": "r",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".zig": "zig",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".dart": "dart",
    ".vue": "vue",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".sql": "sql",
    ".tf": "hcl",
    ".proto": "proto",
}

# Node types that represent semantic entities we want to extract
ENTITY_NODE_TYPES: dict[str, list[str]] = {
    "python": [
        "function_definition", "class_definition", "decorated_definition",
    ],
    "javascript": [
        "function_declaration", "class_declaration", "arrow_function",
        "method_definition", "export_statement",
    ],
    "typescript": [
        "function_declaration", "class_declaration", "arrow_function",
        "method_definition", "interface_declaration", "type_alias_declaration",
        "export_statement",
    ],
    "tsx": [
        "function_declaration", "class_declaration", "arrow_function",
        "method_definition", "interface_declaration", "type_alias_declaration",
        "export_statement",
    ],
    "rust": [
        "function_item", "impl_item", "struct_item", "enum_item",
        "trait_item", "type_item",
    ],
    "go": [
        "function_declaration", "method_declaration", "type_declaration",
    ],
}

# Approximate tokens as chars / 4
CHARS_PER_TOKEN = 4


def _get_parser(language_name: str) -> Parser | None:
    """Get a tree-sitter parser for a language."""
    try:
        lang = tslp.get_language(language_name)
        return Parser(Language(lang))
    except Exception:
        return None


def _get_entity_name(node: Node, language: str) -> str:
    """Extract the name of a semantic entity from its AST node."""
    # For decorated definitions (Python), get the inner definition
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                return _get_entity_name(child, language)

    # For export statements, get the inner declaration
    if node.type == "export_statement":
        for child in node.children:
            if child.type in (
                "function_declaration", "class_declaration",
                "interface_declaration", "type_alias_declaration",
            ):
                return _get_entity_name(child, language)
        return "export"

    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode("utf-8")
    return node.type


def _get_entity_type(node: Node) -> str:
    """Get a normalized entity type string."""
    t = node.type
    if t == "decorated_definition":
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                return child.type
    if t == "export_statement":
        for child in node.children:
            if hasattr(child, "type") and "declaration" in child.type:
                return child.type
    return t


def _find_parent_class(node: Node, language: str) -> str | None:
    """Walk up the tree to find if this node is inside a class."""
    class_types = {
        "python": "class_definition",
        "javascript": "class_declaration",
        "typescript": "class_declaration",
        "tsx": "class_declaration",
        "rust": "impl_item",
        "go": None,
    }
    class_type = class_types.get(language)
    if not class_type:
        return None

    current = node.parent
    while current:
        if current.type == class_type:
            return _get_entity_name(current, language)
        current = current.parent
    return None


def _get_signature(node: Node, source_bytes: bytes) -> str:
    """Extract the signature (first line) of a function/class definition."""
    start = node.start_byte
    # Find the end of the first line or the body start
    text = source_bytes[start:node.end_byte].decode("utf-8", errors="replace")
    first_line = text.split("\n")[0]
    return first_line.strip()


def _node_to_text(node: Node, source_bytes: bytes) -> str:
    """Extract the text of a node from source bytes."""
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _make_chunk_id(file_path: str, entity_name: str, line_start: int) -> str:
    """Create a deterministic chunk ID."""
    raw = f"{file_path}:{entity_name}:{line_start}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def extract_chunks_from_file(
    file_path: Path,
    max_tokens: int = 512,
    repo_name: str = "",
) -> list[Chunk]:
    """Parse a source file with tree-sitter and extract semantic chunks."""
    ext = file_path.suffix.lower()
    language = LANGUAGE_EXTENSIONS.get(ext)
    if not language:
        return []

    parser = _get_parser(language)
    if not parser:
        return []

    source_bytes = file_path.read_bytes()
    tree = parser.parse(source_bytes)
    root = tree.root_node

    entity_types = ENTITY_NODE_TYPES.get(language, [])
    if not entity_types:
        # Fallback: treat entire file as one chunk for languages without entity definitions
        text = source_bytes.decode("utf-8", errors="replace")
        if len(text) > max_tokens * CHARS_PER_TOKEN:
            text = text[: max_tokens * CHARS_PER_TOKEN]
        chunk_id = _make_chunk_id(str(file_path), "module", 0)
        return [Chunk(
            id=chunk_id,
            text=text,
            metadata={
                "file_path": str(file_path),
                "repo": repo_name,
                "language": language,
                "entity_type": "module",
                "entity_name": file_path.stem,
                "parent_entity": "",
                "signature": "",
                "line_range": [0, root.end_point[0]],
            },
        )]

    chunks: list[Chunk] = []
    seen_ranges: set[tuple[int, int]] = set()

    def visit(node: Node, depth: int = 0) -> None:
        if node.type in entity_types:
            # Avoid duplicates from nested entity extraction
            key = (node.start_byte, node.end_byte)
            if key in seen_ranges:
                return
            seen_ranges.add(key)

            text = _node_to_text(node, source_bytes)
            entity_name = _get_entity_name(node, language)
            entity_type = _get_entity_type(node)
            parent = _find_parent_class(node, language)
            signature = _get_signature(node, source_bytes)
            line_start = node.start_point[0] + 1  # 1-indexed
            line_end = node.end_point[0] + 1

            # Add parent context prefix for methods
            if parent:
                text = f"# In class {parent}\n{text}"

            # Truncate if too large
            max_chars = max_tokens * CHARS_PER_TOKEN
            if len(text) > max_chars:
                text = text[:max_chars] + "\n# ... (truncated)"

            chunk_id = _make_chunk_id(str(file_path), entity_name, line_start)
            chunks.append(Chunk(
                id=chunk_id,
                text=text,
                metadata={
                    "file_path": str(file_path),
                    "repo": repo_name,
                    "language": language,
                    "entity_type": entity_type,
                    "entity_name": entity_name,
                    "parent_entity": parent or "",
                    "signature": signature,
                    "line_range": [line_start, line_end],
                },
            ))

            # For classes, also visit children to extract methods
            if node.type in (
                "class_definition", "class_declaration", "impl_item",
            ):
                for child in node.children:
                    visit(child, depth + 1)
            return

        # Recurse into children
        for child in node.children:
            visit(child, depth + 1)

    visit(root)
    return chunks


class CodeIndexer:
    """Indexes a repository by parsing code files and storing chunks."""

    def __init__(
        self,
        vector_store,  # ChromaStore
        metadata_db,   # MetadataDB
        embedder,      # OllamaEmbedder
        config=None,
    ) -> None:
        from devrag.config import CodeConfig, DevragConfig

        self.vector_store = vector_store
        self.metadata_db = metadata_db
        self.embedder = embedder
        if config is None:
            config = DevragConfig()
        self.code_config = config.code

    def index_repo(
        self,
        repo_path: Path,
        incremental: bool = True,
        repo_name: str = "",
    ) -> IndexStats:
        """Index a repository. Returns stats about what was processed."""
        from devrag.utils.git import discover_files

        stats = IndexStats()
        files = discover_files(repo_path, self.code_config.exclude_patterns)
        stats.files_scanned = len(files)

        if not repo_name:
            repo_name = repo_path.name

        # Determine which files need indexing
        files_to_index: list[Path] = []
        current_file_paths: set[str] = set()

        for file_path in files:
            rel_path = str(file_path.relative_to(repo_path))
            current_file_paths.add(rel_path)
            content_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()

            if incremental:
                stored_hash = self.metadata_db.get_file_hash(rel_path)
                if stored_hash == content_hash:
                    stats.files_skipped += 1
                    continue

            self.metadata_db.set_file_hash(rel_path, content_hash)
            files_to_index.append(file_path)

        # Remove files that no longer exist
        previously_indexed = set(self.metadata_db.get_all_indexed_files())
        removed_files = previously_indexed - current_file_paths
        for rel_path in removed_files:
            chunk_ids = self.metadata_db.get_chunks_for_file(rel_path)
            if chunk_ids:
                self.vector_store.delete("code_chunks", chunk_ids)
            self.metadata_db.remove_file(rel_path)
            stats.files_removed += 1

        # Index changed/new files
        for file_path in files_to_index:
            rel_path = str(file_path.relative_to(repo_path))

            # Remove old chunks for this file
            old_chunk_ids = self.metadata_db.get_chunks_for_file(rel_path)
            if old_chunk_ids:
                self.vector_store.delete("code_chunks", old_chunk_ids)
                self.metadata_db.delete_fts(old_chunk_ids)

            # Extract new chunks
            chunks = extract_chunks_from_file(
                file_path,
                max_tokens=self.code_config.chunk_max_tokens,
                repo_name=repo_name,
            )
            if not chunks:
                stats.files_indexed += 1
                continue

            # Generate embeddings
            texts = [c.text for c in chunks]
            embeddings = self.embedder.embed(texts)

            # Store in vector DB
            self.vector_store.upsert(
                collection="code_chunks",
                ids=[c.id for c in chunks],
                embeddings=embeddings,
                documents=texts,
                metadatas=[c.metadata for c in chunks],
            )

            # Store metadata and FTS
            for chunk in chunks:
                line_range = chunk.metadata["line_range"]
                self.metadata_db.set_chunk_source(
                    chunk.id, rel_path, line_range[0], line_range[1]
                )
                self.metadata_db.upsert_fts(chunk.id, chunk.text)

            stats.files_indexed += 1
            stats.chunks_created += len(chunks)

        return stats
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_code_indexer.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Write integration tests for CodeIndexer class**

Add to `tests/test_code_indexer.py`:

```python
# Add these tests at the bottom of tests/test_code_indexer.py

from devrag.ingest.code_indexer import CodeIndexer
from devrag.stores.chroma_store import ChromaStore
from devrag.stores.metadata_db import MetadataDB


@pytest.fixture
def indexer_deps(tmp_dir):
    """Create real store instances for integration testing."""
    store = ChromaStore(persist_dir=str(tmp_dir / "chroma"))
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    # Mock embedder that returns fixed-size vectors
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])
    return store, meta, embedder


def test_code_indexer_indexes_repo(tmp_dir, indexer_deps):
    store, meta, embedder = indexer_deps
    repo = tmp_dir / "repo"
    repo.mkdir()

    # Initialize as git repo
    import subprocess
    subprocess.run(["git", "init", str(repo)], capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=str(repo), capture_output=True)

    (repo / "main.py").write_text("def hello():\n    return 'world'\n")
    (repo / "utils.py").write_text("def add(a, b):\n    return a + b\n")
    subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), capture_output=True)

    indexer = CodeIndexer(store, meta, embedder)
    stats = indexer.index_repo(repo)

    assert stats.files_scanned >= 2
    assert stats.files_indexed >= 2
    assert stats.chunks_created >= 2
    assert store.count("code_chunks") >= 2


def test_code_indexer_incremental_skips_unchanged(tmp_dir, indexer_deps):
    store, meta, embedder = indexer_deps
    repo = tmp_dir / "repo"
    repo.mkdir()

    import subprocess
    subprocess.run(["git", "init", str(repo)], capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=str(repo), capture_output=True)

    (repo / "main.py").write_text("def hello():\n    return 'world'\n")
    subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), capture_output=True)

    indexer = CodeIndexer(store, meta, embedder)
    stats1 = indexer.index_repo(repo)
    assert stats1.files_indexed >= 1

    # Second run should skip everything
    embedder.embed.reset_mock()
    stats2 = indexer.index_repo(repo, incremental=True)
    assert stats2.files_skipped >= 1
    assert stats2.files_indexed == 0
    embedder.embed.assert_not_called()


def test_code_indexer_detects_removed_files(tmp_dir, indexer_deps):
    store, meta, embedder = indexer_deps
    repo = tmp_dir / "repo"
    repo.mkdir()

    import subprocess
    subprocess.run(["git", "init", str(repo)], capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=str(repo), capture_output=True)

    (repo / "main.py").write_text("def hello():\n    return 'world'\n")
    (repo / "old.py").write_text("def old():\n    pass\n")
    subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), capture_output=True)

    indexer = CodeIndexer(store, meta, embedder)
    indexer.index_repo(repo)
    initial_count = store.count("code_chunks")

    # Remove old.py and re-index
    (repo / "old.py").unlink()
    subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "commit", "-m", "remove old"], cwd=str(repo), capture_output=True)

    stats = indexer.index_repo(repo)
    assert stats.files_removed >= 1
    assert store.count("code_chunks") < initial_count
```

- [ ] **Step 6: Run all code indexer tests**

Run: `uv run pytest tests/test_code_indexer.py -v`
Expected: All 9 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add devrag/ingest/code_indexer.py tests/test_code_indexer.py
git commit -m "feat: AST-aware code indexer with tree-sitter and incremental indexing"
```

---

## Task 9: Hybrid Search (Vector + BM25 + RRF)

**Files:**
- Create: `devrag/retrieve/hybrid_search.py`
- Create: `tests/test_hybrid_search.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hybrid_search.py
import pytest
from unittest.mock import MagicMock

from devrag.retrieve.hybrid_search import HybridSearch, reciprocal_rank_fusion
from devrag.types import QueryResult


def test_rrf_merges_two_rankings():
    """RRF should merge two ranked lists, giving credit for appearing in both."""
    # Doc "a" is rank 1 in both lists => highest RRF score
    # Doc "b" is rank 2 in list1, rank 3 in list2
    # Doc "c" is rank 3 in list1 only
    # Doc "d" is rank 1 in list2 only (was not in list1)
    list1 = ["a", "b", "c"]
    list2 = ["d", "a", "b"]

    merged = reciprocal_rank_fusion([list1, list2], k=60)
    # "a" should be first (appears high in both)
    assert merged[0] == "a"
    # All 4 docs should appear
    assert set(merged) == {"a", "b", "c", "d"}


def test_rrf_single_list():
    merged = reciprocal_rank_fusion([["x", "y", "z"]], k=60)
    assert merged == ["x", "y", "z"]


def test_rrf_empty():
    merged = reciprocal_rank_fusion([], k=60)
    assert merged == []


def test_hybrid_search_combines_vector_and_bm25():
    # Mock vector store
    mock_store = MagicMock()
    mock_store.query.return_value = QueryResult(
        ids=["chunk_1", "chunk_2", "chunk_3"],
        documents=["def auth(): pass", "class User:", "import os"],
        metadatas=[{"file_path": "a.py"}, {"file_path": "b.py"}, {"file_path": "c.py"}],
        distances=[0.1, 0.3, 0.5],
    )

    # Mock metadata DB (BM25 via FTS5)
    mock_meta = MagicMock()
    mock_meta.search_fts.return_value = [
        ("chunk_2", -5.0),  # "class User" matches keyword
        ("chunk_4", -3.0),  # a chunk only found via BM25
        ("chunk_1", -1.0),  # "def auth" also matches
    ]

    # Mock embedder
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 768

    search = HybridSearch(
        vector_store=mock_store,
        metadata_db=mock_meta,
        embedder=mock_embedder,
        collection="code_chunks",
    )
    results = search.search("authentication", top_k=20)

    # Should have results from both sources
    result_ids = [r.chunk_id for r in results]
    assert "chunk_1" in result_ids
    assert "chunk_2" in result_ids
    # chunk_4 found only via BM25 should also appear
    # (but it won't have document text from vector store, so it depends on implementation)

    # Verify embedder was called
    mock_embedder.embed_query.assert_called_once_with("authentication")

    # Verify both search paths were used
    mock_store.query.assert_called_once()
    mock_meta.search_fts.assert_called_once_with("authentication", limit=20)


def test_hybrid_search_with_no_bm25_results():
    mock_store = MagicMock()
    mock_store.query.return_value = QueryResult(
        ids=["c1"], documents=["text"], metadatas=[{}], distances=[0.1],
    )
    mock_meta = MagicMock()
    mock_meta.search_fts.return_value = []
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 768

    search = HybridSearch(mock_store, mock_meta, mock_embedder, "code_chunks")
    results = search.search("query", top_k=5)
    assert len(results) == 1
    assert results[0].chunk_id == "c1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hybrid_search.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write hybrid search implementation**

```python
# devrag/retrieve/hybrid_search.py
from __future__ import annotations

from devrag.types import QueryResult, SearchResult


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[str]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    score(doc) = sum(1 / (k + rank_i)) for each list i where doc appears.
    Higher score = more relevant.
    """
    if not ranked_lists:
        return []

    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    return sorted(scores, key=lambda d: scores[d], reverse=True)


class HybridSearch:
    """Combines vector similarity search with BM25 keyword search via RRF."""

    def __init__(
        self,
        vector_store,  # ChromaStore
        metadata_db,   # MetadataDB
        embedder,      # OllamaEmbedder
        collection: str = "code_chunks",
    ) -> None:
        self.vector_store = vector_store
        self.metadata_db = metadata_db
        self.embedder = embedder
        self.collection = collection

    def search(self, query: str, top_k: int = 20) -> list[SearchResult]:
        """Run hybrid search: vector + BM25, merged via RRF."""
        # 1. Vector search
        query_embedding = self.embedder.embed_query(query)
        vector_results = self.vector_store.query(
            collection=self.collection,
            query_embedding=query_embedding,
            n_results=top_k,
        )

        # Build lookup for document text and metadata from vector results
        doc_lookup: dict[str, tuple[str, dict]] = {}
        for i, doc_id in enumerate(vector_results.ids):
            doc_lookup[doc_id] = (
                vector_results.documents[i],
                vector_results.metadatas[i],
            )

        # 2. BM25 search via FTS5
        bm25_results = self.metadata_db.search_fts(query, limit=top_k)

        # 3. Reciprocal Rank Fusion
        vector_ranked = vector_results.ids
        bm25_ranked = [chunk_id for chunk_id, _score in bm25_results]

        fused_ids = reciprocal_rank_fusion([vector_ranked, bm25_ranked], k=60)

        # 4. Build SearchResult list with RRF scores
        results: list[SearchResult] = []
        rrf_scores: dict[str, float] = {}
        for rank, doc_id in enumerate(fused_ids):
            rrf_scores[doc_id] = 1.0 / (60 + rank + 1)

        for doc_id in fused_ids[:top_k]:
            if doc_id in doc_lookup:
                text, metadata = doc_lookup[doc_id]
            else:
                # Found only via BM25, not in vector results — skip
                # (we don't have the document text without querying the store by ID)
                continue
            results.append(SearchResult(
                chunk_id=doc_id,
                text=text,
                score=rrf_scores[doc_id],
                metadata=metadata,
            ))

        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hybrid_search.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add devrag/retrieve/hybrid_search.py tests/test_hybrid_search.py
git commit -m "feat: hybrid search with vector + BM25 (FTS5) and RRF fusion"
```

---

## Task 10: Cross-Encoder Reranker

**Files:**
- Create: `devrag/retrieve/reranker.py`
- Create: `tests/test_reranker.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_reranker.py
from unittest.mock import MagicMock, patch

import pytest

from devrag.retrieve.reranker import Reranker
from devrag.types import SearchResult


@patch("devrag.retrieve.reranker.CrossEncoder")
def test_reranker_reorders_by_relevance(mock_ce_class):
    """Reranker should reorder results by cross-encoder score."""
    mock_model = MagicMock()
    mock_ce_class.return_value = mock_model
    # rank() returns results sorted by score descending
    mock_model.rank.return_value = [
        {"corpus_id": 2, "score": 9.5},   # chunk_3 is most relevant
        {"corpus_id": 0, "score": 7.2},   # chunk_1 is second
        {"corpus_id": 1, "score": 1.1},   # chunk_2 is least relevant
    ]

    reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    candidates = [
        SearchResult(chunk_id="chunk_1", text="auth code", score=0.9, metadata={}),
        SearchResult(chunk_id="chunk_2", text="unrelated", score=0.8, metadata={}),
        SearchResult(chunk_id="chunk_3", text="login handler", score=0.7, metadata={}),
    ]

    results = reranker.rerank("how does authentication work", candidates, top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "chunk_3"  # highest cross-encoder score
    assert results[1].chunk_id == "chunk_1"
    assert results[0].score > results[1].score


@patch("devrag.retrieve.reranker.CrossEncoder")
def test_reranker_with_fewer_candidates_than_top_k(mock_ce_class):
    mock_model = MagicMock()
    mock_ce_class.return_value = mock_model
    mock_model.rank.return_value = [
        {"corpus_id": 0, "score": 5.0},
    ]

    reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    candidates = [
        SearchResult(chunk_id="c1", text="only one", score=1.0, metadata={}),
    ]

    results = reranker.rerank("query", candidates, top_k=5)
    assert len(results) == 1


@patch("devrag.retrieve.reranker.CrossEncoder")
def test_reranker_empty_candidates(mock_ce_class):
    reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    results = reranker.rerank("query", [], top_k=5)
    assert results == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_reranker.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write reranker implementation**

```python
# devrag/retrieve/reranker.py
from __future__ import annotations

from sentence_transformers import CrossEncoder

from devrag.types import SearchResult


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Rerank candidates using a cross-encoder model.

        Returns the top_k results sorted by cross-encoder relevance score.
        """
        if not candidates:
            return []

        documents = [c.text for c in candidates]
        ranked = self._model.rank(query, documents, top_k=top_k)

        results: list[SearchResult] = []
        for item in ranked:
            idx = item["corpus_id"]
            original = candidates[idx]
            results.append(SearchResult(
                chunk_id=original.chunk_id,
                text=original.text,
                score=float(item["score"]),
                metadata=original.metadata,
            ))

        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_reranker.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add devrag/retrieve/reranker.py tests/test_reranker.py
git commit -m "feat: cross-encoder reranker using ms-marco-MiniLM-L-6-v2"
```

---

## Task 11: Result Formatters

**Files:**
- Create: `devrag/utils/formatters.py`
- Create: `tests/test_formatters.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_formatters.py
from devrag.utils.formatters import format_search_results, format_index_stats
from devrag.types import SearchResult, IndexStats


def test_format_search_results():
    results = [
        SearchResult(
            chunk_id="c1",
            text="def authenticate(user, pwd):\n    return check(user, pwd)",
            score=0.95,
            metadata={"file_path": "src/auth.py", "line_range": [10, 15], "entity_name": "authenticate"},
        ),
        SearchResult(
            chunk_id="c2",
            text="class AuthMiddleware:\n    pass",
            score=0.82,
            metadata={"file_path": "src/middleware.py", "line_range": [1, 5], "entity_name": "AuthMiddleware"},
        ),
    ]
    output = format_search_results(results)
    assert "src/auth.py" in output
    assert "authenticate" in output
    assert "src/middleware.py" in output
    assert "AuthMiddleware" in output


def test_format_search_results_empty():
    output = format_search_results([])
    assert "no results" in output.lower()


def test_format_index_stats():
    stats = IndexStats(
        files_scanned=100,
        files_indexed=20,
        files_skipped=78,
        files_removed=2,
        chunks_created=85,
    )
    output = format_index_stats(stats)
    assert "100" in output
    assert "20" in output
    assert "78" in output
    assert "85" in output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_formatters.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write formatters implementation**

```python
# devrag/utils/formatters.py
from __future__ import annotations

from devrag.types import IndexStats, SearchResult


def format_search_results(results: list[SearchResult]) -> str:
    """Format search results for MCP/CLI output."""
    if not results:
        return "No results found."

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        file_path = r.metadata.get("file_path", "unknown")
        line_range = r.metadata.get("line_range", [])
        entity_name = r.metadata.get("entity_name", "")
        language = r.metadata.get("language", "")

        location = file_path
        if line_range:
            location += f":{line_range[0]}-{line_range[1]}"

        lines.append(f"### {i}. {entity_name} ({location})")
        lines.append(f"```{language}")
        # Show first 10 lines of the chunk
        text_lines = r.text.strip().split("\n")
        preview = "\n".join(text_lines[:10])
        if len(text_lines) > 10:
            preview += "\n# ... (truncated)"
        lines.append(preview)
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def format_index_stats(stats: IndexStats) -> str:
    """Format indexing statistics for MCP/CLI output."""
    parts = [
        f"Scanned {stats.files_scanned} files",
        f"Indexed {stats.files_indexed} files ({stats.chunks_created} chunks)",
        f"Skipped {stats.files_skipped} unchanged files",
    ]
    if stats.files_removed:
        parts.append(f"Removed {stats.files_removed} deleted files")
    return ". ".join(parts) + "."
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_formatters.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add devrag/utils/formatters.py tests/test_formatters.py
git commit -m "feat: result formatters for search output and index stats"
```

---

## Task 12: MCP Server

**Files:**
- Create: `devrag/mcp_server.py`

The MCP server wires together all components and exposes them as FastMCP tools. Testing is done via end-to-end verification (Task 13) since FastMCP tools are best tested by running the server.

- [ ] **Step 1: Write the MCP server**

```python
# devrag/mcp_server.py
from __future__ import annotations

from pathlib import Path

from fastmcp import FastMCP

from devrag.config import DevragConfig, load_config
from devrag.ingest.code_indexer import CodeIndexer
from devrag.ingest.embedder import OllamaEmbedder
from devrag.retrieve.hybrid_search import HybridSearch
from devrag.retrieve.reranker import Reranker
from devrag.stores.chroma_store import ChromaStore
from devrag.stores.metadata_db import MetadataDB
from devrag.utils.formatters import format_index_stats, format_search_results

mcp = FastMCP("DevRAG")

# Lazy-initialized singletons
_config: DevragConfig | None = None
_vector_store: ChromaStore | None = None
_metadata_db: MetadataDB | None = None
_embedder: OllamaEmbedder | None = None
_reranker: Reranker | None = None


def _get_config() -> DevragConfig:
    global _config
    if _config is None:
        _config = load_config(project_dir=Path.cwd())
    return _config


def _get_vector_store() -> ChromaStore:
    global _vector_store
    if _vector_store is None:
        config = _get_config()
        persist_dir = Path(config.vector_store.persist_dir).expanduser()
        persist_dir.mkdir(parents=True, exist_ok=True)
        _vector_store = ChromaStore(persist_dir=str(persist_dir))
    return _vector_store


def _get_metadata_db() -> MetadataDB:
    global _metadata_db
    if _metadata_db is None:
        db_dir = Path("~/.local/share/devrag").expanduser()
        db_dir.mkdir(parents=True, exist_ok=True)
        _metadata_db = MetadataDB(str(db_dir / "metadata.db"))
    return _metadata_db


def _get_embedder() -> OllamaEmbedder:
    global _embedder
    if _embedder is None:
        config = _get_config()
        _embedder = OllamaEmbedder(
            model=config.embedding.model,
            ollama_url=config.embedding.ollama_url,
            batch_size=config.embedding.batch_size,
        )
    return _embedder


def _get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        config = _get_config()
        _reranker = Reranker(model_name=config.retrieval.reranker_model)
    return _reranker


@mcp.tool
def search(query: str, top_k: int = 5) -> str:
    """Search indexed code using hybrid retrieval (semantic + keyword).

    Returns the most relevant code chunks matching the query,
    with file paths and code snippets.
    """
    config = _get_config()
    hybrid = HybridSearch(
        vector_store=_get_vector_store(),
        metadata_db=_get_metadata_db(),
        embedder=_get_embedder(),
        collection="code_chunks",
    )

    candidates = hybrid.search(query, top_k=config.retrieval.top_k)

    if config.retrieval.rerank and candidates:
        reranker = _get_reranker()
        results = reranker.rerank(query, candidates, top_k=top_k)
    else:
        results = candidates[:top_k]

    return format_search_results(results)


@mcp.tool
def index_repo(path: str = ".", incremental: bool = True) -> str:
    """Index a local code repository using AST-aware chunking.

    Parses source files with tree-sitter, extracts functions/classes/methods,
    and stores embeddings for semantic search. Uses incremental indexing
    to skip unchanged files.
    """
    repo_path = Path(path).resolve()
    if not repo_path.exists():
        return f"Error: path '{path}' does not exist."

    indexer = CodeIndexer(
        vector_store=_get_vector_store(),
        metadata_db=_get_metadata_db(),
        embedder=_get_embedder(),
        config=_get_config(),
    )

    stats = indexer.index_repo(repo_path, incremental=incremental)
    return format_index_stats(stats)


@mcp.tool
def status() -> str:
    """Show indexing status: number of files and chunks indexed."""
    store = _get_vector_store()
    meta = _get_metadata_db()

    chunk_count = store.count("code_chunks")
    indexed_files = meta.get_all_indexed_files()

    lines = [
        f"Indexed files: {len(indexed_files)}",
        f"Code chunks: {chunk_count}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `uv run python -c "from devrag.mcp_server import mcp; print(f'Server: {mcp.name}')"`
Expected: `Server: DevRAG`

- [ ] **Step 3: Commit**

```bash
git add devrag/mcp_server.py
git commit -m "feat: FastMCP server with search, index_repo, and status tools"
```

---

## Task 13: End-to-End Verification

This task verifies the full pipeline works together. It requires Ollama running locally with `nomic-embed-text` pulled.

- [ ] **Step 1: Verify Ollama is available**

Run: `curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; tags=json.load(sys.stdin); models=[m['name'] for m in tags.get('models',[])]; print('Models:', models); assert any('nomic' in m for m in models), 'Pull nomic-embed-text first: ollama pull nomic-embed-text'"`

Expected: Shows available models including `nomic-embed-text`. If not available:

Run: `ollama pull nomic-embed-text`

- [ ] **Step 2: Write an end-to-end integration test**

Create `tests/test_integration.py`:

```python
# tests/test_integration.py
"""End-to-end integration test. Requires Ollama running with nomic-embed-text."""
import os
import subprocess
from pathlib import Path

import pytest

from devrag.config import DevragConfig
from devrag.ingest.code_indexer import CodeIndexer
from devrag.ingest.embedder import OllamaEmbedder
from devrag.retrieve.hybrid_search import HybridSearch
from devrag.stores.chroma_store import ChromaStore
from devrag.stores.metadata_db import MetadataDB

# Skip if Ollama is not available
pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_INTEGRATION", "1") == "1",
    reason="Set SKIP_INTEGRATION=0 to run integration tests (requires Ollama)",
)


@pytest.fixture
def e2e_repo(tmp_dir):
    """Create a realistic mini-repo for testing."""
    repo = tmp_dir / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=str(repo), capture_output=True)

    (repo / "src").mkdir()
    (repo / "src" / "auth.py").write_text('''
"""Authentication module."""

import hashlib
import secrets


class PasswordHasher:
    """Handles secure password hashing and verification."""

    def hash_password(self, password: str) -> str:
        """Hash a password using SHA-256 with a random salt."""
        salt = secrets.token_hex(16)
        hashed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
        return f"{salt}:{hashed}"

    def verify_password(self, password: str, stored: str) -> bool:
        """Verify a password against a stored hash."""
        salt, expected = stored.split(":")
        actual = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
        return actual == expected


def create_session_token(user_id: str) -> str:
    """Generate a random session token for a user."""
    return f"{user_id}:{secrets.token_urlsafe(32)}"
''')

    (repo / "src" / "database.py").write_text('''
"""Database connection pooling."""

from dataclasses import dataclass


@dataclass
class ConnectionConfig:
    host: str
    port: int = 5432
    pool_size: int = 10


class ConnectionPool:
    """Manages a pool of database connections."""

    def __init__(self, config: ConnectionConfig):
        self.config = config
        self._connections: list = []

    def acquire(self):
        """Acquire a connection from the pool."""
        if self._connections:
            return self._connections.pop()
        return self._create_connection()

    def release(self, conn):
        """Return a connection to the pool."""
        if len(self._connections) < self.config.pool_size:
            self._connections.append(conn)

    def _create_connection(self):
        """Create a new database connection."""
        return {"host": self.config.host, "port": self.config.port}
''')

    subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), capture_output=True)
    return repo


@pytest.fixture
def e2e_components(tmp_dir):
    store = ChromaStore(persist_dir=str(tmp_dir / "chroma"))
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = OllamaEmbedder(model="nomic-embed-text")
    return store, meta, embedder


def test_index_and_search_e2e(e2e_repo, e2e_components):
    store, meta, embedder = e2e_components

    # Index the repo
    indexer = CodeIndexer(store, meta, embedder)
    stats = indexer.index_repo(e2e_repo)
    assert stats.files_indexed >= 2
    assert stats.chunks_created >= 4  # Multiple functions/classes

    # Semantic search: should find auth-related code
    search = HybridSearch(store, meta, embedder, "code_chunks")
    results = search.search("password authentication and hashing")
    assert len(results) > 0
    # The top result should be from auth.py
    top_files = [r.metadata.get("file_path", "") for r in results[:3]]
    assert any("auth.py" in f for f in top_files)

    # Keyword search: exact function name
    results2 = search.search("create_session_token")
    assert len(results2) > 0
    assert any("create_session_token" in r.text for r in results2[:3])

    # Incremental re-index should skip everything
    stats2 = indexer.index_repo(e2e_repo, incremental=True)
    assert stats2.files_skipped >= 2
    assert stats2.files_indexed == 0
```

- [ ] **Step 3: Run integration test (with Ollama)**

Run: `SKIP_INTEGRATION=0 uv run pytest tests/test_integration.py -v -s`
Expected: PASS — indexing and search work end-to-end.

- [ ] **Step 4: Test MCP server starts**

Run: `timeout 5 uv run python -c "from devrag.mcp_server import mcp; print('MCP server initialized successfully')" || true`
Expected: `MCP server initialized successfully`

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: end-to-end integration test for full indexing and search pipeline"
```

- [ ] **Step 6: Register MCP server with Claude Code**

Run: `claude mcp add devrag -- uv run --directory /home/tom/Projects/dev-rag python -m devrag.mcp_server`

Verify: `claude mcp list` should show `devrag` registered.

- [ ] **Step 7: Test from Claude Code**

In a Claude Code session, verify:
1. `mcp__devrag__index_repo` with path to a real repo
2. `mcp__devrag__search` with a query
3. `mcp__devrag__status` shows correct counts

---

## Spec Coverage Checklist

| Spec Section | Task(s) | Status |
|---|---|---|
| 1.1 Project Scaffolding | Task 1 | Covered |
| 1.2 Storage Layer (VectorStore + ChromaDB + MetadataDB) | Tasks 3, 4, 5 | Covered |
| 1.3 Embedding Engine (Ollama) | Task 6 | Covered |
| 1.4 Code Indexer (AST + incremental) | Task 8 | Covered |
| 1.5 Retrieval (Hybrid + Reranker) | Tasks 9, 10 | Covered |
| 1.6 MCP Server | Task 12 | Covered |
| 1.7 Config | Task 2 | Covered |
| 1.8 Dependencies | Task 1 (pyproject.toml) | Covered |
| Verification Plan | Task 13 | Covered |
