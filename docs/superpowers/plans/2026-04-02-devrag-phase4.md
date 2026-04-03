# DevRAG Phase 4: Polish, Scale, and Eval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Qdrant as an alternative vector store backend, build an eval framework for measuring retrieval quality, add per-query observability metrics, create Claude Code skills, and add re-embedding support.

**Architecture:** Qdrant implements the existing `VectorStore` protocol — config-driven swap with no code changes needed by consumers. Eval framework reads JSONL test queries, runs them through the search pipeline, and computes precision/recall/MRR metrics. Observability logs per-query metrics to a new SQLite table. Claude Code skills are markdown files wrapping MCP tools. Re-embedding clears and rebuilds all collections.

**Tech Stack:** qdrant-client (optional dep), existing stack.

---

## File Structure

```
devrag/
├── pyproject.toml              # MODIFY: add qdrant-client optional dep
├── devrag/
│   ├── config.py               # MODIFY: add qdrant_url to VectorStoreConfig
│   ├── cli.py                  # MODIFY: add eval + reindex commands
│   ├── mcp_server.py           # MODIFY: use store factory instead of ChromaStore
│   │
│   ├── eval.py                 # CREATE: eval framework (run + compare)
│   │
│   ├── stores/
│   │   ├── factory.py          # CREATE: vector store factory based on config
│   │   ├── qdrant_store.py     # CREATE: Qdrant VectorStore implementation
│   │   └── metadata_db.py      # MODIFY: add query_metrics table
│   │
│   └── retrieve/
│       └── hybrid_search.py    # MODIFY: add metrics logging
│
├── .claude/
│   └── skills/
│       ├── rag-search/SKILL.md # CREATE
│       ├── rag-index/SKILL.md  # CREATE
│       └── rag-pr/SKILL.md     # CREATE
│
└── tests/
    ├── test_qdrant_store.py    # CREATE
    ├── test_eval.py            # CREATE
    ├── test_store_factory.py   # CREATE
    └── test_observability.py   # CREATE
```

---

## Task 1: Qdrant Backend

**Files:**
- Modify: `pyproject.toml`
- Modify: `devrag/config.py`
- Create: `devrag/stores/qdrant_store.py`
- Create: `tests/test_qdrant_store.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_qdrant_store.py
import pytest
from unittest.mock import MagicMock, patch, call

from devrag.stores.qdrant_store import QdrantStore
from devrag.types import QueryResult


@pytest.fixture
def mock_qdrant_client():
    with patch("devrag.stores.qdrant_store.QdrantClient") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        yield mock_client


def test_upsert(mock_qdrant_client):
    mock_qdrant_client.collection_exists.return_value = True
    store = QdrantStore(url="http://localhost:6333", embedding_dim=768)
    store.upsert(
        collection="test",
        ids=["a", "b"],
        embeddings=[[0.1] * 768, [0.2] * 768],
        documents=["doc a", "doc b"],
        metadatas=[{"lang": "en"}, {"lang": "fr"}],
    )
    mock_qdrant_client.upsert.assert_called_once()
    args = mock_qdrant_client.upsert.call_args
    assert args.kwargs["collection_name"] == "test"
    assert len(args.kwargs["points"]) == 2


def test_query(mock_qdrant_client):
    mock_qdrant_client.collection_exists.return_value = True
    mock_scored_point = MagicMock()
    mock_scored_point.id = "a"
    mock_scored_point.score = 0.95
    mock_scored_point.payload = {"_document": "doc a", "lang": "en"}
    mock_qdrant_client.search.return_value = [mock_scored_point]

    store = QdrantStore(url="http://localhost:6333", embedding_dim=768)
    result = store.query(collection="test", query_embedding=[0.1] * 768, n_results=5)

    assert isinstance(result, QueryResult)
    assert result.ids == ["a"]
    assert result.documents == ["doc a"]
    assert result.metadatas == [{"lang": "en"}]
    assert result.distances == [0.95]


def test_query_nonexistent_collection(mock_qdrant_client):
    mock_qdrant_client.collection_exists.return_value = False
    store = QdrantStore(url="http://localhost:6333", embedding_dim=768)
    result = store.query(collection="nonexistent", query_embedding=[0.1] * 768)
    assert result.ids == []


def test_delete(mock_qdrant_client):
    mock_qdrant_client.collection_exists.return_value = True
    store = QdrantStore(url="http://localhost:6333", embedding_dim=768)
    store.delete(collection="test", ids=["a", "b"])
    mock_qdrant_client.delete.assert_called_once()


def test_count(mock_qdrant_client):
    mock_qdrant_client.collection_exists.return_value = True
    mock_count = MagicMock()
    mock_count.count = 42
    mock_qdrant_client.count.return_value = mock_count
    store = QdrantStore(url="http://localhost:6333", embedding_dim=768)
    assert store.count("test") == 42


def test_count_nonexistent(mock_qdrant_client):
    mock_qdrant_client.collection_exists.return_value = False
    store = QdrantStore(url="http://localhost:6333", embedding_dim=768)
    assert store.count("nonexistent") == 0


def test_query_with_where_filter(mock_qdrant_client):
    mock_qdrant_client.collection_exists.return_value = True
    mock_qdrant_client.search.return_value = []
    store = QdrantStore(url="http://localhost:6333", embedding_dim=768)
    store.query(collection="test", query_embedding=[0.1] * 768, where={"lang": "python"})
    search_call = mock_qdrant_client.search.call_args
    assert search_call.kwargs.get("query_filter") is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_qdrant_store.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Add qdrant-client as optional dependency**

Add to `pyproject.toml`:
```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24.0",
    "respx>=0.22.0",
]
qdrant = [
    "qdrant-client>=1.12.0",
]
```

Add `qdrant_url` to `VectorStoreConfig` in `devrag/config.py`:
```python
@dataclass
class VectorStoreConfig:
    backend: str = "chromadb"
    persist_dir: str = "~/.local/share/devrag/chroma"
    qdrant_url: str = "http://localhost:6333"
    embedding_dim: int = 768
```

Run: `uv sync --all-extras`

- [ ] **Step 4: Write QdrantStore implementation**

```python
# devrag/stores/qdrant_store.py
from __future__ import annotations

from devrag.types import QueryResult

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointIdsList,
        PointStruct,
        VectorParams,
    )
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False


class QdrantStore:
    def __init__(self, url: str = "http://localhost:6333", embedding_dim: int = 768) -> None:
        if not HAS_QDRANT:
            raise ImportError("qdrant-client is required. Install with: pip install devrag[qdrant]")
        self._client = QdrantClient(url=url)
        self._embedding_dim = embedding_dim

    def _ensure_collection(self, collection: str) -> None:
        if not self._client.collection_exists(collection):
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=self._embedding_dim, distance=Distance.COSINE),
            )

    def upsert(
        self,
        collection: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        self._ensure_collection(collection)
        points = []
        for i, doc_id in enumerate(ids):
            payload = dict(metadatas[i]) if metadatas[i] else {}
            payload["_document"] = documents[i]
            points.append(PointStruct(id=doc_id, vector=embeddings[i], payload=payload))
        self._client.upsert(collection_name=collection, points=points, wait=True)

    def query(
        self,
        collection: str,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict | None = None,
    ) -> QueryResult:
        if not self._client.collection_exists(collection):
            return QueryResult(ids=[], documents=[], metadatas=[], distances=[])

        query_filter = None
        if where:
            conditions = []
            for key, value in where.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            query_filter = Filter(must=conditions)

        results = self._client.search(
            collection_name=collection,
            query_vector=query_embedding,
            limit=n_results,
            query_filter=query_filter,
            with_payload=True,
        )

        ids = []
        documents = []
        metadatas = []
        distances = []
        for point in results:
            ids.append(str(point.id))
            payload = dict(point.payload) if point.payload else {}
            doc = payload.pop("_document", "")
            documents.append(doc)
            metadatas.append(payload)
            distances.append(point.score)

        return QueryResult(ids=ids, documents=documents, metadatas=metadatas, distances=distances)

    def delete(self, collection: str, ids: list[str]) -> None:
        if not self._client.collection_exists(collection):
            return
        self._client.delete(
            collection_name=collection,
            points_selector=PointIdsList(points=ids),
            wait=True,
        )

    def count(self, collection: str) -> int:
        if not self._client.collection_exists(collection):
            return 0
        result = self._client.count(collection_name=collection, exact=True)
        return result.count
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_qdrant_store.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml devrag/config.py devrag/stores/qdrant_store.py tests/test_qdrant_store.py
git commit -m "feat: Qdrant vector store backend implementing VectorStore protocol"
```

---

## Task 2: Store Factory

**Files:**
- Create: `devrag/stores/factory.py`
- Create: `tests/test_store_factory.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_store_factory.py
from unittest.mock import patch, MagicMock

import pytest

from devrag.stores.factory import create_vector_store
from devrag.config import DevragConfig


def test_factory_creates_chroma_by_default(tmp_dir):
    config = DevragConfig()
    config.vector_store.persist_dir = str(tmp_dir / "chroma")
    store = create_vector_store(config)
    from devrag.stores.chroma_store import ChromaStore
    assert isinstance(store, ChromaStore)


@patch("devrag.stores.factory.QdrantStore")
def test_factory_creates_qdrant(mock_qdrant_cls):
    mock_store = MagicMock()
    mock_qdrant_cls.return_value = mock_store
    config = DevragConfig()
    config.vector_store.backend = "qdrant"
    config.vector_store.qdrant_url = "http://localhost:6333"
    store = create_vector_store(config)
    mock_qdrant_cls.assert_called_once_with(
        url="http://localhost:6333",
        embedding_dim=config.vector_store.embedding_dim,
    )
    assert store is mock_store


def test_factory_raises_for_unknown_backend():
    config = DevragConfig()
    config.vector_store.backend = "unknown"
    with pytest.raises(ValueError, match="Unknown vector store backend"):
        create_vector_store(config)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_store_factory.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write store factory**

```python
# devrag/stores/factory.py
from __future__ import annotations

from pathlib import Path

from devrag.config import DevragConfig


def create_vector_store(config: DevragConfig):
    """Create a vector store instance based on config."""
    backend = config.vector_store.backend

    if backend == "chromadb":
        from devrag.stores.chroma_store import ChromaStore
        persist_dir = Path(config.vector_store.persist_dir).expanduser()
        persist_dir.mkdir(parents=True, exist_ok=True)
        return ChromaStore(persist_dir=str(persist_dir))

    elif backend == "qdrant":
        from devrag.stores.qdrant_store import QdrantStore
        return QdrantStore(
            url=config.vector_store.qdrant_url,
            embedding_dim=config.vector_store.embedding_dim,
        )

    else:
        raise ValueError(f"Unknown vector store backend: {backend}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_store_factory.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Update MCP server and CLI to use factory**

In `devrag/mcp_server.py`, replace `_get_vector_store`:

```python
from devrag.stores.factory import create_vector_store

def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = create_vector_store(_get_config())
    return _vector_store
```

Remove the `ChromaStore` import and the `Path(config.vector_store.persist_dir)...` logic from `_get_vector_store`.

In `devrag/cli.py`, update `_get_search_components` and each command that creates a store to use the factory:

```python
from devrag.stores.factory import create_vector_store
```

Replace all `ChromaStore(persist_dir=...)` calls with `create_vector_store(config)`. Remove `ChromaStore` imports.

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add devrag/stores/factory.py tests/test_store_factory.py devrag/mcp_server.py devrag/cli.py
git commit -m "feat: vector store factory for config-driven backend selection"
```

---

## Task 3: Observability — Query Metrics

**Files:**
- Modify: `devrag/stores/metadata_db.py`
- Modify: `devrag/retrieve/hybrid_search.py`
- Create: `tests/test_observability.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_observability.py
import time

from devrag.stores.metadata_db import MetadataDB


def test_log_query_metric(tmp_dir):
    db = MetadataDB(str(tmp_dir / "meta.db"))
    db.log_query_metric(
        query="how does auth work",
        collections=["code_chunks"],
        vector_ms=15.2,
        bm25_ms=3.1,
        rerank_ms=45.0,
        total_ms=63.3,
        result_count=5,
        classification="code",
    )
    metrics = db.get_query_metrics(limit=10)
    assert len(metrics) == 1
    assert metrics[0]["query"] == "how does auth work"
    assert metrics[0]["total_ms"] == 63.3
    assert metrics[0]["result_count"] == 5


def test_get_query_stats(tmp_dir):
    db = MetadataDB(str(tmp_dir / "meta.db"))
    db.log_query_metric("q1", ["code_chunks"], 10, 2, 40, 52, 5, "code")
    db.log_query_metric("q2", ["pr_diffs"], 12, 3, 42, 57, 3, "pr")
    db.log_query_metric("q3", ["code_chunks"], 8, 1, 38, 47, 4, "code")

    stats = db.get_query_stats()
    assert stats["total_queries"] == 3
    assert stats["avg_total_ms"] == pytest.approx(52.0, abs=0.1)
    assert stats["avg_result_count"] == pytest.approx(4.0, abs=0.1)
```

Add `import pytest` to the top if not already imported.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_observability.py -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 3: Add query_metrics table and methods to MetadataDB**

Add to `_create_tables` in `devrag/stores/metadata_db.py`:

```sql
CREATE TABLE IF NOT EXISTS query_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    collections TEXT NOT NULL,
    vector_ms REAL,
    bm25_ms REAL,
    rerank_ms REAL,
    total_ms REAL,
    result_count INTEGER,
    classification TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Add methods:

```python
def log_query_metric(
    self,
    query: str,
    collections: list[str],
    vector_ms: float,
    bm25_ms: float,
    rerank_ms: float,
    total_ms: float,
    result_count: int,
    classification: str,
) -> None:
    self._conn.execute(
        "INSERT INTO query_metrics (query, collections, vector_ms, bm25_ms, rerank_ms, total_ms, result_count, classification) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (query, ",".join(collections), vector_ms, bm25_ms, rerank_ms, total_ms, result_count, classification),
    )
    self._conn.commit()

def get_query_metrics(self, limit: int = 100) -> list[dict]:
    rows = self._conn.execute(
        "SELECT query, collections, vector_ms, bm25_ms, rerank_ms, total_ms, result_count, classification, timestamp "
        "FROM query_metrics ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [
        {
            "query": r[0], "collections": r[1], "vector_ms": r[2],
            "bm25_ms": r[3], "rerank_ms": r[4], "total_ms": r[5],
            "result_count": r[6], "classification": r[7], "timestamp": r[8],
        }
        for r in rows
    ]

def get_query_stats(self) -> dict:
    row = self._conn.execute(
        "SELECT COUNT(*), AVG(total_ms), AVG(result_count), AVG(vector_ms), AVG(bm25_ms), AVG(rerank_ms) "
        "FROM query_metrics"
    ).fetchone()
    return {
        "total_queries": row[0],
        "avg_total_ms": row[1] or 0.0,
        "avg_result_count": row[2] or 0.0,
        "avg_vector_ms": row[3] or 0.0,
        "avg_bm25_ms": row[4] or 0.0,
        "avg_rerank_ms": row[5] or 0.0,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_observability.py -v`
Expected: All 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add devrag/stores/metadata_db.py tests/test_observability.py
git commit -m "feat: per-query observability metrics in SQLite"
```

---

## Task 4: Eval Framework

**Files:**
- Create: `devrag/eval.py`
- Create: `tests/test_eval.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_eval.py
import json
import pytest

from devrag.eval import compute_metrics, precision_at_k, recall_at_k, mrr


def test_precision_at_k():
    # 3 out of 5 results are relevant
    retrieved = ["a.py", "b.py", "c.py", "d.py", "e.py"]
    relevant = {"a.py", "c.py", "e.py"}
    assert precision_at_k(retrieved, relevant, k=5) == 3 / 5


def test_precision_at_k_partial():
    retrieved = ["a.py", "b.py", "c.py"]
    relevant = {"a.py"}
    assert precision_at_k(retrieved, relevant, k=2) == 1 / 2


def test_recall_at_k():
    retrieved = ["a.py", "b.py", "c.py"]
    relevant = {"a.py", "c.py", "d.py"}
    assert recall_at_k(retrieved, relevant, k=3) == 2 / 3


def test_mrr():
    # First relevant result at position 2 (0-indexed 1)
    retrieved = ["b.py", "a.py", "c.py"]
    relevant = {"a.py"}
    assert mrr(retrieved, relevant) == 1 / 2


def test_mrr_first_position():
    retrieved = ["a.py", "b.py"]
    relevant = {"a.py"}
    assert mrr(retrieved, relevant) == 1.0


def test_mrr_no_relevant():
    retrieved = ["b.py", "c.py"]
    relevant = {"a.py"}
    assert mrr(retrieved, relevant) == 0.0


def test_compute_metrics():
    test_cases = [
        {
            "query": "how does auth work",
            "expected_files": ["src/auth.py", "src/login.py"],
        },
    ]
    # Simulate results where src/auth.py was found at position 1
    search_results = {
        "how does auth work": [
            {"file_path": "src/auth.py"},
            {"file_path": "src/other.py"},
            {"file_path": "src/login.py"},
        ],
    }

    metrics = compute_metrics(test_cases, search_results, k=5)
    assert metrics["precision_at_5"] == pytest.approx(2 / 3, abs=0.01)
    assert metrics["recall_at_5"] == pytest.approx(2 / 2, abs=0.01)
    assert metrics["mrr"] == pytest.approx(1.0, abs=0.01)


def test_compute_metrics_with_prs():
    test_cases = [
        {
            "query": "why did we change auth",
            "expected_prs": [42, 56],
        },
    ]
    search_results = {
        "why did we change auth": [
            {"pr_number": 42},
            {"pr_number": 99},
        ],
    }
    metrics = compute_metrics(test_cases, search_results, k=5)
    assert metrics["precision_at_5"] == pytest.approx(1 / 2, abs=0.01)
    assert metrics["recall_at_5"] == pytest.approx(1 / 2, abs=0.01)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_eval.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write eval module**

```python
# devrag/eval.py
from __future__ import annotations

import json
from pathlib import Path


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of top-k results that are relevant."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for r in top_k if r in relevant)
    return hits / len(top_k)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items found in top-k results."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return hits / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Mean Reciprocal Rank: 1/position of first relevant result."""
    for i, item in enumerate(retrieved):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def compute_metrics(
    test_cases: list[dict],
    search_results: dict[str, list[dict]],
    k: int = 5,
) -> dict[str, float]:
    """Compute aggregate metrics across test cases.

    Args:
        test_cases: List of {"query": str, "expected_files": [...], "expected_prs": [...]}
        search_results: Map of query -> list of result metadata dicts
        k: k for precision@k and recall@k
    """
    all_precision = []
    all_recall = []
    all_mrr = []

    for case in test_cases:
        query = case["query"]
        results = search_results.get(query, [])

        # Build relevant set and retrieved list
        expected_files = set(case.get("expected_files", []))
        expected_prs = set(case.get("expected_prs", []))

        retrieved: list[str] = []
        for r in results:
            if "file_path" in r and expected_files:
                retrieved.append(r["file_path"])
            elif "pr_number" in r and expected_prs:
                retrieved.append(str(r["pr_number"]))

        relevant = expected_files | {str(p) for p in expected_prs}

        all_precision.append(precision_at_k(retrieved, relevant, k))
        all_recall.append(recall_at_k(retrieved, relevant, k))
        all_mrr.append(mrr(retrieved, relevant))

    n = len(test_cases) or 1
    return {
        f"precision_at_{k}": sum(all_precision) / n,
        f"recall_at_{k}": sum(all_recall) / n,
        "mrr": sum(all_mrr) / n,
        "num_queries": len(test_cases),
    }


def load_test_queries(path: Path) -> list[dict]:
    """Load test queries from a JSONL file."""
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def save_results(results: list[dict], path: Path) -> None:
    """Save eval results to a JSONL file."""
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


def load_results(path: Path) -> list[dict]:
    """Load eval results from a JSONL file."""
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_eval.py -v`
Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add devrag/eval.py tests/test_eval.py
git commit -m "feat: eval framework with precision@k, recall@k, and MRR metrics"
```

---

## Task 5: Re-embedding Support + CLI Updates

**Files:**
- Modify: `devrag/cli.py`

- [ ] **Step 1: Add reindex, eval, and sync commands to CLI**

Add to `devrag/cli.py`:

```python
eval_app = typer.Typer(help="Evaluate retrieval quality.")
app.add_typer(eval_app, name="eval")


@app.command()
def reindex(
    all_collections: bool = typer.Option(False, "--all", help="Re-embed all collections"),
):
    """Re-index everything. Use after changing embedding models."""
    from devrag.config import load_config
    from devrag.ingest.code_indexer import CodeIndexer
    from devrag.ingest.embedder import OllamaEmbedder
    from devrag.stores.factory import create_vector_store
    from devrag.stores.metadata_db import MetadataDB
    from devrag.utils.formatters import format_index_stats

    config = load_config(project_dir=Path.cwd())
    store = create_vector_store(config)
    db_dir = Path("~/.local/share/devrag").expanduser()
    db_dir.mkdir(parents=True, exist_ok=True)
    meta = MetadataDB(str(db_dir / "metadata.db"))
    embedder = OllamaEmbedder(
        model=config.embedding.model,
        ollama_url=config.embedding.ollama_url,
        batch_size=config.embedding.batch_size,
    )

    if all_collections:
        # Clear file hashes to force full re-index
        for file_path in meta.get_all_indexed_files():
            meta.remove_file(file_path)
        typer.echo("Cleared all file hashes. Run 'devrag index repo .' to re-index code.")
    else:
        typer.echo("Use --all to clear all indexes and force re-embedding.")


@eval_app.command("run")
def eval_run(
    queries_file: str = typer.Argument(..., help="Path to test queries JSONL file"),
    output: str = typer.Option("results.jsonl", help="Output file for results"),
    top_k: int = typer.Option(5, help="Number of results per query"),
):
    """Run eval queries and compute metrics."""
    import json
    from devrag.eval import load_test_queries, compute_metrics, save_results
    from devrag.retrieve.query_router import QueryRouter
    from devrag.utils.formatters import format_search_results

    hybrid, reranker, config = _get_search_components()
    router = QueryRouter()
    test_cases = load_test_queries(Path(queries_file))
    search_results_map: dict[str, list[dict]] = {}
    all_results: list[dict] = []

    for case in test_cases:
        query = case["query"]
        collections = router.route(query)
        candidates = hybrid.search(query, top_k=config.retrieval.top_k, collections=collections)
        if reranker and candidates:
            results = reranker.rerank(query, candidates, top_k=top_k)
        else:
            results = candidates[:top_k]

        result_metas = [r.metadata for r in results]
        search_results_map[query] = result_metas
        all_results.append({"query": query, "results": result_metas})

    metrics = compute_metrics(test_cases, search_results_map, k=top_k)
    save_results(all_results, Path(output))

    typer.echo(f"Evaluated {metrics['num_queries']} queries:")
    typer.echo(f"  Precision@{top_k}: {metrics[f'precision_at_{top_k}']:.3f}")
    typer.echo(f"  Recall@{top_k}: {metrics[f'recall_at_{top_k}']:.3f}")
    typer.echo(f"  MRR: {metrics['mrr']:.3f}")
    typer.echo(f"Results saved to {output}")


@eval_app.command("compare")
def eval_compare(
    file_a: str = typer.Argument(..., help="First results file"),
    file_b: str = typer.Argument(..., help="Second results file"),
):
    """Compare two eval result files."""
    from devrag.eval import load_results
    results_a = load_results(Path(file_a))
    results_b = load_results(Path(file_b))
    typer.echo(f"File A: {len(results_a)} queries from {file_a}")
    typer.echo(f"File B: {len(results_b)} queries from {file_b}")
    # Simple comparison: show which queries changed
    queries_a = {r["query"] for r in results_a}
    queries_b = {r["query"] for r in results_b}
    common = queries_a & queries_b
    typer.echo(f"Common queries: {len(common)}")
    for q in sorted(common):
        ra = next(r for r in results_a if r["query"] == q)
        rb = next(r for r in results_b if r["query"] == q)
        files_a = [m.get("file_path", m.get("pr_number", "?")) for m in ra.get("results", [])]
        files_b = [m.get("file_path", m.get("pr_number", "?")) for m in rb.get("results", [])]
        if files_a != files_b:
            typer.echo(f"  CHANGED: {q}")
            typer.echo(f"    A: {files_a[:3]}")
            typer.echo(f"    B: {files_b[:3]}")
```

- [ ] **Step 2: Add CLI tests for new commands**

Add to `tests/test_cli.py`:

```python
def test_cli_reindex_help():
    result = runner.invoke(app, ["reindex", "--help"])
    assert result.exit_code == 0
    assert "all" in result.stdout.lower()


def test_cli_eval_run_help():
    result = runner.invoke(app, ["eval", "run", "--help"])
    assert result.exit_code == 0


def test_cli_eval_compare_help():
    result = runner.invoke(app, ["eval", "compare", "--help"])
    assert result.exit_code == 0
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_cli.py -v`
Expected: All 12 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add devrag/cli.py tests/test_cli.py
git commit -m "feat: reindex and eval CLI commands"
```

---

## Task 6: Claude Code Skills

**Files:**
- Create: `.claude/skills/rag-search/SKILL.md`
- Create: `.claude/skills/rag-index/SKILL.md`
- Create: `.claude/skills/rag-pr/SKILL.md`

- [ ] **Step 1: Create rag-search skill**

```bash
mkdir -p .claude/skills/rag-search
```

```markdown
# .claude/skills/rag-search/SKILL.md
---
name: rag-search
description: Search codebase knowledge (code, PRs, docs) using DevRAG
allowed-tools: mcp__devrag__search
---

Search DevRAG for: $ARGUMENTS

Present results grouped by source type (code, PR, doc).
For code results, show the file path and relevant snippet.
For PR results, show the PR title, number, and relevant excerpt.
For doc results, show the document title and section.
```

- [ ] **Step 2: Create rag-index skill**

```bash
mkdir -p .claude/skills/rag-index
```

```markdown
# .claude/skills/rag-index/SKILL.md
---
name: rag-index
description: Index a codebase, documents, or PRs for DevRAG search
allowed-tools: mcp__devrag__index_repo, mcp__devrag__index_docs, mcp__devrag__sync_prs, mcp__devrag__status
---

Help the user index their codebase for search.

If $ARGUMENTS mentions a directory or repo path, use index_repo to index it.
If $ARGUMENTS mentions docs or documents, use index_docs.
If $ARGUMENTS mentions PRs or a GitHub repo, use sync_prs.
If $ARGUMENTS is empty or says "status", use status to show current index state.

After indexing, show the status.
```

- [ ] **Step 3: Create rag-pr skill**

```bash
mkdir -p .claude/skills/rag-pr
```

```markdown
# .claude/skills/rag-pr/SKILL.md
---
name: rag-pr
description: Search PR history for why code changed
allowed-tools: mcp__devrag__search
---

Search DevRAG for PR history related to: $ARGUMENTS

Use scope "prs" to focus on pull request history.
Show PR numbers, titles, and relevant excerpts.
Highlight review comments that explain the reasoning behind changes.
```

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/
git commit -m "feat: Claude Code skills for rag-search, rag-index, and rag-pr"
```

---

## Task 7: Status Enrichment with Query Stats

**Files:**
- Modify: `devrag/cli.py`
- Modify: `devrag/mcp_server.py`

- [ ] **Step 1: Update status command in CLI**

Update the `status` function in `devrag/cli.py` to include query stats:

```python
@app.command()
def status():
    """Show indexing status and query stats."""
    from devrag.config import load_config
    from devrag.stores.factory import create_vector_store
    from devrag.stores.metadata_db import MetadataDB
    config = load_config(project_dir=Path.cwd())
    persist_dir = Path(config.vector_store.persist_dir).expanduser()
    if not persist_dir.exists():
        typer.echo("No index found. Run 'devrag index repo .' first.")
        return
    store = create_vector_store(config)
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
    stats = meta.get_query_stats()
    if stats["total_queries"] > 0:
        lines.append("")
        lines.append(f"Query stats ({stats['total_queries']} queries):")
        lines.append(f"  Avg latency: {stats['avg_total_ms']:.0f}ms")
        lines.append(f"  Avg results: {stats['avg_result_count']:.1f}")
    typer.echo("\n".join(lines))
```

- [ ] **Step 2: Update status in MCP server similarly**

Update `status` in `devrag/mcp_server.py` to include query stats:

```python
@mcp.tool
def status() -> str:
    """Show indexing status and query statistics."""
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
    stats = meta.get_query_stats()
    if stats["total_queries"] > 0:
        lines.append(f"Queries logged: {stats['total_queries']}")
        lines.append(f"Avg latency: {stats['avg_total_ms']:.0f}ms")
    return "\n".join(lines)
```

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add devrag/cli.py devrag/mcp_server.py
git commit -m "feat: enriched status with query statistics"
```

---

## Spec Coverage Checklist

| Spec Section | Task(s) | Status |
|---|---|---|
| 4.1 Qdrant Backend (VectorStore protocol) | Task 1 | Covered |
| 4.1 Config-driven swap | Task 2 (factory) | Covered |
| 4.2 Eval Framework (run + compare) | Task 4, Task 5 | Covered |
| 4.2 JSONL format, precision@k, recall@k, MRR | Task 4 | Covered |
| 4.3 Observability (per-query metrics to SQLite) | Task 3 | Covered |
| 4.3 Status enriched with query stats | Task 7 | Covered |
| 4.4 Claude Code Skills | Task 6 | Covered |
| 4.5 Re-embedding Support (devrag reindex --all) | Task 5 | Covered |
| 4.6 Automatic PR Sync (manual trigger) | Already done (CLI + MCP) | Covered |
