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
