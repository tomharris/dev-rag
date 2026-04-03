from __future__ import annotations

import os
from pathlib import Path

from fastmcp import FastMCP

from devrag.config import DevragConfig, load_config
from devrag.ingest.code_indexer import CodeIndexer
from devrag.ingest.doc_indexer import DocIndexer
from devrag.ingest.embedder import OllamaEmbedder
from devrag.ingest.pr_indexer import PRIndexer
from devrag.retrieve.hybrid_search import HybridSearch
from devrag.retrieve.query_router import QueryRouter
from devrag.retrieve.reranker import Reranker
from devrag.stores.chroma_store import ChromaStore
from devrag.stores.metadata_db import MetadataDB
from devrag.utils.formatters import format_doc_index_stats, format_index_stats, format_pr_sync_stats, format_search_results
from devrag.utils.github import GitHubClient

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
def search(query: str, scope: str = "all", top_k: int = 5) -> str:
    """Search code, PRs, and docs using hybrid retrieval.

    Args:
        query: The search query.
        scope: What to search. "all" auto-routes by intent,
               "code" searches code only, "prs" searches PRs only.
        top_k: Number of results to return.
    """
    config = _get_config()
    router = QueryRouter()
    collections = router.route(query, scope=scope)
    hybrid = HybridSearch(
        vector_store=_get_vector_store(),
        metadata_db=_get_metadata_db(),
        embedder=_get_embedder(),
    )
    candidates = hybrid.search(query, top_k=config.retrieval.top_k, collections=collections)
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


@mcp.tool
def sync_prs(repo: str, since_days: int = 90) -> str:
    """Sync GitHub PRs for a repository.

    Fetches PR diffs, descriptions, and review comments, then indexes
    them for search. Uses cursor-based sync to avoid re-fetching.

    Requires GITHUB_TOKEN environment variable.
    """
    config = _get_config()
    token = os.environ.get(config.prs.github_token_env)
    if not token:
        return f"Error: {config.prs.github_token_env} environment variable not set."
    github = GitHubClient(token=token)
    indexer = PRIndexer(
        vector_store=_get_vector_store(),
        metadata_db=_get_metadata_db(),
        embedder=_get_embedder(),
        github_client=github,
    )
    stats = indexer.sync(repo, since_days=since_days)
    return format_pr_sync_stats(stats)


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


if __name__ == "__main__":
    mcp.run()
