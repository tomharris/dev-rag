from __future__ import annotations

import os
from pathlib import Path

from fastmcp import FastMCP

from devrag.config import DevragConfig, load_config
from devrag.ingest.code_indexer import CodeIndexer
from devrag.ingest.doc_indexer import DocIndexer
from devrag.ingest.embedder import OllamaEmbedder
from devrag.ingest.issue_indexer import IssueIndexer
from devrag.ingest.jira_indexer import JiraIndexer
from devrag.ingest.pr_indexer import PRIndexer
from devrag.ingest.session_indexer import SessionsIndexer
from devrag.ingest.slite_indexer import SliteIndexer
from devrag.retrieve.hybrid_search import HybridSearch, deduplicate_results
from devrag.retrieve.query_router import QueryRouter
from devrag.retrieve.reranker import Reranker
from devrag.stores.factory import create_vector_store
from devrag.stores.metadata_db import MetadataDB
from devrag.utils.formatters import format_doc_index_stats, format_index_stats, format_issue_sync_stats, format_jira_sync_stats, format_pr_sync_stats, format_search_results, format_session_sync_stats, format_slite_sync_stats
from devrag.utils.github import GitHubClient
from devrag.utils.jira_client import JiraClient
from devrag.utils.slite_client import SliteClient

mcp = FastMCP("DevRAG")

_config: DevragConfig | None = None
_vector_store = None
_metadata_db: MetadataDB | None = None
_embedder: OllamaEmbedder | None = None
_reranker: Reranker | None = None


def _get_config() -> DevragConfig:
    global _config
    if _config is None:
        _config = load_config(project_dir=Path.cwd())
    return _config


def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = create_vector_store(_get_config())
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
            max_tokens=config.embedding.max_tokens,
        )
    return _embedder


def _get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        config = _get_config()
        _reranker = Reranker(model_name=config.retrieval.reranker_model)
    return _reranker


@mcp.tool
def search(
    query: str,
    scope: str = "all",
    top_k: int = 0,
    repo: str = "",
    chunk_type: str = "",
    pr_number: int = 0,
    issue_number: int = 0,
    ticket_key: str = "",
    page_id: str = "",
    session_id: str = "",
    file_path: str = "",
) -> str:
    """Search code, PRs, issues, and docs using hybrid retrieval.

    Args:
        query: The search query.
        scope: What to search. "all" auto-routes by intent,
               "code" searches code only, "prs" searches PRs only,
               "issues" searches issues only, "jira", "slite", "docs",
               "sessions" for Claude Code session logs.
        top_k: Number of results to return (0 = use configured default).
        repo: Optional repo name to filter results (empty = all repos).
        chunk_type: Optional filter by chunk type. Known values:
            "description" (PR/issue/jira), "comment" (issue/jira),
            "diff" (PR), "review_comment" (PR), "slite_page",
            "document", "session_exchange". Code chunks have no
            chunk_type — filtering by it will exclude them.
        pr_number: Optional filter to a specific PR number.
        issue_number: Optional filter to a specific issue number.
        ticket_key: Optional Jira ticket key (e.g. "PROJ-123").
        page_id: Optional Slite page id.
        session_id: Optional Claude Code session UUID.
        file_path: Optional exact file path match.

    All filter params are AND-combined against vector-store metadata.
    Note: the BM25/FTS5 leg of hybrid search does not apply these
    filters; matching vectors still dominate ranking via RRF.
    """
    config = _get_config()
    top_k = top_k if top_k > 0 else config.retrieval.final_k
    router = QueryRouter()
    collections = router.route(query, scope=scope)
    where: dict = {}
    if repo:
        where["repo"] = repo
    if chunk_type:
        where["chunk_type"] = chunk_type
    if pr_number:
        where["pr_number"] = pr_number
    if issue_number:
        where["issue_number"] = issue_number
    if ticket_key:
        where["ticket_key"] = ticket_key
    if page_id:
        where["page_id"] = page_id
    if session_id:
        where["session_id"] = session_id
    if file_path:
        where["file_path"] = file_path
    where = where or None
    hybrid = HybridSearch(
        vector_store=_get_vector_store(),
        metadata_db=_get_metadata_db(),
        embedder=_get_embedder(),
        rrf_k=config.retrieval.rrf_k,
    )
    candidates = hybrid.search(query, top_k=config.retrieval.top_k, collections=collections, where=where)
    if config.retrieval.rerank and candidates:
        reranker = _get_reranker()
        results = reranker.rerank(query, candidates, top_k=top_k)
    else:
        results = candidates[:top_k]
    results = deduplicate_results(results, max_per_source=config.retrieval.max_per_source)
    return format_search_results(results)


@mcp.tool
def index_repo(path: str = ".", incremental: bool = True, name: str = "") -> str:
    """Index a local code repository using AST-aware chunking.

    Parses source files with tree-sitter, extracts functions/classes/methods,
    and stores embeddings for semantic search. Uses incremental indexing
    to skip unchanged files.

    Multiple repos can be indexed — each is tracked independently.

    Args:
        path: Path to the repository root.
        incremental: Skip unchanged files (default True).
        name: Repo name for multi-repo support (default: directory name).
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
    stats = indexer.index_repo(repo_path, incremental=incremental, repo_name=name)
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
def sync_prs(repo: str, since_days: int | None = None) -> str:
    """Sync GitHub PRs for a repository.

    Fetches PR diffs, descriptions, and review comments, then indexes
    them for search. Uses cursor-based sync to avoid re-fetching.

    If `since_days` is provided, it overrides the stored cursor (use for
    backfill). If omitted, sync is incremental from the cursor (or a 90-day
    lookback on first run).

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
        chunk_max_tokens=config.prs.chunk_max_tokens,
    )
    stats = indexer.sync(repo, since_days=since_days)
    return format_pr_sync_stats(stats)


@mcp.tool
def sync_issues(repo: str, since_days: int = 90) -> str:
    """Sync GitHub issues for a repository.

    Fetches issue descriptions and comments, then indexes them for search.
    Uses cursor-based sync to avoid re-fetching. Skips pull requests.

    Requires GITHUB_TOKEN environment variable.
    """
    config = _get_config()
    token = os.environ.get(config.issues.github_token_env)
    if not token:
        return f"Error: {config.issues.github_token_env} environment variable not set."
    github = GitHubClient(token=token)
    indexer = IssueIndexer(
        vector_store=_get_vector_store(),
        metadata_db=_get_metadata_db(),
        embedder=_get_embedder(),
        github_client=github,
        chunk_max_tokens=config.issues.chunk_max_tokens,
        include_labels=config.issues.include_labels,
        exclude_labels=config.issues.exclude_labels,
    )
    stats = indexer.sync(repo, since_days=since_days)
    return format_issue_sync_stats(stats)


@mcp.tool
def sync_jira(since_days: int = 90) -> str:
    """Sync Jira Cloud tickets based on configured JQL filter.

    Fetches ticket descriptions and comments, then indexes them for search.
    Uses cursor-based sync to avoid re-fetching.

    Requires JIRA_EMAIL and JIRA_TOKEN environment variables,
    plus jira.instance_url and jira.jql configured in .devrag.yaml.
    """
    config = _get_config()
    if not config.jira.instance_url:
        return "Error: jira.instance_url not configured in .devrag.yaml."
    if not config.jira.jql:
        return "Error: jira.jql not configured in .devrag.yaml."
    email = os.environ.get(config.jira.jira_email_env)
    token = os.environ.get(config.jira.jira_token_env)
    if not email or not token:
        return f"Error: {config.jira.jira_email_env} and {config.jira.jira_token_env} environment variables must be set."
    jira = JiraClient(instance_url=config.jira.instance_url, email=email, api_token=token)
    indexer = JiraIndexer(
        vector_store=_get_vector_store(),
        metadata_db=_get_metadata_db(),
        embedder=_get_embedder(),
        jira_client=jira,
        chunk_max_tokens=config.jira.chunk_max_tokens,
    )
    stats = indexer.sync(config.jira.instance_url, config.jira.jql, since_days=since_days)
    return format_jira_sync_stats(stats)


@mcp.tool
def sync_slite(since_days: int = 90) -> str:
    """Sync Slite pages for the configured workspace.

    Fetches page content as markdown and indexes with section-aware chunking.
    Uses cursor-based sync to avoid re-fetching unchanged pages.
    Filters by configured channel IDs if set.

    Requires SLITE_TOKEN environment variable.
    """
    config = _get_config()
    token = os.environ.get(config.slite.slite_token_env)
    if not token:
        return f"Error: {config.slite.slite_token_env} environment variable not set."
    slite = SliteClient(api_token=token)
    indexer = SliteIndexer(
        vector_store=_get_vector_store(),
        metadata_db=_get_metadata_db(),
        embedder=_get_embedder(),
        slite_client=slite,
        chunk_max_tokens=config.slite.chunk_max_tokens,
        chunk_overlap_tokens=config.slite.chunk_overlap_tokens,
        channel_ids=config.slite.channel_ids,
    )
    stats = indexer.sync(since_days=since_days)
    return format_slite_sync_stats(stats)


@mcp.tool
def sync_sessions(since_days: int = 0) -> str:
    """Sync local Claude Code JSONL session logs.

    Indexes past conversations with Claude Code as a searchable knowledge
    source. Walks `sessions.logs_dir` (default `~/.claude/projects`) for
    `*.jsonl` files modified since the stored cursor (or `backfill_days`
    on first run) and chunks each user→assistant exchange.

    Args:
        since_days: If > 0, overrides the stored cursor (use for backfill).
            0 means incremental from cursor (or backfill_days on first run).
    """
    config = _get_config()
    logs_dir = Path(config.sessions.logs_dir).expanduser()
    indexer = SessionsIndexer(
        vector_store=_get_vector_store(),
        metadata_db=_get_metadata_db(),
        embedder=_get_embedder(),
        logs_dir=logs_dir,
        chunk_max_tokens=config.sessions.chunk_max_tokens,
        backfill_days=config.sessions.backfill_days,
    )
    stats = indexer.sync(since_days=since_days if since_days > 0 else None)
    return format_session_sync_stats(stats)


@mcp.tool
def status() -> str:
    """Show indexing status: files, code chunks, PRs, issues, and documents."""
    store = _get_vector_store()
    meta = _get_metadata_db()
    chunk_count = store.count("code_chunks")
    pr_diff_count = store.count("pr_diffs")
    pr_disc_count = store.count("pr_discussions")
    issue_desc_count = store.count("issue_descriptions")
    issue_disc_count = store.count("issue_discussions")
    jira_desc_count = store.count("jira_descriptions")
    jira_disc_count = store.count("jira_discussions")
    slite_count = store.count("slite_pages")
    doc_count = store.count("documents")
    session_count = store.count("session_logs")
    indexed_files = meta.get_all_indexed_files()
    repos = meta.get_all_repos()
    lines = [
        f"Indexed files: {len(indexed_files)}",
    ]
    if repos:
        lines.append(f"Indexed repos: {len(repos)}")
        for repo_name, repo_path in repos:
            repo_files = meta.get_indexed_files_for_repo(repo_name)
            lines.append(f"  {repo_name}: {len(repo_files)} files ({repo_path})")
    lines += [
        f"Code chunks: {chunk_count}",
        f"PR diff chunks: {pr_diff_count}",
        f"PR discussion chunks: {pr_disc_count}",
        f"Issue description chunks: {issue_desc_count}",
        f"Issue discussion chunks: {issue_disc_count}",
        f"Jira description chunks: {jira_desc_count}",
        f"Jira discussion chunks: {jira_disc_count}",
        f"Slite page chunks: {slite_count}",
        f"Document chunks: {doc_count}",
        f"Session log chunks: {session_count}",
    ]
    stats = meta.get_query_stats()
    if stats["total_queries"] > 0:
        lines.append(f"Queries logged: {stats['total_queries']}")
        lines.append(f"Avg latency: {stats['avg_total_ms']:.0f}ms")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
