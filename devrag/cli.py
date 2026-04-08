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
    from devrag.config import load_config
    from devrag.ingest.embedder import OllamaEmbedder
    from devrag.retrieve.hybrid_search import HybridSearch
    from devrag.retrieve.reranker import Reranker
    from devrag.stores.factory import create_vector_store
    from devrag.stores.metadata_db import MetadataDB
    config = load_config(project_dir=Path.cwd())
    store = create_vector_store(config)
    db_dir = Path("~/.local/share/devrag").expanduser()
    db_dir.mkdir(parents=True, exist_ok=True)
    meta = MetadataDB(str(db_dir / "metadata.db"))
    embedder = OllamaEmbedder(model=config.embedding.model, ollama_url=config.embedding.ollama_url, batch_size=config.embedding.batch_size, max_tokens=config.embedding.max_tokens)
    hybrid = HybridSearch(store, meta, embedder, rrf_k=config.retrieval.rrf_k)
    reranker = Reranker(model_name=config.retrieval.reranker_model) if config.retrieval.rerank else None
    return hybrid, reranker, config


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query text"),
    scope: str = typer.Option("all", help="Scope: all, code, prs, issues, docs"),
    top_k: int = typer.Option(0, help="Number of results (0 = use configured default)"),
):
    """Search code, PRs, issues, and docs."""
    from devrag.retrieve.hybrid_search import deduplicate_results
    from devrag.retrieve.query_router import QueryRouter
    from devrag.utils.formatters import format_search_results
    hybrid, reranker, config = _get_search_components()
    top_k = top_k if top_k > 0 else config.retrieval.final_k
    router = QueryRouter()
    collections = router.route(query, scope=scope)
    candidates = hybrid.search(query, top_k=config.retrieval.top_k, collections=collections)
    if reranker and candidates:
        results = reranker.rerank(query, candidates, top_k=top_k)
    else:
        results = candidates[:top_k]
    results = deduplicate_results(results, max_per_source=config.retrieval.max_per_source)
    typer.echo(format_search_results(results))


@index_app.command("repo")
def index_repo(
    path: str = typer.Argument(".", help="Path to repository"),
    full: bool = typer.Option(False, "--full", help="Full re-index (skip incremental)"),
    name: str = typer.Option("", "--name", help="Repo name (default: directory name)"),
):
    """Index a local code repository."""
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
    embedder = OllamaEmbedder(model=config.embedding.model, ollama_url=config.embedding.ollama_url, batch_size=config.embedding.batch_size, max_tokens=config.embedding.max_tokens)
    indexer = CodeIndexer(store, meta, embedder, config.code)
    stats = indexer.index_repo(Path(path).resolve(), incremental=not full, repo_name=name)
    typer.echo(format_index_stats(stats))


@index_app.command("remove-repo")
def remove_repo(
    name: str = typer.Argument(..., help="Repo name to remove from index"),
):
    """Remove all indexed data for a repository."""
    from devrag.config import load_config
    from devrag.stores.factory import create_vector_store
    from devrag.stores.metadata_db import MetadataDB
    config = load_config(project_dir=Path.cwd())
    store = create_vector_store(config)
    db_dir = Path("~/.local/share/devrag").expanduser()
    meta = MetadataDB(str(db_dir / "metadata.db"))
    chunk_ids = meta._conn.execute(
        "SELECT chunk_id FROM chunk_sources WHERE repo = ?", (name,)
    ).fetchall()
    if chunk_ids:
        ids = [r[0] for r in chunk_ids]
        store.delete("code_chunks", ids)
    meta.remove_repo(name)
    typer.echo(f"Removed repo '{name}' ({len(chunk_ids)} chunks deleted).")


@index_app.command("docs")
def index_docs_cmd(
    path: str = typer.Argument(..., help="Path to docs directory"),
    glob: str = typer.Option("**/*.md", help="Glob pattern(s), comma-separated"),
):
    """Index a directory of documents."""
    from devrag.config import load_config
    from devrag.ingest.doc_indexer import DocIndexer
    from devrag.ingest.embedder import OllamaEmbedder
    from devrag.stores.factory import create_vector_store
    from devrag.stores.metadata_db import MetadataDB
    from devrag.utils.formatters import format_doc_index_stats
    config = load_config(project_dir=Path.cwd())
    store = create_vector_store(config)
    db_dir = Path("~/.local/share/devrag").expanduser()
    db_dir.mkdir(parents=True, exist_ok=True)
    meta = MetadataDB(str(db_dir / "metadata.db"))
    embedder = OllamaEmbedder(model=config.embedding.model, ollama_url=config.embedding.ollama_url, batch_size=config.embedding.batch_size, max_tokens=config.embedding.max_tokens)
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
    from devrag.stores.factory import create_vector_store
    from devrag.stores.metadata_db import MetadataDB
    from devrag.utils.formatters import format_pr_sync_stats
    from devrag.utils.github import GitHubClient
    config = load_config(project_dir=Path.cwd())
    token = os.environ.get(config.prs.github_token_env)
    if not token:
        typer.echo(f"Error: {config.prs.github_token_env} environment variable not set.", err=True)
        raise typer.Exit(1)
    store = create_vector_store(config)
    db_dir = Path("~/.local/share/devrag").expanduser()
    db_dir.mkdir(parents=True, exist_ok=True)
    meta = MetadataDB(str(db_dir / "metadata.db"))
    embedder = OllamaEmbedder(model=config.embedding.model, ollama_url=config.embedding.ollama_url, batch_size=config.embedding.batch_size, max_tokens=config.embedding.max_tokens)
    days = int(since.rstrip("d"))
    github = GitHubClient(token=token)
    indexer = PRIndexer(store, meta, embedder, github, chunk_max_tokens=config.prs.chunk_max_tokens)
    stats = indexer.sync(repo, since_days=days)
    typer.echo(format_pr_sync_stats(stats))


@index_app.command("issues")
def index_issues(
    repo: str = typer.Argument(..., help="GitHub repo (owner/name)"),
    since: str = typer.Option("90d", help="Lookback period (e.g. 90d)"),
):
    """Sync GitHub issues for a repository."""
    from devrag.config import load_config
    from devrag.ingest.embedder import OllamaEmbedder
    from devrag.ingest.issue_indexer import IssueIndexer
    from devrag.stores.factory import create_vector_store
    from devrag.stores.metadata_db import MetadataDB
    from devrag.utils.formatters import format_issue_sync_stats
    from devrag.utils.github import GitHubClient
    config = load_config(project_dir=Path.cwd())
    token = os.environ.get(config.issues.github_token_env)
    if not token:
        typer.echo(f"Error: {config.issues.github_token_env} environment variable not set.", err=True)
        raise typer.Exit(1)
    store = create_vector_store(config)
    db_dir = Path("~/.local/share/devrag").expanduser()
    db_dir.mkdir(parents=True, exist_ok=True)
    meta = MetadataDB(str(db_dir / "metadata.db"))
    embedder = OllamaEmbedder(model=config.embedding.model, ollama_url=config.embedding.ollama_url, batch_size=config.embedding.batch_size, max_tokens=config.embedding.max_tokens)
    days = int(since.rstrip("d"))
    github = GitHubClient(token=token)
    indexer = IssueIndexer(store, meta, embedder, github, chunk_max_tokens=config.issues.chunk_max_tokens,
                           include_labels=config.issues.include_labels, exclude_labels=config.issues.exclude_labels)
    stats = indexer.sync(repo, since_days=days)
    typer.echo(format_issue_sync_stats(stats))


@index_app.command("jira")
def index_jira(
    since: str = typer.Option("90d", help="Lookback period (e.g. 90d)"),
):
    """Sync Jira Cloud tickets based on configured JQL filter."""
    from devrag.config import load_config
    from devrag.ingest.embedder import OllamaEmbedder
    from devrag.ingest.jira_indexer import JiraIndexer
    from devrag.stores.factory import create_vector_store
    from devrag.stores.metadata_db import MetadataDB
    from devrag.utils.formatters import format_jira_sync_stats
    from devrag.utils.jira_client import JiraClient
    config = load_config(project_dir=Path.cwd())
    if not config.jira.instance_url:
        typer.echo("Error: jira.instance_url not configured in .devrag.yaml", err=True)
        raise typer.Exit(1)
    if not config.jira.jql:
        typer.echo("Error: jira.jql not configured in .devrag.yaml", err=True)
        raise typer.Exit(1)
    email = os.environ.get(config.jira.jira_email_env)
    token = os.environ.get(config.jira.jira_token_env)
    if not email or not token:
        typer.echo(f"Error: {config.jira.jira_email_env} and {config.jira.jira_token_env} environment variables must be set.", err=True)
        raise typer.Exit(1)
    store = create_vector_store(config)
    db_dir = Path("~/.local/share/devrag").expanduser()
    db_dir.mkdir(parents=True, exist_ok=True)
    meta = MetadataDB(str(db_dir / "metadata.db"))
    embedder = OllamaEmbedder(model=config.embedding.model, ollama_url=config.embedding.ollama_url, batch_size=config.embedding.batch_size, max_tokens=config.embedding.max_tokens)
    days = int(since.rstrip("d"))
    jira = JiraClient(instance_url=config.jira.instance_url, email=email, api_token=token)
    indexer = JiraIndexer(store, meta, embedder, jira, chunk_max_tokens=config.jira.chunk_max_tokens)
    stats = indexer.sync(config.jira.instance_url, config.jira.jql, since_days=days)
    typer.echo(format_jira_sync_stats(stats))


@index_app.command("slite")
def index_slite(
    since: str = typer.Option("90d", help="Lookback period (e.g. 90d)"),
):
    """Sync Slite pages for the configured workspace."""
    from devrag.config import load_config
    from devrag.ingest.embedder import OllamaEmbedder
    from devrag.ingest.slite_indexer import SliteIndexer
    from devrag.stores.factory import create_vector_store
    from devrag.stores.metadata_db import MetadataDB
    from devrag.utils.formatters import format_slite_sync_stats
    from devrag.utils.slite_client import SliteClient
    config = load_config(project_dir=Path.cwd())
    token = os.environ.get(config.slite.slite_token_env)
    if not token:
        typer.echo(f"Error: {config.slite.slite_token_env} environment variable not set.", err=True)
        raise typer.Exit(1)
    store = create_vector_store(config)
    db_dir = Path("~/.local/share/devrag").expanduser()
    db_dir.mkdir(parents=True, exist_ok=True)
    meta = MetadataDB(str(db_dir / "metadata.db"))
    embedder = OllamaEmbedder(model=config.embedding.model, ollama_url=config.embedding.ollama_url, batch_size=config.embedding.batch_size, max_tokens=config.embedding.max_tokens)
    days = int(since.rstrip("d"))
    slite = SliteClient(api_token=token)
    indexer = SliteIndexer(store, meta, embedder, slite, chunk_max_tokens=config.slite.chunk_max_tokens,
                           chunk_overlap_tokens=config.slite.chunk_overlap_tokens,
                           channel_ids=config.slite.channel_ids)
    stats = indexer.sync(since_days=days)
    typer.echo(format_slite_sync_stats(stats))


@app.command()
def status():
    """Show indexing status."""
    from devrag.config import load_config
    from devrag.stores.factory import create_vector_store
    from devrag.stores.metadata_db import MetadataDB
    config = load_config(project_dir=Path.cwd())
    store = create_vector_store(config)
    db_dir = Path("~/.local/share/devrag").expanduser()
    db_dir.mkdir(parents=True, exist_ok=True)
    meta = MetadataDB(str(db_dir / "metadata.db"))
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
        f"Code chunks: {store.count('code_chunks')}",
        f"PR diff chunks: {store.count('pr_diffs')}",
        f"PR discussion chunks: {store.count('pr_discussions')}",
        f"Issue description chunks: {store.count('issue_descriptions')}",
        f"Issue discussion chunks: {store.count('issue_discussions')}",
        f"Jira description chunks: {store.count('jira_descriptions')}",
        f"Jira discussion chunks: {store.count('jira_discussions')}",
        f"Slite page chunks: {store.count('slite_pages')}",
        f"Document chunks: {store.count('documents')}",
    ]
    stats = meta.get_query_stats()
    if stats["total_queries"] > 0:
        lines.append("")
        lines.append(f"Query stats ({stats['total_queries']} queries):")
        lines.append(f"  Avg latency: {stats['avg_total_ms']:.0f}ms")
        lines.append(f"  Avg results: {stats['avg_result_count']:.1f}")
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


eval_app = typer.Typer(help="Evaluate retrieval quality.")
app.add_typer(eval_app, name="eval")


@app.command()
def reindex(
    all_collections: bool = typer.Option(False, "--all", help="Clear all indexes, sync cursors, and vector collections, then re-index known code repos"),
    name: str = typer.Option("", "--name", help="Re-index a single repo by name (preserves other repos and external sources)"),
):
    """Reset and re-index everything, or re-index a single repo by name."""
    from devrag.config import load_config
    from devrag.ingest.code_indexer import CodeIndexer
    from devrag.ingest.embedder import OllamaEmbedder
    from devrag.stores.factory import create_vector_store
    from devrag.stores.metadata_db import MetadataDB
    from devrag.utils.formatters import format_index_stats

    if not all_collections and not name:
        typer.echo("Usage: devrag reindex --all | --name <repo>")
        raise typer.Exit(1)

    config = load_config(project_dir=Path.cwd())
    store = create_vector_store(config)
    db_dir = Path("~/.local/share/devrag").expanduser()
    db_dir.mkdir(parents=True, exist_ok=True)
    meta = MetadataDB(str(db_dir / "metadata.db"))

    if name:
        # Single-repo reindex: remove then re-index non-incrementally
        repos = meta.get_all_repos()
        match = next(((n, p) for n, p in repos if n == name), None)
        if not match:
            registered = [n for n, _ in repos]
            typer.echo(f"Repo '{name}' not found. Registered repos: {', '.join(registered) or '(none)'}")
            raise typer.Exit(1)
        repo_name, repo_path = match
        repo_dir = Path(repo_path)
        if not repo_dir.exists():
            typer.echo(f"Directory not found: {repo_path}")
            raise typer.Exit(1)

        # Remove existing chunks and metadata for this repo
        chunk_ids = meta._conn.execute(
            "SELECT chunk_id FROM chunk_sources WHERE repo = ?", (name,)
        ).fetchall()
        if chunk_ids:
            store.delete("code_chunks", [r[0] for r in chunk_ids])
        meta.remove_repo(name)
        typer.echo(f"Cleared {len(chunk_ids)} chunks for '{name}'.")

        # Re-index
        embedder = OllamaEmbedder(model=config.embedding.model, ollama_url=config.embedding.ollama_url,
                                  batch_size=config.embedding.batch_size, max_tokens=config.embedding.max_tokens)
        indexer = CodeIndexer(store, meta, embedder, config.code)
        typer.echo(f"Re-indexing {repo_name} ({repo_path})...")
        stats = indexer.index_repo(repo_dir, incremental=False, repo_name=repo_name)
        typer.echo(format_index_stats(stats))
        return

    # --all: full reset
    from devrag.retrieve.query_router import ALL_COLLECTIONS

    # Save registered repos before clearing
    repos = meta.get_all_repos()

    # Clear all metadata and vector collections
    meta.reset_all()
    for coll in ALL_COLLECTIONS:
        store.delete_collection(coll)
    typer.echo("Cleared all indexes and sync cursors.")

    # Auto-reindex known code repos
    if repos:
        embedder = OllamaEmbedder(model=config.embedding.model, ollama_url=config.embedding.ollama_url,
                                  batch_size=config.embedding.batch_size, max_tokens=config.embedding.max_tokens)
        indexer = CodeIndexer(store, meta, embedder, config.code)
        total_chunks = 0
        for repo_name, repo_path in repos:
            repo_dir = Path(repo_path)
            if not repo_dir.exists():
                typer.echo(f"  Skipping {repo_name} ({repo_path}) — directory not found.")
                continue
            typer.echo(f"  Re-indexing {repo_name} ({repo_path})...")
            stats = indexer.index_repo(repo_dir, incremental=False, repo_name=repo_name)
            total_chunks += stats.chunks_created
            typer.echo(f"    {format_index_stats(stats)}")
        typer.echo(f"Re-indexed {len(repos)} code repo(s) ({total_chunks} chunks).")
    else:
        typer.echo("No code repos registered. Run 'devrag index repo .' to index code.")

    typer.echo("\nRe-sync external sources as needed:")
    typer.echo("  devrag index prs <owner/repo>")
    typer.echo("  devrag index issues <owner/repo>")
    typer.echo("  devrag index jira")
    typer.echo("  devrag index slite")


@eval_app.command("run")
def eval_run(
    queries_file: str = typer.Argument(..., help="Path to test queries JSONL file"),
    output: str = typer.Option("results.jsonl", help="Output file for results"),
    top_k: int = typer.Option(5, help="Number of results per query"),
):
    """Run eval queries and compute metrics."""
    from devrag.eval import load_test_queries, compute_metrics, save_results
    from devrag.retrieve.query_router import QueryRouter
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


if __name__ == "__main__":
    app()
