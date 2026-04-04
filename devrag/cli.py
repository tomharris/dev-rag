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
    stats = indexer.index_repo(Path(path).resolve(), incremental=not full)
    typer.echo(format_index_stats(stats))


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
    all_collections: bool = typer.Option(False, "--all", help="Re-embed all collections"),
):
    """Re-index everything. Use after changing embedding models."""
    from devrag.config import load_config
    from devrag.stores.metadata_db import MetadataDB
    if not all_collections:
        typer.echo("Use --all to clear all indexes and force re-embedding.")
        return
    config = load_config(project_dir=Path.cwd())
    db_dir = Path("~/.local/share/devrag").expanduser()
    db_dir.mkdir(parents=True, exist_ok=True)
    meta = MetadataDB(str(db_dir / "metadata.db"))
    for file_path in meta.get_all_indexed_files():
        meta.remove_file(file_path)
    typer.echo("Cleared all file hashes. Run 'devrag index repo .' to re-index code.")


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
