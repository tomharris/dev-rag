# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Search Strategy

Always use `/rag-search` (DevRAG) as the FIRST tool when answering codebase questions. Only fall back to Grep/Glob/Explore if RAG results are insufficient. DevRAG has semantic understanding of code structure, PR history, and documentation that keyword search misses.

## Build & Development Commands

```bash
uv sync                                    # Install dependencies
uv sync --extra dev                        # Install with dev dependencies
uv sync --extra qdrant                     # Install with Qdrant support
uv run devrag --help                       # CLI help
uv run devrag serve                        # Start MCP server
```

## Testing

```bash
uv run pytest tests/                       # All tests
uv run pytest tests/test_config.py -v      # Single test file
uv run pytest tests/test_config.py::test_name -v  # Single test
SKIP_INTEGRATION=0 uv run pytest tests/test_integration.py -v  # Integration (requires Ollama with nomic-embed-text)
```

Integration tests are skipped by default (`SKIP_INTEGRATION != "0"`). They require a running Ollama instance.

## Architecture

DevRAG is a local RAG system that ingests code, GitHub PRs, GitHub issues, Jira Cloud tickets, Slite pages, and documents, surfaced via CLI (Typer) and MCP server (FastMCP) for Claude Code integration.

### Three-Layer Pipeline

**Ingestion** (`devrag/ingest/`) - Converts sources into embedded chunks:
- `code_indexer.py` - AST-aware chunking via tree-sitter (50+ languages). Parses functions/classes/methods as semantic units, not raw text splits. Tracks file hashes in SQLite for incremental re-indexing. Supports multi-repo indexing — each repo is tracked independently via a `repo` column in `file_hashes`/`chunk_sources` tables and a `code_repos` registry. Default repo name is the directory name; override with `--name`.
- `pr_indexer.py` - GitHub PR sync with cursor-based incremental fetching. Chunks PR descriptions, diff hunks, and review comments separately. Truncates chunks to `chunk_max_tokens` before embedding.
- `issue_indexer.py` - GitHub issue sync with cursor-based incremental fetching. Chunks issue descriptions and comments separately into `issue_descriptions` and `issue_discussions` collections. Skips pull requests (GitHub Issues API returns PRs too). Supports `include_labels` / `exclude_labels` filtering.
- `jira_indexer.py` - Jira Cloud ticket sync with cursor-based incremental fetching via JQL. Chunks ticket descriptions and comments into `jira_descriptions` and `jira_discussions` collections. Uses `JiraClient` (`devrag/utils/jira_client.py`) for REST API v3 with basic auth. Converts Atlassian Document Format (ADF) to plain text for embedding.
- `slite_indexer.py` - Slite page sync via REST API with cursor-based incremental fetching. Fetches page content as markdown and chunks with section-aware splitting (reuses `split_markdown()` from `doc_indexer.py`). Stores in `slite_pages` collection. Uses `SliteClient` (`devrag/utils/slite_client.py`) for Bearer token auth. Supports channel filtering via `channel_ids` config.
- `doc_indexer.py` - Section-aware markdown/text splitting with token-based overlap.
- `embedder.py` - Ollama embedding wrapper (default: `nomic-embed-text`). Truncates oversized text to `max_tokens` (default 8192) before embedding. Skips empty texts and returns zero vectors for their positions.

**Storage** (`devrag/stores/`) - Pluggable vector store with metadata tracking:
- `base.py` defines a `VectorStore` Protocol. `chroma_store.py` (default) and `qdrant_store.py` implement it. `factory.py` selects backend from config.
- `metadata_db.py` - SQLite with WAL for file hashes, chunk-source mappings, PR/issue/Jira/Slite sync cursors, FTS5 index (BM25), and query metrics.
- Nine collections: `code_chunks`, `pr_diffs`, `pr_discussions`, `issue_descriptions`, `issue_discussions`, `jira_descriptions`, `jira_discussions`, `slite_pages`, `documents`.

**Retrieval** (`devrag/retrieve/`) - Query processing and result ranking:
- `query_router.py` - Regex-based intent classification routes queries to relevant collections.
- `hybrid_search.py` - Vector + BM25 search merged via Reciprocal Rank Fusion (RRF).
- `reranker.py` - Cross-encoder reranking of top-K candidates.

### Entry Points

- **CLI**: `devrag/cli.py` - Typer app with subcommands: `search`, `index` (repo/docs/prs/issues/jira/slite), `status`, `config`, `serve`, `reindex`, `eval`. `reindex --all` clears all vector collections, metadata, and sync cursors, then re-indexes all registered code repos (non-incremental). External sources (PRs, issues, Jira, Slite) must be re-synced manually after. `reindex --name <repo>` re-indexes a single repo without affecting other repos or external sources.
- **MCP Server**: `devrag/mcp_server.py` - FastMCP server with lazy-initialized global singletons for stores/embedders. Tools: `search()`, `index_repo()`, `index_docs()`, `sync_prs()`, `sync_issues()`, `sync_jira()`, `sync_slite()`, `status()`.
- **Types**: `devrag/types.py` - Core dataclasses (`Chunk`, `SearchResult`, `IndexStats`).

### Configuration

Nested dataclass hierarchy in `devrag/config.py`. Loaded from `~/.config/devrag/devrag.yaml` (user) merged with `.devrag.yaml` (project). Key sections: `EmbeddingConfig` (including `max_tokens` for context limit), `VectorStoreConfig`, `RetrievalConfig`, `CodeConfig`, `PrsConfig`, `IssuesConfig`, `JiraConfig`, `SliteConfig`, `DocumentsConfig`.

### Evaluation

`devrag/eval.py` computes precision@k, recall@k, and MRR. Test queries are JSONL with `expected_files`/`expected_prs`. CLI commands: `devrag eval run` and `devrag eval compare`.

## Key Patterns

- **Multi-repo indexing**: Multiple repos can be indexed into a unified search space. Each repo is tracked independently — indexing repo-b won't delete repo-a's data. Repos are identified by directory name (or `--name` override). `code_repos` table serves as a registry. `devrag index remove-repo <name>` removes a specific repo.
- **Incremental indexing**: File content hashes in SQLite skip unchanged files. PR, issue, Jira, and Slite sync use stored cursors.
- **VectorStore Protocol**: Adding a new backend means implementing `upsert()`, `query()`, `delete()` and registering in `factory.py`.
- **Text truncation safety**: PR chunks are truncated at creation time (`chunk_max_tokens`), and the embedder has a safety-net truncation at the model context limit (`embedding.max_tokens`). Empty/whitespace texts produce zero vectors.
- **Git-aware file discovery**: `devrag/utils/git.py` respects `.gitignore` and `.devragignore`.
- GitHub tokens come from env vars (configured via `prs.github_token_env` / `issues.github_token_env`), never stored in config files. Jira credentials similarly use `jira.jira_email_env` / `jira.jira_token_env`. Slite uses `slite.slite_token_env` (default: `SLITE_TOKEN`).
- **RAG-first routing**: The `rag-first` skill (`.claude/skills/rag-first/`) auto-triggers on codebase questions. A hookify rule (`.claude/hookify.rag-first-reminder.local.md`) warns if Grep/Glob are used without searching DevRAG first.

## Dependencies

Python 3.12+. Uses `hatchling` build backend. Key deps: `chromadb`, `fastmcp`, `tree-sitter` + `tree-sitter-language-pack`, `sentence-transformers`, `typer`, `httpx`, `gitpython`. Dev: `pytest`, `pytest-asyncio`, `respx`. Optional: `qdrant-client`.
