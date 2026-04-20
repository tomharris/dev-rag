# DevRAG

A local RAG system for developer teams that ingests **code**, **GitHub PRs**, **GitHub issues**, **Jira Cloud tickets**, **Slite pages**, **documents**, and **Claude Code session logs** — surfaced through Claude Code slash commands and a standalone CLI. Designed for zero-friction adoption: index your repo, search immediately.

## Why DevRAG?

Most code search tools treat source files as plain text. DevRAG uses **tree-sitter AST parsing** to chunk code along semantic boundaries (functions, classes, methods), so searches return meaningful units instead of fragments split mid-function.

It combines **vector search** (semantic) with **BM25** (keyword) via Reciprocal Rank Fusion, which is critical for code where exact identifiers like `refreshToken` matter as much as the concept of "token refresh."

PR history, issues, Jira tickets, and Slite wiki pages are first-class data sources — encoding *why* code changed, what bugs and features are tracked, what project management context exists, and what internal knowledge base content says.

## Quick Start

### Install

Install directly from GitHub (no clone required):

```bash
# As a global CLI tool (recommended)
uv tool install git+https://github.com/tomharris/dev-rag.git

# Or run without installing
uvx --from git+https://github.com/tomharris/dev-rag.git devrag --help

# Or add as a project dependency
uv add git+https://github.com/tomharris/dev-rag.git
# pip install git+https://github.com/tomharris/dev-rag.git
```

For development (from source):

```bash
git clone https://github.com/tomharris/dev-rag.git
cd dev-rag
uv sync
uv run devrag --help
```

### Index and Search

```bash
# Index the current repository
devrag index repo .

# Search across code, PRs, issues, and docs
devrag search "how does authentication work"

# Check what's indexed
devrag status
```

### Requirements

- Python 3.12+
- [Ollama](https://ollama.com/) running locally with `nomic-embed-text` pulled:
  ```bash
  ollama pull nomic-embed-text
  ```

## CLI Reference

### Search

```bash
devrag search "query"                      # Search all sources
devrag search "query" --scope code         # Code only
devrag search "query" --scope prs          # PR history only
devrag search "query" --scope issues       # Issues only
devrag search "query" --scope slite        # Slite pages only
devrag search "query" --scope docs         # Documents only
devrag search "query" --scope sessions     # Claude Code session logs only
devrag search "query" --top-k 10           # More results

# Narrow with metadata filters (all optional, AND-combined)
devrag search "auth" --repo myrepo --chunk-type diff
devrag search "bug" --pr-number 1234
devrag search "refund flow" --ticket-key PROJ-123
devrag search "deploy" --session-id <uuid>
```

The query router automatically classifies intent and targets relevant collections. "Why did we switch to Redis?" routes to PR history; "is there a bug with login?" routes to issues and Jira tickets; "what sprint is auth in?" routes to Jira; "what does our wiki say about deploys?" routes to Slite pages and documents; "how does the auth middleware work?" routes to code.

### Indexing

```bash
# Code (AST-aware, incremental by default)
devrag index repo .                        # Current directory
devrag index repo /path/to/repo            # Specific path
devrag index repo . --full                 # Force full re-index
devrag index repo . --name my-repo         # Name for multi-repo support
devrag index remove-repo my-repo           # Remove a repo from the index

# Documents (section-aware splitting)
devrag index docs ./specs                  # Default: **/*.md
devrag index docs ./docs --glob "**/*.txt,**/*.rst"

# GitHub PRs (incremental with cursor tracking)
export GITHUB_TOKEN=ghp_xxx
devrag index prs owner/repo                # Incremental from cursor (90d on first run)
devrag index prs owner/repo --since 180d   # Override cursor for backfill

# GitHub Issues (incremental, skips PRs)
devrag index issues owner/repo             # Last 90 days
devrag index issues owner/repo --since 30d # Custom lookback

# Jira Cloud tickets (incremental via JQL)
export JIRA_EMAIL=you@company.com
export JIRA_TOKEN=your-api-token
devrag index jira                          # Uses JQL from .devrag.yaml
devrag index jira --since 180d             # Custom lookback

# Slite pages (incremental, section-aware chunking)
export SLITE_TOKEN=your-api-token
devrag index slite                         # Last 90 days, all channels
devrag index slite --since 30d             # Custom lookback

# Claude Code session logs (local JSONL, incremental via file mtime)
devrag index sessions                      # Incremental from cursor (60d first run)
devrag index sessions --since 30d          # Override cursor
devrag index sessions --logs-dir /custom/path
```

Code indexing is **incremental** — file content hashes are tracked in SQLite, so unchanged files are skipped on re-index.

### Other Commands

```bash
devrag status                              # Show index stats
devrag serve                               # Start MCP server for Claude Code
devrag reindex --all                       # Clear all indexes and re-embed code repos (re-sync PRs/issues/Jira/Slite manually)
devrag reindex --name my_project           # Re-index a single repo (preserves other repos and external sources)
devrag config set embedding.model nomic-embed-text
devrag config get vector_store.backend
```

### Evaluation

Test retrieval quality with a JSONL file of queries and expected results:

```bash
devrag eval run test_queries.jsonl --output results.jsonl --top-k 5
devrag eval compare results_v1.jsonl results_v2.jsonl
```

Query file format:

```jsonl
{"query": "How does rate limiting work?", "expected_files": ["src/middleware/rate_limit.ts"]}
{"query": "Why did we migrate to Redis?", "expected_prs": [1234, 1301]}
```

Metrics: precision@k, recall@k, and MRR (Mean Reciprocal Rank).

## Claude Code Integration

DevRAG ships as a [FastMCP](https://github.com/jlowin/fastmcp) server that integrates directly with Claude Code.

### Setup

```bash
# If installed globally via `uv tool install`:
claude mcp add devrag -- devrag serve

# Or run on demand without installing:
claude mcp add devrag -- uvx --from git+https://github.com/tomharris/dev-rag.git devrag serve
```

### Skills

The repo includes Claude Code skills in `.claude/skills/`:

| Skill | Description |
|-------|-------------|
| `/rag-search <query>` | Search code, PRs, and docs with results grouped by source type |
| `/rag-index [path]` | Index a repo, docs directory, or sync PRs |
| `/rag-pr <query>` | Search PR history for why code changed |
| `rag-first` (auto) | Auto-triggers on codebase questions to search DevRAG before Grep/Glob/Explore |

The `rag-first` skill fires automatically when Claude detects a codebase question — no slash command needed. A hookify warning rule (`.claude/hookify.rag-first-reminder.local.md`) provides a safety-net nudge if Grep/Glob are used without searching DevRAG first.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DEVELOPER INTERFACES                            │
│  CLI (Typer)                          MCP Server (FastMCP)         │
│  devrag search|index|status           Claude Code slash commands    │
└────────────┬──────────────────────────────────┬─────────────────────┘
             │                                  │
             ▼                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     RETRIEVAL LAYER                                 │
│  Query Router ──→ Hybrid Search (Vector + BM25 / RRF) ──→ Reranker│
│  (intent → collections)                          (cross-encoder)   │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STORAGE LAYER                                   │
│  VectorStore Protocol                                              │
│    ├─ ChromaDB (default, embedded, zero-config)                    │
│    └─ Qdrant (optional, for scale)                                 │
│  SQLite: file hashes, chunk mappings, PR cursors, FTS5, metrics    │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     INGESTION LAYER                                 │
│  Code     PR       Issue    Jira     Slite     Doc                   │
│  Indexer  Indexer  Indexer  Indexer  Indexer   Indexer                │
│  (AST)   (GitHub) (GitHub) (Cloud)  (REST)   (sections)             │
│                                                                     │
│  Ollama Embedder (nomic-embed-text, 768d)                          │
└─────────────────────────────────────────────────────────────────────┘
```

### Collections

| Collection | Source | Contents |
|-----------|--------|----------|
| `code_chunks` | Code indexer | AST-extracted functions, classes, methods |
| `pr_diffs` | PR indexer | PR descriptions and file-level diff hunks |
| `pr_discussions` | PR indexer | Review comments and threads |
| `issue_descriptions` | Issue indexer | Issue titles and bodies |
| `issue_discussions` | Issue indexer | Issue comments |
| `jira_descriptions` | Jira indexer | Jira ticket summaries and descriptions |
| `jira_discussions` | Jira indexer | Jira ticket comments |
| `slite_pages` | Slite indexer | Slite page sections (markdown) |
| `documents` | Doc indexer | Markdown/text sections |
| `session_logs` | Sessions indexer | Claude Code user↔assistant exchanges from local JSONL |

## Configuration

DevRAG uses YAML configuration with two layers (project overrides user):

- **User config:** `~/.config/devrag/devrag.yaml`
- **Project config:** `.devrag.yaml` (commit this to share with your team)

```yaml
embedding:
  model: nomic-embed-text        # Embedding model name
  provider: ollama               # ollama or sentence-transformers
  ollama_url: http://localhost:11434
  batch_size: 64
  max_tokens: 8192                   # Truncation limit for embedding context

vector_store:
  backend: chromadb              # chromadb or qdrant
  persist_dir: ~/.local/share/devrag/chroma
  qdrant_url: http://localhost:6333  # if backend: qdrant
  embedding_dim: 768                 # Vector dimensionality

retrieval:
  top_k: 20                     # Candidates before reranking
  final_k: 5                    # Results after reranking
  rerank: true
  reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2

code:
  chunk_max_tokens: 512
  respect_gitignore: true
  exclude_patterns:
    - "*.min.js"
    - "vendor/**"
    - "node_modules/**"
    - "*.lock"
    - "*.generated.*"

prs:
  github_token_env: GITHUB_TOKEN  # Env var name (token never stored in config)
  backfill_days: 90
  include_draft: false
  chunk_max_tokens: 512              # Max tokens per PR chunk

issues:
  github_token_env: GITHUB_TOKEN  # Reuses same token as PRs
  backfill_days: 90
  chunk_max_tokens: 512
  include_labels: []              # Only index issues with these labels (empty = all)
  exclude_labels: ["wontfix"]     # Skip issues with these labels

jira:
  jira_token_env: JIRA_TOKEN      # Env var for API token
  jira_email_env: JIRA_EMAIL      # Env var for email (basic auth)
  instance_url: https://yoursite.atlassian.net
  jql: "project = DEV"            # JQL filter for scoping tickets
  backfill_days: 90
  chunk_max_tokens: 512

slite:
  slite_token_env: SLITE_TOKEN    # Env var for API token
  channel_ids: []                 # Filter to specific channels (empty = all)
  backfill_days: 90
  chunk_max_tokens: 512
  chunk_overlap_tokens: 50

sessions:
  logs_dir: ~/.claude/projects    # Claude Code JSONL log directory
  backfill_days: 60
  chunk_max_tokens: 512

documents:
  glob_patterns: ["**/*.md", "**/*.mdx", "**/*.txt", "**/*.rst", "**/*.html", "**/*.adoc"]
  chunk_max_tokens: 512
  chunk_overlap_tokens: 50
```

### File Exclusion

DevRAG respects `.gitignore` by default. For additional exclusions specific to the index, create a `.devragignore` file (same syntax as `.gitignore`).

## Scaling Up

DevRAG starts with ChromaDB (embedded, zero-config) and can scale to Qdrant when needed:

```bash
# Start Qdrant
docker run -d -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant

# Switch backend
devrag config set vector_store.backend qdrant
devrag config set vector_store.qdrant_url http://localhost:6333

# Re-index with new backend
devrag reindex --all
```

The vector store is abstracted behind a `VectorStore` Protocol — switching backends is a config change, not a code change.

## Development

```bash
git clone https://github.com/tomharris/dev-rag.git
cd dev-rag
uv sync --extra dev

# Run tests
uv run pytest tests/

# Run a single test
uv run pytest tests/test_config.py::test_load_config -v

# Integration tests (requires Ollama running with nomic-embed-text)
SKIP_INTEGRATION=0 uv run pytest tests/test_integration.py -v
```

## License

MIT
