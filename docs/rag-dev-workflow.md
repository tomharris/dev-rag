# DevRAG: Local RAG Workflow for Developer Teams

## Architecture Overview

A codebase-agnostic, locally-run RAG system that ingests code, GitHub PRs, and internal documents — surfaced through Claude Code slash commands and a standalone CLI agent. Designed for performance, scalability, and zero-friction developer adoption.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        DEVELOPER INTERFACES                         │
│                                                                     │
│   Claude Code Slash Commands          Standalone CLI Agent          │
│   /rag search "auth flow"             devrag search "auth flow"    │
│   /rag index .                        devrag index .               │
│   /rag pr 1234                        devrag pr 1234               │
│   /rag docs ./specs                   devrag docs ./specs          │
│   /rag status                         devrag status                │
└──────────────┬───────────────────────────────────┬──────────────────┘
               │                                   │
               │        MCP Server (FastMCP)        │
               │        localhost:8741              │
               ▼                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         RETRIEVAL LAYER                              │
│                                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────────┐ │
│  │ Hybrid       │  │ Reranker     │  │ Query Router               │ │
│  │ Search       │  │ (cross-      │  │ • code → AST-aware search  │ │
│  │ (vector +    │  │  encoder)    │  │ • PR → metadata + diff     │ │
│  │  BM25)       │  │              │  │ • doc → semantic search    │ │
│  └──────┬───────┘  └──────┬───────┘  └────────────┬───────────────┘ │
│         │                 │                        │                 │
└─────────┼─────────────────┼────────────────────────┼─────────────────┘
          │                 │                        │
┌─────────▼─────────────────▼────────────────────────▼─────────────────┐
│                         STORAGE LAYER                                │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  ChromaDB (dev/small teams) ──or── Qdrant (scale)           │    │
│  │  Collections:                                                │    │
│  │    • code_chunks     (AST-aware, per-repo)                  │    │
│  │    • pr_diffs        (per-repo, enriched metadata)          │    │
│  │    • documents       (markdown, confluence, notion, etc.)   │    │
│  │    • pr_discussions  (review comments, threads)             │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  SQLite (metadata store)                                     │    │
│  │    • file hashes, last-indexed timestamps                   │    │
│  │    • PR sync cursors, repo config                           │    │
│  │    • chunk→source mappings for citation                     │    │
│  └──────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
          ▲                 ▲                        ▲
          │                 │                        │
┌─────────┴─────────────────┴────────────────────────┴─────────────────┐
│                        INGESTION LAYER                               │
│                                                                      │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐ │
│  │ Code Indexer    │  │ PR Indexer     │  │ Doc Indexer            │ │
│  │ • tree-sitter   │  │ • GitHub API   │  │ • Markdown/HTML/PDF    │ │
│  │   AST chunking  │  │ • diff parsing │  │ • recursive chunking   │ │
│  │ • dependency    │  │ • review       │  │ • section-aware splits │ │
│  │   extraction    │  │   comments     │  │ • metadata extraction  │ │
│  │ • git-aware     │  │ • label/status │  │                        │ │
│  │   (ignores      │  │   enrichment   │  │                        │ │
│  │   .gitignore)   │  │                │  │                        │ │
│  └────────────────┘  └────────────────┘  └────────────────────────┘ │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  Embedding Engine                                            │    │
│  │  Local: sentence-transformers/all-MiniLM-L6-v2 (384d, fast) │    │
│  │    ──or──                                                    │    │
│  │  Local: nomic-embed-text via Ollama (768d, better quality)   │    │
│  │    ──or──                                                    │    │
│  │  API: Voyage Code 3 / OpenAI text-embedding-3-small          │    │
│  └──────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Component Deep-Dive

### 1. Ingestion Pipeline

#### Code Indexer (AST-Aware)

The single most impactful architectural decision is **not** treating code as plain text. Use tree-sitter to parse source into ASTs and chunk along syntactic boundaries (functions, classes, methods, type definitions). This prevents mid-function splits that destroy semantic meaning.

**Chunking strategy:**

- Parse each file with tree-sitter (supports 100+ languages).
- Extract semantic entities: functions, methods, classes, interfaces, type aliases, imports.
- For each entity, build a **contextualized chunk** that includes the entity's signature, docstring, parent context (e.g., which class it belongs to), and the body — up to a configurable token limit (default: 512 tokens).
- Large entities that exceed the limit are recursively split at the next AST level down.
- Orphan code (module-level statements, config blocks) is grouped with adjacent entities.

**Metadata per chunk:**

```json
{
  "file_path": "src/auth/oauth.ts",
  "repo": "acme/backend",
  "language": "typescript",
  "entity_type": "method",
  "entity_name": "refreshToken",
  "parent_entity": "OAuthClient",
  "signature": "async refreshToken(token: string): Promise<TokenPair>",
  "line_range": [142, 189],
  "imports": ["jsonwebtoken", "redis"],
  "last_modified": "2026-03-15T10:22:00Z",
  "git_blame_author": "alice@acme.com"
}
```

**Incremental indexing:** Track file content hashes in SQLite. On re-index, only process files whose hash changed since last run. For large repos, this reduces re-indexing from minutes to seconds.

**Implementation — recommended libraries:**

| Component | Python | TypeScript |
|-----------|--------|------------|
| AST parsing | `tree-sitter` + `tree-sitter-languages` | `code-chunk` or `web-tree-sitter` |
| Chunking pipeline | `chonkie` (CodeChunker) or custom | custom with `code-chunk` npm |
| Git integration | `gitpython` | `simple-git` |

#### PR Indexer

PRs are a uniquely valuable knowledge source — they encode *why* code changed, not just *what* changed.

**What to ingest per PR:**

- **Diff chunks**: The actual code changes, chunked per-file with before/after context. Each diff hunk becomes a chunk with metadata about the file, line range, and change type (add/modify/delete).
- **PR description**: Title + body as a single document chunk.
- **Review comments**: Each review thread (comment + replies) as a chunk, associated with the file and line range it references.
- **Labels, status, reviewers**: Stored as filterable metadata.

**Sync strategy:**

- Use the GitHub REST API (or GraphQL) to poll for PRs updated since the last sync cursor.
- Store the sync cursor (timestamp) in SQLite per repo.
- Run sync on a cron (every 15 min) or trigger via `/rag sync`.
- For initial backfill, paginate through closed/merged PRs going back N days (configurable, default 90).

**Metadata per PR chunk:**

```json
{
  "repo": "acme/backend",
  "pr_number": 1234,
  "pr_title": "Migrate auth to OAuth2 PKCE flow",
  "pr_state": "merged",
  "pr_author": "alice",
  "pr_labels": ["security", "breaking-change"],
  "merged_at": "2026-03-10T14:00:00Z",
  "chunk_type": "review_comment",
  "file_path": "src/auth/oauth.ts",
  "reviewer": "bob"
}
```

#### Document Indexer

For internal docs (Markdown files, Confluence exports, Notion exports, PDFs, design docs), use recursive text splitting with section-awareness.

**Strategy:**

- Detect document structure: headings (H1–H4) as primary split boundaries, then paragraph breaks, then sentence breaks.
- Target chunk size: 400–512 tokens with 50-token overlap.
- Preserve heading hierarchy as metadata (e.g., `section_path: "Architecture > Auth > OAuth Flow"`).
- For Markdown: split on `##` / `###` headers first, then by paragraph within sections.
- For PDFs: extract text per page, then apply recursive splitting.

**Supported formats:** `.md`, `.mdx`, `.txt`, `.html`, `.pdf`, `.docx`, `.rst`, `.adoc`

---

### 2. Embedding Engine

The embedding model is the most consequential performance knob. Recommendations tiered by use case:

| Tier | Model | Dimensions | Speed | Quality | Notes |
|------|-------|-----------|-------|---------|-------|
| **Fast/Local** | `all-MiniLM-L6-v2` | 384 | ~14k chunks/min on CPU | Good | Best for prototyping and small teams. Runs on any laptop. |
| **Balanced/Local** | `nomic-embed-text` via Ollama | 768 | ~4k chunks/min on CPU | Better | Strong code understanding. Ollama makes deployment trivial. |
| **Quality/API** | `voyage-code-3` | 1024 | API-bound | Best for code | Purpose-built for code retrieval. Requires API key. |
| **Quality/API** | `text-embedding-3-small` | 1536 | API-bound | Best general | OpenAI's workhorse. Good all-rounder. |

**Recommendation:** Start with `all-MiniLM-L6-v2` for zero-dependency local setup. Upgrade to `nomic-embed-text` via Ollama when you want better quality without leaving local. The embedding model is swappable — the system stores the model name in metadata so you can re-embed collections when upgrading.

---

### 3. Vector Store

**Primary recommendation: ChromaDB** for teams up to ~10M vectors (which covers most dev team use cases — a large monorepo with 100k files, 5k PRs, and 10k doc pages produces roughly 500k–2M chunks).

**Why ChromaDB:**

- Zero-config: `pip install chromadb` and you're running. No Docker, no server process in the default embedded mode.
- Built-in hybrid search (vector + full-text) in a single query.
- Built-in metadata filtering (filter by repo, file path, language, entity type, date range).
- The 2025 Rust rewrite made it 4x faster than the original Python implementation.
- Persists to disk by default — survives restarts.

**Scale path: Qdrant** when you outgrow ChromaDB (>10M vectors, need multi-user access, or want client-server separation).

- Run via Docker: `docker run -p 6333:6333 qdrant/qdrant`
- Rust-native, HNSW indexing, sub-5ms queries under load.
- Superior filtered search for complex metadata queries.
- Horizontal scaling when needed.

**The system abstracts the vector store behind an interface**, so swapping ChromaDB → Qdrant is a config change, not a rewrite.

```python
# devrag/stores/base.py
class VectorStore(Protocol):
    def upsert(self, collection: str, ids: list[str], embeddings: list[list[float]],
               documents: list[str], metadatas: list[dict]) -> None: ...
    def query(self, collection: str, query_embedding: list[float], n_results: int = 10,
              where: dict | None = None) -> QueryResult: ...
    def delete(self, collection: str, ids: list[str]) -> None: ...
```

---

### 4. Retrieval Layer

#### Hybrid Search

Every query runs through both vector similarity (semantic) and BM25 (keyword) search, then merges results using Reciprocal Rank Fusion (RRF). This is critical for code, where exact identifiers (`refreshToken`, `OAuth2Client`) matter as much as semantic meaning.

```
score(doc) = Σ  1 / (k + rank_i(doc))
             i∈{vector, bm25}
```

Where `k = 60` (standard RRF constant). This naturally balances semantic relevance with keyword precision.

#### Query Router

Not all queries should search the same collections. The router classifies the query intent and targets appropriate collections:

| Query Pattern | Target Collections | Example |
|---------------|-------------------|---------|
| Code structure / "how does X work" | `code_chunks` | "How does the auth middleware work?" |
| Change history / "why did we" | `pr_diffs`, `pr_discussions` | "Why did we switch from JWT to PKCE?" |
| Process / architecture / spec | `documents` | "What's our API versioning policy?" |
| Debugging / "where is X used" | `code_chunks` + `pr_diffs` | "Where is `refreshToken` called?" |
| Broad / unclear | All collections | "Tell me about authentication" |

The router can be a simple keyword classifier (fast, no LLM call) or an LLM-based classifier for ambiguous queries. Start with keyword rules, upgrade to LLM routing if precision matters.

#### Reranking

After hybrid search returns the top-K candidates (default K=20), pass them through a cross-encoder reranker to re-score by actual query-document relevance:

- **Local:** `cross-encoder/ms-marco-MiniLM-L-6-v2` — fast, good quality.
- **API:** Cohere Rerank or Voyage Rerank — better quality, adds latency.

Reranking typically improves precision@5 by 10–20% over raw retrieval, at the cost of ~50–100ms per query. Worth it for production; optional for prototyping.

---

### 5. Developer Interfaces

#### Claude Code Integration (Primary)

Expose the RAG system as a **FastMCP server** that registers with Claude Code. This gives developers slash commands and `@` resource mentions natively in their terminal.

**MCP Server setup:**

```python
# devrag/mcp_server.py
from fastmcp import FastMCP

mcp = FastMCP(name="DevRAG")

@mcp.tool
def search(query: str, scope: str = "all", repo: str = "auto", top_k: int = 5) -> str:
    """Search code, PRs, and docs. Scope: all|code|prs|docs"""
    results = retriever.search(query, scope=scope, repo=repo, top_k=top_k)
    return format_results(results)

@mcp.tool
def index_repo(path: str = ".", incremental: bool = True) -> str:
    """Index a local repository. Uses git to detect changes for incremental indexing."""
    stats = indexer.index_repo(path, incremental=incremental)
    return f"Indexed {stats.files} files, {stats.chunks} chunks ({stats.new} new, {stats.updated} updated)"

@mcp.tool
def index_docs(path: str, glob: str = "**/*.md") -> str:
    """Index a directory of documents."""
    stats = indexer.index_docs(path, glob=glob)
    return f"Indexed {stats.files} documents, {stats.chunks} chunks"

@mcp.tool
def sync_prs(repo: str, since_days: int = 90) -> str:
    """Sync GitHub PRs for a repository."""
    stats = pr_indexer.sync(repo, since_days=since_days)
    return f"Synced {stats.new_prs} new PRs, {stats.updated_prs} updated"

@mcp.tool
def status() -> str:
    """Show indexing status across all sources."""
    return status_report()
```

**Registration in Claude Code:**

```bash
# One-time setup
claude mcp add devrag -- uv run --with fastmcp fastmcp run devrag/mcp_server.py
```

After registration, developers use it naturally in Claude Code:

```
> /mcp__devrag__search "how does rate limiting work"
> /mcp__devrag__index_repo .
> /mcp__devrag__sync_prs acme/backend
```

Or via Claude Code skills that wrap the MCP calls with richer prompts:

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

#### Standalone CLI (Secondary)

For use outside Claude Code, or for scripting/CI:

```bash
# Search
devrag search "auth flow" --scope code --top-k 10

# Index
devrag index repo .
devrag index docs ./specs --glob "**/*.md"
devrag index prs acme/backend --since 90d

# Status
devrag status

# Config
devrag config set embedding_model nomic-embed-text
devrag config set vector_store qdrant
devrag config set github_token ghp_xxx
```

Built with `click` or `typer` in Python. Shares the same core library as the MCP server.

---

## Project Structure

```
devrag/
├── pyproject.toml              # Package config, dependencies
├── devrag/
│   ├── __init__.py
│   ├── cli.py                  # Standalone CLI (typer)
│   ├── mcp_server.py           # FastMCP server for Claude Code
│   ├── config.py               # YAML-based config management
│   │
│   ├── ingest/
│   │   ├── code_indexer.py     # tree-sitter AST chunking
│   │   ├── pr_indexer.py       # GitHub API PR ingestion
│   │   ├── doc_indexer.py      # Document chunking (recursive/section-aware)
│   │   └── embedder.py         # Embedding engine abstraction
│   │
│   ├── retrieve/
│   │   ├── hybrid_search.py    # Vector + BM25 fusion
│   │   ├── query_router.py     # Intent classification → collection targeting
│   │   └── reranker.py         # Cross-encoder reranking
│   │
│   ├── stores/
│   │   ├── base.py             # VectorStore protocol
│   │   ├── chroma_store.py     # ChromaDB implementation
│   │   ├── qdrant_store.py     # Qdrant implementation
│   │   └── metadata_db.py      # SQLite metadata store
│   │
│   └── utils/
│       ├── git.py              # Git operations, .gitignore handling
│       ├── github.py           # GitHub API client
│       └── formatters.py       # Result formatting for MCP/CLI
│
├── config/
│   └── devrag.yaml             # Default config template
│
├── .claude/
│   └── skills/
│       ├── rag-search/SKILL.md
│       ├── rag-index/SKILL.md
│       └── rag-pr/SKILL.md
│
└── tests/
    ├── test_code_indexer.py
    ├── test_pr_indexer.py
    ├── test_retrieval.py
    └── fixtures/
```

---

## Configuration

```yaml
# ~/.config/devrag/devrag.yaml  (or .devrag.yaml in project root)

embedding:
  model: all-MiniLM-L6-v2          # or: nomic-embed-text, voyage-code-3
  provider: sentence-transformers    # or: ollama, voyage, openai
  batch_size: 64
  cache_enabled: true

vector_store:
  backend: chromadb                  # or: qdrant
  persist_dir: ~/.local/share/devrag/chroma
  # qdrant_url: http://localhost:6333  # if backend: qdrant

retrieval:
  top_k: 20                         # candidates before reranking
  final_k: 5                        # results after reranking
  hybrid_ratio: 0.7                  # 70% vector, 30% BM25
  rerank: true
  reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2

code:
  chunk_max_tokens: 512
  languages: auto                    # or explicit list: [python, typescript, rust]
  respect_gitignore: true
  exclude_patterns:
    - "*.min.js"
    - "vendor/**"
    - "node_modules/**"
    - "*.lock"
    - "*.generated.*"

prs:
  github_token_env: GITHUB_TOKEN     # env var name
  sync_interval_minutes: 15
  backfill_days: 90
  include_draft: false

documents:
  glob_patterns:
    - "**/*.md"
    - "**/*.mdx"
    - "**/*.txt"
    - "**/*.rst"
  chunk_max_tokens: 512
  chunk_overlap_tokens: 50

mcp:
  host: localhost
  port: 8741
```

---

## Adoption Playbook

### Day 1: Solo developer, one repo (5 minutes)

```bash
# Install
pip install devrag

# Index current repo
cd ~/projects/myapp
devrag index repo .

# Search
devrag search "database connection pooling"

# Add to Claude Code
claude mcp add devrag -- devrag serve
```

### Week 1: Add PRs and docs

```bash
# Set GitHub token
export GITHUB_TOKEN=ghp_xxx

# Sync PRs
devrag index prs myorg/myapp --since 90d

# Index team docs
devrag index docs ~/docs/engineering --glob "**/*.md"
```

### Week 2: Team rollout

```bash
# Share config via repo
cp ~/.config/devrag/devrag.yaml .devrag.yaml
git add .devrag.yaml .claude/skills/rag-*
git commit -m "Add DevRAG config and Claude Code skills"

# Each teammate runs:
pip install devrag
devrag index repo .
```

### Month 1: Scale up

```bash
# Switch to Qdrant for shared team index
docker run -d -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant
devrag config set vector_store.backend qdrant
devrag config set vector_store.qdrant_url http://localhost:6333

# Upgrade embedding model
devrag config set embedding.model nomic-embed-text
devrag config set embedding.provider ollama
devrag reindex --all  # re-embed everything with new model
```

---

## Performance Characteristics

| Operation | Expected Latency | Notes |
|-----------|-----------------|-------|
| Index 1k files (incremental, no changes) | < 2s | Hash comparison only |
| Index 1k files (full, all-MiniLM) | ~70s on CPU | ~14k chunks/min |
| Index 1k files (full, nomic via Ollama) | ~250s on CPU | ~4k chunks/min, better quality |
| Search (hybrid + rerank, ChromaDB) | 100–300ms | Dominated by reranker |
| Search (hybrid, no rerank, ChromaDB) | 20–50ms | Fast path for interactive use |
| PR sync (100 PRs, GitHub API) | 30–60s | API rate-limited |

### Scalability Targets

| Scale | Vector Store | Embedding | Index Time (full) |
|-------|-------------|-----------|-------------------|
| 1 repo, 10k files | ChromaDB embedded | all-MiniLM (CPU) | ~12 min |
| 5 repos, 50k files | ChromaDB embedded | nomic-embed (Ollama, GPU) | ~20 min |
| 20 repos, 200k files | Qdrant (Docker) | API embeddings (batched) | ~45 min |
| Monorepo, 500k+ files | Qdrant (dedicated) | API embeddings (parallel) | ~2 hr |

---

## Key Design Decisions & Rationale

**1. MCP over custom protocol.** MCP is the standard Claude Code extension mechanism. It gives you slash commands, resource mentions, and tool integration without building a custom plugin. FastMCP makes the Python implementation trivial.

**2. AST chunking over naive text splitting.** Research (cAST, 2025) shows AST-aware chunking improves Recall@5 by 4+ points on code retrieval benchmarks. The implementation cost is low (tree-sitter is battle-tested) and the quality improvement is substantial.

**3. Hybrid search (vector + BM25) over vector-only.** Production RAG systems consistently show 1–9% recall improvement with hybrid approaches. For code, where exact identifiers matter, the keyword component is essential — vector search alone will miss `refreshToken` when the query says "refresh token."

**4. ChromaDB default over Qdrant.** ChromaDB is the right starting point because it has zero operational overhead (embedded, no server). The abstraction layer means teams can upgrade to Qdrant when they actually need it, rather than paying the ops cost upfront.

**5. SQLite metadata store alongside vector DB.** Vector stores are great at similarity search, bad at relational queries ("which files haven't been indexed in 7 days?"). SQLite handles the bookkeeping without adding another service.

**6. Config-file driven, not code-driven.** Developers shouldn't need to write Python to configure the system. YAML config in the project root (shareable via git) or user home (personal defaults) covers all common customization.

---

## Evaluation & Observability

### Built-in Metrics

The system should log per-query metrics to a local SQLite table for analysis:

- **Retrieval latency** (vector search, BM25, rerank, total)
- **Result sources** (which collections contributed results)
- **Query classification** (what the router decided)
- **Chunk hit rate** (how often the top-1 result was from a chunk vs. full document)

### Manual Eval Workflow

```bash
# Run a batch of test queries and inspect results
devrag eval run ./test_queries.jsonl --output results.jsonl

# Compare two configs (e.g., before/after changing chunk size)
devrag eval compare results_v1.jsonl results_v2.jsonl
```

Test queries file format:

```jsonl
{"query": "How does rate limiting work?", "expected_files": ["src/middleware/rate_limit.ts"]}
{"query": "Why did we migrate to Redis?", "expected_prs": [1234, 1301]}
```

---

## Security Notes

- **GitHub tokens** are read from environment variables, never stored in config files.
- **Embeddings are local by default** — no data leaves the machine unless you opt into API-based embedding models.
- **Vector store data** lives on local disk (ChromaDB) or a local Docker container (Qdrant). No cloud dependency.
- **`.gitignore` is respected** during code indexing. Secrets in ignored files won't be indexed.
- Add a `.devragignore` file (same syntax as `.gitignore`) for additional exclusion patterns specific to the RAG index.
