# DevRAG Phased Implementation Design

## Context

DevRAG is a local RAG system for developer teams that ingests code, GitHub PRs, and internal documents, surfaced through Claude Code (via FastMCP) and a standalone CLI. The full architecture is specified in `docs/rag-dev-workflow.md`.

This document defines the phased implementation strategy. The approach is **vertical slice**: build the complete code search pipeline end-to-end first, then add data sources horizontally. This gets a genuinely useful tool working on a monorepo (50k+ files) as fast as possible.

### Key Decisions

- **Interface priority:** MCP server first (Claude Code integration), CLI later
- **Embedding model:** Ollama + nomic-embed-text (768d) — already available locally, better code quality than all-MiniLM
- **Retrieval:** Full pipeline from day one — hybrid search (vector + BM25) with cross-encoder reranking
- **Incremental indexing:** Essential from Phase 1 due to monorepo scale (50k+ files)
- **Vector store:** ChromaDB embedded (zero-ops), with Qdrant as a Phase 4 scale path

---

## Phase 1: Code Search End-to-End

**Goal:** Index a monorepo with AST-aware chunking and search it from Claude Code via MCP.

### 1.1 Project Scaffolding

- Python package managed with `uv` and `pyproject.toml`
- Source in `devrag/` following the structure from the architecture doc
- YAML-based config with sensible defaults

### 1.2 Storage Layer

**VectorStore protocol** (`devrag/stores/base.py`):
- `upsert(collection, ids, embeddings, documents, metadatas)`
- `query(collection, query_embedding, n_results, where) -> QueryResult`
- `delete(collection, ids)`

**ChromaStore** (`devrag/stores/chroma_store.py`):
- Embedded ChromaDB, persists to `~/.local/share/devrag/chroma`
- `code_chunks` collection
- Built-in full-text search used for BM25 component

**MetadataDB** (`devrag/stores/metadata_db.py`):
- SQLite at `~/.local/share/devrag/metadata.db`
- Tables: `file_hashes` (path, sha256, last_indexed), `chunk_sources` (chunk_id, file_path, line_range)
- Used for incremental indexing change detection

### 1.3 Embedding Engine

**Embedder abstraction** (`devrag/ingest/embedder.py`):
- `embed(texts: list[str]) -> list[list[float]]`
- `embed_query(text: str) -> list[float]`

**Ollama provider**:
- Uses `nomic-embed-text` model (768 dimensions)
- Batch embedding with configurable batch size (default 64)
- Stores model name in chunk metadata for future re-embedding support

### 1.4 Code Indexer (AST-Aware)

**Code indexer** (`devrag/ingest/code_indexer.py`):
- tree-sitter parsing via `tree-sitter` + `tree-sitter-languages`
- Extracts semantic entities: functions, methods, classes, interfaces, type aliases, imports
- Builds contextualized chunks: signature + docstring + parent context + body
- Max chunk size: 512 tokens (configurable)
- Large entities recursively split at next AST level down
- Orphan code (module-level statements) grouped with adjacent entities

**Metadata per chunk:**
- `file_path`, `repo`, `language`, `entity_type`, `entity_name`, `parent_entity`
- `signature`, `line_range`, `imports`, `last_modified`, `git_blame_author`

**Incremental indexing:**
- SHA-256 hash of file contents stored in SQLite
- On re-index: hash each file, skip unchanged files, process only changed/new files
- Delete chunks for removed files
- `.gitignore`-aware file discovery via `gitpython`
- Config-based exclusions: `*.min.js`, `vendor/**`, `node_modules/**`, `*.lock`, `*.generated.*`

### 1.5 Retrieval Layer

**Hybrid search** (`devrag/retrieve/hybrid_search.py`):
- Vector similarity search via ChromaDB
- BM25 keyword search via ChromaDB's built-in full-text
- Reciprocal Rank Fusion (RRF) to merge: `score(doc) = sum(1 / (60 + rank_i(doc)))` for i in {vector, bm25}
- Returns top-20 candidates

**Reranker** (`devrag/retrieve/reranker.py`):
- `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers`
- Re-scores top-20 candidates by query-document relevance
- Returns top-5 (configurable via `retrieval.final_k`)

### 1.6 MCP Server

**FastMCP server** (`devrag/mcp_server.py`):

Tools:
- `search(query: str, top_k: int = 5) -> str` — hybrid search + rerank against `code_chunks`
- `index_repo(path: str = ".", incremental: bool = True) -> str` — index a local repo, returns stats
- `status() -> str` — show indexing stats (files indexed, chunks, last indexed time)

Registration:
```bash
claude mcp add devrag -- uv run --with fastmcp fastmcp run devrag/mcp_server.py
```

### 1.7 Config

**Config file** (`devrag/config.py`):
- YAML-based, loaded from `.devrag.yaml` (project root) or `~/.config/devrag/devrag.yaml` (user)
- Defaults embedded in code for zero-config startup
- Covers: embedding model/provider, vector store backend/path, retrieval params, code exclusions

### 1.8 Dependencies (Phase 1)

```
tree-sitter, tree-sitter-languages
chromadb
sentence-transformers  # for cross-encoder reranker
httpx                  # for Ollama API calls
gitpython
pyyaml
fastmcp
```

### 1.9 Not in Phase 1

- PR indexing, doc indexing
- Query router (single collection, not needed)
- Standalone CLI
- Qdrant backend, eval framework, observability
- `.devragignore`

---

## Phase 2: PR Knowledge

**Goal:** Add GitHub PR history as a searchable knowledge source. Answer "why did we change X" queries.

### 2.1 PR Indexer

**PR indexer** (`devrag/ingest/pr_indexer.py`):
- GitHub REST API integration
- Per PR ingests: diff chunks (per-file hunks with before/after context), PR description (title + body), review comment threads
- Labels, status, reviewers stored as filterable metadata

**Metadata per PR chunk:**
- `repo`, `pr_number`, `pr_title`, `pr_state`, `pr_author`, `pr_labels`, `merged_at`
- `chunk_type` (diff | description | review_comment), `file_path`, `reviewer`

**Sync strategy:**
- Cursor-based: last-sync timestamp stored in SQLite per repo
- Fetches PRs updated since cursor
- Backfill: paginate through closed/merged PRs, configurable lookback (default 90 days)
- GitHub token from `GITHUB_TOKEN` env var

**New collections:** `pr_diffs`, `pr_discussions`

### 2.2 GitHub API Client

**GitHub client** (`devrag/utils/github.py`):
- Rate-limit aware (`X-RateLimit-Remaining`, automatic backoff)
- Pagination helper for PR list/detail endpoints
- Diff parsing to extract per-file hunks

### 2.3 Query Router

**Keyword-based router** (`devrag/retrieve/query_router.py`):
- Fast, deterministic, no LLM call
- Routes by intent:
  - "how does X work" → `code_chunks`
  - "why did we change/switch/migrate X" → `pr_diffs` + `pr_discussions`
  - "where is X used" → `code_chunks` + `pr_diffs`
  - Ambiguous → all collections
- Results merged across collections, then reranked

### 2.4 Result Formatting

**Formatters** (`devrag/utils/formatters.py`):
- Code results: file path + snippet
- PR results: PR number + title + relevant excerpt
- Grouped by source type in output

### 2.5 MCP Server Updates

- New tool: `sync_prs(repo: str, since_days: int = 90) -> str`
- `search()` gains `scope` parameter: `all|code|prs`
- Results include source type indicators

### 2.6 Not in Phase 2

- Document indexing, standalone CLI
- Cron-based automatic PR sync (manual trigger only)

---

## Phase 3: Document Search + CLI

**Goal:** Add internal docs as a third knowledge source. Build the standalone CLI.

### 3.1 Document Indexer

**Doc indexer** (`devrag/ingest/doc_indexer.py`):
- Recursive text splitting with section-awareness
- Split hierarchy: headings (H1-H4) → paragraphs → sentences
- Target: 400-512 tokens per chunk, 50-token overlap
- Preserves heading hierarchy as metadata (`section_path: "Architecture > Auth > OAuth Flow"`)
- Formats: `.md`, `.mdx`, `.txt`, `.rst`, `.html`, `.adoc`
- PDF support via `pymupdf` or `pdfplumber`
- New collection: `documents`

### 3.2 Full Query Router

- Extends Phase 2 router to cover documents:
  - Process/architecture/spec queries → `documents`
  - Code queries → `code_chunks`
  - Change history → `pr_diffs`, `pr_discussions`
  - Broad/unclear → all collections

### 3.3 Standalone CLI

**CLI** (`devrag/cli.py`) built with `typer`:
- `devrag search "query" --scope all|code|prs|docs --top-k 10`
- `devrag index repo . [--full]`
- `devrag index docs ./specs --glob "**/*.md"`
- `devrag index prs org/repo --since 90d`
- `devrag status`
- `devrag config set key value` / `devrag config get key`
- `devrag serve` — starts the MCP server

Shares the same core library as the MCP server. The CLI is a thin wrapper.

### 3.4 Config System

- Full YAML config management per the architecture doc
- Resolution order: project `.devrag.yaml` → user `~/.config/devrag/devrag.yaml` → embedded defaults
- CLI commands for get/set

### 3.5 `.devragignore`

- Additional exclusion patterns beyond `.gitignore`
- Same syntax as `.gitignore`
- Checked during code indexing alongside `.gitignore`

### 3.6 Not in Phase 3

- Qdrant backend, eval framework, observability
- Claude Code skills

---

## Phase 4: Polish, Scale, and Eval

**Goal:** Hardening, alternative backends, evaluation framework, and developer experience polish.

### 4.1 Qdrant Backend

**Qdrant store** (`devrag/stores/qdrant_store.py`):
- Implements `VectorStore` protocol
- Config-driven swap: `vector_store.backend: qdrant`
- For teams outgrowing ChromaDB (>10M vectors, multi-user access)

### 4.2 Eval Framework

- `devrag eval run ./test_queries.jsonl --output results.jsonl`
- `devrag eval compare results_v1.jsonl results_v2.jsonl`
- JSONL format: `{"query": "...", "expected_files": [...], "expected_prs": [...]}`
- Metrics: precision@k, recall@k, MRR

### 4.3 Observability

- Per-query metrics to SQLite: latency breakdown (vector, BM25, rerank, total), result sources, query classification, chunk hit rate
- `devrag status` enriched with query statistics

### 4.4 Claude Code Skills

- `.claude/skills/rag-search/SKILL.md` — wraps MCP search with formatting prompts
- `.claude/skills/rag-index/SKILL.md` — guided indexing workflow
- `.claude/skills/rag-pr/SKILL.md` — PR-focused search

### 4.5 Re-embedding Support

- `devrag reindex --all` — re-embed everything when switching models
- Model name tracked in chunk metadata; mixed-model collections detectable and flagged

### 4.6 Automatic PR Sync

- Optional cron-based sync (configurable interval, default 15 min)
- Or manual trigger via `devrag sync` / MCP tool

---

## Verification Plan

### Phase 1 Verification
1. Index via MCP: `mcp__devrag__index_repo` with path to target monorepo — confirm stats show file/chunk counts. Run again to confirm incremental indexing skips unchanged files.
2. Search via MCP: `mcp__devrag__search "how does auth work"` — confirm relevant code chunks returned with file paths and snippets
3. Verify hybrid search: search for an exact function name (should rank high via BM25) and a conceptual query (should rank high via vector similarity)
4. Check `mcp__devrag__status` reports accurate file/chunk counts and last-indexed timestamp

### Phase 2 Verification
1. Sync PRs: `mcp__devrag__sync_prs org/repo` — confirm PR count and chunk creation
2. Search for "why did we migrate to X" — confirm PR results appear with PR numbers and titles
3. Verify query router: code query returns code results, PR query returns PR results

### Phase 3 Verification
1. Index docs: `devrag index docs ./specs` — confirm document chunks with section hierarchy
2. CLI search matches MCP search results for same query
3. Config precedence: project config overrides user config overrides defaults

### Phase 4 Verification
1. Swap to Qdrant backend via config — confirm search results are equivalent
2. Run eval suite against test queries — confirm metrics are computed correctly
3. Re-index with different embedding model — confirm chunks are re-embedded
