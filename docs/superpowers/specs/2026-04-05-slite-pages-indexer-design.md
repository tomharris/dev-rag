# Slite Pages Indexer — Design Spec

## Context

DevRAG indexes code, GitHub PRs, GitHub issues, Jira tickets, and documents as knowledge sources. Slite is a knowledge base tool used for internal documentation — adding it as a source lets DevRAG surface wiki/knowledge base content alongside code and tickets in search results.

## Overview

Add Slite pages as a new knowledge source following the established indexer pattern. Uses the Slite REST API to fetch pages as markdown, chunks them with section-aware splitting, and stores them in a `slite_pages` vector collection.

## Components

### 1. SliteClient (`devrag/utils/slite_client.py`)

HTTP client for the Slite REST API, following the `JiraClient` pattern.

- **Auth**: Bearer token via `Authorization: Bearer <token>` header
- **Base URL**: `https://api.slite.com/v1/`
- **HTTP library**: `httpx.Client` with 30s timeout
- **Rate limiting**: Detect 429, sleep `Retry-After` (capped at 60s), retry once

**Methods:**

| Method | API Endpoint | Purpose |
|--------|-------------|---------|
| `list_notes(channel_ids, since_days_ago, cursor)` | `GET /knowledge-management/notes` | Paginated note listing with channel + date filtering. Yields note stubs (id, title, url, updatedAt). Uses cursor pagination (`hasNextPage` / `nextCursor`). |
| `get_note(note_id, format="md")` | `GET /notes/{noteId}?format=md` | Fetch full note content as markdown. Returns dict with id, title, url, content, updatedAt. |

**`list_notes` parameters mapping:**
- `channel_ids` → `channelIdList[]` query param (omitted if empty = all channels)
- `since_days_ago` → `sinceDaysAgo` query param
- `cursor` → `cursor` query param for pagination
- `first` → hardcode to 50 (API max)

### 2. SliteConfig (`devrag/config.py`)

```python
@dataclass
class SliteConfig:
    slite_token_env: str = "SLITE_TOKEN"
    channel_ids: list[str] = field(default_factory=list)
    backfill_days: int = 90
    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 50
```

- Added to `DevragConfig` as `slite: SliteConfig = field(default_factory=SliteConfig)`
- `channel_ids`: empty list means index all channels; otherwise only index pages in these channels
- Token resolved at runtime from env var named in `slite_token_env`

**Example `.devrag.yaml`:**
```yaml
slite:
  slite_token_env: SLITE_TOKEN
  channel_ids:
    - "abc123"
    - "def456"
  backfill_days: 90
```

### 3. SliteIndexer (`devrag/ingest/slite_indexer.py`)

Main indexer class. Constructor: `SliteIndexer(vector_store, metadata_db, embedder, slite_client, chunk_max_tokens, chunk_overlap_tokens)`.

**`sync(since_days)` flow:**

1. Get sync cursor from `metadata_db.get_slite_sync_cursor("default")` (or use `since_days` if no cursor)
2. Convert stored ISO timestamp to `since_days_ago` integer (with +1 day overlap buffer to handle edge cases)
3. Call `slite_client.list_notes(channel_ids, since_days_ago)` — iterate all pages
4. For each note:
   a. Fetch full markdown: `slite_client.get_note(note_id, format="md")`
   b. Skip if content is empty/whitespace
   c. Delete old chunks: `metadata_db.delete_chunks_for_slite_page("default", note_id)`
   d. Chunk markdown using `chunk_slite_page()` (see below)
   e. Batch embed all chunks
   f. Upsert to vector store collection `slite_pages`
   g. Store chunk sources and FTS index
   h. Track `latest_updated` timestamp
5. Store cursor: `metadata_db.set_slite_sync_cursor("default", latest_updated)`
6. Return `SliteSyncStats`

**`chunk_slite_page(note, max_tokens, overlap_tokens)` function:**

Reuses `split_markdown()` from `doc_indexer.py` for section-aware splitting, then applies the same oversized-section handling (paragraph splitting with token overlap).

Chunk ID: `sha256(f"slite:{page_id}:{section_path}:{index}")[:16]`

Chunk metadata:
```python
{
    "page_id": str,
    "page_title": str,
    "page_url": str,
    "updated_at": str,       # ISO timestamp
    "section_path": str,     # "Heading > Subheading"
    "chunk_type": "slite_page",
    "entity_name": str,      # leaf section title or page title
}
```

### 4. MetadataDB Changes (`devrag/stores/metadata_db.py`)

**New tables** (added to `_create_tables()`):

```sql
CREATE TABLE IF NOT EXISTS slite_sync_cursors (
    workspace_id TEXT PRIMARY KEY,
    last_synced TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS slite_chunk_sources (
    chunk_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    page_id TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_slite_chunk_sources_workspace_page
    ON slite_chunk_sources(workspace_id, page_id);
```

**New methods:**
- `get_slite_sync_cursor(workspace_id) -> str | None`
- `set_slite_sync_cursor(workspace_id, last_synced)`
- `set_slite_chunk_source(chunk_id, workspace_id, page_id)`
- `get_chunks_for_slite_page(workspace_id, page_id) -> list[str]`
- `delete_chunks_for_slite_page(workspace_id, page_id)` — deletes chunk sources + FTS entries, returns chunk IDs for vector store deletion

### 5. Query Router (`devrag/retrieve/query_router.py`)

- Add `SLITE_COLLECTIONS = ["slite_pages"]`
- Add to `ALL_COLLECTIONS`
- Add `_SLITE_PATTERNS`: `r"\bslite\b"`, `r"\bwiki\b"`, `r"\bknowledge\s+base\b"`, `r"\binternal\s+doc\b"`
- Route `scope == "slite"` to `SLITE_COLLECTIONS`
- Slite pattern match returns `SLITE_COLLECTIONS + DOC_COLLECTIONS` (Slite content is similar to docs)

### 6. CLI (`devrag/cli.py`)

New subcommand under `index`:

```
devrag index slite [--since 90d]
```

- Load config, resolve `SLITE_TOKEN` from env
- Create `SliteClient`, `SliteIndexer`
- Call `indexer.sync(since_days=days)`
- Print `format_slite_sync_stats(stats)`

### 7. MCP Server (`devrag/mcp_server.py`)

```python
@mcp.tool
def sync_slite(since_days: int = 90) -> str:
    """Sync Slite pages for the configured workspace.

    Fetches page content as markdown and indexes with section-aware chunking.
    Uses cursor-based sync to avoid re-fetching unchanged pages.
    Filters by configured channel IDs if set.

    Requires SLITE_TOKEN environment variable.
    """
```

### 8. Types (`devrag/types.py`)

```python
@dataclass
class SliteSyncStats:
    pages_fetched: int = 0
    pages_indexed: int = 0
    pages_skipped: int = 0
    chunks_created: int = 0
```

### 9. Formatters (`devrag/utils/formatters.py`)

**`format_slite_sync_stats(stats)`** — same pattern as other formatters:
```
Fetched 42 Slite pages. Indexed 38 pages (156 chunks). Skipped 4 pages.
```

**`format_search_results()`** — add Slite block that detects `chunk_type == "slite_page"`:
```
### 1. [Page Title] Section Path
*Slite page — last updated 2026-04-01*
```preview```
```

## Files Modified

| File | Change |
|------|--------|
| `devrag/utils/slite_client.py` | **New** — Slite REST API client |
| `devrag/ingest/slite_indexer.py` | **New** — Slite page indexer |
| `devrag/config.py` | Add `SliteConfig` dataclass + `slite` field on `DevragConfig` |
| `devrag/types.py` | Add `SliteSyncStats` dataclass |
| `devrag/stores/metadata_db.py` | Add cursor/chunk-source tables and methods |
| `devrag/retrieve/query_router.py` | Add Slite collections and routing patterns |
| `devrag/cli.py` | Add `index slite` subcommand |
| `devrag/mcp_server.py` | Add `sync_slite()` MCP tool |
| `devrag/utils/formatters.py` | Add `format_slite_sync_stats()` + search result formatting |
| `CLAUDE.md` | Document Slite indexer in Architecture section |
| `README.md` | Add Slite to knowledge sources |

## Verification

1. **Unit tests**: Test `SliteClient` with `respx` mocking (pagination, rate limiting, error handling). Test `chunk_slite_page()` with sample markdown. Test `SliteIndexer.sync()` with mocked client.
2. **Integration test**: With a real Slite API token, run `devrag index slite --since 7d` and verify pages appear in `devrag search "slite query"`.
3. **Query routing**: Verify `"what does our wiki say about X"` routes to `slite_pages + documents`.
