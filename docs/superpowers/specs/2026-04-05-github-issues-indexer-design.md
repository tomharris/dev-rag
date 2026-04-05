# GitHub Issues Indexer Design

## Context

DevRAG indexes code, PRs, and documents as knowledge sources. GitHub issues contain valuable project context — bug reports, feature requests, design discussions — that is currently invisible to RAG queries. Adding issues as a knowledge source fills this gap, enabling both historical context retrieval ("why was this built?") and active issue awareness ("what bugs are known here?").

## Approach

Mirror the existing PR indexer pattern: dedicated `IssueIndexer` class, two vector store collections, cursor-based incremental sync. This follows the proven architecture without introducing abstractions or refactoring existing code.

## Data Model

### Chunk Types

| Type | Source | Collection | Content Format |
|------|--------|------------|----------------|
| Description | Issue title + body | `issue_descriptions` | `[{repo} #{number}] {title}\n\n{body}` |
| Comment | Individual comment | `issue_discussions` | `Comment by {author} on #{number}:\n\n{body}` |

### Chunk Metadata

All chunks carry: `repo`, `issue_number`, `issue_title`, `issue_state`, `issue_author`, `issue_labels`, `created_at`, `updated_at`.

Comment chunks additionally carry: `comment_author`, `comment_created_at`.

### Chunk IDs

`sha256(f"{repo}:issue{number}:{chunk_type}:{index}")[:16]` — same scheme as PR chunks.

### Token Truncation

Chunks are truncated to `chunk_max_tokens` (default 512) before embedding, matching the PR indexer safety pattern.

### PR Filtering

The GitHub Issues API returns pull requests as issues. Items with a `pull_request` key present in the API response are skipped to avoid duplicate content with the existing PR indexer.

## Configuration

```python
@dataclass
class IssuesConfig:
    github_token_env: str = "GITHUB_TOKEN"
    backfill_days: int = 90
    chunk_max_tokens: int = 512
```

Added as `issues: IssuesConfig` field on `DevragConfig`. Reuses the same GitHub token as PR indexing.

## New Files

- `devrag/ingest/issue_indexer.py` — `IssueIndexer` class

## Modified Files

- `devrag/utils/github.py` — Add `list_issues()`, `get_issue_comments()` to `GitHubClient`
- `devrag/config.py` — Add `IssuesConfig`, add `issues` field to `DevragConfig`
- `devrag/types.py` — Add `IssueSyncStats` dataclass
- `devrag/stores/metadata_db.py` — Add `issue_sync_cursors` and `issue_chunk_sources` tables + methods
- `devrag/retrieve/query_router.py` — Add `ISSUE_COLLECTIONS`, update routing patterns and `ALL_COLLECTIONS`
- `devrag/cli.py` — Add `devrag index issues <repo> --since 90d` command
- `devrag/mcp_server.py` — Add `sync_issues()` MCP tool
- `devrag/utils/formatters.py` — Add `format_issue_sync_stats()`

## Sync Flow

1. Load GitHub token from env var (`issues.github_token_env`)
2. Read `issue_sync_cursors` table for last sync timestamp for this repo
3. Call `GitHubClient.list_issues(repo, state="all", sort="updated", since=cursor)`
4. For each issue (skip if `pull_request` key present):
   a. Fetch comments via `get_issue_comments(repo, issue_number)`
   b. Delete existing chunks for this issue (handles updates/edits)
   c. Create description chunk + one chunk per comment
   d. Truncate to `chunk_max_tokens`, embed, upsert to vector store
   e. Index in FTS5 via `metadata_db.upsert_fts()`
   f. Track in `issue_chunk_sources` table
5. Update `issue_sync_cursors` with latest `updated_at` timestamp
6. Return `IssueSyncStats`

## Query Routing

Add `ISSUE_COLLECTIONS = ["issue_descriptions", "issue_discussions"]` to the query router.

Issue-intent patterns: queries mentioning "bug", "issue", "feature request", "reported", "filed" route to issue collections.

Issue collections are also added to `ALL_COLLECTIONS` so general queries search issues too.

## Metadata DB Schema Additions

```sql
CREATE TABLE IF NOT EXISTS issue_sync_cursors (
    repo TEXT PRIMARY KEY,
    last_synced TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS issue_chunk_sources (
    chunk_id TEXT PRIMARY KEY,
    repo TEXT NOT NULL,
    issue_number INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_issue_chunk_sources_repo_issue
    ON issue_chunk_sources(repo, issue_number);
```

Methods to add: `get_issue_sync_cursor()`, `set_issue_sync_cursor()`, `set_issue_chunk_source()`, `delete_chunks_for_issue()`.

## Verification

1. **Unit tests:** Test issue chunking logic, PR filtering, metadata construction, token truncation
2. **Integration test:** With a mock GitHub API (using `respx`), verify full sync flow — fetch, chunk, embed, upsert, cursor update
3. **CLI smoke test:** `devrag index issues <test-repo> --since 30d` against a real repo
4. **MCP tool test:** Verify `sync_issues` tool is registered and callable via `devrag serve`
5. **Query routing test:** Verify "bug report" queries hit issue collections, general queries include issues
6. **Incremental sync test:** Run sync twice, verify second run skips unchanged issues
