# Jira Tickets Indexer Design

## Context

DevRAG indexes code, PRs, GitHub issues, and documents as knowledge sources. Many teams use Jira Cloud for project management — tickets contain bug reports, feature specs, sprint planning context, and design decisions that are invisible to RAG queries. Adding Jira as a knowledge source enables retrieval of project management context alongside code ("what's the acceptance criteria for this feature?", "what bugs were filed against this module?").

## Approach

Mirror the existing issue/PR indexer pattern: dedicated `JiraIndexer` class, two vector store collections, cursor-based incremental sync via JQL. Add a new `JiraClient` (httpx-based) following the `GitHubClient` pattern. Target Jira Cloud REST API v3 only.

## Data Model

### Chunk Types

| Type | Source | Collection | Content Format |
|------|--------|------------|----------------|
| Description | Ticket summary + description | `jira_descriptions` | `[{instance} {key}] {summary}\n\n{description_text}` |
| Comment | Individual comment | `jira_discussions` | `Comment by {author} on {key}:\n\n{body}` |

### Chunk Metadata

All chunks carry: `jira_instance`, `ticket_key`, `ticket_summary`, `ticket_status`, `ticket_type`, `reporter`, `assignee`, `priority`, `labels`, `created_at`, `updated_at`.

Comment chunks additionally carry: `comment_author`, `comment_created_at`.

### Chunk IDs

`sha256(f"{instance}:{key}:{chunk_type}:{index}")[:16]` — same scheme as issue/PR chunks.

### Token Truncation

Chunks are truncated to `chunk_max_tokens` (default 512) before embedding, matching the existing safety pattern.

### ADF-to-Text Conversion

Jira Cloud returns descriptions in Atlassian Document Format (a JSON-based rich text format). A recursive extractor walks the ADF tree and collects `text` nodes, joining with appropriate whitespace. This is lossy (drops images, complex table structure) but sufficient for semantic embedding. The converter lives in `JiraClient` as a static method.

## Configuration

```python
@dataclass
class JiraConfig:
    jira_token_env: str = "JIRA_TOKEN"       # env var for API token
    jira_email_env: str = "JIRA_EMAIL"       # env var for email (basic auth)
    instance_url: str = ""                    # e.g. "https://yoursite.atlassian.net"
    jql: str = ""                             # JQL filter for scoping tickets
    backfill_days: int = 90
    chunk_max_tokens: int = 512
```

Added as `jira: JiraConfig` field on `DevragConfig`.

Auth uses Jira Cloud basic auth: `base64(email:api_token)`. Both values come from environment variables (configured via `jira.jira_token_env` and `jira.jira_email_env`), never stored in config files.

The `instance_url` and `jql` fields are required for sync. The CLI and MCP tool should error clearly if either is missing.

## New Files

- `devrag/ingest/jira_indexer.py` — `JiraIndexer` class with `sync()` method
- `devrag/utils/jira_client.py` — `JiraClient` (httpx, auth, pagination, rate limit retry, ADF-to-text)

## Modified Files

- `devrag/config.py` — Add `JiraConfig` dataclass, add `jira` field to `DevragConfig`
- `devrag/types.py` — Add `JiraSyncStats` dataclass
- `devrag/stores/metadata_db.py` — Add `jira_sync_cursors` and `jira_chunk_sources` tables + methods
- `devrag/retrieve/query_router.py` — Add `JIRA_COLLECTIONS`, update routing patterns and `ALL_COLLECTIONS`
- `devrag/cli.py` — Add `devrag index jira --since 90d` command
- `devrag/mcp_server.py` — Add `sync_jira()` MCP tool
- `devrag/utils/formatters.py` — Add Jira search result formatting + `format_jira_sync_stats()`

## Sync Flow

1. Load credentials from env vars (`jira.jira_token_env`, `jira.jira_email_env`)
2. Validate `instance_url` and `jql` are configured
3. Read `jira_sync_cursors` table for last sync timestamp for this instance+JQL combo
4. Build search JQL: user's JQL + ` AND updated >= "cursor_date"` (Jira datetime format: `"2026/04/01 00:00"`)
5. Call `JiraClient.search_issues(jql, fields=["summary","description","status","issuetype","reporter","assignee","priority","labels","comment","created","updated"])` with pagination (`startAt`/`maxResults`, 100 per page)
6. For each ticket:
   a. Extract comments from the response (included via `fields=comment`)
   b. Delete existing chunks for this ticket via `metadata_db.delete_chunks_for_jira_ticket()`
   c. Convert ADF description to plain text
   d. Create description chunk (summary + plain text body)
   e. Create one chunk per comment
   f. Truncate all chunks to `chunk_max_tokens`, embed, upsert to vector store
   g. Index in FTS5 via `metadata_db.upsert_fts()`
   h. Track in `jira_chunk_sources` table
7. Update `jira_sync_cursors` with latest `updated` timestamp from the batch
8. Return `JiraSyncStats`

## JiraClient API

```python
class JiraClient:
    def __init__(self, instance_url: str, email: str, api_token: str):
        ...

    def search_issues(self, jql: str, fields: list[str], max_results: int = 100) -> Iterator[dict]:
        """Paginated JQL search. Yields individual issues."""
        ...

    @staticmethod
    def adf_to_text(adf: dict | None) -> str:
        """Recursively extract text from Atlassian Document Format JSON."""
        ...
```

Uses `httpx.Client` with:
- Basic auth header: `Authorization: Basic base64(email:api_token)`
- Rate limit handling: retry on 429 with `Retry-After` header
- Timeout: 30s per request
- Base URL: `{instance_url}/rest/api/3/`

## Query Routing

Add `JIRA_COLLECTIONS = ["jira_descriptions", "jira_discussions"]` to the query router.

Jira-intent patterns: queries mentioning "jira", "ticket", "sprint", "epic", "story" route to Jira collections. The existing "bug", "issue" patterns already route to issue collections; these will also apply to Jira collections since both contain issue-like content.

Jira collections are added to `ALL_COLLECTIONS` so general queries search Jira tickets too.

## CLI Command

```
devrag index jira [--since 90d]
```

Reads `instance_url` and `jql` from config (`.devrag.yaml`). The `--since` flag overrides `backfill_days`. No positional arguments needed since the instance and scope come from config.

## MCP Tool

```python
@mcp.tool
def sync_jira(since_days: int = 90) -> str:
    """Sync Jira tickets based on configured JQL filter."""
    ...
```

Reads instance/JQL from config, same lazy-init pattern as other MCP tools.

## Metadata DB Schema Additions

```sql
CREATE TABLE IF NOT EXISTS jira_sync_cursors (
    cursor_key TEXT PRIMARY KEY,  -- instance_url + hash(jql) for uniqueness
    last_synced TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jira_chunk_sources (
    chunk_id TEXT PRIMARY KEY,
    instance_url TEXT NOT NULL,
    ticket_key TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jira_chunk_sources_instance_ticket
    ON jira_chunk_sources(instance_url, ticket_key);
```

Methods to add: `get_jira_sync_cursor()`, `set_jira_sync_cursor()`, `set_jira_chunk_source()`, `delete_chunks_for_jira_ticket()`.

The cursor key is `sha256(f"{instance_url}:{jql}")[:16]`, so different JQL filters on the same instance track independent cursors.

## Verification

1. **Unit tests:** ADF-to-text conversion (nested structures, empty docs, plain text fallback), chunking logic, metadata construction, JQL cursor date injection
2. **Integration test:** Mock Jira API with `respx`, verify full sync flow — search, chunk, embed, upsert, cursor update
3. **CLI smoke test:** `devrag index jira --since 30d` against a real Jira Cloud instance
4. **MCP tool test:** Verify `sync_jira` tool is registered and callable via `devrag serve`
5. **Query routing test:** Verify "jira ticket about auth" hits Jira collections, general queries include Jira
6. **Incremental sync test:** Run sync twice, verify second run only processes tickets updated since first sync
