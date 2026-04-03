# DevRAG Phase 2: PR Knowledge — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add GitHub PR history (diffs, descriptions, review comments) as a searchable knowledge source, with a query router that directs searches to the right collections.

**Architecture:** GitHub REST API client fetches PRs, diffs, and review comments. PR indexer chunks them and stores in two new ChromaDB collections (`pr_diffs`, `pr_discussions`) + SQLite FTS5. A keyword-based query router classifies intent and targets appropriate collections. HybridSearch is extended to search multiple collections. MCP server gains `sync_prs` tool and `scope` parameter on `search`.

**Tech Stack:** httpx (GitHub API), existing ChromaDB + SQLite + Ollama stack from Phase 1.

---

## File Structure

```
devrag/
├── devrag/
│   ├── config.py               # MODIFY: add PrsConfig
│   ├── types.py                # MODIFY: add PRSyncStats
│   ├── mcp_server.py           # MODIFY: add sync_prs tool, scope on search
│   │
│   ├── ingest/
│   │   └── pr_indexer.py       # CREATE: PR ingestion and chunking
│   │
│   ├── retrieve/
│   │   ├── hybrid_search.py    # MODIFY: multi-collection search
│   │   └── query_router.py     # CREATE: keyword-based intent routing
│   │
│   ├── stores/
│   │   └── metadata_db.py      # MODIFY: add pr_sync_cursors table
│   │
│   └── utils/
│       ├── github.py           # CREATE: GitHub API client
│       └── formatters.py       # MODIFY: PR result formatting
│
└── tests/
    ├── test_github.py          # CREATE
    ├── test_pr_indexer.py      # CREATE
    ├── test_query_router.py    # CREATE
    ├── test_hybrid_search.py   # MODIFY: multi-collection tests
    └── test_formatters.py      # MODIFY: PR formatting tests
```

---

## Task 1: Config + Types Updates

**Files:**
- Modify: `devrag/config.py`
- Modify: `devrag/types.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write the failing test for PrsConfig**

Add to `tests/test_config.py`:

```python
def test_default_config_has_prs_section():
    config = DevragConfig()
    assert config.prs.github_token_env == "GITHUB_TOKEN"
    assert config.prs.backfill_days == 90
    assert config.prs.include_draft is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_default_config_has_prs_section -v`
Expected: FAIL — `AttributeError: 'DevragConfig' object has no attribute 'prs'`

- [ ] **Step 3: Add PrsConfig to config.py**

Add after `CodeConfig` in `devrag/config.py`:

```python
@dataclass
class PrsConfig:
    github_token_env: str = "GITHUB_TOKEN"
    backfill_days: int = 90
    include_draft: bool = False
```

Add to `DevragConfig`:

```python
@dataclass
class DevragConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    code: CodeConfig = field(default_factory=CodeConfig)
    prs: PrsConfig = field(default_factory=PrsConfig)
```

- [ ] **Step 4: Add PRSyncStats to types.py**

Add to `devrag/types.py`:

```python
@dataclass
class PRSyncStats:
    """Statistics from a PR sync run."""
    prs_fetched: int = 0
    prs_indexed: int = 0
    prs_skipped: int = 0
    chunks_created: int = 0
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add devrag/config.py devrag/types.py tests/test_config.py
git commit -m "feat: add PrsConfig and PRSyncStats for Phase 2 PR support"
```

---

## Task 2: MetadataDB — PR Sync Cursors

**Files:**
- Modify: `devrag/stores/metadata_db.py`
- Modify: `tests/test_metadata_db.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_metadata_db.py`:

```python
def test_pr_sync_cursor_set_and_get(tmp_dir):
    db = MetadataDB(str(tmp_dir / "meta.db"))
    db.set_pr_sync_cursor("acme/backend", "2026-03-15T10:00:00Z")
    assert db.get_pr_sync_cursor("acme/backend") == "2026-03-15T10:00:00Z"
    assert db.get_pr_sync_cursor("other/repo") is None


def test_pr_sync_cursor_update(tmp_dir):
    db = MetadataDB(str(tmp_dir / "meta.db"))
    db.set_pr_sync_cursor("acme/backend", "2026-03-01T00:00:00Z")
    db.set_pr_sync_cursor("acme/backend", "2026-03-15T10:00:00Z")
    assert db.get_pr_sync_cursor("acme/backend") == "2026-03-15T10:00:00Z"


def test_pr_chunk_source_mapping(tmp_dir):
    db = MetadataDB(str(tmp_dir / "meta.db"))
    db.set_pr_chunk_source("chunk_pr_1", "acme/backend", 1234)
    db.set_pr_chunk_source("chunk_pr_2", "acme/backend", 1234)
    db.set_pr_chunk_source("chunk_pr_3", "acme/backend", 5678)
    chunks = db.get_chunks_for_pr("acme/backend", 1234)
    assert set(chunks) == {"chunk_pr_1", "chunk_pr_2"}


def test_delete_chunks_for_pr(tmp_dir):
    db = MetadataDB(str(tmp_dir / "meta.db"))
    db.set_pr_chunk_source("c1", "acme/backend", 100)
    db.upsert_fts("c1", "some pr text")
    db.delete_chunks_for_pr("acme/backend", 100)
    assert db.get_chunks_for_pr("acme/backend", 100) == []
    assert db.search_fts("some pr text", limit=5) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_metadata_db.py::test_pr_sync_cursor_set_and_get -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 3: Add PR tables and methods to MetadataDB**

Add to `_create_tables` in `devrag/stores/metadata_db.py`:

```sql
CREATE TABLE IF NOT EXISTS pr_sync_cursors (
    repo TEXT PRIMARY KEY,
    last_synced TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pr_chunk_sources (
    chunk_id TEXT PRIMARY KEY,
    repo TEXT NOT NULL,
    pr_number INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pr_chunk_sources_repo_pr
    ON pr_chunk_sources(repo, pr_number);
```

Add methods to `MetadataDB`:

```python
# --- PR sync cursors ---

def get_pr_sync_cursor(self, repo: str) -> str | None:
    row = self._conn.execute(
        "SELECT last_synced FROM pr_sync_cursors WHERE repo = ?", (repo,)
    ).fetchone()
    return row[0] if row else None

def set_pr_sync_cursor(self, repo: str, last_synced: str) -> None:
    self._conn.execute(
        "INSERT OR REPLACE INTO pr_sync_cursors (repo, last_synced) VALUES (?, ?)",
        (repo, last_synced),
    )
    self._conn.commit()

# --- PR chunk sources ---

def set_pr_chunk_source(self, chunk_id: str, repo: str, pr_number: int) -> None:
    self._conn.execute(
        "INSERT OR REPLACE INTO pr_chunk_sources (chunk_id, repo, pr_number) VALUES (?, ?, ?)",
        (chunk_id, repo, pr_number),
    )
    self._conn.commit()

def get_chunks_for_pr(self, repo: str, pr_number: int) -> list[str]:
    rows = self._conn.execute(
        "SELECT chunk_id FROM pr_chunk_sources WHERE repo = ? AND pr_number = ?",
        (repo, pr_number),
    ).fetchall()
    return [r[0] for r in rows]

def delete_chunks_for_pr(self, repo: str, pr_number: int) -> None:
    chunk_ids = self.get_chunks_for_pr(repo, pr_number)
    if chunk_ids:
        self.delete_fts(chunk_ids)
        placeholders = ",".join("?" for _ in chunk_ids)
        self._conn.execute(
            f"DELETE FROM pr_chunk_sources WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )
        self._conn.commit()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_metadata_db.py -v`
Expected: All 14 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add devrag/stores/metadata_db.py tests/test_metadata_db.py
git commit -m "feat: PR sync cursors and chunk source tracking in MetadataDB"
```

---

## Task 3: GitHub API Client

**Files:**
- Create: `devrag/utils/github.py`
- Create: `tests/test_github.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_github.py
import json
import time

import httpx
import pytest
import respx

from devrag.utils.github import GitHubClient, parse_diff_hunks


@respx.mock
def test_list_prs():
    respx.get("https://api.github.com/repos/acme/backend/pulls").respond(json=[
        {
            "number": 1,
            "title": "Add auth",
            "body": "Adds authentication module",
            "state": "closed",
            "user": {"login": "alice"},
            "labels": [{"name": "feature"}],
            "created_at": "2026-03-01T00:00:00Z",
            "updated_at": "2026-03-02T00:00:00Z",
            "merged_at": "2026-03-02T12:00:00Z",
            "draft": False,
        },
    ])
    client = GitHubClient(token="test-token")
    prs = client.list_prs("acme/backend", state="all", per_page=100)
    assert len(prs) == 1
    assert prs[0]["number"] == 1
    assert prs[0]["title"] == "Add auth"


@respx.mock
def test_get_pr_files():
    respx.get("https://api.github.com/repos/acme/backend/pulls/1/files").respond(json=[
        {
            "filename": "src/auth.py",
            "status": "added",
            "additions": 50,
            "deletions": 0,
            "patch": "@@ -0,0 +1,50 @@\n+def authenticate():\n+    pass",
        },
    ])
    client = GitHubClient(token="test-token")
    files = client.get_pr_files("acme/backend", 1)
    assert len(files) == 1
    assert files[0]["filename"] == "src/auth.py"
    assert "patch" in files[0]


@respx.mock
def test_get_pr_comments():
    respx.get("https://api.github.com/repos/acme/backend/pulls/1/comments").respond(json=[
        {
            "id": 101,
            "body": "Consider using bcrypt here",
            "user": {"login": "bob"},
            "path": "src/auth.py",
            "line": 10,
            "created_at": "2026-03-02T10:00:00Z",
        },
    ])
    client = GitHubClient(token="test-token")
    comments = client.get_pr_comments("acme/backend", 1)
    assert len(comments) == 1
    assert comments[0]["body"] == "Consider using bcrypt here"


@respx.mock
def test_pagination():
    page1_url = "https://api.github.com/repos/acme/backend/pulls"
    page2_url = "https://api.github.com/repos/acme/backend/pulls?page=2"

    respx.get(page1_url).respond(
        json=[{"number": 1, "title": "PR 1"}],
        headers={"link": f'<{page2_url}>; rel="next"'},
    )
    respx.get(page2_url).respond(
        json=[{"number": 2, "title": "PR 2"}],
        headers={},
    )

    client = GitHubClient(token="test-token")
    all_items = client.paginate(page1_url, params={"per_page": 1})
    assert len(all_items) == 2
    assert all_items[0]["number"] == 1
    assert all_items[1]["number"] == 2


@respx.mock
def test_rate_limit_backoff():
    """Client should wait and retry when rate limited."""
    reset_time = str(int(time.time()) + 1)  # 1 second from now

    call_count = 0

    def handler(request):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(
                403,
                json={"message": "API rate limit exceeded"},
                headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": reset_time},
            )
        return httpx.Response(200, json=[{"number": 1}])

    respx.get("https://api.github.com/repos/acme/backend/pulls").mock(side_effect=handler)
    client = GitHubClient(token="test-token")
    result = client.list_prs("acme/backend")
    assert len(result) == 1
    assert call_count == 2


def test_parse_diff_hunks():
    patch = """@@ -10,4 +10,6 @@ class Auth:
     def login(self):
         pass
+    def logout(self):
+        pass
@@ -20,3 +22,5 @@ class Auth:
     def refresh(self):
         pass
+    def revoke(self):
+        pass"""
    hunks = parse_diff_hunks(patch, "src/auth.py")
    assert len(hunks) == 2
    assert hunks[0]["file_path"] == "src/auth.py"
    assert "+    def logout" in hunks[0]["content"]
    assert "+    def revoke" in hunks[1]["content"]


def test_github_client_no_token():
    """Client without token should still work (unauthenticated)."""
    client = GitHubClient(token=None)
    assert "Authorization" not in client._headers
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_github.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write GitHub client implementation**

```python
# devrag/utils/github.py
from __future__ import annotations

import re
import time

import httpx


API_BASE = "https://api.github.com"


def parse_diff_hunks(patch: str, file_path: str) -> list[dict]:
    """Parse a unified diff patch into individual hunks with metadata."""
    if not patch:
        return []

    hunks: list[dict] = []
    current_header = ""
    current_lines: list[str] = []

    for line in patch.split("\n"):
        if line.startswith("@@"):
            if current_lines:
                hunks.append({
                    "file_path": file_path,
                    "header": current_header,
                    "content": "\n".join(current_lines),
                })
            current_header = line
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        hunks.append({
            "file_path": file_path,
            "header": current_header,
            "content": "\n".join(current_lines),
        })

    return hunks


def _get_next_url(response: httpx.Response) -> str | None:
    """Extract the 'next' URL from GitHub's Link header."""
    link = response.headers.get("link", "")
    match = re.search(r'<([^>]+)>;\s*rel="next"', link)
    return match.group(1) if match else None


class GitHubClient:
    def __init__(self, token: str | None = None) -> None:
        self._headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            self._headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.Client(headers=self._headers, timeout=30.0)

    def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make a request with rate-limit handling."""
        resp = self._client.request(method, url, **kwargs)

        if resp.status_code in (403, 429):
            remaining = int(resp.headers.get("x-ratelimit-remaining", "1"))
            if remaining == 0:
                reset_at = int(resp.headers.get("x-ratelimit-reset", "0"))
                retry_after = int(resp.headers.get("retry-after", "0"))
                wait = retry_after if retry_after else max(reset_at - time.time(), 0) + 1
                time.sleep(min(wait, 60))  # cap at 60s
                resp = self._client.request(method, url, **kwargs)

        resp.raise_for_status()
        return resp

    def paginate(self, url: str, params: dict | None = None) -> list[dict]:
        """Fetch all pages from a paginated GitHub endpoint."""
        all_items: list[dict] = []
        current_url = url
        current_params = params

        while current_url:
            resp = self._request("GET", current_url, params=current_params)
            all_items.extend(resp.json())
            current_url = _get_next_url(resp)
            current_params = None  # params are embedded in the next URL

        return all_items

    def list_prs(
        self,
        repo: str,
        state: str = "all",
        sort: str = "updated",
        direction: str = "desc",
        per_page: int = 100,
        since: str | None = None,
    ) -> list[dict]:
        """List pull requests for a repo."""
        params: dict = {
            "state": state,
            "sort": sort,
            "direction": direction,
            "per_page": per_page,
        }
        url = f"{API_BASE}/repos/{repo}/pulls"
        return self.paginate(url, params=params)

    def get_pr_files(self, repo: str, pr_number: int) -> list[dict]:
        """Get files changed in a PR (includes patch/diff content)."""
        url = f"{API_BASE}/repos/{repo}/pulls/{pr_number}/files"
        return self.paginate(url, params={"per_page": 100})

    def get_pr_comments(self, repo: str, pr_number: int) -> list[dict]:
        """Get review comments on a PR."""
        url = f"{API_BASE}/repos/{repo}/pulls/{pr_number}/comments"
        return self.paginate(url, params={"per_page": 100})

    def close(self) -> None:
        self._client.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_github.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add devrag/utils/github.py tests/test_github.py
git commit -m "feat: GitHub API client with rate limiting and pagination"
```

---

## Task 4: PR Indexer

**Files:**
- Create: `devrag/ingest/pr_indexer.py`
- Create: `tests/test_pr_indexer.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pr_indexer.py
import hashlib
from unittest.mock import MagicMock, patch

import pytest

from devrag.ingest.pr_indexer import PRIndexer, chunk_pr


def _make_pr(number=1, title="Add auth", body="Adds authentication", state="closed",
             author="alice", labels=None, merged_at="2026-03-02T12:00:00Z",
             updated_at="2026-03-02T12:00:00Z", draft=False):
    return {
        "number": number,
        "title": title,
        "body": body,
        "state": state,
        "user": {"login": author},
        "labels": [{"name": l} for l in (labels or [])],
        "merged_at": merged_at,
        "updated_at": updated_at,
        "draft": draft,
    }


def _make_file(filename="src/auth.py", status="added", patch="@@ -0,0 +1,3 @@\n+def auth():\n+    pass"):
    return {"filename": filename, "status": status, "patch": patch}


def _make_comment(body="Looks good", user="bob", path="src/auth.py", line=10):
    return {
        "body": body,
        "user": {"login": user},
        "path": path,
        "line": line,
        "created_at": "2026-03-02T10:00:00Z",
    }


def test_chunk_pr_creates_description_chunk():
    pr = _make_pr()
    files = []
    comments = []
    chunks = chunk_pr(pr, files, comments, repo="acme/backend")
    desc_chunks = [c for c in chunks if c.metadata["chunk_type"] == "description"]
    assert len(desc_chunks) == 1
    assert "Add auth" in desc_chunks[0].text
    assert "Adds authentication" in desc_chunks[0].text
    assert desc_chunks[0].metadata["pr_number"] == 1
    assert desc_chunks[0].metadata["pr_author"] == "alice"
    assert desc_chunks[0].metadata["repo"] == "acme/backend"


def test_chunk_pr_creates_diff_chunks():
    pr = _make_pr()
    files = [_make_file()]
    comments = []
    chunks = chunk_pr(pr, files, comments, repo="acme/backend")
    diff_chunks = [c for c in chunks if c.metadata["chunk_type"] == "diff"]
    assert len(diff_chunks) >= 1
    assert diff_chunks[0].metadata["file_path"] == "src/auth.py"
    assert "+def auth()" in diff_chunks[0].text


def test_chunk_pr_creates_comment_chunks():
    pr = _make_pr()
    files = []
    comments = [_make_comment()]
    chunks = chunk_pr(pr, files, comments, repo="acme/backend")
    comment_chunks = [c for c in chunks if c.metadata["chunk_type"] == "review_comment"]
    assert len(comment_chunks) == 1
    assert "Looks good" in comment_chunks[0].text
    assert comment_chunks[0].metadata["reviewer"] == "bob"
    assert comment_chunks[0].metadata["file_path"] == "src/auth.py"


def test_chunk_pr_metadata_fields():
    pr = _make_pr(labels=["security", "breaking-change"])
    files = [_make_file()]
    comments = []
    chunks = chunk_pr(pr, files, comments, repo="acme/backend")
    for chunk in chunks:
        assert chunk.metadata["repo"] == "acme/backend"
        assert chunk.metadata["pr_number"] == 1
        assert chunk.metadata["pr_title"] == "Add auth"
        assert chunk.metadata["pr_state"] == "closed"
        assert chunk.metadata["pr_author"] == "alice"


def test_chunk_pr_skips_empty_body():
    pr = _make_pr(body=None)
    files = []
    comments = []
    chunks = chunk_pr(pr, files, comments, repo="acme/backend")
    desc_chunks = [c for c in chunks if c.metadata["chunk_type"] == "description"]
    # Should still create a description chunk with title only
    assert len(desc_chunks) == 1
    assert "Add auth" in desc_chunks[0].text


def test_chunk_ids_are_deterministic():
    pr = _make_pr()
    files = [_make_file()]
    comments = [_make_comment()]
    chunks1 = chunk_pr(pr, files, comments, repo="acme/backend")
    chunks2 = chunk_pr(pr, files, comments, repo="acme/backend")
    assert [c.id for c in chunks1] == [c.id for c in chunks2]


def test_pr_indexer_sync(tmp_dir):
    """Integration test for PRIndexer.sync with mocked GitHub client."""
    from devrag.stores.chroma_store import ChromaStore
    from devrag.stores.metadata_db import MetadataDB

    store = ChromaStore(persist_dir=str(tmp_dir / "chroma"))
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])

    mock_github = MagicMock()
    mock_github.list_prs.return_value = [_make_pr(number=42, title="Fix bug")]
    mock_github.get_pr_files.return_value = [_make_file()]
    mock_github.get_pr_comments.return_value = [_make_comment()]

    indexer = PRIndexer(store, meta, embedder, mock_github)
    stats = indexer.sync("acme/backend", since_days=90)

    assert stats.prs_fetched == 1
    assert stats.prs_indexed == 1
    assert stats.chunks_created >= 3  # description + diff + comment
    assert store.count("pr_diffs") >= 1
    assert store.count("pr_discussions") >= 1

    # Verify sync cursor was set
    cursor = meta.get_pr_sync_cursor("acme/backend")
    assert cursor is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pr_indexer.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write PR indexer implementation**

```python
# devrag/ingest/pr_indexer.py
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

from devrag.types import Chunk, PRSyncStats
from devrag.utils.github import GitHubClient, parse_diff_hunks


def _make_pr_chunk_id(repo: str, pr_number: int, chunk_type: str, index: int) -> str:
    """Create a deterministic chunk ID for a PR chunk."""
    raw = f"{repo}:pr{pr_number}:{chunk_type}:{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _pr_base_metadata(pr: dict, repo: str) -> dict:
    """Build the common metadata fields for all chunks of a PR."""
    labels = [l["name"] for l in pr.get("labels", [])]
    return {
        "repo": repo,
        "pr_number": pr["number"],
        "pr_title": pr["title"],
        "pr_state": pr["state"],
        "pr_author": pr["user"]["login"],
        "pr_labels": labels,
        "merged_at": pr.get("merged_at") or "",
    }


def chunk_pr(
    pr: dict,
    files: list[dict],
    comments: list[dict],
    repo: str,
) -> list[Chunk]:
    """Chunk a single PR into description, diff, and comment chunks."""
    chunks: list[Chunk] = []
    base_meta = _pr_base_metadata(pr, repo)

    # 1. Description chunk (title + body)
    title = pr["title"] or ""
    body = pr.get("body") or ""
    desc_text = f"# PR #{pr['number']}: {title}\n\n{body}".strip()
    chunks.append(Chunk(
        id=_make_pr_chunk_id(repo, pr["number"], "description", 0),
        text=desc_text,
        metadata={**base_meta, "chunk_type": "description", "file_path": ""},
    ))

    # 2. Diff chunks (one per file hunk)
    diff_index = 0
    for file_info in files:
        patch = file_info.get("patch")
        if not patch:
            continue
        filename = file_info["filename"]
        hunks = parse_diff_hunks(patch, filename)
        for hunk in hunks:
            diff_text = f"File: {filename} ({file_info.get('status', 'modified')})\n{hunk['content']}"
            chunks.append(Chunk(
                id=_make_pr_chunk_id(repo, pr["number"], "diff", diff_index),
                text=diff_text,
                metadata={**base_meta, "chunk_type": "diff", "file_path": filename},
            ))
            diff_index += 1

    # 3. Review comment chunks
    for i, comment in enumerate(comments):
        reviewer = comment.get("user", {}).get("login", "unknown") if comment.get("user") else "unknown"
        comment_path = comment.get("path", "")
        comment_text = f"Review comment by {reviewer} on {comment_path}:\n{comment['body']}"
        chunks.append(Chunk(
            id=_make_pr_chunk_id(repo, pr["number"], "review_comment", i),
            text=comment_text,
            metadata={
                **base_meta,
                "chunk_type": "review_comment",
                "file_path": comment_path,
                "reviewer": reviewer,
            },
        ))

    return chunks


class PRIndexer:
    """Indexes GitHub PRs by fetching, chunking, embedding, and storing."""

    def __init__(
        self,
        vector_store,   # ChromaStore
        metadata_db,    # MetadataDB
        embedder,       # OllamaEmbedder
        github_client,  # GitHubClient
    ) -> None:
        self.vector_store = vector_store
        self.metadata_db = metadata_db
        self.embedder = embedder
        self.github = github_client

    def sync(self, repo: str, since_days: int = 90) -> PRSyncStats:
        """Sync PRs for a repository. Fetches PRs updated since last sync cursor."""
        stats = PRSyncStats()

        # Determine start date
        cursor = self.metadata_db.get_pr_sync_cursor(repo)
        if cursor:
            since_date = cursor
        else:
            since_dt = datetime.now(timezone.utc) - timedelta(days=since_days)
            since_date = since_dt.isoformat()

        # Fetch PRs
        prs = self.github.list_prs(repo, state="all", sort="updated")
        stats.prs_fetched = len(prs)

        latest_updated = since_date
        for pr in prs:
            updated_at = pr.get("updated_at", "")
            if updated_at and updated_at < since_date:
                stats.prs_skipped += 1
                continue

            if updated_at and updated_at > latest_updated:
                latest_updated = updated_at

            # Remove old chunks for this PR (re-index)
            self.metadata_db.delete_chunks_for_pr(repo, pr["number"])

            # Fetch details
            files = self.github.get_pr_files(repo, pr["number"])
            comments = self.github.get_pr_comments(repo, pr["number"])

            # Chunk
            chunks = chunk_pr(pr, files, comments, repo=repo)
            if not chunks:
                stats.prs_indexed += 1
                continue

            # Embed
            texts = [c.text for c in chunks]
            embeddings = self.embedder.embed(texts)

            # Store — diffs go to pr_diffs, comments go to pr_discussions,
            # descriptions go to pr_diffs (they're about the change)
            diff_chunks = [c for c in chunks if c.metadata["chunk_type"] in ("diff", "description")]
            discussion_chunks = [c for c in chunks if c.metadata["chunk_type"] == "review_comment"]

            # Map chunk IDs to their embeddings
            embed_map = {c.id: embeddings[i] for i, c in enumerate(chunks)}

            if diff_chunks:
                self.vector_store.upsert(
                    collection="pr_diffs",
                    ids=[c.id for c in diff_chunks],
                    embeddings=[embed_map[c.id] for c in diff_chunks],
                    documents=[c.text for c in diff_chunks],
                    metadatas=[c.metadata for c in diff_chunks],
                )

            if discussion_chunks:
                self.vector_store.upsert(
                    collection="pr_discussions",
                    ids=[c.id for c in discussion_chunks],
                    embeddings=[embed_map[c.id] for c in discussion_chunks],
                    documents=[c.text for c in discussion_chunks],
                    metadatas=[c.metadata for c in discussion_chunks],
                )

            # Store chunk sources and FTS
            for i, chunk in enumerate(chunks):
                self.metadata_db.set_pr_chunk_source(chunk.id, repo, pr["number"])
                self.metadata_db.upsert_fts(chunk.id, chunk.text)

            stats.prs_indexed += 1
            stats.chunks_created += len(chunks)

        # Update sync cursor
        self.metadata_db.set_pr_sync_cursor(repo, latest_updated)

        return stats
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pr_indexer.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add devrag/ingest/pr_indexer.py tests/test_pr_indexer.py
git commit -m "feat: PR indexer with diff, description, and review comment chunking"
```

---

## Task 5: Query Router

**Files:**
- Create: `devrag/retrieve/query_router.py`
- Create: `tests/test_query_router.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_query_router.py
from devrag.retrieve.query_router import QueryRouter


def test_code_query():
    router = QueryRouter()
    collections = router.route("how does the auth middleware work")
    assert collections == ["code_chunks"]


def test_pr_query_why():
    router = QueryRouter()
    collections = router.route("why did we switch from JWT to PKCE")
    assert "pr_diffs" in collections
    assert "pr_discussions" in collections


def test_pr_query_change():
    router = QueryRouter()
    collections = router.route("when did we change the database schema")
    assert "pr_diffs" in collections


def test_pr_query_migrate():
    router = QueryRouter()
    collections = router.route("why did we migrate to Redis")
    assert "pr_diffs" in collections
    assert "pr_discussions" in collections


def test_usage_query():
    router = QueryRouter()
    collections = router.route("where is refreshToken used")
    assert "code_chunks" in collections
    assert "pr_diffs" in collections


def test_ambiguous_query():
    router = QueryRouter()
    collections = router.route("tell me about authentication")
    assert "code_chunks" in collections
    assert "pr_diffs" in collections
    assert "pr_discussions" in collections


def test_code_only_scope():
    router = QueryRouter()
    collections = router.route("how does auth work", scope="code")
    assert collections == ["code_chunks"]


def test_prs_only_scope():
    router = QueryRouter()
    collections = router.route("how does auth work", scope="prs")
    assert set(collections) == {"pr_diffs", "pr_discussions"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_query_router.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write query router implementation**

```python
# devrag/retrieve/query_router.py
from __future__ import annotations

import re

ALL_COLLECTIONS = ["code_chunks", "pr_diffs", "pr_discussions"]
CODE_COLLECTIONS = ["code_chunks"]
PR_COLLECTIONS = ["pr_diffs", "pr_discussions"]

# Patterns that indicate PR/change-history intent
_PR_PATTERNS = [
    r"\bwhy\s+did\s+we\b",
    r"\bwhy\s+was\b",
    r"\bwhy\s+were\b",
    r"\bwhen\s+did\s+we\b",
    r"\bwho\s+changed\b",
    r"\bwho\s+added\b",
    r"\bwho\s+removed\b",
    r"\bswitch(?:ed)?\s+(?:from|to)\b",
    r"\bmigrat(?:e|ed|ion)\b",
    r"\bchange(?:d|s)?\s+(?:the|to|from)\b",
    r"\bremov(?:e|ed)\b.*\bwhy\b",
    r"\bwhy\b.*\bremov(?:e|ed)\b",
    r"\bintroduc(?:e|ed)\b",
    r"\brevert(?:ed)?\b",
    r"\bdeprecated?\b",
]

# Patterns that indicate code-usage/structure intent
_CODE_PATTERNS = [
    r"\bhow\s+does\b",
    r"\bhow\s+do\b",
    r"\bhow\s+is\b",
    r"\bwhat\s+does\b",
    r"\bimplement(?:s|ed|ation)?\b",
    r"\bdefin(?:e|ed|ition)\b",
]

# Patterns that indicate "where is X used" → code + PR diffs
_USAGE_PATTERNS = [
    r"\bwhere\s+is\b",
    r"\bwhere\s+are\b",
    r"\bwho\s+uses\b",
    r"\busage\s+of\b",
    r"\bcall(?:s|ed)\s+(?:from|by|in)\b",
]


class QueryRouter:
    """Routes queries to appropriate collections based on keyword intent classification."""

    def route(self, query: str, scope: str = "all") -> list[str]:
        """Classify query intent and return target collections.

        Args:
            query: The search query text.
            scope: Override scope. "all" uses intent classification,
                   "code" forces code_chunks only, "prs" forces PR collections only.
        """
        if scope == "code":
            return CODE_COLLECTIONS
        if scope == "prs":
            return PR_COLLECTIONS

        q = query.lower()

        # Check for PR/change-history patterns
        for pattern in _PR_PATTERNS:
            if re.search(pattern, q):
                return PR_COLLECTIONS

        # Check for usage patterns → code + diffs
        for pattern in _USAGE_PATTERNS:
            if re.search(pattern, q):
                return ["code_chunks", "pr_diffs"]

        # Check for code structure patterns
        for pattern in _CODE_PATTERNS:
            if re.search(pattern, q):
                return CODE_COLLECTIONS

        # Ambiguous → search everything
        return ALL_COLLECTIONS
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_query_router.py -v`
Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add devrag/retrieve/query_router.py tests/test_query_router.py
git commit -m "feat: keyword-based query router for intent classification"
```

---

## Task 6: Multi-Collection Hybrid Search

**Files:**
- Modify: `devrag/retrieve/hybrid_search.py`
- Modify: `tests/test_hybrid_search.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_hybrid_search.py`:

```python
def test_hybrid_search_multiple_collections():
    mock_store = MagicMock()

    def mock_query(collection, query_embedding, n_results):
        if collection == "code_chunks":
            return QueryResult(
                ids=["code_1"], documents=["def auth(): pass"],
                metadatas=[{"file_path": "a.py", "source_type": "code"}], distances=[0.1],
            )
        elif collection == "pr_diffs":
            return QueryResult(
                ids=["pr_1"], documents=["diff: added auth"],
                metadatas=[{"pr_number": 1, "source_type": "pr"}], distances=[0.2],
            )
        return QueryResult(ids=[], documents=[], metadatas=[], distances=[])

    mock_store.query = MagicMock(side_effect=mock_query)
    mock_meta = MagicMock()
    mock_meta.search_fts.return_value = []
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 768

    search = HybridSearch(mock_store, mock_meta, mock_embedder)
    results = search.search("auth", top_k=10, collections=["code_chunks", "pr_diffs"])

    result_ids = [r.chunk_id for r in results]
    assert "code_1" in result_ids
    assert "pr_1" in result_ids


def test_hybrid_search_defaults_to_code_chunks():
    """When no collections specified, defaults to code_chunks for backward compat."""
    mock_store = MagicMock()
    mock_store.query.return_value = QueryResult(
        ids=["c1"], documents=["text"], metadatas=[{}], distances=[0.1],
    )
    mock_meta = MagicMock()
    mock_meta.search_fts.return_value = []
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 768

    search = HybridSearch(mock_store, mock_meta, mock_embedder)
    results = search.search("query", top_k=5)
    # Should have called query with "code_chunks"
    mock_store.query.assert_called_once()
    call_args = mock_store.query.call_args
    assert call_args.kwargs.get("collection") == "code_chunks" or call_args[1].get("collection") == "code_chunks"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hybrid_search.py::test_hybrid_search_multiple_collections -v`
Expected: FAIL — `TypeError` (unexpected `collections` kwarg)

- [ ] **Step 3: Update HybridSearch to support multiple collections**

Replace the `HybridSearch` class in `devrag/retrieve/hybrid_search.py`:

```python
class HybridSearch:
    """Combines vector similarity search with BM25 keyword search via RRF."""

    def __init__(self, vector_store, metadata_db, embedder, collection: str = "code_chunks") -> None:
        self.vector_store = vector_store
        self.metadata_db = metadata_db
        self.embedder = embedder
        self.collection = collection  # default collection for backward compat

    def search(self, query: str, top_k: int = 20, collections: list[str] | None = None) -> list[SearchResult]:
        """Run hybrid search across one or more collections.

        Args:
            query: Search query text.
            top_k: Number of candidates per collection before RRF.
            collections: Collections to search. Defaults to [self.collection].
        """
        if collections is None:
            collections = [self.collection]

        query_embedding = self.embedder.embed_query(query)

        # Collect results from all collections
        doc_lookup: dict[str, tuple[str, dict]] = {}
        all_vector_ranked: list[str] = []

        for coll in collections:
            vector_results = self.vector_store.query(
                collection=coll, query_embedding=query_embedding, n_results=top_k,
            )
            for i, doc_id in enumerate(vector_results.ids):
                doc_lookup[doc_id] = (vector_results.documents[i], vector_results.metadatas[i])
            all_vector_ranked.extend(vector_results.ids)

        # BM25 search (FTS5 spans all collections since it's one table)
        bm25_results = self.metadata_db.search_fts(query, limit=top_k)
        bm25_ranked = [chunk_id for chunk_id, _score in bm25_results]

        # RRF fusion
        fused_ids = reciprocal_rank_fusion([all_vector_ranked, bm25_ranked], k=60)

        rrf_scores: dict[str, float] = {}
        for rank, doc_id in enumerate(fused_ids):
            rrf_scores[doc_id] = 1.0 / (60 + rank + 1)

        results: list[SearchResult] = []
        for doc_id in fused_ids[:top_k]:
            if doc_id in doc_lookup:
                text, metadata = doc_lookup[doc_id]
            else:
                continue
            results.append(SearchResult(chunk_id=doc_id, text=text, score=rrf_scores[doc_id], metadata=metadata))

        return results
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `uv run pytest tests/test_hybrid_search.py -v`
Expected: All 7 tests PASS (5 existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add devrag/retrieve/hybrid_search.py tests/test_hybrid_search.py
git commit -m "feat: multi-collection hybrid search support"
```

---

## Task 7: PR Result Formatting

**Files:**
- Modify: `devrag/utils/formatters.py`
- Modify: `tests/test_formatters.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_formatters.py`:

```python
def test_format_search_results_with_pr():
    results = [
        SearchResult(
            chunk_id="pr1",
            text="@@ -1,3 +1,5 @@\n+def new_auth():\n+    pass",
            score=0.9,
            metadata={
                "pr_number": 42,
                "pr_title": "Add new auth flow",
                "chunk_type": "diff",
                "file_path": "src/auth.py",
                "pr_author": "alice",
            },
        ),
        SearchResult(
            chunk_id="pr2",
            text="Consider using bcrypt here",
            score=0.8,
            metadata={
                "pr_number": 42,
                "pr_title": "Add new auth flow",
                "chunk_type": "review_comment",
                "reviewer": "bob",
                "file_path": "src/auth.py",
            },
        ),
    ]
    output = format_search_results(results)
    assert "PR #42" in output
    assert "Add new auth flow" in output
    assert "alice" in output or "auth.py" in output
    assert "bob" in output or "bcrypt" in output


def test_format_pr_sync_stats():
    from devrag.types import PRSyncStats
    stats = PRSyncStats(prs_fetched=50, prs_indexed=45, prs_skipped=5, chunks_created=200)
    output = format_pr_sync_stats(stats)
    assert "50" in output
    assert "45" in output
    assert "200" in output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_formatters.py::test_format_search_results_with_pr -v`
Expected: FAIL — assertion errors (current formatter doesn't handle PR metadata)

- [ ] **Step 3: Update formatters**

Replace `format_search_results` in `devrag/utils/formatters.py` and add `format_pr_sync_stats`:

```python
from devrag.types import IndexStats, PRSyncStats, SearchResult


def format_search_results(results: list[SearchResult]) -> str:
    if not results:
        return "No results found."
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        chunk_type = r.metadata.get("chunk_type", "")

        if chunk_type in ("diff", "description", "review_comment"):
            # PR result
            pr_num = r.metadata.get("pr_number", "?")
            pr_title = r.metadata.get("pr_title", "")
            file_path = r.metadata.get("file_path", "")

            if chunk_type == "review_comment":
                reviewer = r.metadata.get("reviewer", "")
                lines.append(f"### {i}. [PR #{pr_num}] Review comment by {reviewer} on {file_path}")
            elif chunk_type == "description":
                pr_author = r.metadata.get("pr_author", "")
                lines.append(f"### {i}. [PR #{pr_num}] {pr_title} (by {pr_author})")
            else:
                lines.append(f"### {i}. [PR #{pr_num}] {pr_title} — {file_path}")

            lines.append("```diff")
            text_lines = r.text.strip().split("\n")
            preview = "\n".join(text_lines[:10])
            if len(text_lines) > 10:
                preview += "\n# ... (truncated)"
            lines.append(preview)
            lines.append("```")
            lines.append("")
        else:
            # Code result (existing behavior)
            file_path = r.metadata.get("file_path", "unknown")
            line_range = r.metadata.get("line_range", [])
            entity_name = r.metadata.get("entity_name", "")
            language = r.metadata.get("language", "")
            location = file_path
            if line_range:
                location += f":{line_range[0]}-{line_range[1]}"
            lines.append(f"### {i}. {entity_name} ({location})")
            lines.append(f"```{language}")
            text_lines = r.text.strip().split("\n")
            preview = "\n".join(text_lines[:10])
            if len(text_lines) > 10:
                preview += "\n# ... (truncated)"
            lines.append(preview)
            lines.append("```")
            lines.append("")

    return "\n".join(lines)


def format_pr_sync_stats(stats: PRSyncStats) -> str:
    parts = [
        f"Fetched {stats.prs_fetched} PRs",
        f"Indexed {stats.prs_indexed} PRs ({stats.chunks_created} chunks)",
    ]
    if stats.prs_skipped:
        parts.append(f"Skipped {stats.prs_skipped} unchanged PRs")
    return ". ".join(parts) + "."
```

Update the import in `tests/test_formatters.py`:

```python
from devrag.utils.formatters import format_search_results, format_index_stats, format_pr_sync_stats
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `uv run pytest tests/test_formatters.py -v`
Expected: All 5 tests PASS (3 existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add devrag/utils/formatters.py tests/test_formatters.py
git commit -m "feat: PR-aware result formatting and sync stats"
```

---

## Task 8: MCP Server Updates

**Files:**
- Modify: `devrag/mcp_server.py`

- [ ] **Step 1: Update MCP server with sync_prs tool and scope on search**

Add imports at top of `devrag/mcp_server.py`:

```python
import os
from devrag.ingest.pr_indexer import PRIndexer
from devrag.retrieve.query_router import QueryRouter
from devrag.utils.github import GitHubClient
from devrag.utils.formatters import format_pr_sync_stats
```

Add `sync_prs` tool and update `search` tool:

```python
@mcp.tool
def search(query: str, scope: str = "all", top_k: int = 5) -> str:
    """Search code, PRs, and docs using hybrid retrieval.

    Args:
        query: The search query.
        scope: What to search. "all" auto-routes by intent,
               "code" searches code only, "prs" searches PRs only.
        top_k: Number of results to return.
    """
    config = _get_config()
    router = QueryRouter()
    collections = router.route(query, scope=scope)

    hybrid = HybridSearch(
        vector_store=_get_vector_store(),
        metadata_db=_get_metadata_db(),
        embedder=_get_embedder(),
    )
    candidates = hybrid.search(query, top_k=config.retrieval.top_k, collections=collections)

    if config.retrieval.rerank and candidates:
        reranker = _get_reranker()
        results = reranker.rerank(query, candidates, top_k=top_k)
    else:
        results = candidates[:top_k]

    return format_search_results(results)


@mcp.tool
def sync_prs(repo: str, since_days: int = 90) -> str:
    """Sync GitHub PRs for a repository.

    Fetches PR diffs, descriptions, and review comments, then indexes
    them for search. Uses cursor-based sync to avoid re-fetching.

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
    )

    stats = indexer.sync(repo, since_days=since_days)
    return format_pr_sync_stats(stats)
```

Update `status` tool to include PR counts:

```python
@mcp.tool
def status() -> str:
    """Show indexing status: number of files, chunks, and PRs indexed."""
    store = _get_vector_store()
    meta = _get_metadata_db()
    chunk_count = store.count("code_chunks")
    pr_diff_count = store.count("pr_diffs")
    pr_disc_count = store.count("pr_discussions")
    indexed_files = meta.get_all_indexed_files()
    lines = [
        f"Indexed files: {len(indexed_files)}",
        f"Code chunks: {chunk_count}",
        f"PR diff chunks: {pr_diff_count}",
        f"PR discussion chunks: {pr_disc_count}",
    ]
    return "\n".join(lines)
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `uv run python -c "from devrag.mcp_server import mcp; print(f'Server: {mcp.name}')"`
Expected: `Server: DevRAG`

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add devrag/mcp_server.py
git commit -m "feat: MCP server with sync_prs tool, scope-aware search, and query routing"
```

---

## Spec Coverage Checklist

| Spec Section | Task(s) | Status |
|---|---|---|
| 2.1 PR Indexer (diff, description, comment chunks) | Task 4 | Covered |
| 2.1 Metadata per PR chunk | Task 4 | Covered |
| 2.1 Sync strategy (cursor, backfill) | Tasks 2, 4 | Covered |
| 2.1 New collections (pr_diffs, pr_discussions) | Task 4 | Covered |
| 2.2 GitHub API client (rate-limit, pagination, diff parsing) | Task 3 | Covered |
| 2.3 Query Router (keyword-based, intent classification) | Task 5 | Covered |
| 2.4 Result Formatting (PR results) | Task 7 | Covered |
| 2.5 MCP: sync_prs tool | Task 8 | Covered |
| 2.5 MCP: scope parameter on search | Task 8 | Covered |
| 2.5 MCP: source type indicators | Task 7 | Covered |
| Config: PrsConfig | Task 1 | Covered |
| Types: PRSyncStats | Task 1 | Covered |
| MetadataDB: PR sync cursors | Task 2 | Covered |
| HybridSearch: multi-collection | Task 6 | Covered |
