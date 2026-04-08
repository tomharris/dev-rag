from devrag.utils.formatters import format_search_results, format_index_stats, format_pr_sync_stats, format_doc_index_stats
from devrag.types import SearchResult, IndexStats


def test_format_search_results():
    results = [
        SearchResult(chunk_id="c1", text="def authenticate(user, pwd):\n    return check(user, pwd)",
            score=0.95, metadata={"file_path": "src/auth.py", "line_range": [10, 15], "entity_name": "authenticate"}),
        SearchResult(chunk_id="c2", text="class AuthMiddleware:\n    pass",
            score=0.82, metadata={"file_path": "src/middleware.py", "line_range": [1, 5], "entity_name": "AuthMiddleware"}),
    ]
    output = format_search_results(results)
    assert "src/auth.py" in output
    assert "authenticate" in output
    assert "src/middleware.py" in output
    assert "AuthMiddleware" in output


def test_format_search_results_empty():
    output = format_search_results([])
    assert "no results" in output.lower()


def test_format_index_stats():
    stats = IndexStats(files_scanned=100, files_indexed=20, files_skipped=78, files_removed=2, chunks_created=85)
    output = format_index_stats(stats)
    assert "100" in output
    assert "20" in output
    assert "78" in output
    assert "85" in output
    assert "no chunks" not in output  # hidden when zero


def test_format_index_stats_with_empty():
    stats = IndexStats(files_scanned=100, files_indexed=15, files_skipped=78, files_empty=5, chunks_created=60)
    output = format_index_stats(stats)
    assert "5 files produced no chunks" in output


def test_format_search_results_with_pr():
    results = [
        SearchResult(chunk_id="pr1", text="@@ -1,3 +1,5 @@\n+def new_auth():\n+    pass", score=0.9,
            metadata={"pr_number": 42, "pr_title": "Add new auth flow", "chunk_type": "diff",
                       "file_path": "src/auth.py", "pr_author": "alice"}),
        SearchResult(chunk_id="pr2", text="Consider using bcrypt here", score=0.8,
            metadata={"pr_number": 42, "pr_title": "Add new auth flow", "chunk_type": "review_comment",
                       "reviewer": "bob", "file_path": "src/auth.py"}),
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


def test_format_search_results_with_document():
    results = [
        SearchResult(
            chunk_id="doc1",
            text="Use Bearer tokens for all API requests.",
            score=0.9,
            metadata={
                "file_path": "docs/api.md",
                "chunk_type": "document",
                "section_path": "API Guide > Authentication",
                "entity_name": "Authentication",
            },
        ),
    ]
    output = format_search_results(results)
    assert "API Guide > Authentication" in output or "Authentication" in output
    assert "docs/api.md" in output
    assert "Bearer tokens" in output


def test_format_doc_index_stats():
    from devrag.types import DocIndexStats
    stats = DocIndexStats(files_scanned=10, files_indexed=8, chunks_created=42)
    output = format_doc_index_stats(stats)
    assert "10" in output
    assert "8" in output
    assert "42" in output
