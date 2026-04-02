from devrag.utils.formatters import format_search_results, format_index_stats
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
