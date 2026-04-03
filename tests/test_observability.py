import pytest
from devrag.stores.metadata_db import MetadataDB


def test_log_query_metric(tmp_dir):
    db = MetadataDB(str(tmp_dir / "meta.db"))
    db.log_query_metric(
        query="how does auth work", collections=["code_chunks"],
        vector_ms=15.2, bm25_ms=3.1, rerank_ms=45.0, total_ms=63.3,
        result_count=5, classification="code",
    )
    metrics = db.get_query_metrics(limit=10)
    assert len(metrics) == 1
    assert metrics[0]["query"] == "how does auth work"
    assert metrics[0]["total_ms"] == 63.3
    assert metrics[0]["result_count"] == 5


def test_get_query_stats(tmp_dir):
    db = MetadataDB(str(tmp_dir / "meta.db"))
    db.log_query_metric("q1", ["code_chunks"], 10, 2, 40, 52, 5, "code")
    db.log_query_metric("q2", ["pr_diffs"], 12, 3, 42, 57, 3, "pr")
    db.log_query_metric("q3", ["code_chunks"], 8, 1, 38, 47, 4, "code")
    stats = db.get_query_stats()
    assert stats["total_queries"] == 3
    assert stats["avg_total_ms"] == pytest.approx(52.0, abs=0.1)
    assert stats["avg_result_count"] == pytest.approx(4.0, abs=0.1)
