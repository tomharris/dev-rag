from unittest.mock import MagicMock

from devrag.retrieve.metrics import log_search


def test_log_search_maps_timings_to_schema_columns():
    db = MagicMock()
    log_search(db, "q", ["code_chunks"], "code",
               {"embed_ms": 10.0, "sparse_ms": 2.0, "vector_ms": 30.0,
                "rerank_ms": 50.0, "total_ms": 95.0}, result_count=5)
    kwargs = db.log_query_metric.call_args.kwargs
    # vector_ms folds in the dense embed round trip; the fused Qdrant call has no
    # separable BM25 leg, so bm25_ms is query-encoding time only.
    assert kwargs["vector_ms"] == 40.0
    assert kwargs["bm25_ms"] == 2.0
    assert kwargs["rerank_ms"] == 50.0
    assert kwargs["total_ms"] == 95.0
    assert kwargs["classification"] == "code"
    assert kwargs["result_count"] == 5


def test_log_search_tolerates_missing_timing_keys():
    db = MagicMock()
    log_search(db, "q", ["code_chunks"], "code", {}, result_count=0)
    assert db.log_query_metric.call_args.kwargs["total_ms"] == 0.0


def test_log_search_noop_without_db():
    log_search(None, "q", ["code_chunks"], "code", {}, result_count=0)


def test_log_search_swallows_db_errors():
    """Metrics are a convenience; a failed insert must not break a search."""
    db = MagicMock()
    db.log_query_metric.side_effect = RuntimeError("database is locked")
    log_search(db, "q", ["code_chunks"], "code", {}, result_count=0)
