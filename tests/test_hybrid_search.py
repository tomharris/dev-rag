import pytest
from unittest.mock import MagicMock

from devrag.retrieve.hybrid_search import HybridSearch, reciprocal_rank_fusion, deduplicate_results
from devrag.types import QueryResult, SearchResult


def test_rrf_merges_two_rankings():
    list1 = ["a", "b", "c"]
    list2 = ["d", "a", "b"]
    merged = reciprocal_rank_fusion([list1, list2], k=60)
    assert merged[0] == "a"
    assert set(merged) == {"a", "b", "c", "d"}


def test_rrf_single_list():
    merged = reciprocal_rank_fusion([["x", "y", "z"]], k=60)
    assert merged == ["x", "y", "z"]


def test_rrf_empty():
    merged = reciprocal_rank_fusion([], k=60)
    assert merged == []


def test_hybrid_search_combines_vector_and_bm25():
    mock_store = MagicMock()
    mock_store.query.return_value = QueryResult(
        ids=["chunk_1", "chunk_2", "chunk_3"],
        documents=["def auth(): pass", "class User:", "import os"],
        metadatas=[{"file_path": "a.py"}, {"file_path": "b.py"}, {"file_path": "c.py"}],
        distances=[0.1, 0.3, 0.5],
    )
    # chunk_4 is a BM25-only result — should now be resolved via get_by_ids
    mock_store.get_by_ids.return_value = QueryResult(
        ids=["chunk_4"], documents=["def login(): pass"],
        metadatas=[{"file_path": "d.py"}], distances=[],
    )
    mock_meta = MagicMock()
    mock_meta.search_fts_scoped.return_value = [
        ("chunk_2", -5.0), ("chunk_4", -3.0), ("chunk_1", -1.0),
    ]
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 768

    search = HybridSearch(mock_store, mock_meta, mock_embedder, "code_chunks")
    results = search.search("authentication", top_k=20)

    result_ids = [r.chunk_id for r in results]
    assert "chunk_1" in result_ids
    assert "chunk_2" in result_ids
    # BM25-only result should now be included (was silently dropped before)
    assert "chunk_4" in result_ids
    mock_embedder.embed_query.assert_called_once_with("authentication")
    mock_store.query.assert_called_once()
    mock_meta.search_fts_scoped.assert_called_once()


def test_hybrid_search_with_no_bm25_results():
    mock_store = MagicMock()
    mock_store.query.return_value = QueryResult(
        ids=["c1"], documents=["text"], metadatas=[{}], distances=[0.1],
    )
    mock_meta = MagicMock()
    mock_meta.search_fts_scoped.return_value = []
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 768

    search = HybridSearch(mock_store, mock_meta, mock_embedder, "code_chunks")
    results = search.search("query", top_k=5)
    assert len(results) == 1
    assert results[0].chunk_id == "c1"


def test_hybrid_search_multiple_collections():
    mock_store = MagicMock()
    def mock_query(collection, query_embedding, n_results, where=None):
        if collection == "code_chunks":
            return QueryResult(ids=["code_1"], documents=["def auth(): pass"],
                metadatas=[{"file_path": "a.py", "source_type": "code"}], distances=[0.1])
        elif collection == "pr_diffs":
            return QueryResult(ids=["pr_1"], documents=["diff: added auth"],
                metadatas=[{"pr_number": 1, "source_type": "pr"}], distances=[0.2])
        return QueryResult(ids=[], documents=[], metadatas=[], distances=[])
    mock_store.query = MagicMock(side_effect=mock_query)
    mock_meta = MagicMock()
    mock_meta.search_fts_scoped.return_value = []
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 768
    search = HybridSearch(mock_store, mock_meta, mock_embedder)
    results = search.search("auth", top_k=10, collections=["code_chunks", "pr_diffs"])
    result_ids = [r.chunk_id for r in results]
    assert "code_1" in result_ids
    assert "pr_1" in result_ids


def test_hybrid_search_defaults_to_code_chunks():
    mock_store = MagicMock()
    mock_store.query.return_value = QueryResult(ids=["c1"], documents=["text"], metadatas=[{}], distances=[0.1])
    mock_meta = MagicMock()
    mock_meta.search_fts_scoped.return_value = []
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 768
    search = HybridSearch(mock_store, mock_meta, mock_embedder)
    results = search.search("query", top_k=5)
    mock_store.query.assert_called_once()
    call_kwargs = mock_store.query.call_args
    assert call_kwargs.kwargs.get("collection", call_kwargs[1].get("collection")) == "code_chunks"


def test_hybrid_search_custom_rrf_k():
    mock_store = MagicMock()
    mock_store.query.return_value = QueryResult(
        ids=["c1"], documents=["text"], metadatas=[{"file_path": "a.py"}], distances=[0.1],
    )
    mock_meta = MagicMock()
    mock_meta.search_fts_scoped.return_value = []
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 768
    search = HybridSearch(mock_store, mock_meta, mock_embedder, rrf_k=10)
    results = search.search("query", top_k=5)
    # With rrf_k=10, first result score should be 1/(10+0+1) = 1/11
    assert abs(results[0].score - 1.0 / 11) < 1e-9


def test_deduplicate_results_limits_per_source():
    results = [
        SearchResult(chunk_id="c1", text="a", score=0.9, metadata={"file_path": "foo.py"}),
        SearchResult(chunk_id="c2", text="b", score=0.8, metadata={"file_path": "foo.py"}),
        SearchResult(chunk_id="c3", text="c", score=0.7, metadata={"file_path": "foo.py"}),
        SearchResult(chunk_id="c4", text="d", score=0.6, metadata={"file_path": "bar.py"}),
    ]
    deduped = deduplicate_results(results, max_per_source=2)
    assert len(deduped) == 3
    assert [r.chunk_id for r in deduped] == ["c1", "c2", "c4"]


def test_deduplicate_results_groups_by_pr():
    results = [
        SearchResult(chunk_id="p1", text="a", score=0.9, metadata={"pr_number": 42, "repo": "r"}),
        SearchResult(chunk_id="p2", text="b", score=0.8, metadata={"pr_number": 42, "repo": "r"}),
        SearchResult(chunk_id="p3", text="c", score=0.7, metadata={"pr_number": 42, "repo": "r"}),
        SearchResult(chunk_id="p4", text="d", score=0.6, metadata={"pr_number": 99, "repo": "r"}),
    ]
    deduped = deduplicate_results(results, max_per_source=1)
    assert len(deduped) == 2
    assert [r.chunk_id for r in deduped] == ["p1", "p4"]


def test_deduplicate_preserves_order():
    results = [
        SearchResult(chunk_id="c1", text="a", score=0.9, metadata={"file_path": "a.py"}),
        SearchResult(chunk_id="c2", text="b", score=0.8, metadata={"file_path": "b.py"}),
        SearchResult(chunk_id="c3", text="c", score=0.7, metadata={"file_path": "c.py"}),
    ]
    deduped = deduplicate_results(results, max_per_source=2)
    assert deduped == results  # All different sources, nothing removed
