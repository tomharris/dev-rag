import pytest
from unittest.mock import MagicMock

from devrag.retrieve.hybrid_search import HybridSearch, reciprocal_rank_fusion
from devrag.types import QueryResult


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
    mock_meta = MagicMock()
    mock_meta.search_fts.return_value = [
        ("chunk_2", -5.0), ("chunk_4", -3.0), ("chunk_1", -1.0),
    ]
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 768

    search = HybridSearch(mock_store, mock_meta, mock_embedder, "code_chunks")
    results = search.search("authentication", top_k=20)

    result_ids = [r.chunk_id for r in results]
    assert "chunk_1" in result_ids
    assert "chunk_2" in result_ids
    mock_embedder.embed_query.assert_called_once_with("authentication")
    mock_store.query.assert_called_once()
    mock_meta.search_fts.assert_called_once_with("authentication", limit=20)


def test_hybrid_search_with_no_bm25_results():
    mock_store = MagicMock()
    mock_store.query.return_value = QueryResult(
        ids=["c1"], documents=["text"], metadatas=[{}], distances=[0.1],
    )
    mock_meta = MagicMock()
    mock_meta.search_fts.return_value = []
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
    mock_meta.search_fts.return_value = []
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
    mock_meta.search_fts.return_value = []
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 768
    search = HybridSearch(mock_store, mock_meta, mock_embedder)
    results = search.search("query", top_k=5)
    mock_store.query.assert_called_once()
    call_kwargs = mock_store.query.call_args
    assert call_kwargs.kwargs.get("collection", call_kwargs[1].get("collection")) == "code_chunks"
