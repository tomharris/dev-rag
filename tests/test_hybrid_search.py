from unittest.mock import MagicMock

from qdrant_client.models import SparseVector

from devrag.retrieve.hybrid_search import HybridSearch, deduplicate_results
from devrag.types import QueryResult, SearchResult


def _mock_sparse_encoder():
    enc = MagicMock()
    enc.encode_query.return_value = SparseVector(indices=[1, 2], values=[0.5, 0.3])
    return enc


def test_hybrid_search_calls_hybrid_query():
    mock_store = MagicMock()
    mock_store.hybrid_query.return_value = QueryResult(
        ids=["chunk_1", "chunk_2"],
        documents=["def auth(): pass", "class User:"],
        metadatas=[{"file_path": "a.py"}, {"file_path": "b.py"}],
        distances=[0.9, 0.8],
    )
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 768
    sparse_enc = _mock_sparse_encoder()

    search = HybridSearch(mock_store, mock_embedder, sparse_enc, "code_chunks")
    results = search.search("authentication", top_k=20)

    result_ids = [r.chunk_id for r in results]
    assert result_ids == ["chunk_1", "chunk_2"]
    mock_embedder.embed_query.assert_called_once_with("authentication")
    sparse_enc.encode_query.assert_called_once_with("authentication")
    mock_store.hybrid_query.assert_called_once()


def test_hybrid_search_empty_results():
    mock_store = MagicMock()
    mock_store.hybrid_query.return_value = QueryResult(ids=[], documents=[], metadatas=[], distances=[])
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 768

    search = HybridSearch(mock_store, mock_embedder, _mock_sparse_encoder(), "code_chunks")
    results = search.search("query", top_k=5)
    assert results == []


def test_hybrid_search_multiple_collections_merged_by_score():
    mock_store = MagicMock()
    def mock_hybrid_query(collection, dense_embedding, sparse_embedding, n_results, where=None):
        if collection == "code_chunks":
            return QueryResult(ids=["code_1"], documents=["def auth(): pass"],
                metadatas=[{"file_path": "a.py"}], distances=[0.7])
        elif collection == "pr_diffs":
            return QueryResult(ids=["pr_1"], documents=["diff: added auth"],
                metadatas=[{"pr_number": 1}], distances=[0.9])
        return QueryResult(ids=[], documents=[], metadatas=[], distances=[])
    mock_store.hybrid_query = MagicMock(side_effect=mock_hybrid_query)
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 768

    search = HybridSearch(mock_store, mock_embedder, _mock_sparse_encoder())
    results = search.search("auth", top_k=10, collections=["code_chunks", "pr_diffs"])

    assert [r.chunk_id for r in results] == ["pr_1", "code_1"]


def test_hybrid_search_defaults_to_code_chunks():
    mock_store = MagicMock()
    mock_store.hybrid_query.return_value = QueryResult(
        ids=["c1"], documents=["text"], metadatas=[{}], distances=[0.9],
    )
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 768
    search = HybridSearch(mock_store, mock_embedder, _mock_sparse_encoder())
    search.search("query", top_k=5)

    call_kwargs = mock_store.hybrid_query.call_args
    assert call_kwargs.kwargs["collection"] == "code_chunks"


def test_hybrid_search_propagates_filters():
    mock_store = MagicMock()
    mock_store.hybrid_query.return_value = QueryResult(ids=[], documents=[], metadatas=[], distances=[])
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 768

    search = HybridSearch(mock_store, mock_embedder, _mock_sparse_encoder())
    search.search("query", top_k=5, where={"repo": "my-repo"})

    call_kwargs = mock_store.hybrid_query.call_args
    assert call_kwargs.kwargs["where"] == {"repo": "my-repo"}


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
    assert deduped == results
