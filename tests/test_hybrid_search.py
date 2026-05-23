from unittest.mock import MagicMock

from qdrant_client.models import SparseVector

from devrag.config import DevragConfig
from devrag.retrieve.hybrid_search import (
    HybridSearch,
    apply_repo_preference,
    deduplicate_results,
    search_rank_dedupe,
)
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


def test_hybrid_search_parallel_three_collections():
    mock_store = MagicMock()
    seen: list[str] = []

    def mock_hybrid_query(collection, dense_embedding, sparse_embedding, n_results, where=None):
        seen.append(collection)
        return QueryResult(ids=[f"{collection}_1"], documents=[f"doc-{collection}"],
            metadatas=[{"file_path": f"{collection}.py"}], distances=[0.5])
    mock_store.hybrid_query = MagicMock(side_effect=mock_hybrid_query)
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 768

    search = HybridSearch(mock_store, mock_embedder, _mock_sparse_encoder())
    results = search.search("x", top_k=10, collections=["code_chunks", "pr_diffs", "documents"])

    assert {r.chunk_id for r in results} == {"code_chunks_1", "pr_diffs_1", "documents_1"}
    assert set(seen) == {"code_chunks", "pr_diffs", "documents"}


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


def _config(top_k=20, max_per_source=1):
    config = DevragConfig()
    config.retrieval.top_k = top_k
    config.retrieval.max_per_source = max_per_source
    return config


def test_search_rank_dedupe_dedupes_full_pool_before_truncating():
    """final_k distinct-source results survive even when the top reranked hits share a source."""
    hybrid = MagicMock()
    hybrid.search.return_value = [
        SearchResult(chunk_id="c1", text="a", score=0.9, metadata={"file_path": "foo.py"}),
        SearchResult(chunk_id="c2", text="b", score=0.8, metadata={"file_path": "foo.py"}),
        SearchResult(chunk_id="c3", text="c", score=0.7, metadata={"file_path": "bar.py"}),
    ]
    reranker = MagicMock()
    # Identity reranker: returns candidates in given order, honoring top_k.
    reranker.rerank.side_effect = lambda q, cands, top_k: cands[:top_k]

    results = search_rank_dedupe(
        hybrid, reranker, "q", ["code_chunks"], None, _config(max_per_source=1), final_k=2
    )

    # Without the fix, rerank-to-2 then dedup yields only ["c1"]. With it: ["c1", "c3"].
    assert [r.chunk_id for r in results] == ["c1", "c3"]
    # The reranker must score the WHOLE pool, not just final_k.
    assert reranker.rerank.call_args.kwargs["top_k"] == 3


def test_apply_repo_preference_promotes_close_in_repo_result():
    """A near-tie in-repo result is nudged above a slightly better cross-repo one.

    Scores span 0.10..0.60 (spread 0.50); boost 0.15 -> bonus 0.075, enough to lift
    "b" (0.55, app) above "a" (0.60, lib) but not the far-behind "c".
    """
    results = [
        SearchResult(chunk_id="a", text="x", score=0.60, metadata={"repo": "lib", "file_path": "a.py"}),
        SearchResult(chunk_id="b", text="y", score=0.55, metadata={"repo": "app", "file_path": "b.py"}),
        SearchResult(chunk_id="c", text="z", score=0.10, metadata={"repo": "lib", "file_path": "c.py"}),
    ]
    boosted = apply_repo_preference(results, prefer_repo="app", boost=0.15)
    assert [r.chunk_id for r in boosted] == ["b", "a", "c"]


def test_apply_repo_preference_does_not_rescue_far_behind_in_repo():
    """Moderate boost must not bury strong cross-repo context (spread-relative)."""
    results = [
        SearchResult(chunk_id="a", text="x", score=0.90, metadata={"repo": "lib", "file_path": "a.py"}),
        SearchResult(chunk_id="b", text="y", score=0.20, metadata={"repo": "app", "file_path": "b.py"}),
    ]
    boosted = apply_repo_preference(results, prefer_repo="app", boost=0.15)
    assert [r.chunk_id for r in boosted] == ["a", "b"]


def test_apply_repo_preference_is_noop_when_disabled():
    results = [
        SearchResult(chunk_id="a", text="x", score=0.50, metadata={"repo": "lib", "file_path": "x.py"}),
        SearchResult(chunk_id="b", text="y", score=0.45, metadata={"repo": "app", "file_path": "y.py"}),
    ]
    assert apply_repo_preference(results, prefer_repo="app", boost=0.0) == results
    assert apply_repo_preference(results, prefer_repo="", boost=0.15) == results


def test_apply_repo_preference_keeps_cross_repo_results_and_scores():
    """Reorder only: same set out, and displayed scores stay un-inflated."""
    results = [
        SearchResult(chunk_id="a", text="x", score=0.60, metadata={"repo": "lib", "file_path": "a.py"}),
        SearchResult(chunk_id="b", text="y", score=0.55, metadata={"repo": "app", "file_path": "b.py"}),
    ]
    boosted = apply_repo_preference(results, prefer_repo="app", boost=0.15)
    assert {r.chunk_id for r in boosted} == {"a", "b"}
    assert {r.chunk_id: r.score for r in boosted} == {"a": 0.60, "b": 0.55}


def test_apply_repo_preference_empty_results():
    assert apply_repo_preference([], prefer_repo="app", boost=0.15) == []


def test_search_rank_dedupe_without_reranker_dedupes_then_truncates():
    hybrid = MagicMock()
    hybrid.search.return_value = [
        SearchResult(chunk_id="c1", text="a", score=0.9, metadata={"file_path": "foo.py"}),
        SearchResult(chunk_id="c2", text="b", score=0.8, metadata={"file_path": "foo.py"}),
        SearchResult(chunk_id="c3", text="c", score=0.7, metadata={"file_path": "bar.py"}),
    ]
    results = search_rank_dedupe(
        hybrid, None, "q", ["code_chunks"], None, _config(max_per_source=1), final_k=2
    )
    assert [r.chunk_id for r in results] == ["c1", "c3"]
