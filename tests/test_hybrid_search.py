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


def _config(top_k=20, max_per_source=1, expand_related=False):
    config = DevragConfig()
    config.retrieval.top_k = top_k
    config.retrieval.max_per_source = max_per_source
    # Expansion ships off by default (see RetrievalConfig); the tests that
    # exercise it opt in explicitly.
    config.retrieval.expand_related = expand_related
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


def test_search_rank_dedupe_fills_timings():
    hybrid = MagicMock()
    hybrid.search.return_value = [
        SearchResult(chunk_id="c1", text="a", score=0.9, metadata={"file_path": "foo.py"}),
    ]

    def _fake_search(query, top_k, collections, where, timings=None):
        if timings is not None:
            timings.update({"embed_ms": 1.0, "sparse_ms": 2.0, "vector_ms": 3.0})
        return hybrid.search.return_value

    hybrid.search.side_effect = _fake_search
    timings: dict = {}
    search_rank_dedupe(hybrid, None, "q", ["code_chunks"], None, _config(), final_k=5,
                       timings=timings)
    assert timings["embed_ms"] == 1.0
    assert "rerank_ms" in timings and "total_ms" in timings
    assert timings["total_ms"] >= 0.0


def test_search_rank_dedupe_without_timings_is_unchanged():
    """timings is an out-param; omitting it must not alter behaviour."""
    hybrid = MagicMock()
    hybrid.search.return_value = [
        SearchResult(chunk_id="c1", text="a", score=0.9, metadata={"file_path": "foo.py"}),
    ]
    results = search_rank_dedupe(hybrid, None, "q", ["code_chunks"], None, _config(), final_k=5)
    assert [r.chunk_id for r in results] == ["c1"]
    assert hybrid.search.call_args.kwargs["timings"] is None


def test_hybrid_search_records_stage_timings():
    mock_store = MagicMock()
    mock_store.hybrid_query.return_value = QueryResult(
        ids=["c1"], documents=["x"], metadatas=[{"file_path": "a.py"}], distances=[0.9],
    )
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 768
    hs = HybridSearch(mock_store, mock_embedder, _mock_sparse_encoder())
    timings: dict = {}
    hs.search("q", top_k=5, timings=timings)
    assert set(timings) == {"embed_ms", "sparse_ms", "vector_ms"}
    assert all(v >= 0.0 for v in timings.values())


def _expandable(chunk_id, file_path, score, repo="app"):
    return SearchResult(chunk_id=chunk_id, text="code", score=score,
                        metadata={"repo": repo, "file_path": file_path})


def _store_returning_pr(chunk_id="p1"):
    store = MagicMock()
    store.fetch_by_filter.side_effect = lambda coll, where, limit: (
        QueryResult(ids=[chunk_id], documents=["diff"],
                    metadatas=[{"repo": "app", "file_path": where["file_path"],
                                "pr_number": 12, "chunk_type": "diff"}],
                    distances=[0.0])
        if coll == "pr_diffs" else QueryResult(ids=[], documents=[], metadatas=[], distances=[])
    )
    return store


def test_pipeline_expands_a_code_hit_into_the_pr_that_touched_it():
    hybrid = MagicMock()
    hybrid.vector_store = _store_returning_pr()
    hybrid.search.return_value = [_expandable("c1", "src/auth.py", 0.9)]
    results = search_rank_dedupe(hybrid, None, "q", ["code_chunks"], None, _config(expand_related=True), final_k=5)
    assert [r.chunk_id for r in results] == ["c1", "p1"]


def test_expansion_can_be_turned_off():
    hybrid = MagicMock()
    hybrid.vector_store = _store_returning_pr()
    hybrid.search.return_value = [_expandable("c1", "src/auth.py", 0.9)]
    config = _config(expand_related=False)
    results = search_rank_dedupe(hybrid, None, "q", ["code_chunks"], None, config, final_k=5)
    assert [r.chunk_id for r in results] == ["c1"]
    hybrid.vector_store.fetch_by_filter.assert_not_called()


def test_expansion_is_skipped_when_the_caller_pinned_a_source():
    """--pr-number 12 or --chunk-type diff is a statement of intent; expansion
    crosses collections and would quietly widen it."""
    hybrid = MagicMock()
    hybrid.vector_store = _store_returning_pr()
    hybrid.search.return_value = [_expandable("c1", "src/auth.py", 0.9)]
    for where in [{"pr_number": 12}, {"chunk_type": "diff"}, {"session_id": "s"}]:
        hybrid.vector_store.fetch_by_filter.reset_mock()
        search_rank_dedupe(hybrid, None, "q", ["code_chunks"], where, _config(expand_related=True), final_k=5)
        hybrid.vector_store.fetch_by_filter.assert_not_called()


def test_expansion_still_runs_for_a_repo_or_file_path_filter():
    """Those narrow *which* file, which is exactly what expansion joins on."""
    hybrid = MagicMock()
    hybrid.vector_store = _store_returning_pr()
    hybrid.search.return_value = [_expandable("c1", "src/auth.py", 0.9)]
    search_rank_dedupe(hybrid, None, "q", ["code_chunks"], {"repo": "app"}, _config(expand_related=True), final_k=5)
    hybrid.vector_store.fetch_by_filter.assert_called()


def test_expanded_candidates_are_reranked_then_reseated_under_their_anchor():
    """They enter the pool before reranking so the cross-encoder can drop them,
    but a survivor is placed under its anchor, never above it."""
    hybrid = MagicMock()
    hybrid.vector_store = _store_returning_pr()
    hybrid.search.return_value = [_expandable("c1", "src/auth.py", 0.9)]
    reranker = MagicMock()
    reranker.rerank.side_effect = lambda q, cands, top_k: sorted(
        cands, key=lambda r: 0 if r.chunk_id == "p1" else 1
    )[:top_k]
    results = search_rank_dedupe(hybrid, reranker, "q", ["code_chunks"], None, _config(expand_related=True),
                                 final_k=5)
    # The reranker put the expanded PR first; re-seating keeps context under the
    # result that pulled it in.
    assert [r.chunk_id for r in results] == ["c1", "p1"]
    assert {c.chunk_id for c in reranker.rerank.call_args.args[1]} == {"c1", "p1"}


def test_expansion_records_timings():
    hybrid = MagicMock()
    hybrid.vector_store = _store_returning_pr()
    hybrid.search.return_value = [_expandable("c1", "src/auth.py", 0.9)]
    timings: dict = {}
    search_rank_dedupe(hybrid, None, "q", ["code_chunks"], None, _config(expand_related=True), final_k=5,
                       timings=timings)
    assert timings["expanded_count"] == 1
    assert timings["expand_ms"] >= 0.0


def test_expansion_is_suppressed_by_an_explicit_scope():
    """`--scope code` means "search code only"; expansion crosses collections."""
    hybrid = MagicMock()
    hybrid.vector_store = _store_returning_pr()
    hybrid.search.return_value = [_expandable("c1", "src/auth.py", 0.9)]
    results = search_rank_dedupe(hybrid, None, "q", ["code_chunks"], None, _config(expand_related=True),
                                 final_k=5, expand=False)
    assert [r.chunk_id for r in results] == ["c1"]
    hybrid.vector_store.fetch_by_filter.assert_not_called()


def _auto_config(mode="auto", max_results=2):
    config = _config()
    config.retrieval.expand_related = mode
    config.retrieval.expand_max_results = max_results
    return config


def test_auto_mode_expands_a_history_question():
    hybrid = MagicMock()
    hybrid.vector_store = _store_returning_pr()
    hybrid.search.return_value = [_expandable("c1", "src/auth.py", 0.9)]
    results = search_rank_dedupe(hybrid, None, "why did we change auth", ["code_chunks"],
                                 None, _auto_config(), final_k=5)
    assert [r.chunk_id for r in results] == ["c1", "p1"]


def test_auto_mode_does_not_expand_a_how_question():
    """The measured trade-off: on 'how does X work', expanded chunks cost recall."""
    hybrid = MagicMock()
    hybrid.vector_store = _store_returning_pr()
    hybrid.search.return_value = [_expandable("c1", "src/auth.py", 0.9)]
    results = search_rank_dedupe(hybrid, None, "how does auth work", ["code_chunks"],
                                 None, _auto_config(), final_k=5)
    assert [r.chunk_id for r in results] == ["c1"]
    hybrid.vector_store.fetch_by_filter.assert_not_called()


def test_always_mode_expands_regardless_of_shape():
    hybrid = MagicMock()
    hybrid.vector_store = _store_returning_pr()
    hybrid.search.return_value = [_expandable("c1", "src/auth.py", 0.9)]
    results = search_rank_dedupe(hybrid, None, "how does auth work", ["code_chunks"],
                                 None, _auto_config(mode="always"), final_k=5)
    assert [r.chunk_id for r in results] == ["c1", "p1"]


def test_explicit_expand_overrides_auto_on_a_how_question():
    hybrid = MagicMock()
    hybrid.vector_store = _store_returning_pr()
    hybrid.search.return_value = [_expandable("c1", "src/auth.py", 0.9)]
    results = search_rank_dedupe(hybrid, None, "how does auth work", ["code_chunks"],
                                 None, _auto_config(), final_k=5, expand=True)
    assert [r.chunk_id for r in results] == ["c1", "p1"]


def test_slot_budget_caps_how_many_expanded_chunks_outrank_real_results():
    """Bounds the damage when the history signal misfires: the failure mode is
    always expanded chunks eating final_k slots from real results."""
    hybrid = MagicMock()
    store = MagicMock()
    store.fetch_by_filter.side_effect = lambda coll, where, limit: (
        QueryResult(ids=[f"p{i}" for i in range(limit)], documents=["diff"] * limit,
                    metadatas=[{"repo": "app", "file_path": where["file_path"],
                                "pr_number": i} for i in range(limit)],
                    distances=[0.0] * limit)
        if coll == "pr_diffs" else QueryResult(ids=[], documents=[], metadatas=[], distances=[])
    )
    hybrid.vector_store = store
    # Four real results, so the budget actually binds on the final_k=5 slice.
    hybrid.search.return_value = [
        _expandable(f"c{i}", f"src/f{i}.py", 0.9 - i / 10) for i in range(4)
    ]
    config = _auto_config(max_results=1)
    config.retrieval.max_per_source = 5
    results = search_rank_dedupe(hybrid, None, "why did auth change", ["code_chunks"],
                                 None, config, final_k=5)
    real = [r for r in results if "expanded_from" not in r.metadata]
    expanded = [r for r in results if "expanded_from" in r.metadata]
    assert len(real) == 4, "real results must not lose their slots"
    assert len(expanded) == 1


def test_slot_budget_of_zero_lets_real_results_take_every_slot():
    hybrid = MagicMock()
    hybrid.vector_store = _store_returning_pr()
    hybrid.search.return_value = [_expandable("c1", "src/auth.py", 0.9),
                                  _expandable("c2", "src/b.py", 0.8)]
    config = _auto_config(max_results=0)
    config.retrieval.max_per_source = 5
    results = search_rank_dedupe(hybrid, None, "why did auth change", ["code_chunks"],
                                 None, config, final_k=2)
    assert [r.chunk_id for r in results] == ["c1", "c2"]
