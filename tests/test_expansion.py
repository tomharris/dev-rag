from unittest.mock import MagicMock

from devrag.retrieve.expansion import expand_related, interleave_after_anchors
from devrag.types import QueryResult, SearchResult


def _code(chunk_id="c1", file_path="src/auth.py", repo="app", score=0.9):
    return SearchResult(chunk_id=chunk_id, text="def auth(): pass", score=score,
                        metadata={"repo": repo, "file_path": file_path, "entity_name": "auth"})


def _pr_hit(chunk_id="p1", file_path="src/auth.py", repo="app", pr_number=12):
    return QueryResult(
        ids=[chunk_id], documents=["@@ -1 +1 @@"],
        metadatas=[{"repo": repo, "file_path": file_path, "pr_number": pr_number,
                    "chunk_type": "diff"}],
        distances=[0.0],
    )


def _empty():
    return QueryResult(ids=[], documents=[], metadatas=[], distances=[])


def test_code_hit_pulls_in_the_prs_that_touched_the_file():
    store = MagicMock()
    store.fetch_by_filter.side_effect = lambda coll, where, limit: (
        _pr_hit() if coll == "pr_diffs" else _empty()
    )
    extra = expand_related(store, [_code()])
    assert [e.chunk_id for e in extra] == ["p1"]
    assert extra[0].metadata["expanded_from"] == "c1"
    # The join key is the pair, not the path alone — another repo's src/auth.py
    # is a different file.
    coll, where = store.fetch_by_filter.call_args_list[0].args
    assert where == {"repo": "app", "file_path": "src/auth.py"}


def test_pr_hit_pulls_in_the_current_code():
    store = MagicMock()
    store.fetch_by_filter.side_effect = lambda coll, where, limit: (
        QueryResult(ids=["c9"], documents=["def auth(): pass"],
                    metadatas=[{"repo": "app", "file_path": "src/auth.py"}], distances=[0.0])
        if coll == "code_chunks" else _empty()
    )
    pr = SearchResult(chunk_id="p1", text="diff", score=0.8,
                      metadata={"repo": "app", "file_path": "src/auth.py", "pr_number": 12})
    extra = expand_related(store, [pr])
    assert [e.chunk_id for e in extra] == ["c9"]
    assert store.fetch_by_filter.call_args_list[0].args[0] == "code_chunks"


def test_expanded_candidate_inherits_its_anchor_score():
    """Expanded chunks have no query score of their own; without a reranker the
    anchor's score is the only defensible placement."""
    store = MagicMock()
    store.fetch_by_filter.side_effect = lambda coll, where, limit: (
        _pr_hit() if coll == "pr_diffs" else _empty()
    )
    extra = expand_related(store, [_code(score=0.77)])
    assert extra[0].score == 0.77


def test_never_returns_a_chunk_already_in_the_results():
    store = MagicMock()
    store.fetch_by_filter.side_effect = lambda coll, where, limit: (
        _pr_hit(chunk_id="already") if coll == "pr_diffs" else _empty()
    )
    existing = SearchResult(chunk_id="already", text="x", score=0.5,
                            metadata={"repo": "app", "file_path": "src/auth.py", "pr_number": 12})
    assert expand_related(store, [_code(), existing]) == []


def test_chunks_without_a_file_edge_are_not_expanded():
    """Issues, Jira tickets, Slack threads and repo-less docs have no file edge."""
    store = MagicMock()
    results = [
        SearchResult(chunk_id="i1", text="x", score=0.9,
                     metadata={"repo": "app", "file_path": "src/a.py", "issue_number": 3}),
        SearchResult(chunk_id="s1", text="x", score=0.8, metadata={"channel_id": "C1"}),
        SearchResult(chunk_id="d1", text="x", score=0.7, metadata={"file_path": "/abs/notes.md"}),
    ]
    assert expand_related(store, results) == []
    store.fetch_by_filter.assert_not_called()


def test_respects_top_n_per_anchor_and_max_total():
    store = MagicMock()
    store.fetch_by_filter.side_effect = lambda coll, where, limit: (
        QueryResult(ids=[f"{where['file_path']}-{coll}-{i}" for i in range(limit)],
                    documents=["d"] * limit,
                    metadatas=[{"repo": "app", "file_path": where["file_path"],
                                "pr_number": 1}] * limit,
                    distances=[0.0] * limit)
        if coll == "pr_diffs" else _empty()
    )
    results = [_code(chunk_id=f"c{i}", file_path=f"src/f{i}.py") for i in range(10)]
    extra = expand_related(store, results, top_n=3, per_anchor=2, max_total=5)
    assert len(extra) == 5
    assert len({e.metadata["expanded_from"] for e in extra}) == 3


def test_store_failure_is_skipped_not_raised():
    store = MagicMock()
    store.fetch_by_filter.side_effect = RuntimeError("collection missing")
    assert expand_related(store, [_code()]) == []


def test_disabled_by_zero_limits():
    store = MagicMock()
    assert expand_related(store, [_code()], top_n=0) == []
    assert expand_related(store, [_code()], per_anchor=0) == []
    assert expand_related(store, [_code()], max_total=0) == []
    store.fetch_by_filter.assert_not_called()


def test_interleave_places_each_candidate_after_its_anchor():
    a, b = _code(chunk_id="a"), _code(chunk_id="b")
    extra = SearchResult(chunk_id="x", text="d", score=a.score,
                         metadata={"expanded_from": "a"})
    merged = interleave_after_anchors([a, b], [extra])
    assert [r.chunk_id for r in merged] == ["a", "x", "b"]


def test_interleave_drops_a_candidate_whose_anchor_did_not_survive():
    """It was only ever context for that anchor."""
    a = _code(chunk_id="a")
    orphan = SearchResult(chunk_id="x", text="d", score=0.1,
                          metadata={"expanded_from": "gone"})
    assert [r.chunk_id for r in interleave_after_anchors([a], [orphan])] == ["a"]


def test_interleave_is_a_noop_without_expansion():
    results = [_code(chunk_id="a")]
    assert interleave_after_anchors(results, []) is results



