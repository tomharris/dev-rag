from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from devrag.retrieve.expansion import expand_related, interleave_after_anchors
from devrag.retrieve.query_router import QueryRouter
from devrag.types import SearchResult


def deduplicate_results(results: list[SearchResult], max_per_source: int = 2) -> list[SearchResult]:
    """Limit results per source to prevent a single file/PR/issue dominating."""
    source_counts: dict[str, int] = defaultdict(int)
    deduped: list[SearchResult] = []
    for r in results:
        key = _source_key(r.metadata)
        if source_counts[key] < max_per_source:
            deduped.append(r)
            source_counts[key] += 1
    return deduped


def _source_key(metadata: dict) -> str:
    """Derive a grouping key from chunk metadata."""
    if "pr_number" in metadata:
        return f"pr:{metadata.get('repo', '')}:{metadata['pr_number']}"
    if "issue_number" in metadata:
        return f"issue:{metadata.get('repo', '')}:{metadata['issue_number']}"
    if "ticket_key" in metadata:
        return f"jira:{metadata['ticket_key']}"
    if "page_id" in metadata:
        return f"slite:{metadata['page_id']}"
    if "session_id" in metadata:
        return f"session:{metadata['session_id']}"
    if "file_path" in metadata:
        return f"file:{metadata.get('repo', '')}:{metadata['file_path']}"
    return f"unknown:{id(metadata)}"


def apply_repo_preference(
    results: list[SearchResult], prefer_repo: str, boost: float
) -> list[SearchResult]:
    """Re-rank so results from ``prefer_repo`` get a soft score bonus.

    Goal: when a small repo shares the index with a much larger one, results from
    the repo the user is actively working in shouldn't get drowned out — but
    cross-repo results must still appear (this is a *soft* boost, not a filter).

    The boost is *spread-relative*: ``bonus = boost * (max_score - min_score)``.
    This keeps the preference "moderate" regardless of the active score scale —
    cross-encoder logits (~±11) when reranking is on, or RRF scores (~0.01–0.03)
    when it's off — so a near-tie in-repo result gets lifted while strong cross-repo
    context is preserved. Reorders only; scores are left untouched for display.

    Contract (see tests in tests/test_hybrid_search.py):
      - No-op (return ``results`` unchanged) when ``boost == 0`` or ``prefer_repo`` is "".
      - Otherwise rank by ``score + bonus`` for in-repo results, descending (stable).
      - Never add or drop results — same set in, same set out, only reordered.
    """
    if not boost or not prefer_repo or not results:
        return results
    scores = [r.score for r in results]
    bonus = boost * ((max(scores) - min(scores)) or 1.0)
    return sorted(
        results,
        key=lambda r: r.score + (bonus if r.metadata.get("repo") == prefer_repo else 0.0),
        reverse=True,
    )


# Filters that name one specific source. Expansion crosses collections by
# design, so honouring one of these means not expanding at all: a user who asked
# for PR #12, or for one chunk_type, has said what they want.
_PINNING_FILTER_KEYS = frozenset({
    "chunk_type", "pr_number", "issue_number", "ticket_key", "page_id",
    "session_id", "channel_id",
})


def _filter_pins_a_source(where: dict | None) -> bool:
    return bool(where) and bool(_PINNING_FILTER_KEYS & set(where))


def _should_expand(config, query: str, expand: bool | None) -> bool:
    """Resolve the expansion decision: explicit override, else the config mode."""
    if expand is not None:
        return expand
    mode = str(config.retrieval.expand_related).lower()
    if mode in ("always", "true"):
        return True
    if mode == "auto":
        return QueryRouter().wants_history(query)
    return False


def _apply_slot_budget(results, max_results: int):
    """Demote expanded chunks beyond *max_results* behind every real result.

    The measured failure mode is expanded chunks taking `final_k` slots from
    results that were actually retrieved, so the budget caps how many may hold a
    position *ahead* of real results. Excess ones are moved behind them rather
    than deleted: if slots remain unfilled, related context still beats nothing.
    Order is otherwise preserved, so a promoted chunk stays under its anchor.
    """
    if max_results is None or max_results < 0:
        return results
    kept, dropped, taken = [], [], 0
    for r in results:
        if "expanded_from" in r.metadata:
            if taken >= max_results:
                dropped.append(r)
                continue
            taken += 1
        kept.append(r)
    return kept + dropped


def search_rank_dedupe(hybrid, reranker, query, collections, where, config, final_k,
                       prefer_repo="", timings=None, expand=None):
    """Run the full retrieval pipeline: hybrid search, rerank, dedupe, truncate.

    Dedupe runs on the *full ranked pool* before the final-k slice, so a query
    whose top hits share a source still returns up to final_k distinct sources.

    ``timings`` is an optional dict filled in place with millisecond stage
    timings (see ``HybridSearch.search`` for the retrieval-side keys, plus
    ``rerank_ms`` and ``total_ms``). It is an out-param rather than a second
    return value so every existing caller keeps working unchanged.

    ``expand`` decides related-chunk expansion: None follows
    ``config.retrieval.expand_related`` (``"auto"`` spends slots on history only
    for queries that ask about history), and an explicit bool overrides it —
    that is how ``search --expand`` forces it on for one query. Callers also fold
    in ``scope == "all"``: an explicit ``--scope code`` means "search code only",
    and expansion crosses collections by design, so honouring the scope means not
    expanding. See ``_filter_pins_a_source`` for the filter-side equivalent.
    """
    started = time.perf_counter()
    candidates = hybrid.search(query, top_k=config.retrieval.top_k, collections=collections,
                               where=where, timings=timings)
    # One hop over the file-path edge, before reranking, so pulled-in chunks
    # compete on merit instead of being appended as unranked extras. Skipped
    # when the caller pinned a filter that expansion would quietly widen.
    want_expand = _should_expand(config, query, expand)
    if want_expand and not _filter_pins_a_source(where):
        expand_started = time.perf_counter()
        extra = expand_related(
            hybrid.vector_store, candidates,
            top_n=config.retrieval.expand_top_n,
            per_anchor=config.retrieval.expand_per_anchor,
            max_total=config.retrieval.expand_max_total,
        )
        candidates = candidates + extra
        if timings is not None:
            timings["expand_ms"] = (time.perf_counter() - expand_started) * 1000
            timings["expanded_count"] = len(extra)
    rerank_started = time.perf_counter()
    if reranker and candidates:
        ranked = reranker.rerank(query, candidates, top_k=len(candidates))
    else:
        ranked = candidates
    if timings is not None:
        timings["rerank_ms"] = (time.perf_counter() - rerank_started) * 1000
    ranked = apply_repo_preference(ranked, prefer_repo, config.retrieval.repo_boost)
    # Re-seat expanded chunks under their anchors *after* ranking: the reranker
    # scores them on merit, but context must not outrank what pulled it in.
    ranked = interleave_after_anchors(
        [r for r in ranked if "expanded_from" not in r.metadata],
        [r for r in ranked if "expanded_from" in r.metadata],
    )
    deduped = deduplicate_results(ranked, max_per_source=config.retrieval.max_per_source)
    # Budget applied before the final slice, so a capped-out expanded chunk frees
    # its slot for a real result rather than just vanishing.
    if any("expanded_from" in r.metadata for r in deduped):
        deduped = _apply_slot_budget(deduped, config.retrieval.expand_max_results)
    if timings is not None:
        timings["total_ms"] = (time.perf_counter() - started) * 1000
    return deduped[:final_k]


class HybridSearch:
    def __init__(self, vector_store, embedder, sparse_encoder, collection: str = "code_chunks") -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder
        self.collection = collection

    def search(self, query: str, top_k: int = 20, collections: list[str] | None = None,
               where: dict | None = None, timings: dict | None = None) -> list[SearchResult]:
        """Hybrid-search ``collections``, returning the top_k fused results.

        ``timings``, if given, is filled in place with:
          - ``embed_ms``  — dense query embedding (Ollama round trip)
          - ``sparse_ms`` — BM25 query encoding (local FastEmbed)
          - ``vector_ms`` — the Qdrant ``query_points`` calls

        There is no separable "BM25 search" time: Qdrant prefetches both legs and
        fuses them server-side inside a single call, so ``vector_ms`` covers both.
        """
        if collections is None:
            collections = [self.collection]
        embed_started = time.perf_counter()
        dense = self.embedder.embed_query(query)
        sparse_started = time.perf_counter()
        sparse = self.sparse_encoder.encode_query(query)
        query_started = time.perf_counter()
        if timings is not None:
            timings["embed_ms"] = (sparse_started - embed_started) * 1000
            timings["sparse_ms"] = (query_started - sparse_started) * 1000

        def _query_one(coll: str):
            return self.vector_store.hybrid_query(
                collection=coll,
                dense_embedding=dense,
                sparse_embedding=sparse,
                n_results=top_k,
                where=where,
            )

        if len(collections) == 1:
            per_collection = [_query_one(collections[0])]
        else:
            with ThreadPoolExecutor(max_workers=min(len(collections), 8)) as pool:
                per_collection = list(pool.map(_query_one, collections))

        if timings is not None:
            timings["vector_ms"] = (time.perf_counter() - query_started) * 1000

        all_results: list[SearchResult] = []
        for hits in per_collection:
            for i, doc_id in enumerate(hits.ids):
                all_results.append(SearchResult(
                    chunk_id=doc_id,
                    text=hits.documents[i],
                    score=hits.distances[i] if hits.distances else 0.0,
                    metadata=hits.metadatas[i],
                ))

        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:top_k]
