from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

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


def search_rank_dedupe(hybrid, reranker, query, collections, where, config, final_k, prefer_repo=""):
    """Run the full retrieval pipeline: hybrid search, rerank, dedupe, truncate.

    Dedupe runs on the *full ranked pool* before the final-k slice, so a query
    whose top hits share a source still returns up to final_k distinct sources.
    """
    candidates = hybrid.search(query, top_k=config.retrieval.top_k, collections=collections, where=where)
    if reranker and candidates:
        ranked = reranker.rerank(query, candidates, top_k=len(candidates))
    else:
        ranked = candidates
    ranked = apply_repo_preference(ranked, prefer_repo, config.retrieval.repo_boost)
    deduped = deduplicate_results(ranked, max_per_source=config.retrieval.max_per_source)
    return deduped[:final_k]


class HybridSearch:
    def __init__(self, vector_store, embedder, sparse_encoder, collection: str = "code_chunks") -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder
        self.collection = collection

    def search(self, query: str, top_k: int = 20, collections: list[str] | None = None, where: dict | None = None) -> list[SearchResult]:
        if collections is None:
            collections = [self.collection]
        dense = self.embedder.embed_query(query)
        sparse = self.sparse_encoder.encode_query(query)

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
