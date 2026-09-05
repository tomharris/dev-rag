"""Query-metric logging for the search path.

`MetadataDB.log_query_metric` existed since the first observability commit but
was never wired to a caller, so `query_metrics` stayed empty and there was no
record of what real queries do. This module is the single chokepoint both the
CLI `search` command and the MCP `search()` tool call, so the two can't drift
(same reasoning as `refresh_all_repos` and `request_with_retries`).

Logging must never break a search: every failure here is swallowed. A metrics
table is a convenience, not part of the contract of returning results.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def log_search(meta_db, query: str, collections: list[str], classification: str,
               timings: dict, result_count: int) -> None:
    """Record one search into `query_metrics`, best-effort.

    The table's four timing columns predate the current Qdrant pipeline, where
    dense and sparse are fused inside one server-side call. They are filled as:

      - ``vector_ms`` — dense query embedding + the Qdrant ``query_points`` calls
      - ``bm25_ms``   — BM25 *query encoding* only (there is no separable BM25
                        search time; the fused call is counted in ``vector_ms``)
      - ``rerank_ms`` — cross-encoder reranking, 0.0 when reranking is off
      - ``total_ms``  — the whole pipeline including dedupe

    Args:
        meta_db: a MetadataDB, or None to skip logging entirely.
        classification: the routed intent label from ``QueryRouter.classify``.
        timings: the out-param dict filled by ``search_rank_dedupe``.
    """
    if meta_db is None:
        return
    try:
        meta_db.log_query_metric(
            query=query,
            collections=collections,
            vector_ms=timings.get("embed_ms", 0.0) + timings.get("vector_ms", 0.0),
            bm25_ms=timings.get("sparse_ms", 0.0),
            rerank_ms=timings.get("rerank_ms", 0.0),
            total_ms=timings.get("total_ms", 0.0),
            result_count=result_count,
            classification=classification,
        )
    except Exception:  # pragma: no cover - defensive; metrics must not break search
        logger.debug("failed to log query metric", exc_info=True)
