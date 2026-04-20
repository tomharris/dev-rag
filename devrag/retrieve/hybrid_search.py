from __future__ import annotations

from collections import defaultdict

from devrag.types import QueryResult, SearchResult


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


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    if not ranked_lists:
        return []
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda d: scores[d], reverse=True)


class HybridSearch:
    def __init__(self, vector_store, metadata_db, embedder, collection: str = "code_chunks", rrf_k: int = 60) -> None:
        self.vector_store = vector_store
        self.metadata_db = metadata_db
        self.embedder = embedder
        self.collection = collection
        self._rrf_k = rrf_k

    def search(self, query: str, top_k: int = 20, collections: list[str] | None = None, where: dict | None = None) -> list[SearchResult]:
        if collections is None:
            collections = [self.collection]
        query_embedding = self.embedder.embed_query(query)
        doc_lookup: dict[str, tuple[str, dict]] = {}
        all_vector_ranked: list[str] = []
        for coll in collections:
            vector_results = self.vector_store.query(
                collection=coll, query_embedding=query_embedding, n_results=top_k, where=where,
            )
            for i, doc_id in enumerate(vector_results.ids):
                doc_lookup[doc_id] = (vector_results.documents[i], vector_results.metadatas[i])
            all_vector_ranked.extend(vector_results.ids)
        bm25_results = self.metadata_db.search_fts_scoped(query, collections=collections, limit=top_k)
        bm25_ranked = [chunk_id for chunk_id, _score in bm25_results]
        fused_ids = reciprocal_rank_fusion([all_vector_ranked, bm25_ranked], k=self._rrf_k)

        # Resolve BM25-only results: fetch text+metadata from vector store
        missing_ids = [doc_id for doc_id in fused_ids[:top_k] if doc_id not in doc_lookup]
        if missing_ids:
            for coll in collections:
                fetched = self.vector_store.get_by_ids(collection=coll, ids=missing_ids)
                for i, doc_id in enumerate(fetched.ids):
                    if doc_id not in doc_lookup:
                        doc_lookup[doc_id] = (fetched.documents[i], fetched.metadatas[i])

        results: list[SearchResult] = []
        for rank, doc_id in enumerate(fused_ids[:top_k]):
            if doc_id not in doc_lookup:
                continue
            text, metadata = doc_lookup[doc_id]
            results.append(SearchResult(chunk_id=doc_id, text=text, score=1.0 / (self._rrf_k + rank + 1), metadata=metadata))
        return results
