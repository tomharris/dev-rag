from __future__ import annotations

from devrag.types import QueryResult, SearchResult


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    if not ranked_lists:
        return []
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda d: scores[d], reverse=True)


class HybridSearch:
    def __init__(self, vector_store, metadata_db, embedder, collection: str = "code_chunks") -> None:
        self.vector_store = vector_store
        self.metadata_db = metadata_db
        self.embedder = embedder
        self.collection = collection

    def search(self, query: str, top_k: int = 20) -> list[SearchResult]:
        query_embedding = self.embedder.embed_query(query)
        vector_results = self.vector_store.query(
            collection=self.collection, query_embedding=query_embedding, n_results=top_k,
        )
        doc_lookup: dict[str, tuple[str, dict]] = {}
        for i, doc_id in enumerate(vector_results.ids):
            doc_lookup[doc_id] = (vector_results.documents[i], vector_results.metadatas[i])

        bm25_results = self.metadata_db.search_fts(query, limit=top_k)
        vector_ranked = vector_results.ids
        bm25_ranked = [chunk_id for chunk_id, _score in bm25_results]
        fused_ids = reciprocal_rank_fusion([vector_ranked, bm25_ranked], k=60)

        rrf_scores: dict[str, float] = {}
        for rank, doc_id in enumerate(fused_ids):
            rrf_scores[doc_id] = 1.0 / (60 + rank + 1)

        results: list[SearchResult] = []
        for doc_id in fused_ids[:top_k]:
            if doc_id in doc_lookup:
                text, metadata = doc_lookup[doc_id]
            else:
                continue
            results.append(SearchResult(chunk_id=doc_id, text=text, score=rrf_scores[doc_id], metadata=metadata))
        return results
