from __future__ import annotations

from sentence_transformers import CrossEncoder

from devrag.types import SearchResult


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        if not candidates:
            return []
        documents = [c.text for c in candidates]
        ranked = self._model.rank(query, documents, top_k=top_k)
        results: list[SearchResult] = []
        for item in ranked[:top_k]:
            idx = item["corpus_id"]
            original = candidates[idx]
            results.append(SearchResult(
                chunk_id=original.chunk_id, text=original.text,
                score=float(item["score"]), metadata=original.metadata,
            ))
        return results
