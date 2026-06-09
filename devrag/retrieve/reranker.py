from __future__ import annotations

import logging

import transformers.utils.logging as tf_logging
from sentence_transformers import CrossEncoder

from devrag.types import SearchResult


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length: int = 512) -> None:
        prev_tf = tf_logging.get_verbosity()
        hf_logger = logging.getLogger("huggingface_hub.utils._http")
        prev_hf = hf_logger.level
        tf_logging.set_verbosity_error()
        hf_logger.setLevel(logging.ERROR)
        try:
            # Prefer the local HF cache: a cache-only load makes zero network
            # calls, so it works behind networks that block huggingface.co and
            # sidesteps the huggingface_hub closed-client bug (a connect error
            # during the online etag HEAD surfaces as the misleading
            # "Cannot send a request, as the client has been closed").
            try:
                self._model = CrossEncoder(model_name, max_length=max_length, local_files_only=True)
            except Exception:
                # Not in the local cache — allow a network download (first run,
                # HF reachable).
                try:
                    self._model = CrossEncoder(model_name, max_length=max_length)
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"Reranker model '{model_name}' is not cached locally and huggingface.co "
                        f"could not be reached. Pre-download it on a network with HF access, or set "
                        f"retrieval.rerank: false to disable reranking."
                    ) from exc
        finally:
            tf_logging.set_verbosity(prev_tf)
            hf_logger.setLevel(prev_hf)

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
