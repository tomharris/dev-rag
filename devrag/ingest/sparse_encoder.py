from __future__ import annotations

import logging

from qdrant_client.models import SparseVector

logger = logging.getLogger(__name__)


class BM25SparseEncoder:
    def __init__(self, model_name: str = "Qdrant/bm25", batch_size: int = 64) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            from fastembed import SparseTextEmbedding
            self._model = SparseTextEmbedding(model_name=self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> list[SparseVector]:
        if not texts:
            return []

        results: list[SparseVector | None] = [None] * len(texts)
        non_empty_indices = [i for i, t in enumerate(texts) if t.strip()]
        if not non_empty_indices:
            return [SparseVector(indices=[], values=[])] * len(texts)

        non_empty_texts = [texts[i] for i in non_empty_indices]
        model = self._get_model()
        embeddings = list(model.embed(non_empty_texts, batch_size=self.batch_size))

        for idx, emb in zip(non_empty_indices, embeddings):
            results[idx] = SparseVector(
                indices=emb.indices.tolist(),
                values=emb.values.tolist(),
            )
        for i in range(len(texts)):
            if results[i] is None:
                results[i] = SparseVector(indices=[], values=[])
        return results  # type: ignore[return-value]

    def encode_query(self, text: str) -> SparseVector:
        if not text.strip():
            return SparseVector(indices=[], values=[])
        model = self._get_model()
        emb = next(model.query_embed(text))
        return SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())
