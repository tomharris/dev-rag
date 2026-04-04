from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    def __init__(
        self,
        model: str = "nomic-embed-text",
        ollama_url: str = "http://localhost:11434",
        batch_size: int = 64,
    ) -> None:
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # Filter out empty/whitespace-only texts, tracking their positions
        non_empty_indices = [i for i, t in enumerate(texts) if t.strip()]
        if len(non_empty_indices) < len(texts):
            empty_indices = [i for i in range(len(texts)) if i not in set(non_empty_indices)]
            logger.warning("Skipping empty texts at indices %s", empty_indices)

        if not non_empty_indices:
            logger.warning("All %d texts are empty; returning no embeddings", len(texts))
            return []

        non_empty_texts = [texts[i] for i in non_empty_indices]

        all_embeddings: list[list[float]] = []
        for i in range(0, len(non_empty_texts), self.batch_size):
            batch = non_empty_texts[i : i + self.batch_size]
            response = httpx.post(
                f"{self.ollama_url}/api/embed",
                json={"model": self.model, "input": batch},
                timeout=120.0,
            )
            if not response.is_success:
                logger.error("Ollama embed failed (%s): %s", response.status_code, response.text)
                response.raise_for_status()
            data = response.json()
            all_embeddings.extend(data["embeddings"])

        # Reconstruct full results with zero vectors for empty inputs
        if len(non_empty_indices) == len(texts):
            return all_embeddings

        dim = len(all_embeddings[0])
        result: list[list[float]] = [[] for _ in texts]
        for idx, emb in zip(non_empty_indices, all_embeddings):
            result[idx] = emb
        for i in range(len(texts)):
            if not result[i]:
                result[i] = [0.0] * dim
        return result

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]
