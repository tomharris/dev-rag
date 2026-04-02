from __future__ import annotations

import httpx


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

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = httpx.post(
                f"{self.ollama_url}/api/embed",
                json={"model": self.model, "input": batch},
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()
            all_embeddings.extend(data["embeddings"])

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]
