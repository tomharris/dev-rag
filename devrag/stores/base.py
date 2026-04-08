from __future__ import annotations

from typing import Protocol

from devrag.types import QueryResult


class VectorStore(Protocol):
    def upsert(
        self,
        collection: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None: ...

    def query(
        self,
        collection: str,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict | None = None,
    ) -> QueryResult: ...

    def get_by_ids(self, collection: str, ids: list[str]) -> QueryResult: ...

    def delete(self, collection: str, ids: list[str]) -> None: ...

    def delete_collection(self, collection: str) -> None: ...

    def count(self, collection: str) -> int: ...
