from __future__ import annotations

import chromadb

from devrag.types import QueryResult


class ChromaStore:
    def __init__(self, persist_dir: str) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)

    def _get_or_create(self, collection: str) -> chromadb.Collection:
        return self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        collection: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        coll = self._get_or_create(collection)
        # ChromaDB rejects empty dicts; use None for empty metadata entries
        safe_metadatas = [m if m else None for m in metadatas]
        coll.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=safe_metadatas)

    def query(
        self,
        collection: str,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict | None = None,
    ) -> QueryResult:
        try:
            coll = self._client.get_collection(name=collection)
        except Exception:
            return QueryResult(ids=[], documents=[], metadatas=[], distances=[])

        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, coll.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        if coll.count() == 0:
            return QueryResult(ids=[], documents=[], metadatas=[], distances=[])

        results = coll.query(**kwargs)
        return QueryResult(
            ids=results["ids"][0],
            documents=results["documents"][0],
            metadatas=results["metadatas"][0],
            distances=results["distances"][0],
        )

    def get_by_ids(self, collection: str, ids: list[str]) -> QueryResult:
        if not ids:
            return QueryResult(ids=[], documents=[], metadatas=[], distances=[])
        try:
            coll = self._client.get_collection(name=collection)
        except Exception:
            return QueryResult(ids=[], documents=[], metadatas=[], distances=[])
        results = coll.get(ids=ids, include=["documents", "metadatas"])
        return QueryResult(
            ids=results["ids"],
            documents=results["documents"],
            metadatas=results["metadatas"],
            distances=[],
        )

    def delete(self, collection: str, ids: list[str]) -> None:
        try:
            coll = self._client.get_collection(name=collection)
        except Exception:
            return
        coll.delete(ids=ids)

    def delete_collection(self, collection: str) -> None:
        try:
            self._client.delete_collection(name=collection)
        except Exception:
            pass

    def count(self, collection: str) -> int:
        try:
            coll = self._client.get_collection(name=collection)
            return coll.count()
        except Exception:
            return 0
