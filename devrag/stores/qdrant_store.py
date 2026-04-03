from __future__ import annotations
from devrag.types import QueryResult

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, FieldCondition, Filter, MatchValue,
        PointIdsList, PointStruct, VectorParams,
    )
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False


class QdrantStore:
    def __init__(self, url: str = "http://localhost:6333", embedding_dim: int = 768) -> None:
        if not HAS_QDRANT:
            raise ImportError("qdrant-client is required. Install with: pip install devrag[qdrant]")
        self._client = QdrantClient(url=url)
        self._embedding_dim = embedding_dim

    def _ensure_collection(self, collection: str) -> None:
        if not self._client.collection_exists(collection):
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=self._embedding_dim, distance=Distance.COSINE),
            )

    def upsert(self, collection: str, ids: list[str], embeddings: list[list[float]],
               documents: list[str], metadatas: list[dict]) -> None:
        self._ensure_collection(collection)
        points = []
        for i, doc_id in enumerate(ids):
            payload = dict(metadatas[i]) if metadatas[i] else {}
            payload["_document"] = documents[i]
            points.append(PointStruct(id=doc_id, vector=embeddings[i], payload=payload))
        self._client.upsert(collection_name=collection, points=points, wait=True)

    def query(self, collection: str, query_embedding: list[float],
              n_results: int = 10, where: dict | None = None) -> QueryResult:
        if not self._client.collection_exists(collection):
            return QueryResult(ids=[], documents=[], metadatas=[], distances=[])
        query_filter = None
        if where:
            conditions = [FieldCondition(key=k, match=MatchValue(value=v)) for k, v in where.items()]
            query_filter = Filter(must=conditions)
        results = self._client.search(
            collection_name=collection, query_vector=query_embedding,
            limit=n_results, query_filter=query_filter, with_payload=True,
        )
        ids, documents, metadatas, distances = [], [], [], []
        for point in results:
            ids.append(str(point.id))
            payload = dict(point.payload) if point.payload else {}
            doc = payload.pop("_document", "")
            documents.append(doc)
            metadatas.append(payload)
            distances.append(point.score)
        return QueryResult(ids=ids, documents=documents, metadatas=metadatas, distances=distances)

    def delete(self, collection: str, ids: list[str]) -> None:
        if not self._client.collection_exists(collection):
            return
        self._client.delete(collection_name=collection, points_selector=PointIdsList(points=ids), wait=True)

    def count(self, collection: str) -> int:
        if not self._client.collection_exists(collection):
            return 0
        result = self._client.count(collection_name=collection, exact=True)
        return result.count
