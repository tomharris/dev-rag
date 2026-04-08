from __future__ import annotations
import uuid
from devrag.types import QueryResult


def _to_uuid(string_id: str) -> str:
    """Convert a string ID to a deterministic UUID5 for Qdrant compatibility."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, string_id))

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
            payload["_original_id"] = doc_id
            points.append(PointStruct(id=_to_uuid(doc_id), vector=embeddings[i], payload=payload))
        self._client.upsert(collection_name=collection, points=points, wait=True)

    def query(self, collection: str, query_embedding: list[float],
              n_results: int = 10, where: dict | None = None) -> QueryResult:
        if not self._client.collection_exists(collection):
            return QueryResult(ids=[], documents=[], metadatas=[], distances=[])
        query_filter = None
        if where:
            conditions = [FieldCondition(key=k, match=MatchValue(value=v)) for k, v in where.items()]
            query_filter = Filter(must=conditions)
        response = self._client.query_points(
            collection_name=collection, query=query_embedding,
            limit=n_results, query_filter=query_filter, with_payload=True,
        )
        results = response.points
        ids, documents, metadatas, distances = [], [], [], []
        for point in results:
            payload = dict(point.payload) if point.payload else {}
            original_id = payload.pop("_original_id", str(point.id))
            doc = payload.pop("_document", "")
            ids.append(original_id)
            documents.append(doc)
            metadatas.append(payload)
            distances.append(point.score)
        return QueryResult(ids=ids, documents=documents, metadatas=metadatas, distances=distances)

    def get_by_ids(self, collection: str, ids: list[str]) -> QueryResult:
        if not ids or not self._client.collection_exists(collection):
            return QueryResult(ids=[], documents=[], metadatas=[], distances=[])
        qdrant_ids = [_to_uuid(id) for id in ids]
        points = self._client.retrieve(collection_name=collection, ids=qdrant_ids, with_payload=True)
        out_ids, documents, metadatas = [], [], []
        for point in points:
            payload = dict(point.payload) if point.payload else {}
            original_id = payload.pop("_original_id", str(point.id))
            doc = payload.pop("_document", "")
            out_ids.append(original_id)
            documents.append(doc)
            metadatas.append(payload)
        return QueryResult(ids=out_ids, documents=documents, metadatas=metadatas, distances=[])

    def delete(self, collection: str, ids: list[str]) -> None:
        if not self._client.collection_exists(collection):
            return
        qdrant_ids = [_to_uuid(id) for id in ids]
        self._client.delete(collection_name=collection, points_selector=PointIdsList(points=qdrant_ids), wait=True)

    def delete_collection(self, collection: str) -> None:
        if self._client.collection_exists(collection):
            self._client.delete_collection(collection_name=collection)

    def count(self, collection: str) -> int:
        if not self._client.collection_exists(collection):
            return 0
        result = self._client.count(collection_name=collection, exact=True)
        return result.count
