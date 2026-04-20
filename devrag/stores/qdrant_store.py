from __future__ import annotations
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient
from qdrant_client.models import (
    BinaryQuantization, BinaryQuantizationConfig, Distance, FieldCondition,
    Filter, FusionQuery, Fusion, MatchValue, Modifier, PayloadSchemaType,
    PointIdsList, PointStruct, Prefetch, ScalarQuantization,
    ScalarQuantizationConfig, ScalarType, SparseVector, SparseVectorParams,
    VectorParams,
)
from devrag.types import QueryResult

if TYPE_CHECKING:
    from devrag.config import DevragConfig


DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"

PAYLOAD_INDEX_FIELDS: dict[str, PayloadSchemaType] = {
    "repo": PayloadSchemaType.KEYWORD,
    "file_path": PayloadSchemaType.KEYWORD,
    "pr_number": PayloadSchemaType.INTEGER,
    "issue_number": PayloadSchemaType.INTEGER,
    "ticket_key": PayloadSchemaType.KEYWORD,
    "page_id": PayloadSchemaType.KEYWORD,
    "session_id": PayloadSchemaType.KEYWORD,
    "chunk_type": PayloadSchemaType.KEYWORD,
}


def _to_uuid(string_id: str) -> str:
    """Convert a string ID to a deterministic UUID5 for Qdrant compatibility."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, string_id))


def _build_quantization_config(kind: str, always_ram: bool):
    if kind == "scalar":
        return ScalarQuantization(
            scalar=ScalarQuantizationConfig(type=ScalarType.INT8, always_ram=always_ram),
        )
    if kind == "binary":
        return BinaryQuantization(binary=BinaryQuantizationConfig(always_ram=always_ram))
    return None


class QdrantStore:
    def __init__(
        self,
        url: str | None = None,
        path: str | None = None,
        embedding_dim: int = 768,
        quantization: str = "",
        quantization_always_ram: bool = True,
    ) -> None:
        if path:
            self._client = QdrantClient(path=path)
        else:
            self._client = QdrantClient(url=url or "http://localhost:6333")
        self._embedding_dim = embedding_dim
        self._quantization_config = _build_quantization_config(quantization, quantization_always_ram)

    @classmethod
    def from_config(cls, config: "DevragConfig") -> "QdrantStore":
        vs = config.vector_store
        if vs.qdrant_path:
            path = Path(vs.qdrant_path).expanduser()
            path.mkdir(parents=True, exist_ok=True)
            return cls(
                path=str(path), embedding_dim=vs.embedding_dim,
                quantization=vs.quantization, quantization_always_ram=vs.quantization_always_ram,
            )
        return cls(
            url=vs.qdrant_url, embedding_dim=vs.embedding_dim,
            quantization=vs.quantization, quantization_always_ram=vs.quantization_always_ram,
        )

    def _ensure_collection(self, collection: str) -> None:
        if self._client.collection_exists(collection):
            return
        kwargs: dict = {
            "collection_name": collection,
            "vectors_config": {
                DENSE_VECTOR: VectorParams(size=self._embedding_dim, distance=Distance.COSINE),
            },
            "sparse_vectors_config": {
                SPARSE_VECTOR: SparseVectorParams(modifier=Modifier.IDF),
            },
        }
        if self._quantization_config is not None:
            kwargs["quantization_config"] = self._quantization_config
        self._client.create_collection(**kwargs)
        for field_name, schema in PAYLOAD_INDEX_FIELDS.items():
            self._client.create_payload_index(
                collection_name=collection, field_name=field_name, field_schema=schema,
            )

    def upsert(
        self,
        collection: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
        sparse_embeddings: list[SparseVector] | None = None,
        wait: bool = True,
    ) -> None:
        self._ensure_collection(collection)
        points = []
        for i, doc_id in enumerate(ids):
            payload = dict(metadatas[i]) if metadatas[i] else {}
            payload["_document"] = documents[i]
            payload["_original_id"] = doc_id
            vector: dict = {DENSE_VECTOR: embeddings[i]}
            if sparse_embeddings is not None:
                vector[SPARSE_VECTOR] = sparse_embeddings[i]
            points.append(PointStruct(id=_to_uuid(doc_id), vector=vector, payload=payload))
        self._client.upsert(collection_name=collection, points=points, wait=wait)

    def _build_filter(self, where: dict | None) -> Filter | None:
        if not where:
            return None
        conditions = [FieldCondition(key=k, match=MatchValue(value=v)) for k, v in where.items()]
        return Filter(must=conditions)

    def _points_to_result(self, points) -> QueryResult:
        ids, documents, metadatas, distances = [], [], [], []
        for point in points:
            payload = dict(point.payload) if point.payload else {}
            original_id = payload.pop("_original_id", str(point.id))
            doc = payload.pop("_document", "")
            ids.append(original_id)
            documents.append(doc)
            metadatas.append(payload)
            distances.append(point.score)
        return QueryResult(ids=ids, documents=documents, metadatas=metadatas, distances=distances)

    def query(
        self,
        collection: str,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict | None = None,
    ) -> QueryResult:
        if not self._client.collection_exists(collection):
            return QueryResult(ids=[], documents=[], metadatas=[], distances=[])
        response = self._client.query_points(
            collection_name=collection,
            query=query_embedding,
            using=DENSE_VECTOR,
            limit=n_results,
            query_filter=self._build_filter(where),
            with_payload=True,
        )
        return self._points_to_result(response.points)

    def hybrid_query(
        self,
        collection: str,
        dense_embedding: list[float],
        sparse_embedding: SparseVector,
        n_results: int = 10,
        where: dict | None = None,
    ) -> QueryResult:
        if not self._client.collection_exists(collection):
            return QueryResult(ids=[], documents=[], metadatas=[], distances=[])
        query_filter = self._build_filter(where)
        prefetch = [
            Prefetch(query=dense_embedding, using=DENSE_VECTOR, limit=n_results, filter=query_filter),
            Prefetch(query=sparse_embedding, using=SPARSE_VECTOR, limit=n_results, filter=query_filter),
        ]
        response = self._client.query_points(
            collection_name=collection,
            prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=n_results,
            with_payload=True,
        )
        return self._points_to_result(response.points)

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
