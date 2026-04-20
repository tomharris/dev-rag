import pytest
from unittest.mock import MagicMock, patch

from qdrant_client.models import SparseVector

from devrag.stores.qdrant_store import QdrantStore
from devrag.types import QueryResult


@pytest.fixture
def mock_qdrant_client():
    with patch("devrag.stores.qdrant_store.QdrantClient") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        yield mock_client


def test_upsert(mock_qdrant_client):
    mock_qdrant_client.collection_exists.return_value = True
    store = QdrantStore(url="http://localhost:6333", embedding_dim=768)
    store.upsert(collection="test", ids=["a", "b"],
        embeddings=[[0.1] * 768, [0.2] * 768],
        documents=["doc a", "doc b"],
        metadatas=[{"lang": "en"}, {"lang": "fr"}])
    mock_qdrant_client.upsert.assert_called_once()
    args = mock_qdrant_client.upsert.call_args
    assert args.kwargs["collection_name"] == "test"
    assert len(args.kwargs["points"]) == 2


def test_query(mock_qdrant_client):
    mock_qdrant_client.collection_exists.return_value = True
    mock_scored_point = MagicMock()
    mock_scored_point.id = "a"
    mock_scored_point.score = 0.95
    mock_scored_point.payload = {"_document": "doc a", "lang": "en"}
    mock_response = MagicMock()
    mock_response.points = [mock_scored_point]
    mock_qdrant_client.query_points.return_value = mock_response
    store = QdrantStore(url="http://localhost:6333", embedding_dim=768)
    result = store.query(collection="test", query_embedding=[0.1] * 768, n_results=5)
    assert isinstance(result, QueryResult)
    assert result.ids == ["a"]
    assert result.documents == ["doc a"]
    assert result.metadatas == [{"lang": "en"}]
    assert result.distances == [0.95]


def test_query_nonexistent_collection(mock_qdrant_client):
    mock_qdrant_client.collection_exists.return_value = False
    store = QdrantStore(url="http://localhost:6333", embedding_dim=768)
    result = store.query(collection="nonexistent", query_embedding=[0.1] * 768)
    assert result.ids == []


def test_delete(mock_qdrant_client):
    mock_qdrant_client.collection_exists.return_value = True
    store = QdrantStore(url="http://localhost:6333", embedding_dim=768)
    store.delete(collection="test", ids=["a", "b"])
    mock_qdrant_client.delete.assert_called_once()


def test_count(mock_qdrant_client):
    mock_qdrant_client.collection_exists.return_value = True
    mock_count = MagicMock()
    mock_count.count = 42
    mock_qdrant_client.count.return_value = mock_count
    store = QdrantStore(url="http://localhost:6333", embedding_dim=768)
    assert store.count("test") == 42


def test_count_nonexistent(mock_qdrant_client):
    mock_qdrant_client.collection_exists.return_value = False
    store = QdrantStore(url="http://localhost:6333", embedding_dim=768)
    assert store.count("nonexistent") == 0


def test_query_with_where_filter(mock_qdrant_client):
    mock_qdrant_client.collection_exists.return_value = True
    mock_response = MagicMock()
    mock_response.points = []
    mock_qdrant_client.query_points.return_value = mock_response
    store = QdrantStore(url="http://localhost:6333", embedding_dim=768)
    store.query(collection="test", query_embedding=[0.1] * 768, where={"lang": "python"})
    search_call = mock_qdrant_client.query_points.call_args
    assert search_call.kwargs.get("query_filter") is not None


def test_ensure_collection_creates_sparse_and_payload_indexes(mock_qdrant_client):
    mock_qdrant_client.collection_exists.return_value = False
    store = QdrantStore(url="http://localhost:6333", embedding_dim=768)
    store._ensure_collection("test")
    create_call = mock_qdrant_client.create_collection.call_args
    assert "dense" in create_call.kwargs["vectors_config"]
    assert "bm25" in create_call.kwargs["sparse_vectors_config"]
    assert "quantization_config" not in create_call.kwargs
    indexed_fields = {c.kwargs["field_name"] for c in mock_qdrant_client.create_payload_index.call_args_list}
    assert {"repo", "file_path", "pr_number", "issue_number", "ticket_key",
            "page_id", "session_id", "chunk_type"} <= indexed_fields


def test_ensure_collection_with_scalar_quantization(mock_qdrant_client):
    from qdrant_client.models import ScalarQuantization
    mock_qdrant_client.collection_exists.return_value = False
    store = QdrantStore(url="http://localhost:6333", embedding_dim=768, quantization="scalar")
    store._ensure_collection("test")
    create_call = mock_qdrant_client.create_collection.call_args
    assert isinstance(create_call.kwargs["quantization_config"], ScalarQuantization)


def test_ensure_collection_with_binary_quantization(mock_qdrant_client):
    from qdrant_client.models import BinaryQuantization
    mock_qdrant_client.collection_exists.return_value = False
    store = QdrantStore(url="http://localhost:6333", embedding_dim=768, quantization="binary")
    store._ensure_collection("test")
    create_call = mock_qdrant_client.create_collection.call_args
    assert isinstance(create_call.kwargs["quantization_config"], BinaryQuantization)


def test_hybrid_query_uses_fusion(mock_qdrant_client):
    mock_qdrant_client.collection_exists.return_value = True
    mock_response = MagicMock()
    mock_response.points = []
    mock_qdrant_client.query_points.return_value = mock_response
    store = QdrantStore(url="http://localhost:6333", embedding_dim=768)
    store.hybrid_query(
        collection="test",
        dense_embedding=[0.1] * 768,
        sparse_embedding=SparseVector(indices=[1], values=[1.0]),
        n_results=5,
    )
    call = mock_qdrant_client.query_points.call_args
    assert call.kwargs["collection_name"] == "test"
    assert len(call.kwargs["prefetch"]) == 2
    assert call.kwargs["query"] is not None  # FusionQuery


def test_from_config_uses_path(tmp_dir):
    from devrag.config import DevragConfig
    config = DevragConfig()
    config.vector_store.qdrant_path = str(tmp_dir / "qdrant")
    store = QdrantStore.from_config(config)
    assert store is not None


def test_from_config_uses_url(mock_qdrant_client):
    from devrag.config import DevragConfig
    config = DevragConfig()
    config.vector_store.qdrant_url = "http://localhost:6333"
    config.vector_store.qdrant_path = ""
    QdrantStore.from_config(config)
    # Client constructed with URL, not path
    from devrag.stores.qdrant_store import QdrantClient
    assert QdrantClient.call_args.kwargs.get("url") == "http://localhost:6333" or \
           "http://localhost:6333" in str(QdrantClient.call_args)


def test_upsert_with_sparse_embeddings(mock_qdrant_client):
    mock_qdrant_client.collection_exists.return_value = True
    store = QdrantStore(url="http://localhost:6333", embedding_dim=768)
    store.upsert(
        collection="test", ids=["a"], embeddings=[[0.1] * 768],
        documents=["doc"], metadatas=[{}],
        sparse_embeddings=[SparseVector(indices=[1, 2], values=[0.5, 0.3])],
    )
    points = mock_qdrant_client.upsert.call_args.kwargs["points"]
    assert "dense" in points[0].vector
    assert "bm25" in points[0].vector


def test_upsert_wait_flag_forwards(mock_qdrant_client):
    mock_qdrant_client.collection_exists.return_value = True
    store = QdrantStore(url="http://localhost:6333", embedding_dim=768)
    store.upsert(collection="test", ids=["a"], embeddings=[[0.1] * 768],
                 documents=["doc"], metadatas=[{}], wait=False)
    assert mock_qdrant_client.upsert.call_args.kwargs["wait"] is False
