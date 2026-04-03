import pytest
from unittest.mock import MagicMock, patch

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
    mock_qdrant_client.search.return_value = [mock_scored_point]
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
    mock_qdrant_client.search.return_value = []
    store = QdrantStore(url="http://localhost:6333", embedding_dim=768)
    store.query(collection="test", query_embedding=[0.1] * 768, where={"lang": "python"})
    search_call = mock_qdrant_client.search.call_args
    assert search_call.kwargs.get("query_filter") is not None
