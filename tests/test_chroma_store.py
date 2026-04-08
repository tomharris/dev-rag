import pytest

from devrag.stores.chroma_store import ChromaStore
from devrag.types import QueryResult


@pytest.fixture
def store(tmp_dir):
    return ChromaStore(persist_dir=str(tmp_dir / "chroma"))


def test_upsert_and_count(store):
    store.upsert(
        collection="test",
        ids=["a", "b"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        documents=["hello world", "foo bar"],
        metadatas=[{"lang": "en"}, {"lang": "en"}],
    )
    assert store.count("test") == 2


def test_query_returns_closest(store):
    store.upsert(
        collection="test",
        ids=["a", "b", "c"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        documents=["doc a", "doc b", "doc c"],
        metadatas=[{"x": 1}, {"x": 2}, {"x": 3}],
    )
    result = store.query(
        collection="test",
        query_embedding=[1.0, 0.1, 0.0],
        n_results=2,
    )
    assert isinstance(result, QueryResult)
    assert len(result.ids) == 2
    assert result.ids[0] == "a"


def test_query_with_where_filter(store):
    store.upsert(
        collection="test",
        ids=["a", "b"],
        embeddings=[[1.0, 0.0], [0.9, 0.1]],
        documents=["doc a", "doc b"],
        metadatas=[{"lang": "python"}, {"lang": "typescript"}],
    )
    result = store.query(
        collection="test",
        query_embedding=[1.0, 0.0],
        n_results=2,
        where={"lang": "typescript"},
    )
    assert len(result.ids) == 1
    assert result.ids[0] == "b"


def test_upsert_overwrites_existing(store):
    store.upsert(
        collection="test",
        ids=["a"],
        embeddings=[[1.0, 0.0]],
        documents=["original"],
        metadatas=[{"v": 1}],
    )
    store.upsert(
        collection="test",
        ids=["a"],
        embeddings=[[0.0, 1.0]],
        documents=["updated"],
        metadatas=[{"v": 2}],
    )
    assert store.count("test") == 1
    result = store.query("test", query_embedding=[0.0, 1.0], n_results=1)
    assert result.documents[0] == "updated"
    assert result.metadatas[0]["v"] == 2


def test_delete(store):
    store.upsert(
        collection="test",
        ids=["a", "b"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=["a", "b"],
        metadatas=[{}, {}],
    )
    store.delete("test", ids=["a"])
    assert store.count("test") == 1


def test_delete_collection(store):
    store.upsert(
        collection="test",
        ids=["a", "b"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=["a", "b"],
        metadatas=[{}, {}],
    )
    assert store.count("test") == 2
    store.delete_collection("test")
    assert store.count("test") == 0


def test_delete_collection_nonexistent(store):
    store.delete_collection("nonexistent")  # Should not raise


def test_count_nonexistent_collection(store):
    assert store.count("nonexistent") == 0
