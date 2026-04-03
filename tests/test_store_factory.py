from unittest.mock import patch, MagicMock
import pytest
from devrag.stores.factory import create_vector_store
from devrag.config import DevragConfig


def test_factory_creates_chroma_by_default(tmp_dir):
    config = DevragConfig()
    config.vector_store.persist_dir = str(tmp_dir / "chroma")
    store = create_vector_store(config)
    from devrag.stores.chroma_store import ChromaStore
    assert isinstance(store, ChromaStore)


@patch("devrag.stores.factory.QdrantStore")
def test_factory_creates_qdrant(mock_qdrant_cls):
    mock_store = MagicMock()
    mock_qdrant_cls.return_value = mock_store
    config = DevragConfig()
    config.vector_store.backend = "qdrant"
    config.vector_store.qdrant_url = "http://localhost:6333"
    store = create_vector_store(config)
    mock_qdrant_cls.assert_called_once_with(
        url="http://localhost:6333",
        embedding_dim=config.vector_store.embedding_dim,
    )
    assert store is mock_store


def test_factory_raises_for_unknown_backend():
    config = DevragConfig()
    config.vector_store.backend = "unknown"
    with pytest.raises(ValueError, match="Unknown vector store backend"):
        create_vector_store(config)
