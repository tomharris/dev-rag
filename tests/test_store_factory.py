from unittest.mock import MagicMock, patch

from devrag.config import DevragConfig
from devrag.stores.factory import create_vector_store


@patch("devrag.stores.factory.QdrantStore")
def test_factory_creates_qdrant_from_url(mock_qdrant_cls):
    mock_store = MagicMock()
    mock_qdrant_cls.return_value = mock_store
    config = DevragConfig()
    config.vector_store.qdrant_url = "http://localhost:6333"
    store = create_vector_store(config)
    mock_qdrant_cls.assert_called_once_with(
        url="http://localhost:6333",
        embedding_dim=config.vector_store.embedding_dim,
    )
    assert store is mock_store


@patch("devrag.stores.factory.QdrantStore")
def test_factory_creates_qdrant_from_path(mock_qdrant_cls, tmp_dir):
    mock_store = MagicMock()
    mock_qdrant_cls.return_value = mock_store
    config = DevragConfig()
    config.vector_store.qdrant_path = str(tmp_dir / "qdrant")
    store = create_vector_store(config)
    mock_qdrant_cls.assert_called_once_with(
        path=str(tmp_dir / "qdrant"),
        embedding_dim=config.vector_store.embedding_dim,
    )
    assert store is mock_store
