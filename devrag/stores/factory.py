from __future__ import annotations

from pathlib import Path

from devrag.config import DevragConfig
from devrag.stores.qdrant_store import QdrantStore


def create_vector_store(config: DevragConfig) -> QdrantStore:
    vs = config.vector_store
    if vs.qdrant_path:
        path = Path(vs.qdrant_path).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return QdrantStore(path=str(path), embedding_dim=vs.embedding_dim)
    return QdrantStore(url=vs.qdrant_url, embedding_dim=vs.embedding_dim)
