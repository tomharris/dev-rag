from __future__ import annotations
from pathlib import Path
from devrag.config import DevragConfig
from devrag.stores.qdrant_store import QdrantStore


def create_vector_store(config: DevragConfig):
    backend = config.vector_store.backend
    if backend == "chromadb":
        from devrag.stores.chroma_store import ChromaStore
        persist_dir = Path(config.vector_store.persist_dir).expanduser()
        persist_dir.mkdir(parents=True, exist_ok=True)
        return ChromaStore(persist_dir=str(persist_dir))
    elif backend == "qdrant":
        return QdrantStore(
            url=config.vector_store.qdrant_url,
            embedding_dim=config.vector_store.embedding_dim,
        )
    else:
        raise ValueError(f"Unknown vector store backend: {backend}")
