# tests/test_integration.py
"""End-to-end integration test. Requires Ollama running with nomic-embed-text."""
import os
import subprocess
from pathlib import Path

import pytest

from devrag.config import DevragConfig
from devrag.ingest.code_indexer import CodeIndexer
from devrag.ingest.embedder import OllamaEmbedder
from devrag.retrieve.hybrid_search import HybridSearch
from devrag.stores.chroma_store import ChromaStore
from devrag.stores.metadata_db import MetadataDB

pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_INTEGRATION", "1") == "1",
    reason="Set SKIP_INTEGRATION=0 to run integration tests (requires Ollama)",
)


@pytest.fixture
def e2e_repo(tmp_dir):
    repo = tmp_dir / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=str(repo), capture_output=True)

    (repo / "src").mkdir()
    (repo / "src" / "auth.py").write_text('''
"""Authentication module."""

import hashlib
import secrets


class PasswordHasher:
    """Handles secure password hashing and verification."""

    def hash_password(self, password: str) -> str:
        """Hash a password using SHA-256 with a random salt."""
        salt = secrets.token_hex(16)
        hashed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
        return f"{salt}:{hashed}"

    def verify_password(self, password: str, stored: str) -> bool:
        """Verify a password against a stored hash."""
        salt, expected = stored.split(":")
        actual = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
        return actual == expected


def create_session_token(user_id: str) -> str:
    """Generate a random session token for a user."""
    return f"{user_id}:{secrets.token_urlsafe(32)}"
''')

    (repo / "src" / "database.py").write_text('''
"""Database connection pooling."""

from dataclasses import dataclass


@dataclass
class ConnectionConfig:
    host: str
    port: int = 5432
    pool_size: int = 10


class ConnectionPool:
    """Manages a pool of database connections."""

    def __init__(self, config: ConnectionConfig):
        self.config = config
        self._connections: list = []

    def acquire(self):
        """Acquire a connection from the pool."""
        if self._connections:
            return self._connections.pop()
        return self._create_connection()

    def release(self, conn):
        """Return a connection to the pool."""
        if len(self._connections) < self.config.pool_size:
            self._connections.append(conn)

    def _create_connection(self):
        """Create a new database connection."""
        return {"host": self.config.host, "port": self.config.port}
''')

    subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), capture_output=True)
    return repo


@pytest.fixture
def e2e_components(tmp_dir):
    store = ChromaStore(persist_dir=str(tmp_dir / "chroma"))
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = OllamaEmbedder(model="nomic-embed-text")
    return store, meta, embedder


def test_index_and_search_e2e(e2e_repo, e2e_components):
    store, meta, embedder = e2e_components

    # Index the repo
    indexer = CodeIndexer(store, meta, embedder)
    stats = indexer.index_repo(e2e_repo)
    assert stats.files_indexed >= 2
    assert stats.chunks_created >= 4

    # Semantic search: should find auth-related code
    search = HybridSearch(store, meta, embedder, "code_chunks")
    results = search.search("password authentication and hashing")
    assert len(results) > 0
    top_files = [r.metadata.get("file_path", "") for r in results[:3]]
    assert any("auth.py" in f for f in top_files)

    # Keyword search: exact function name
    results2 = search.search("create_session_token")
    assert len(results2) > 0
    assert any("create_session_token" in r.text for r in results2[:3])

    # Incremental re-index should skip everything
    stats2 = indexer.index_repo(e2e_repo, incremental=True)
    assert stats2.files_skipped >= 2
    assert stats2.files_indexed == 0
