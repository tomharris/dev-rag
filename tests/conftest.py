import tempfile
from pathlib import Path

import pytest
from qdrant_client.models import SparseVector


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def vector_store(tmp_dir):
    from devrag.stores.qdrant_store import QdrantStore
    return QdrantStore(path=str(tmp_dir / "qdrant"), embedding_dim=768)


class _StubSparseEncoder:
    """Lightweight sparse encoder for tests — avoids FastEmbed model download."""

    def encode(self, texts):
        return [SparseVector(indices=[hash(t) % 1000], values=[1.0]) if t.strip()
                else SparseVector(indices=[], values=[]) for t in texts]

    def encode_query(self, text):
        if not text.strip():
            return SparseVector(indices=[], values=[])
        return SparseVector(indices=[hash(text) % 1000], values=[1.0])


@pytest.fixture
def sparse_encoder():
    return _StubSparseEncoder()


@pytest.fixture
def frozen_now(monkeypatch):
    """Freeze "now" in the PR/issue/Jira sync indexers to a fixed instant.

    These indexers derive a lookback floor (``now - since_days``) and skip items
    older than it. The test fixtures use hardcoded dates around 2026-03-02, so
    without a frozen clock the tests rot the moment wall-clock time drifts past
    ``fixture_date + since_days`` (the item falls outside the window and is
    skipped / the Jira cursor collapses to the floor). Freezing to 2026-03-10 —
    just after the fixtures' dates — keeps a 90-day window inclusive of them
    forever.

    Subclassing ``datetime`` and overriding ``now`` is the dependency-free way to
    freeze time: arithmetic with ``timedelta`` and ``.isoformat()`` still work
    because the returned value is a real ``datetime``.
    """
    from datetime import datetime, timezone

    class _FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 3, 10, tzinfo=tz or timezone.utc)

    for module in ("pr_indexer", "issue_indexer", "jira_indexer"):
        monkeypatch.setattr(f"devrag.ingest.{module}.datetime", _FrozenDateTime)
    return _FrozenDateTime


@pytest.fixture
def sample_python_file(tmp_dir):
    code = '''"""Module docstring."""

import os
from pathlib import Path


class FileProcessor:
    """Processes files from disk."""

    def __init__(self, root: str):
        self.root = Path(root)

    def read_file(self, name: str) -> str:
        """Read a file by name."""
        path = self.root / name
        return path.read_text()

    def list_files(self) -> list[str]:
        """List all files in root."""
        return [f.name for f in self.root.iterdir() if f.is_file()]


def standalone_function(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y
'''
    p = tmp_dir / "sample.py"
    p.write_text(code)
    return p


@pytest.fixture
def sample_ts_file(tmp_dir):
    code = '''import { readFileSync } from "fs";

interface Config {
    host: string;
    port: number;
}

export class Server {
    private config: Config;

    constructor(config: Config) {
        this.config = config;
    }

    start(): void {
        console.log(`Starting on ${this.config.host}:${this.config.port}`);
    }
}

export function loadConfig(path: string): Config {
    const raw = readFileSync(path, "utf-8");
    return JSON.parse(raw);
}
'''
    p = tmp_dir / "sample.ts"
    p.write_text(code)
    return p
