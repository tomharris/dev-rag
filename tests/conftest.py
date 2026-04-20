import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def vector_store(tmp_dir):
    from devrag.stores.qdrant_store import QdrantStore
    return QdrantStore(path=str(tmp_dir / "qdrant"), embedding_dim=768)


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
