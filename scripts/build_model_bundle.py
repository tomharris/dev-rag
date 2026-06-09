"""Build the DevRAG model bundle on an HF-reachable machine.

Warms the default reranker + BM25 models into temp caches, tars them into the
hub/ + fastembed/ layout the extractor expects, and prints the sha256. Upload
the result to a dev-rag GitHub release, then update DEFAULT_BUNDLE_URL /
DEFAULT_BUNDLE_SHA256 in devrag/ingest/model_bundle.py.

Usage: uv run python scripts/build_model_bundle.py
"""
from __future__ import annotations

import hashlib
import tarfile
import tempfile
from pathlib import Path

RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BM25 = "Qdrant/bm25"
OUT = Path("dist/devrag-models.tar.gz")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        hub = Path(tmp) / "hub"
        fe = Path(tmp) / "fastembed"
        hub.mkdir()
        fe.mkdir()

        from sentence_transformers import CrossEncoder
        CrossEncoder(RERANKER, cache_folder=str(hub))

        from fastembed import SparseTextEmbedding
        m = SparseTextEmbedding(model_name=BM25, cache_dir=str(fe))
        next(m.query_embed("warm"))

        with tarfile.open(OUT, "w:gz") as tar:
            tar.add(hub, arcname="hub")
            tar.add(fe, arcname="fastembed")

    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")
    print(f"sha256: {digest}")
    print("Next: gh release create models-v1 dist/devrag-models.tar.gz  (or `gh release upload models-v1 ...`)")
    print("Then set DEFAULT_BUNDLE_SHA256 in devrag/ingest/model_bundle.py to the sha256 above.")


if __name__ == "__main__":
    main()
