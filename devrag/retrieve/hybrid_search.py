from __future__ import annotations

from collections import defaultdict

from devrag.types import SearchResult


def deduplicate_results(results: list[SearchResult], max_per_source: int = 2) -> list[SearchResult]:
    """Limit results per source to prevent a single file/PR/issue dominating."""
    source_counts: dict[str, int] = defaultdict(int)
    deduped: list[SearchResult] = []
    for r in results:
        key = _source_key(r.metadata)
        if source_counts[key] < max_per_source:
            deduped.append(r)
            source_counts[key] += 1
    return deduped


def _source_key(metadata: dict) -> str:
    """Derive a grouping key from chunk metadata."""
    if "pr_number" in metadata:
        return f"pr:{metadata.get('repo', '')}:{metadata['pr_number']}"
    if "issue_number" in metadata:
        return f"issue:{metadata.get('repo', '')}:{metadata['issue_number']}"
    if "ticket_key" in metadata:
        return f"jira:{metadata['ticket_key']}"
    if "page_id" in metadata:
        return f"slite:{metadata['page_id']}"
    if "session_id" in metadata:
        return f"session:{metadata['session_id']}"
    if "file_path" in metadata:
        return f"file:{metadata.get('repo', '')}:{metadata['file_path']}"
    return f"unknown:{id(metadata)}"


class HybridSearch:
    def __init__(self, vector_store, embedder, sparse_encoder, collection: str = "code_chunks") -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder
        self.collection = collection

    def search(self, query: str, top_k: int = 20, collections: list[str] | None = None, where: dict | None = None) -> list[SearchResult]:
        if collections is None:
            collections = [self.collection]
        dense = self.embedder.embed_query(query)
        sparse = self.sparse_encoder.encode_query(query)

        all_results: list[SearchResult] = []
        for coll in collections:
            hits = self.vector_store.hybrid_query(
                collection=coll,
                dense_embedding=dense,
                sparse_embedding=sparse,
                n_results=top_k,
                where=where,
            )
            for i, doc_id in enumerate(hits.ids):
                all_results.append(SearchResult(
                    chunk_id=doc_id,
                    text=hits.documents[i],
                    score=hits.distances[i] if hits.distances else 0.0,
                    metadata=hits.metadatas[i],
                ))

        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:top_k]
