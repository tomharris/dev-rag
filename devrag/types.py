from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A single indexed chunk of code or text."""
    id: str
    text: str
    metadata: dict[str, str | int | float | bool | list[str]]
    embedding: list[float] | None = None


@dataclass
class SearchResult:
    """A single search result with score and source chunk."""
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, str | int | float | bool | list[str]]


@dataclass
class QueryResult:
    """Results from a vector store query."""
    ids: list[str]
    documents: list[str]
    metadatas: list[dict]
    distances: list[float]


@dataclass
class IndexStats:
    """Statistics from an indexing run."""
    files_scanned: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    files_removed: int = 0
    chunks_created: int = 0


@dataclass
class DocIndexStats:
    """Statistics from a document indexing run."""
    files_scanned: int = 0
    files_indexed: int = 0
    chunks_created: int = 0


@dataclass
class PRSyncStats:
    """Statistics from a PR sync run."""
    prs_fetched: int = 0
    prs_indexed: int = 0
    prs_skipped: int = 0
    chunks_created: int = 0


@dataclass
class IssueSyncStats:
    """Statistics from an issue sync run."""
    issues_fetched: int = 0
    issues_indexed: int = 0
    issues_skipped: int = 0
    chunks_created: int = 0


@dataclass
class JiraSyncStats:
    """Statistics from a Jira ticket sync run."""
    tickets_fetched: int = 0
    tickets_indexed: int = 0
    tickets_skipped: int = 0
    chunks_created: int = 0


@dataclass
class SliteSyncStats:
    """Statistics from a Slite page sync run."""
    pages_fetched: int = 0
    pages_indexed: int = 0
    pages_skipped: int = 0
    chunks_created: int = 0
