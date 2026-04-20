from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class EmbeddingConfig:
    model: str = "nomic-embed-text"
    provider: str = "ollama"
    ollama_url: str = "http://localhost:11434"
    batch_size: int = 64
    max_tokens: int = 8192


@dataclass
class SparseEmbeddingConfig:
    model: str = "Qdrant/bm25"
    batch_size: int = 64


@dataclass
class VectorStoreConfig:
    qdrant_url: str = "http://localhost:6333"
    qdrant_path: str = ""
    embedding_dim: int = 768


@dataclass
class RetrievalConfig:
    top_k: int = 20
    final_k: int = 5
    rerank: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rrf_k: int = 60
    max_per_source: int = 2


@dataclass
class CodeConfig:
    chunk_max_tokens: int = 512
    respect_gitignore: bool = True
    exclude_patterns: list[str] = field(default_factory=lambda: [
        "*.min.js",
        "vendor/**",
        "node_modules/**",
        "*.lock",
        "*.generated.*",
    ])


@dataclass
class PrsConfig:
    github_token_env: str = "GITHUB_TOKEN"
    backfill_days: int = 90
    include_draft: bool = False
    chunk_max_tokens: int = 512


@dataclass
class IssuesConfig:
    github_token_env: str = "GITHUB_TOKEN"
    backfill_days: int = 90
    chunk_max_tokens: int = 512
    include_labels: list[str] = field(default_factory=list)
    exclude_labels: list[str] = field(default_factory=list)


@dataclass
class JiraConfig:
    jira_token_env: str = "JIRA_TOKEN"
    jira_email_env: str = "JIRA_EMAIL"
    instance_url: str = ""
    jql: str = ""
    backfill_days: int = 90
    chunk_max_tokens: int = 512


@dataclass
class SliteConfig:
    slite_token_env: str = "SLITE_TOKEN"
    channel_ids: list[str] = field(default_factory=list)
    backfill_days: int = 90
    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 50


@dataclass
class DocumentsConfig:
    glob_patterns: list[str] = field(default_factory=lambda: [
        "**/*.md", "**/*.mdx", "**/*.txt", "**/*.rst", "**/*.html", "**/*.adoc",
    ])
    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 50


@dataclass
class SessionsConfig:
    logs_dir: str = "~/.claude/projects"
    backfill_days: int = 60
    chunk_max_tokens: int = 512


@dataclass
class DevragConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    sparse_embedding: SparseEmbeddingConfig = field(default_factory=SparseEmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    code: CodeConfig = field(default_factory=CodeConfig)
    prs: PrsConfig = field(default_factory=PrsConfig)
    issues: IssuesConfig = field(default_factory=IssuesConfig)
    jira: JiraConfig = field(default_factory=JiraConfig)
    slite: SliteConfig = field(default_factory=SliteConfig)
    documents: DocumentsConfig = field(default_factory=DocumentsConfig)
    sessions: SessionsConfig = field(default_factory=SessionsConfig)


def _merge_dict_into_dataclass(dc: object, overrides: dict) -> None:
    """Recursively merge a dict of overrides into a dataclass instance."""
    for key, value in overrides.items():
        if not hasattr(dc, key):
            continue
        current = getattr(dc, key)
        if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
            _merge_dict_into_dataclass(current, value)
        else:
            setattr(dc, key, value)


def load_config(
    project_dir: Path | None = None,
    user_config_dir: Path | None = None,
) -> DevragConfig:
    """Load config with priority: project .devrag.yaml > user config > defaults."""
    config = DevragConfig()

    if user_config_dir is None:
        xdg = Path.home() / ".config" / "devrag"
        user_config_dir = xdg

    user_config_path = user_config_dir / "devrag.yaml"
    if user_config_path.exists():
        with open(user_config_path) as f:
            data = yaml.safe_load(f) or {}
        _merge_dict_into_dataclass(config, data)

    if project_dir is not None:
        project_config_path = project_dir / ".devrag.yaml"
        if project_config_path.exists():
            with open(project_config_path) as f:
                data = yaml.safe_load(f) or {}
            _merge_dict_into_dataclass(config, data)

    return config
