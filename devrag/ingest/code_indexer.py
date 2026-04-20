from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import tree_sitter_language_pack as tslp
from tree_sitter import Node, Parser

from devrag.config import CodeConfig
from devrag.stores.metadata_db import MetadataDB
from devrag.stores.qdrant_store import QdrantStore
from devrag.types import Chunk, IndexStats
from devrag.utils.git import discover_files

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4

# ---------------------------------------------------------------------------
# Language configuration
# ---------------------------------------------------------------------------

LANGUAGE_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".rs": "rust",
    ".go": "go",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "c_sharp",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".lua": "lua",
    ".r": "r",
    ".R": "r",
    ".jl": "julia",
    ".ex": "elixir",
    ".exs": "elixir",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".mli": "ocaml",
    ".sh": "bash",
    ".bash": "bash",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
}

# Node types that represent named entities we want to extract per language.
# Values are tuples of (node_type, is_function_like).
ENTITY_NODE_TYPES: dict[str, list[str]] = {
    "python": [
        "function_definition",
        "class_definition",
        "decorated_definition",
    ],
    "javascript": [
        "function_declaration",
        "function_expression",
        "arrow_function",
        "class_declaration",
        "class_expression",
        "method_definition",
        "export_statement",
    ],
    "typescript": [
        "function_declaration",
        "function_expression",
        "arrow_function",
        "class_declaration",
        "class_expression",
        "method_definition",
        "interface_declaration",
        "type_alias_declaration",
    ],
    "tsx": [
        "function_declaration",
        "function_expression",
        "arrow_function",
        "class_declaration",
        "class_expression",
        "method_definition",
        "interface_declaration",
        "type_alias_declaration",
    ],
    "rust": [
        "function_item",
        "struct_item",
        "enum_item",
        "impl_item",
        "trait_item",
        "mod_item",
    ],
    "go": [
        "function_declaration",
        "method_declaration",
        "type_declaration",
    ],
}


# ---------------------------------------------------------------------------
# Parser cache
# ---------------------------------------------------------------------------

_parser_cache: dict[str, Parser] = {}


def _get_parser(language: str) -> Parser | None:
    """Get (or create and cache) a tree-sitter Parser for *language*."""
    if language in _parser_cache:
        return _parser_cache[language]
    try:
        lang = tslp.get_language(language)
        parser = Parser(lang)
        _parser_cache[language] = parser
        return parser
    except Exception as exc:
        logger.info("Cannot load tree-sitter grammar for %r: %s", language, exc)
        return None


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _get_entity_name(node: Node, language: str) -> str | None:
    """Return the identifier/name for an entity node, or None."""
    # Direct name field (works for most languages)
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return name_node.text.decode("utf-8", errors="replace")

    # TypeScript/JS export_statement wraps the real declaration
    if node.type == "export_statement":
        for child in node.children:
            if child.type not in ("export", "default", "declare"):
                inner = _get_entity_name(child, language)
                if inner:
                    return inner

    # Go method_declaration: receiver + function name
    if node.type == "method_declaration":
        field_id = node.child_by_field_name("name")
        if field_id:
            return field_id.text.decode("utf-8", errors="replace")

    # Go type_declaration wraps type_spec
    if node.type == "type_declaration":
        for child in node.children:
            if child.type == "type_spec":
                spec_name = child.child_by_field_name("name")
                if spec_name:
                    return spec_name.text.decode("utf-8", errors="replace")

    return None


def _get_entity_type(node: Node) -> str:
    """Return a normalised entity type string."""
    t = node.type
    # Normalise to simpler names
    if t in ("function_definition", "function_declaration", "function_item",
             "function_expression", "arrow_function", "method_definition",
             "method_declaration"):
        return t  # keep specific; tests check membership in set
    if t in ("class_definition", "class_declaration", "class_expression"):
        return t
    return t


def _find_parent_class(node: Node, language: str) -> str | None:
    """Walk up the tree to find an enclosing class, if any."""
    class_types = {
        "python": {"class_definition"},
        "javascript": {"class_declaration", "class_expression"},
        "typescript": {"class_declaration", "class_expression"},
        "tsx": {"class_declaration", "class_expression"},
        "rust": {"impl_item", "trait_item"},
        "go": set(),
    }
    enclosing = class_types.get(language, set())
    parent = node.parent
    while parent is not None:
        if parent.type in enclosing:
            return _get_entity_name(parent, language)
        parent = parent.parent
    return None


def _node_to_text(node: Node, source: bytes) -> str:
    """Extract the source text for a node."""
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _get_signature(node: Node, source_bytes: bytes) -> str:
    """Return the first line of an entity node as its signature."""
    text = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
    return text.split("\n")[0].strip()


def _make_chunk_id(file_path: str, entity_name: str, line_start: int, repo: str = "") -> str:
    """Deterministic SHA-256-based chunk ID."""
    if repo:
        key = f"{repo}:{file_path}:{entity_name}:{line_start}"
    else:
        key = f"{file_path}:{entity_name}:{line_start}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _collect_entity_nodes(
    root: Node,
    target_types: list[str],
    language: str,
) -> list[Node]:
    """BFS/DFS walk collecting nodes whose type is in *target_types*.

    We stop descending into entity nodes to avoid double-collecting nested
    entities (e.g. methods inside a class) — EXCEPT we *do* descend into
    class bodies so that methods are included.
    """
    results: list[Node] = []
    class_body_types = {
        "block",           # Python
        "class_body",      # JS/TS
        "declaration_list",  # Rust impl
        "field_declaration_list",  # Rust struct
    }

    def walk(node: Node, inside_class: bool) -> None:
        # Treat export_statement as transparent wrapper — descend into
        # children so the real declaration is collected, not the wrapper.
        if node.type == "export_statement":
            for child in node.children:
                walk(child, inside_class=False)
            return
        if node.type in target_types:
            name = _get_entity_name(node, language)
            if name:
                results.append(node)
                # Descend into children only for class-like nodes
                is_class = "class" in node.type or node.type in (
                    "impl_item", "trait_item", "type_declaration"
                )
                if is_class:
                    for child in node.children:
                        walk(child, inside_class=True)
                return  # don't double-count children of functions
        # Descend normally
        for child in node.children:
            walk(child, inside_class)

    walk(root, inside_class=False)
    return results


# ---------------------------------------------------------------------------
# Public extraction function
# ---------------------------------------------------------------------------

def extract_chunks_from_file(
    file_path: Path,
    max_tokens: int = 512,
    repo_name: str = "",
) -> list[Chunk]:
    """Parse *file_path* with tree-sitter and return a list of Chunks.

    Returns an empty list for unsupported file types or parse errors.
    """
    suffix = file_path.suffix.lower()
    # Keep original suffix for lookup (e.g. .R vs .r)
    if suffix not in LANGUAGE_EXTENSIONS and file_path.suffix not in LANGUAGE_EXTENSIONS:
        return []
    language = LANGUAGE_EXTENSIONS.get(suffix) or LANGUAGE_EXTENSIONS.get(file_path.suffix)
    if language is None:
        return []

    parser = _get_parser(language)
    if parser is None:
        return []

    try:
        source = file_path.read_bytes()
    except OSError as exc:
        logger.warning("Cannot read %s: %s", file_path, exc)
        return []

    try:
        tree = parser.parse(source)
    except Exception as exc:
        logger.warning("Parse error for %s: %s", file_path, exc)
        return []

    target_types = ENTITY_NODE_TYPES.get(language)
    if not target_types:
        # Fallback: whole-file chunk
        return _whole_file_chunk(file_path, source, language, repo_name, max_tokens)

    entity_nodes = _collect_entity_nodes(tree.root_node, target_types, language)

    if not entity_nodes:
        return _whole_file_chunk(file_path, source, language, repo_name, max_tokens)

    chunks: list[Chunk] = []
    str_file_path = str(file_path)

    for node in entity_nodes:
        entity_name = _get_entity_name(node, language)
        if not entity_name:
            continue

        entity_type = _get_entity_type(node)
        parent_class = _find_parent_class(node, language)

        line_start = node.start_point[0] + 1  # 1-based
        line_end = node.end_point[0] + 1

        raw_text = _node_to_text(node, source)
        if not raw_text.strip():
            continue

        # Add class context prefix for methods
        if parent_class:
            text = f"# In class {parent_class}\n{raw_text}"
        else:
            text = raw_text

        # Truncate if exceeds max_tokens
        max_chars = max_tokens * CHARS_PER_TOKEN
        if len(text) > max_chars:
            text = text[:max_chars] + "\n# ... (truncated)"

        chunk_id = _make_chunk_id(str_file_path, entity_name, line_start, repo=repo_name)

        metadata: dict[str, Any] = {
            "file_path": str_file_path,
            "language": language,
            "entity_name": entity_name,
            "entity_type": entity_type,
            "line_range": f"{line_start}-{line_end}",
            "line_start": line_start,
            "line_end": line_end,
            "signature": _get_signature(node, source),
        }
        if parent_class:
            metadata["parent_entity"] = parent_class
        if repo_name:
            metadata["repo"] = repo_name

        chunks.append(Chunk(id=chunk_id, text=text, metadata=metadata))

    return chunks


def _whole_file_chunk(
    file_path: Path,
    source: bytes,
    language: str,
    repo_name: str,
    max_tokens: int = 512,
) -> list[Chunk]:
    """Return a single whole-file Chunk when no entity nodes are found."""
    str_file_path = str(file_path)
    text = source.decode("utf-8", errors="replace")
    if not text.strip():
        return []
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) > max_chars:
        text = text[:max_chars] + "\n# ... (truncated)"
    chunk_id = _make_chunk_id(str_file_path, "__file__", 1, repo=repo_name)
    metadata: dict[str, Any] = {
        "file_path": str_file_path,
        "language": language,
        "entity_name": file_path.name,
        "entity_type": "file",
        "line_range": f"1-{text.count(chr(10)) + 1}",
        "line_start": 1,
        "line_end": text.count("\n") + 1,
    }
    if repo_name:
        metadata["repo"] = repo_name
    return [Chunk(id=chunk_id, text=text, metadata=metadata)]


# ---------------------------------------------------------------------------
# CodeIndexer class
# ---------------------------------------------------------------------------

_DEFAULT_EXCLUDE = [
    "*.min.js",
    "vendor/**",
    "node_modules/**",
    "*.lock",
    "*.generated.*",
    "__pycache__/**",
    "*.pyc",
    ".git/**",
]

_COLLECTION = "code_chunks"


class CodeIndexer:
    """Indexes a repository's source code into the vector store + MetadataDB."""

    def __init__(
        self,
        store: QdrantStore,
        meta: MetadataDB,
        embedder: Any,
        sparse_encoder: Any,
        config: CodeConfig | None = None,
    ) -> None:
        self._store = store
        self._meta = meta
        self._embedder = embedder
        self._sparse_encoder = sparse_encoder
        self._config = config or CodeConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_repo(
        self,
        repo_path: Path,
        incremental: bool = False,
        repo_name: str = "",
    ) -> IndexStats:
        """Scan *repo_path* and index all supported source files.

        When *incremental* is True, files whose SHA-256 content hash has not
        changed since the last run are skipped.
        """
        repo_name = repo_name or repo_path.name

        self._meta.register_repo(repo_name, str(repo_path))

        stats = IndexStats()
        exclude = list(self._config.exclude_patterns) + _DEFAULT_EXCLUDE

        files = discover_files(repo_path, exclude_patterns=exclude)
        # Only keep files with supported extensions
        supported_files = [
            f for f in files
            if f.suffix.lower() in LANGUAGE_EXTENSIONS or f.suffix in LANGUAGE_EXTENSIONS
        ]
        stats.files_scanned = len(supported_files)

        current_paths = {str(f) for f in supported_files}

        # Detect removed files — scoped to this repo only
        previously_indexed = set(self._meta.get_indexed_files_for_repo(repo_name))
        removed = previously_indexed - current_paths
        for removed_path in removed:
            self._remove_file(removed_path, repo=repo_name)
            stats.files_removed += 1

        # Index / skip each file
        for file_path in supported_files:
            str_path = str(file_path)
            file_hash = self._hash_file(file_path)

            if incremental:
                stored_hash = self._meta.get_file_hash(str_path, repo=repo_name)
                if stored_hash == file_hash:
                    stats.files_skipped += 1
                    continue

            chunks = extract_chunks_from_file(
                file_path,
                max_tokens=self._config.chunk_max_tokens,
                repo_name=repo_name,
            )
            if not chunks:
                stats.files_empty += 1
                logger.info("No chunks extracted from %s", file_path)
                continue

            self._index_chunks(chunks, str_path, file_hash, repo=repo_name)
            stats.files_indexed += 1
            stats.chunks_created += len(chunks)

        return stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _hash_file(self, file_path: Path) -> str:
        h = hashlib.sha256()
        h.update(file_path.read_bytes())
        return h.hexdigest()

    def _remove_file(self, file_path: str, repo: str = "") -> None:
        chunk_ids = self._meta.get_chunks_for_file(file_path, repo=repo)
        if chunk_ids:
            self._store.delete(_COLLECTION, chunk_ids)
        self._meta.remove_file(file_path, repo=repo)

    def _index_chunks(
        self,
        chunks: list[Chunk],
        file_path: str,
        file_hash: str,
        repo: str = "",
    ) -> None:
        chunks = [c for c in chunks if c.text.strip()]
        if not chunks:
            return

        # Remove old chunks for this file first (handles re-indexing)
        old_chunk_ids = self._meta.get_chunks_for_file(file_path, repo=repo)
        if old_chunk_ids:
            self._store.delete(_COLLECTION, old_chunk_ids)
            self._meta.remove_file(file_path, repo=repo)

        # Deduplicate by chunk ID (keep last occurrence) as safety net
        seen: dict[str, int] = {}
        for i, c in enumerate(chunks):
            seen[c.id] = i
        if len(seen) < len(chunks):
            chunks = [chunks[i] for i in sorted(seen.values())]

        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed(texts)
        sparse_embeddings = self._sparse_encoder.encode(texts)

        ids = [c.id for c in chunks]
        metadatas = [c.metadata for c in chunks]

        self._store.upsert(
            collection=_COLLECTION,
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            sparse_embeddings=sparse_embeddings,
            wait=False,
        )

        for chunk in chunks:
            self._meta.set_chunk_source(
                chunk_id=chunk.id,
                file_path=file_path,
                line_start=chunk.metadata.get("line_start", 0),
                line_end=chunk.metadata.get("line_end", 0),
                repo=repo,
            )

        self._meta.set_file_hash(file_path, file_hash, repo=repo)
