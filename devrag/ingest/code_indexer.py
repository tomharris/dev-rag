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

# Files above this size are skipped: generated typed-DataSets, minified bundles
# and vendored blobs cost real parse time and yield one truncated chunk.
MAX_FILE_BYTES = 2_000_000

# A declaration's doc comment is trimmed to this many lines (from the front,
# keeping the lines nearest the declaration), and may claim at most this share
# of the chunk's char budget so it can never truncate away the code it documents.
DOC_COMMENT_MAX_LINES = 30
DOC_COMMENT_BUDGET_RATIO = 0.25

# A file header shorter than this is boilerplate (a shebang, `# tmp`), not
# documentation, and would only add a near-empty chunk per file.
FILE_HEADER_MIN_CHARS = 40

# Phrases that only appear in a license *grant*, not in prose that happens to
# name a license. Vendored files carry near-identical full license texts, so
# without this every one of them contributes a duplicate header chunk.
_LICENSE_HEADER_MARKERS = (
    "permission is hereby granted",
    "redistribution and use in source and binary",
    "licensed under the apache license",
    "gnu general public license",
    "without warranties or conditions of any kind",
    "spdx-license-identifier",
)

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
    ".cs": "csharp",
    ".vb": "vb",
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
    ".tf": "terraform",
    ".tfvars": "terraform",
    # web.config / app.config. The `xml` grammar exposes no node with a `name`
    # field, so these can only ever produce a whole-file chunk.
    ".config": "xml",
}

# Node types that represent named entities we want to extract per language.
# A language absent from this map falls back to a single truncated whole-file
# chunk, so adding a LANGUAGE_EXTENSIONS entry without one here is half a feature.
# The line drawn is: declares a member with a signature or body = entity;
# binds a value (field, const, enum member) = not an entity.
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
    "terraform": [
        "block",
    ],
    # Namespaces are deliberately absent for VB/C#: a namespace is not a
    # container we descend into, so matching it would collapse a whole file to
    # one truncated namespace chunk and lose every class and method in it.
    "vb": [
        "class_block",
        "module_block",
        "structure_block",
        "interface_block",
        "enum_block",
        "method_declaration",
        "property_declaration",
        "event_declaration",
    ],
    "csharp": [
        "class_declaration",
        "struct_declaration",
        "record_declaration",
        "interface_declaration",
        "enum_declaration",
        "method_declaration",
        "constructor_declaration",
        "property_declaration",
    ],
}

# Entity nodes we still descend into, because their members are separate
# entities (methods in a class/module/struct/interface). Nodes matching
# `"class" in node.type` qualify implicitly.
_CONTAINER_NODE_TYPES = frozenset({
    "impl_item", "trait_item", "type_declaration",  # Rust / Go
    "module_block", "structure_block", "interface_block",  # VB
    "struct_declaration", "interface_declaration", "record_declaration",  # C#
})

# Wrappers that are never an entity in their own right: we descend through them
# in `_collect_entity_nodes` to reach the real declaration, and rise back
# through them in `_hoist_to_doc_anchor` to find that declaration's doc comment.
# Shared so the two directions cannot drift — `export_statement` is listed in
# ENTITY_NODE_TYPES for javascript *and* yields a truthy `_get_entity_name`, so
# neither side can infer its transparency from the tables alone.
_TRANSPARENT_WRAPPER_NODE_TYPES = frozenset({"export_statement"})


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

def _terraform_block_name(node: Node) -> str | None:
    """Build a dotted name for a Terraform ``block`` node.

    Terraform blocks have no ``name`` field; the shape is
    ``identifier string_lit* block_start body block_end``. Examples:
    ``resource "aws_s3_bucket" "foo"`` → ``resource.aws_s3_bucket.foo``;
    ``variable "region"`` → ``variable.region``; ``locals`` → ``locals``.
    """
    parts: list[str] = []
    for child in node.children:
        if child.type == "identifier":
            parts.append(child.text.decode("utf-8", errors="replace"))
        elif child.type == "string_lit":
            # string_lit wraps quoted_template_start, template_literal, quoted_template_end
            for sub in child.children:
                if sub.type == "template_literal":
                    parts.append(sub.text.decode("utf-8", errors="replace"))
                    break
        elif child.type in ("block_start", "body", "block_end"):
            break
    return ".".join(parts) if parts else None


def _get_entity_name(node: Node, language: str) -> str | None:
    """Return the identifier/name for an entity node, or None."""
    # Terraform blocks: identifier + string_lit labels, no name field
    if language == "terraform" and node.type == "block":
        return _terraform_block_name(node)

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
        "vb": {"class_block", "module_block", "structure_block", "interface_block"},
        "csharp": {
            "class_declaration", "struct_declaration",
            "record_declaration", "interface_declaration",
        },
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


def _is_comment_node(node_type: str) -> bool:
    """True for any grammar's comment node.

    Probed across every grammar in LANGUAGE_EXTENSIONS, this covers `comment`,
    `line_comment`, `block_comment`, `doc_comment` and `multiline_comment` while
    excluding `comment_content` (a *child* of Lua's `comment`) and
    `outer_doc_comment_marker` (a child of Rust's `doc_comment`). Deliberately a
    rule rather than a fifth per-language dict to maintain.
    """
    return node_type == "comment" or node_type.endswith("_comment")


def _hoist_to_doc_anchor(node: Node, target_types: list[str], language: str) -> Node:
    """Rise to the outermost wrapper that shares *node*'s end, for doc lookup.

    A doc comment precedes the whole declaration, which may be wrapped: Python's
    `decorated_definition` (the comment sits above the `@decorator`), TS/JS
    `export_statement`, and VB's `type_declaration` around `class_block`. Walking
    `prev_sibling` from the bare entity node misses the comment in all three.

    The `end_byte` equality test is what keeps this grammar-agnostic: a real
    wrapper ends exactly where its single child ends, whereas a member's parent
    (`declaration_list`, `class_body`) extends past it. We stop at any parent
    that would itself be emitted as its own chunk, so we never steal a
    container's doc comment for its first member.
    """
    current = node
    while True:
        parent = current.parent
        if parent is None or parent.end_byte != current.end_byte:
            return current
        if (
            parent.type not in _TRANSPARENT_WRAPPER_NODE_TYPES
            and parent.type in target_types
            and _get_entity_name(parent, language)
        ):
            return current
        current = parent


def _leading_doc_comment(
    node: Node,
    source: bytes,
    target_types: list[str],
    language: str,
    max_lines: int,
) -> str | None:
    """Return the comment block immediately above *node*, or None.

    tree-sitter attaches a preceding comment as an extra *sibling* in the
    parent, not as a child of the declaration it documents, so it has to be
    collected explicitly or every "doc comment above the declaration" language
    (JSDoc, Go, Rust `///`, C#/VB XML docs) loses its prose from the chunk.

    Two traps, both load-bearing:

    * The `vb` grammar emits an explicit `blank_line` node per newline, so the
      comment is never the immediate `prev_sibling` — whitespace-only siblings
      must be skipped or VB (where ~90% of files are error-recovered) gets
      nothing.
    * The blank-line gap is measured against `anchor_row`, not against the
      sibling we just skipped. Advancing the cursor over `blank_line` trivia and
      measuring from there silently defeats the check, which makes an unrelated
      comment two lines above a declaration look adjacent.
    """
    anchor = _hoist_to_doc_anchor(node, target_types, language)
    anchor_row = anchor.start_point[0]
    blocks: list[Node] = []
    current = anchor
    while True:
        prev = current.prev_sibling
        if prev is None:
            break
        if not source[prev.start_byte:prev.end_byte].strip():
            current = prev  # whitespace-only trivia (VB `blank_line`)
            continue
        if not _is_comment_node(prev.type):
            break
        # A blank line between comment and declaration means it documents
        # nothing. Measured from the anchor, never from skipped trivia.
        if anchor_row - prev.end_point[0] > 1:
            break
        blocks.append(prev)
        current = prev
        anchor_row = prev.start_point[0]

    if not blocks:
        return None
    blocks.reverse()
    # Trim from the front: the lines nearest the declaration describe it.
    while len(blocks) > 1 and blocks[-1].end_point[0] - blocks[0].start_point[0] + 1 > max_lines:
        blocks.pop(0)
    text = source[blocks[0].start_byte:blocks[-1].end_byte].decode("utf-8", errors="replace")
    # Rust's `doc_comment` span includes its trailing newline.
    return text.rstrip() or None


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
    """Iterative walk collecting nodes whose type is in *target_types*.

    We stop descending into entity nodes to avoid double-collecting nested
    entities (e.g. methods inside a class) — EXCEPT we *do* descend into
    class-like containers so that their members are included.

    Iterative rather than recursive on purpose: the walk only stops at matched
    entity nodes, so a deep expression outside one (a field initializer, or
    anything inside an ERROR node — ~90% of real VB.NET files have them) is
    walked to the bottom. Real VB files reach AST depths over 1000 via long `&`
    concatenation chains, which overflows Python's recursion limit.
    """
    results: list[Node] = []
    # Children are pushed reversed so siblings pop in document order.
    stack: list[Node] = [root]
    while stack:
        node = stack.pop()
        # Treat export_statement as transparent wrapper — descend into
        # children so the real declaration is collected, not the wrapper.
        if node.type in _TRANSPARENT_WRAPPER_NODE_TYPES:
            stack.extend(reversed(node.children))
            continue
        if node.type in target_types and _get_entity_name(node, language):
            results.append(node)
            # Descend into children only for class-like containers, so we don't
            # double-count the innards of functions.
            if "class" in node.type or node.type in _CONTAINER_NODE_TYPES:
                stack.extend(reversed(node.children))
            continue
        stack.extend(reversed(node.children))
    return results


# ---------------------------------------------------------------------------
# Public extraction function
# ---------------------------------------------------------------------------

def extract_chunks_from_file(
    file_path: Path,
    max_tokens: int = 512,
    repo_name: str = "",
    max_file_bytes: int = MAX_FILE_BYTES,
    include_doc_comments: bool = True,
    doc_comment_max_lines: int = DOC_COMMENT_MAX_LINES,
    index_file_headers: bool = True,
) -> list[Chunk]:
    """Parse *file_path* with tree-sitter and return a list of Chunks.

    Returns an empty list for unsupported file types, parse errors, or files
    larger than *max_file_bytes* (0 disables the cap).
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
        # stat() before read_bytes() so a multi-megabyte generated file is
        # never loaded into memory just to be discarded.
        if max_file_bytes:
            size = file_path.stat().st_size
            if size > max_file_bytes:
                logger.info(
                    "Skipping %s: %d bytes exceeds max_file_bytes=%d",
                    file_path, size, max_file_bytes,
                )
                return []
        source = file_path.read_bytes()
    except OSError as exc:
        logger.warning("Cannot read %s: %s", file_path, exc)
        return []

    # Strip a UTF-8 BOM before parsing: .NET source is routinely BOM'd and the
    # U+FEFF otherwise leads the first chunk's text and its signature line.
    # Only column numbers shift, not line numbers, and _hash_file reads the
    # raw bytes independently.
    if source.startswith(b"\xef\xbb\xbf"):
        source = source[3:]

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

    if index_file_headers:
        header = _file_header_chunk(
            tree.root_node, file_path, source, language, repo_name, max_tokens
        )
        if header is not None:
            chunks.append(header)

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

        max_chars = max_tokens * CHARS_PER_TOKEN

        parts: list[str] = []
        # Add class context prefix for methods
        if parent_class:
            parts.append(f"# In class {parent_class}")
        if include_doc_comments:
            doc = _leading_doc_comment(
                node, source, target_types, language, doc_comment_max_lines
            )
            if doc:
                # Cap the doc's share of the budget so a long comment block can
                # never truncate the code it documents out of its own chunk.
                doc_budget = int(max_chars * DOC_COMMENT_BUDGET_RATIO)
                if len(doc) > doc_budget:
                    doc = doc[:doc_budget].rstrip() + " …"
                parts.append(doc)
        parts.append(raw_text)
        text = "\n".join(parts)

        # Truncate if exceeds max_tokens
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


def _file_header_chunk(
    root: Node,
    file_path: Path,
    source: bytes,
    language: str,
    repo_name: str,
    max_tokens: int = 512,
) -> Chunk | None:
    """Return a chunk for the file's leading comment block / module docstring.

    Entity chunks only ever cover declarations, so a file's top-level "what is
    this for" prose belongs to no chunk at all. Collected as one chunk per file
    rather than prefixed onto every entity, which would duplicate it N times.

    Returns None when the header is only boilerplate (a shebang, an encoding
    line), which is the common case and not worth a chunk.
    """
    blocks: list[Node] = []
    for child in root.children:
        text_bytes = source[child.start_byte:child.end_byte]
        if not text_bytes.strip():
            continue
        # A bare leading `string` is Python's module docstring (a direct `module`
        # child in this grammar, not wrapped in an expression_statement).
        if _is_comment_node(child.type) or child.type == "string":
            blocks.append(child)
            continue
        break

    if not blocks:
        return None
    text = source[blocks[0].start_byte:blocks[-1].end_byte].decode(
        "utf-8", errors="replace"
    ).strip()
    # Strip a shebang / encoding line before judging substance.
    lines = [
        ln for ln in text.splitlines()
        if not ln.startswith("#!") and "coding:" not in ln
    ]
    text = "\n".join(lines).strip()
    if len(text) < FILE_HEADER_MIN_CHARS:
        return None
    lowered = text.lower()
    if any(marker in lowered for marker in _LICENSE_HEADER_MARKERS):
        return None

    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) > max_chars:
        text = text[:max_chars] + "\n# ... (truncated)"

    str_file_path = str(file_path)
    line_start = blocks[0].start_point[0] + 1
    line_end = blocks[-1].end_point[0] + 1
    metadata: dict[str, Any] = {
        "file_path": str_file_path,
        "language": language,
        "entity_name": file_path.name,
        "entity_type": "module_doc",
        "line_range": f"{line_start}-{line_end}",
        "line_start": line_start,
        "line_end": line_end,
    }
    if repo_name:
        metadata["repo"] = repo_name
    return Chunk(
        id=_make_chunk_id(str_file_path, "__module_doc__", line_start, repo=repo_name),
        text=text,
        metadata=metadata,
    )


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
    # .NET / Visual Studio generated code. The `**/` prefix is required:
    # gitignore semantics root-anchor any pattern with an internal slash, so
    # "My Project/**" would miss every nested occurrence.
    "*.Designer.vb",
    "*.Designer.cs",
    "**/My Project/**",
    "**/Web References/**",
    "**/Crystal Reports Backup Files/**",
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

        # Detect removed files — scoped to this repo and to code extensions only.
        # Docs indexed by DocIndexer share the same repo namespace in MetadataDB, so
        # we must not treat a doc file as a "removed code file" here.
        previously_indexed = set(self._meta.get_indexed_files_for_repo(repo_name))
        previously_code = {
            p for p in previously_indexed
            if Path(p).suffix.lower() in LANGUAGE_EXTENSIONS or Path(p).suffix in LANGUAGE_EXTENSIONS
        }
        removed = previously_code - current_paths
        for removed_path in removed:
            self._remove_file(removed_path, repo=repo_name)
            stats.files_removed += 1

        # Index / skip each file. A single file's failure (e.g. an embed 400 on
        # an oversized chunk) is logged and counted, not propagated — otherwise
        # one bad file aborts the whole repo, and under `index refresh` every
        # later repo too. _index_chunks persists the file hash only after a
        # successful upsert, so a failed file is retried on the next run.
        for file_path in supported_files:
            str_path = str(file_path)
            try:
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
                    max_file_bytes=self._config.max_file_bytes,
                    include_doc_comments=self._config.include_doc_comments,
                    doc_comment_max_lines=self._config.doc_comment_max_lines,
                    index_file_headers=self._config.index_file_headers,
                )
                if not chunks:
                    stats.files_empty += 1
                    logger.info("No chunks extracted from %s", file_path)
                    continue

                self._index_chunks(chunks, str_path, file_hash, repo=repo_name)
                stats.files_indexed += 1
                stats.chunks_created += len(chunks)
            except Exception as exc:
                stats.files_failed += 1
                logger.warning("Failed to index %s: %s", file_path, exc)

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
