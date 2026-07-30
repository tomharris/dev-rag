import hashlib
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from devrag.config import CodeConfig
from devrag.ingest import code_indexer as code_indexer_mod
from devrag.ingest.code_indexer import (
    ENTITY_NODE_TYPES,
    LANGUAGE_EXTENSIONS,
    CodeIndexer,
    extract_chunks_from_file,
)
from devrag.types import Chunk


def test_extract_chunks_from_python_file(sample_python_file):
    chunks = extract_chunks_from_file(sample_python_file, max_tokens=512)
    entity_names = [c.metadata["entity_name"] for c in chunks]
    assert "FileProcessor" in entity_names or any(
        c.metadata.get("parent_entity") == "FileProcessor" for c in chunks
    )
    assert "standalone_function" in entity_names

    func_chunk = next(c for c in chunks if c.metadata["entity_name"] == "standalone_function")
    assert func_chunk.metadata["language"] == "python"
    assert func_chunk.metadata["entity_type"] in ("function", "function_definition")
    assert "line_range" in func_chunk.metadata
    assert "def standalone_function" in func_chunk.text


def test_extract_chunks_from_typescript_file(sample_ts_file):
    chunks = extract_chunks_from_file(sample_ts_file, max_tokens=512)
    entity_names = [c.metadata["entity_name"] for c in chunks]
    assert "Server" in entity_names or "loadConfig" in entity_names
    assert any(c.metadata["language"] == "typescript" for c in chunks)


def test_extract_chunks_unsupported_language(tmp_dir):
    path = tmp_dir / "data.csv"
    path.write_text("a,b,c\n1,2,3\n")
    chunks = extract_chunks_from_file(path, max_tokens=512)
    assert chunks == []


def test_language_extensions_mapping():
    assert LANGUAGE_EXTENSIONS[".py"] == "python"
    assert LANGUAGE_EXTENSIONS[".ts"] == "typescript"
    assert LANGUAGE_EXTENSIONS[".js"] == "javascript"
    assert LANGUAGE_EXTENSIONS[".rs"] == "rust"
    assert LANGUAGE_EXTENSIONS[".go"] == "go"
    assert LANGUAGE_EXTENSIONS[".tf"] == "terraform"
    assert LANGUAGE_EXTENSIONS[".tfvars"] == "terraform"
    assert LANGUAGE_EXTENSIONS[".vb"] == "vb"
    assert LANGUAGE_EXTENSIONS[".cs"] == "csharp"
    assert LANGUAGE_EXTENSIONS[".config"] == "xml"


def test_entity_node_types_keys_are_mapped_languages():
    """Every ENTITY_NODE_TYPES key must be a grammar name some extension maps to.

    A key that no extension maps to is dead config: the language silently falls
    back to a single truncated whole-file chunk. This is exactly how `.cs` lost
    its AST chunking (mapped to "c_sharp", keyed as something else).
    """
    unreachable = set(ENTITY_NODE_TYPES) - set(LANGUAGE_EXTENSIONS.values())
    assert not unreachable, f"ENTITY_NODE_TYPES keys no extension maps to: {unreachable}"


def test_extract_chunks_from_terraform_file(tmp_dir):
    tf = tmp_dir / "main.tf"
    tf.write_text(
        'resource "aws_s3_bucket" "foo" {\n'
        '  bucket = "my-bucket"\n'
        '}\n'
        '\n'
        'variable "region" {\n'
        '  type    = string\n'
        '  default = "us-east-1"\n'
        '}\n'
        '\n'
        'module "vpc" {\n'
        '  source = "./vpc"\n'
        '}\n'
        '\n'
        'locals {\n'
        '  env = "prod"\n'
        '}\n'
    )
    chunks = extract_chunks_from_file(tf, max_tokens=512)
    names = [c.metadata["entity_name"] for c in chunks]
    assert "resource.aws_s3_bucket.foo" in names
    assert "variable.region" in names
    assert "module.vpc" in names
    assert "locals" in names
    for c in chunks:
        assert c.metadata["language"] == "terraform"
    # Each block occupies a distinct line range
    line_ranges = {c.metadata["line_range"] for c in chunks}
    assert len(line_ranges) == len(chunks)
    # Body is preserved in the chunk text
    resource_chunk = next(c for c in chunks if c.metadata["entity_name"] == "resource.aws_s3_bucket.foo")
    assert "my-bucket" in resource_chunk.text


VB_SOURCE = '''Imports System.Windows.Forms

Namespace Trax.Payroll
    Public Class PayrollForm
        Inherits System.Windows.Forms.Form

        Private Const MaxRows As Integer = 50
        Private _total As Decimal

        Public Event Recalculated(ByVal amount As Decimal)

#Region "Properties"
        Public Property Total() As Decimal
            Get
                Return _total
            End Get
            Set(ByVal value As Decimal)
                _total = value
            End Set
        End Property
#End Region

        Private Sub btnCalc_Click(ByVal sender As Object, ByVal e As EventArgs) Handles btnCalc.Click
            RecalcTotals()
        End Sub

        Public Sub RecalcTotals()
            With Me.grid
                .Refresh()
            End With
        End Sub
    End Class

    Public Module PayrollHelpers
        Public Function FormatCurrency(ByVal v As Decimal) As String
            Return v.ToString("C")
        End Function
    End Module
End Namespace
'''


def test_extract_chunks_from_vb_file(tmp_dir):
    """VB.NET entities survive the `vb` grammar's parse failures.

    The grammar cannot parse ``#Region``, ``Inherits``, ``Handles`` or ``With`` —
    this fixture has all four, so ``root_node.has_error`` is True. We rely on
    tree-sitter's error recovery to still yield correctly named entities.
    Written with a BOM (``utf-8-sig``) because .NET source routinely carries one.
    """
    p = tmp_dir / "PayrollForm.vb"
    p.write_text(VB_SOURCE, encoding="utf-8-sig")
    chunks = extract_chunks_from_file(p, max_tokens=512)
    names = [c.metadata["entity_name"] for c in chunks]

    for expected in (
        "PayrollForm", "Total", "btnCalc_Click", "RecalcTotals",
        "Recalculated", "PayrollHelpers", "FormatCurrency",
    ):
        assert expected in names, f"{expected} missing from {names}"

    for c in chunks:
        assert c.metadata["language"] == "vb"
        # Falling through to a whole-file chunk would mean AST chunking failed.
        assert c.metadata["entity_type"] != "file"

    by_name = {c.metadata["entity_name"]: c for c in chunks}
    assert by_name["btnCalc_Click"].metadata["parent_entity"] == "PayrollForm"
    assert by_name["RecalcTotals"].metadata["parent_entity"] == "PayrollForm"
    # A Module is a container: its members must not be swallowed into it.
    assert by_name["FormatCurrency"].metadata["parent_entity"] == "PayrollHelpers"

    # Consts and fields bind values; they are not entities.
    assert "MaxRows" not in names
    assert "_total" not in names

    # The BOM must not leak into chunk text or the signature line.
    for c in chunks:
        assert "﻿" not in c.text
        assert "﻿" not in c.metadata["signature"]
    assert by_name["PayrollForm"].metadata["signature"] == "Public Class PayrollForm"


CSHARP_SOURCE = '''using System;

namespace Trax.Util
{
    public class StringUtils
    {
        private const int MaxLen = 255;
        private string _name;

        public StringUtils(string name)
        {
            _name = name;
        }

        public string Name { get; set; }

        public string Other
        {
            get { return _name; }
        }

        public static string Trim(string input)
        {
            return input == null ? null : input.Trim();
        }
    }

    public interface IThing
    {
        void Do(int n);
    }

    public struct Point
    {
        public int X;
        public void Move(int dx) { X += dx; }
    }

    public enum Color { Red, Green }
}
'''


def test_extract_chunks_from_csharp_file(tmp_dir):
    """C# gets real AST chunking, not one truncated whole-file chunk.

    Regression test: `.cs` used to map to a grammar name with no
    ENTITY_NODE_TYPES entry, so every file collapsed to a single `file` chunk.
    """
    p = tmp_dir / "StringUtils.cs"
    p.write_text(CSHARP_SOURCE, encoding="utf-8-sig")
    chunks = extract_chunks_from_file(p, max_tokens=512)
    names = [c.metadata["entity_name"] for c in chunks]

    for expected in (
        "StringUtils", "Name", "Other", "Trim", "IThing", "Do", "Point", "Move", "Color",
    ):
        assert expected in names, f"{expected} missing from {names}"

    for c in chunks:
        assert c.metadata["language"] == "csharp"
        assert c.metadata["entity_type"] != "file"

    # Interfaces and structs are containers too.
    by_name = {c.metadata["entity_name"]: c for c in chunks}
    assert by_name["Do"].metadata["parent_entity"] == "IThing"
    assert by_name["Move"].metadata["parent_entity"] == "Point"

    # Consts, fields and enum members are not entities.
    for absent in ("MaxLen", "_name", "Red", "Green", "X"):
        assert absent not in names

    assert "﻿" not in by_name["StringUtils"].metadata["signature"]


def test_extract_chunks_survives_deeply_nested_expressions(tmp_dir):
    """Deep ASTs must not overflow the entity walker.

    Real VB.NET files reach AST depths over 1000 via long `&` concatenation
    chains building SQL/HTML (measured: 1036 in trax-apps' PayrollPendingCtl.vb).
    The walker only stops descending at matched entity nodes, so a deep
    expression that sits outside one — in a field initializer here, inside an
    ERROR node in the real file — is walked to the bottom. A recursive walker
    raises RecursionError and the file is lost on every run.
    """
    chain = " & ".join(f'"p{i}"' for i in range(2000))
    p = tmp_dir / "Deep.vb"
    p.write_text(
        "Public Class Big\n"
        "    Inherits Form\n"
        f"    Private s As String = {chain}\n"
        "End Class\n"
    )
    chunks = extract_chunks_from_file(p, max_tokens=512)
    names = [c.metadata["entity_name"] for c in chunks]
    assert "Big" in names


def test_extract_chunks_skips_oversized_file(tmp_dir):
    """Files over max_file_bytes are skipped; 0 disables the cap."""
    p = tmp_dir / "huge.py"
    p.write_text("def f():\n    return 1\n" + ("# pad\n" * 20_000))
    assert extract_chunks_from_file(p, max_tokens=512, max_file_bytes=1000) == []
    assert extract_chunks_from_file(p, max_tokens=512, max_file_bytes=0) != []


def test_chunk_ids_are_deterministic(sample_python_file):
    chunks1 = extract_chunks_from_file(sample_python_file, max_tokens=512)
    chunks2 = extract_chunks_from_file(sample_python_file, max_tokens=512)
    ids1 = [c.id for c in chunks1]
    ids2 = [c.id for c in chunks2]
    assert ids1 == ids2


def test_chunk_text_includes_context(sample_python_file):
    chunks = extract_chunks_from_file(sample_python_file, max_tokens=512)
    method_chunks = [c for c in chunks if c.metadata.get("parent_entity") == "FileProcessor"]
    if method_chunks:
        assert any("FileProcessor" in c.text or "read_file" in c.text for c in method_chunks)


# --- Integration tests for CodeIndexer class ---

from devrag.ingest.code_indexer import CodeIndexer
from devrag.stores.metadata_db import MetadataDB


@pytest.fixture
def indexer_deps(tmp_dir, vector_store, sparse_encoder):
    meta = MetadataDB(str(tmp_dir / "meta.db"))
    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])
    return vector_store, meta, embedder, sparse_encoder


def test_code_indexer_indexes_repo(tmp_dir, indexer_deps):
    store, meta, embedder, sparse_encoder = indexer_deps
    repo = tmp_dir / "repo"
    repo.mkdir()
    import subprocess
    subprocess.run(["git", "init", str(repo)], capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=str(repo), capture_output=True)
    (repo / "main.py").write_text("def hello():\n    return 'world'\n")
    (repo / "utils.py").write_text("def add(a, b):\n    return a + b\n")
    subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), capture_output=True)
    indexer = CodeIndexer(store, meta, embedder, sparse_encoder)
    stats = indexer.index_repo(repo)
    assert stats.files_scanned >= 2
    assert stats.files_indexed >= 2
    assert stats.chunks_created >= 2
    assert store.count("code_chunks") >= 2


def test_code_indexer_incremental_skips_unchanged(tmp_dir, indexer_deps):
    store, meta, embedder, sparse_encoder = indexer_deps
    repo = tmp_dir / "repo"
    repo.mkdir()
    import subprocess
    subprocess.run(["git", "init", str(repo)], capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=str(repo), capture_output=True)
    (repo / "main.py").write_text("def hello():\n    return 'world'\n")
    subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), capture_output=True)
    indexer = CodeIndexer(store, meta, embedder, sparse_encoder)
    stats1 = indexer.index_repo(repo)
    assert stats1.files_indexed >= 1
    embedder.embed.reset_mock()
    stats2 = indexer.index_repo(repo, incremental=True)
    assert stats2.files_skipped >= 1
    assert stats2.files_indexed == 0
    embedder.embed.assert_not_called()


def test_code_indexer_isolates_failing_file(tmp_dir, indexer_deps):
    """One file that fails to embed is counted and skipped, not fatal — the rest
    of the repo still indexes, and the failed file is retried next run (no hash
    persisted)."""
    store, meta, embedder, sparse_encoder = indexer_deps
    repo = tmp_dir / "repo"
    repo.mkdir()
    import subprocess
    subprocess.run(["git", "init", str(repo)], capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=str(repo), capture_output=True)
    (repo / "good.py").write_text("def hello():\n    return 'world'\n")
    (repo / "bad.py").write_text("def explode():\n    return 'boom'\n")
    subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), capture_output=True)

    def embed(texts):
        if any("boom" in t for t in texts):
            raise RuntimeError("Ollama embed failed (400): input length exceeds context length")
        return [[0.1] * 768 for _ in texts]

    embedder.embed = MagicMock(side_effect=embed)
    indexer = CodeIndexer(store, meta, embedder, sparse_encoder)
    stats = indexer.index_repo(repo)

    assert stats.files_failed == 1
    assert stats.files_indexed >= 1  # good.py still indexed
    assert meta.get_file_hash(str(repo / "bad.py"), repo=repo.name) is None  # retried next run


def test_extract_chunks_skips_empty_text_nodes(tmp_dir):
    """Nodes whose source text is empty/whitespace-only should be excluded."""
    code = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
    p = tmp_dir / "test.py"
    p.write_text(code)

    # Baseline: both functions produce chunks
    normal_chunks = extract_chunks_from_file(p, max_tokens=512)
    assert len(normal_chunks) == 2

    # Simulate an empty-text node by patching _node_to_text to return
    # whitespace for the first entity while leaving the rest unchanged.
    original_fn = code_indexer_mod._node_to_text
    call_count = 0

    def _fake_node_to_text(node, source):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "   "
        return original_fn(node, source)

    with patch.object(code_indexer_mod, "_node_to_text", side_effect=_fake_node_to_text):
        filtered_chunks = extract_chunks_from_file(p, max_tokens=512)

    assert len(filtered_chunks) == 1
    assert filtered_chunks[0].metadata["entity_name"] == "bar"


def test_exported_ts_declarations_no_duplicate_ids(tmp_dir):
    """Exported TS declarations should not produce duplicate chunk IDs."""
    code = '''export function processData(input: string): string {
    return input.trim();
}

export class DataProcessor {
    run(): void {
        console.log("running");
    }
}

export interface Config {
    host: string;
}
'''
    p = tmp_dir / "exports.ts"
    p.write_text(code)
    chunks = extract_chunks_from_file(p, max_tokens=512)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), f"Duplicate chunk IDs found: {ids}"
    entity_names = [c.metadata["entity_name"] for c in chunks]
    assert "processData" in entity_names
    assert "DataProcessor" in entity_names


def test_whole_file_chunk_skips_empty_file(tmp_dir):
    """Empty files should produce no chunks."""
    p = tmp_dir / "empty.py"
    p.write_text("")
    chunks = extract_chunks_from_file(p, max_tokens=512)
    assert chunks == []


def test_whole_file_chunk_skips_whitespace_only_file(tmp_dir):
    """Files containing only whitespace should produce no chunks."""
    p = tmp_dir / "blank.py"
    p.write_text("   \n\n  \t  \n")
    chunks = extract_chunks_from_file(p, max_tokens=512)
    assert chunks == []


def test_code_indexer_skips_empty_file(tmp_dir, indexer_deps):
    """Empty files should not cause embedding or upsert errors."""
    store, meta, embedder, sparse_encoder = indexer_deps
    repo = tmp_dir / "repo"
    repo.mkdir()
    import subprocess
    subprocess.run(["git", "init", str(repo)], capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=str(repo), capture_output=True)
    (repo / "empty.py").write_text("")
    (repo / "real.py").write_text("def hello():\n    return 'world'\n")
    subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), capture_output=True)
    indexer = CodeIndexer(store, meta, embedder, sparse_encoder)
    stats = indexer.index_repo(repo)
    assert stats.files_indexed >= 1
    assert stats.chunks_created >= 1


def test_code_indexer_skips_oversized_file(tmp_dir, indexer_deps):
    """An oversized file is counted in files_empty, not files_failed."""
    store, meta, embedder, sparse_encoder = indexer_deps
    repo = tmp_dir / "repo"
    repo.mkdir()
    import subprocess
    subprocess.run(["git", "init", str(repo)], capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=str(repo), capture_output=True)
    (repo / "huge.py").write_text("def big():\n    return 1\n" + ("# pad\n" * 20_000))
    (repo / "real.py").write_text("def hello():\n    return 'world'\n")
    subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), capture_output=True)

    config = CodeConfig(max_file_bytes=1000)
    indexer = CodeIndexer(store, meta, embedder, sparse_encoder, config=config)
    stats = indexer.index_repo(repo)
    assert stats.files_empty == 1
    assert stats.files_failed == 0
    assert stats.files_indexed == 1


@pytest.mark.skipif(
    os.environ.get("SKIP_INTEGRATION", "1") == "1",
    reason="Set SKIP_INTEGRATION=0 to run; downloads tree-sitter grammars on a cold cache",
)
def test_every_language_extension_resolves():
    """Every grammar name in LANGUAGE_EXTENSIONS must actually load.

    A typo'd name is caught by _get_parser's bare except, logged only at INFO,
    and the extension silently indexes nothing.
    """
    import tree_sitter_language_pack as tslp
    for lang in sorted(set(LANGUAGE_EXTENSIONS.values())):
        assert tslp.get_language(lang) is not None, lang


def test_code_indexer_multi_repo_no_cross_deletion(tmp_dir, indexer_deps):
    """Indexing repo-b should not delete repo-a's data."""
    store, meta, embedder, sparse_encoder = indexer_deps
    import subprocess

    # Create repo-a
    repo_a = tmp_dir / "repo-a"
    repo_a.mkdir()
    subprocess.run(["git", "init", str(repo_a)], capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=str(repo_a), capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=str(repo_a), capture_output=True)
    (repo_a / "main.py").write_text("def hello_a():\n    return 'a'\n")
    subprocess.run(["git", "add", "."], cwd=str(repo_a), capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo_a), capture_output=True)

    # Create repo-b
    repo_b = tmp_dir / "repo-b"
    repo_b.mkdir()
    subprocess.run(["git", "init", str(repo_b)], capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=str(repo_b), capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=str(repo_b), capture_output=True)
    (repo_b / "app.py").write_text("def hello_b():\n    return 'b'\n")
    subprocess.run(["git", "add", "."], cwd=str(repo_b), capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo_b), capture_output=True)

    indexer = CodeIndexer(store, meta, embedder, sparse_encoder)
    stats_a = indexer.index_repo(repo_a, repo_name="repo-a")
    assert stats_a.files_indexed >= 1
    count_after_a = store.count("code_chunks")

    stats_b = indexer.index_repo(repo_b, repo_name="repo-b")
    assert stats_b.files_indexed >= 1
    assert stats_b.files_removed == 0  # Must not remove repo-a's files

    # Both repos' chunks should be present
    assert store.count("code_chunks") >= count_after_a + stats_b.chunks_created

    # Repo registry should have both
    repos = meta.get_all_repos()
    repo_names = {r[0] for r in repos}
    assert "repo-a" in repo_names
    assert "repo-b" in repo_names


def test_code_indexer_detects_removed_files(tmp_dir, indexer_deps):
    store, meta, embedder, sparse_encoder = indexer_deps
    repo = tmp_dir / "repo"
    repo.mkdir()
    import subprocess
    subprocess.run(["git", "init", str(repo)], capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=str(repo), capture_output=True)
    (repo / "main.py").write_text("def hello():\n    return 'world'\n")
    (repo / "old.py").write_text("def old():\n    pass\n")
    subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), capture_output=True)
    indexer = CodeIndexer(store, meta, embedder, sparse_encoder)
    indexer.index_repo(repo)
    initial_count = store.count("code_chunks")
    (repo / "old.py").unlink()
    subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "commit", "-m", "remove old"], cwd=str(repo), capture_output=True)
    stats = indexer.index_repo(repo)
    assert stats.files_removed >= 1
    assert store.count("code_chunks") < initial_count


def test_code_and_docs_coexist_without_cross_deletion(tmp_dir, indexer_deps):
    """Code and docs share the (repo, file_path) namespace; neither indexer's
    removed-file detection may delete the other's rows."""
    import subprocess

    from devrag.ingest.doc_indexer import DocIndexer

    store, meta, embedder, sparse_encoder = indexer_deps
    repo = tmp_dir / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=str(repo), capture_output=True)
    (repo / "main.py").write_text("def hello():\n    return 'world'\n")
    (repo / "README.md").write_text("# Project\n\nDocs here.\n")

    code = CodeIndexer(store, meta, embedder, sparse_encoder)
    docs = DocIndexer(store, meta, embedder, sparse_encoder)

    code.index_repo(repo, repo_name="r")
    docs.index_repo_docs(repo, repo_name="r")

    readme_chunks = meta.get_chunks_for_file(str(repo / "README.md"), repo="r")
    main_chunks = meta.get_chunks_for_file(str(repo / "main.py"), repo="r")
    assert readme_chunks and main_chunks

    # Re-running the code indexer must NOT see README.md as a removed code file.
    stats = code.index_repo(repo, incremental=True, repo_name="r")
    assert stats.files_removed == 0
    assert meta.get_chunks_for_file(str(repo / "README.md"), repo="r") == readme_chunks

    # Re-running the doc indexer must NOT see main.py as a removed doc file.
    doc_stats = docs.index_repo_docs(repo, repo_name="r", incremental=True)
    assert doc_stats.files_removed == 0
    assert meta.get_chunks_for_file(str(repo / "main.py"), repo="r") == main_chunks


# --- Leading doc-comment capture ---
#
# tree-sitter parses a comment that precedes a declaration as an *extra sibling
# node in the parent*, not a child of the declaration. Every language whose
# convention is "doc comment above the declaration" therefore lost its
# documentation from the entity chunk until _leading_doc_comment was added.

def _one(chunks, name):
    return next(c for c in chunks if c.metadata["entity_name"] == name)


def test_doc_comment_attached_python_through_decorator(tmp_dir):
    """The comment sits above the *decorator*, so the walk must hoist through
    `decorated_definition` (which is in ENTITY_NODE_TYPES but yields no name,
    so `_collect_entity_nodes` emits the inner `function_definition`)."""
    p = tmp_dir / "m.py"
    p.write_text(
        "# Registers the retry handler.\n"
        "# Second line of the note.\n"
        "@decorator\n"
        "def helper(x):\n"
        "    return x\n"
    )
    text = _one(extract_chunks_from_file(p, max_tokens=512), "helper").text
    assert "Registers the retry handler." in text
    assert "Second line of the note." in text


def test_doc_comment_attached_typescript_jsdoc(tmp_dir):
    """JSDoc precedes the `export_statement` wrapper, not the function."""
    p = tmp_dir / "a.ts"
    p.write_text(
        "/**\n * Refreshes the OAuth token.\n */\n"
        "export function refresh(t: string) { return t; }\n"
    )
    assert "Refreshes the OAuth token." in _one(
        extract_chunks_from_file(p, max_tokens=512), "refresh"
    ).text


def test_doc_comment_attached_javascript_export(tmp_dir):
    """`_get_entity_name(export_statement, "javascript")` returns a truthy name,
    so export_statement needs explicit transparency during hoisting."""
    p = tmp_dir / "a.js"
    p.write_text("/** Serializes the payload. */\nexport function g() {}\n")
    assert "Serializes the payload." in _one(
        extract_chunks_from_file(p, max_tokens=512), "g"
    ).text


def test_doc_comment_attached_go(tmp_dir):
    p = tmp_dir / "m.go"
    p.write_text("package m\n\n// Foo dials the upstream.\n// Retries twice.\nfunc Foo() {}\n")
    text = _one(extract_chunks_from_file(p, max_tokens=512), "Foo").text
    assert "Foo dials the upstream." in text
    assert "Retries twice." in text


def test_doc_comment_attached_rust(tmp_dir):
    p = tmp_dir / "m.rs"
    p.write_text("/// Parses the manifest.\npub fn f() {}\n")
    assert "Parses the manifest." in _one(
        extract_chunks_from_file(p, max_tokens=512), "f"
    ).text


def test_doc_comment_attached_csharp_class_and_method(tmp_dir):
    p = tmp_dir / "C.cs"
    p.write_text(
        "namespace N {\n"
        "  /// <summary>Wraps the audit log.</summary>\n"
        "  public class C {\n"
        "    /// <summary>Flushes pending writes.</summary>\n"
        "    [Obsolete]\n"
        "    public void M() { }\n"
        "  }\n"
        "}\n"
    )
    chunks = extract_chunks_from_file(p, max_tokens=512)
    assert "Wraps the audit log." in _one(chunks, "C").text
    assert "Flushes pending writes." in _one(chunks, "M").text


def test_doc_comment_attached_vb_despite_parse_errors(tmp_dir):
    """VB is the language where this matters most and is hardest: the grammar
    emits explicit `blank_line` sibling nodes, wraps `class_block` in
    `type_declaration`, and ~90% of real files carry ERROR nodes."""
    p = tmp_dir / "C.vb"
    p.write_text(
        '#Region "Helpers"\n'
        "''' <summary>Holds the payroll batch.</summary>\n"
        "Public Class C\n"
        "    ''' <summary>Posts the batch to the ledger.</summary>\n"
        "    Public Sub M()\n"
        '        Dim x = 1 & "a"\n'
        "    End Sub\n"
        "End Class\n"
        "#End Region\n"
    )
    chunks = extract_chunks_from_file(p, max_tokens=512)
    assert "Holds the payroll batch." in _one(chunks, "C").text
    assert "Posts the batch to the ledger." in _one(chunks, "M").text


def test_non_adjacent_comment_not_attached(tmp_dir):
    """A comment separated by a blank line documents nothing. Skipping VB's
    `blank_line` trivia must not defeat the line-gap check — measure the gap
    against a tracked anchor, not the trivia node."""
    py = tmp_dir / "gap.py"
    py.write_text("def first(): pass\n\n# UNRELATED_MARKER\n\ndef second(): pass\n")
    assert "UNRELATED_MARKER" not in _one(
        extract_chunks_from_file(py, max_tokens=512), "second"
    ).text

    vb = tmp_dir / "gap.vb"
    vb.write_text(
        "Public Class C\n"
        "    Public Sub M()\n"
        "    End Sub\n"
        "\n"
        "    ' UNRELATED_MARKER\n"
        "\n"
        "    Public Sub N()\n"
        "    End Sub\n"
        "End Class\n"
    )
    assert "UNRELATED_MARKER" not in _one(
        extract_chunks_from_file(vb, max_tokens=512), "N"
    ).text


def test_doc_comment_preserves_line_start_and_signature(tmp_dir):
    """line_start feeds _make_chunk_id, so it must stay the declaration line —
    shifting it to the comment would orphan every existing chunk."""
    p = tmp_dir / "s.py"
    p.write_text("# Doc line one.\n# Doc line two.\ndef helper(x):\n    return x\n")
    chunk = _one(extract_chunks_from_file(p, max_tokens=512), "helper")
    assert chunk.metadata["line_start"] == 3
    assert chunk.metadata["signature"] == "def helper(x):"


def test_doc_comments_do_not_change_chunk_ids(tmp_dir):
    """Chunk IDs must be identical whether or not doc comments are captured."""
    p = tmp_dir / "id.py"
    p.write_text("# Doc.\ndef helper(x):\n    return x\n")
    with_docs = extract_chunks_from_file(p, max_tokens=512, include_doc_comments=True)
    without = extract_chunks_from_file(p, max_tokens=512, include_doc_comments=False)
    assert [c.id for c in with_docs] == [c.id for c in without]
    assert "Doc." in _one(with_docs, "helper").text
    assert "Doc." not in _one(without, "helper").text


def test_doc_comment_truncated_from_front_past_max_lines(tmp_dir):
    """Keep the lines nearest the declaration — those describe it."""
    p = tmp_dir / "long.py"
    p.write_text(
        "# FIRST_MARKER\n"
        + "".join(f"# filler {i}\n" for i in range(10))
        + "# LAST_MARKER\n"
        "def helper(): pass\n"
    )
    text = _one(
        extract_chunks_from_file(p, max_tokens=512, doc_comment_max_lines=3), "helper"
    ).text
    assert "LAST_MARKER" in text
    assert "FIRST_MARKER" not in text


def test_doc_comment_cannot_crowd_out_code(tmp_dir):
    """A huge comment block above a declaration must not consume the chunk's
    whole token budget and truncate away the code itself."""
    p = tmp_dir / "big.py"
    p.write_text(
        "".join(f"# pad pad pad pad pad pad pad pad line {i}\n" for i in range(400))
        + "def helper():\n    return 'BODY_MARKER'\n"
    )
    text = _one(
        extract_chunks_from_file(p, max_tokens=128, doc_comment_max_lines=500), "helper"
    ).text
    assert "def helper" in text
    assert "BODY_MARKER" in text


def test_file_header_chunk_captures_module_docstring(tmp_dir):
    p = tmp_dir / "h.py"
    p.write_text(
        '"""This module reconciles the ledger against upstream."""\n'
        "# Owned by the payments team.\n"
        "\n"
        "def helper(): pass\n"
    )
    chunks = extract_chunks_from_file(p, max_tokens=512)
    header = [c for c in chunks if c.metadata["entity_type"] == "module_doc"]
    assert len(header) == 1
    assert "reconciles the ledger" in header[0].text
    assert "Owned by the payments team." in header[0].text


def test_file_header_chunk_skips_trivial_headers(tmp_dir):
    """A shebang or a one-word comment is not documentation."""
    for name, src in [
        ("sh.py", "#!/usr/bin/env python\ndef f(): pass\n"),
        ("short.py", "# tmp\ndef f(): pass\n"),
        ("none.py", "def f(): pass\n"),
    ]:
        p = tmp_dir / name
        p.write_text(src)
        chunks = extract_chunks_from_file(p, max_tokens=512)
        assert not [c for c in chunks if c.metadata["entity_type"] == "module_doc"], name


def test_file_header_chunk_disabled_by_flag(tmp_dir):
    p = tmp_dir / "off.py"
    p.write_text('"""A reasonably long module-level explanation here."""\ndef f(): pass\n')
    chunks = extract_chunks_from_file(p, max_tokens=512, index_file_headers=False)
    assert not [c for c in chunks if c.metadata["entity_type"] == "module_doc"]


def test_code_indexer_honors_doc_comment_config(tmp_dir, indexer_deps):
    """CodeConfig flags must reach extract_chunks_from_file."""
    store, meta, embedder, sparse_encoder = indexer_deps
    repo = tmp_dir / "cfgrepo"
    repo.mkdir()
    (repo / "m.py").write_text("# Reticulates the splines.\ndef helper(): pass\n")

    captured = {}
    real = code_indexer_mod.extract_chunks_from_file

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return real(*args, **kwargs)

    config = CodeConfig(include_doc_comments=False, doc_comment_max_lines=7,
                        index_file_headers=False)
    with patch.object(code_indexer_mod, "extract_chunks_from_file", spy):
        CodeIndexer(store, meta, embedder, sparse_encoder, config=config).index_repo(
            repo, repo_name="cfgrepo"
        )

    assert captured["include_doc_comments"] is False
    assert captured["doc_comment_max_lines"] == 7
    assert captured["index_file_headers"] is False


def test_file_header_chunk_skips_license_boilerplate(tmp_dir):
    """A license grant is not documentation. Vendored files carry near-identical
    license headers, which would otherwise add a duplicate chunk each."""
    p = tmp_dir / "lic.py"
    p.write_text(
        '"""\n'
        "Copyright 2020-2021, CCL Forensics\n"
        "\n"
        "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
        'of this software and associated documentation files, to deal in the Software.\n'
        '"""\n'
        "def f(): pass\n"
    )
    chunks = extract_chunks_from_file(p, max_tokens=512)
    assert not [c for c in chunks if c.metadata["entity_type"] == "module_doc"]


def test_file_header_chunk_kept_when_license_is_incidental(tmp_dir):
    """A real description that merely mentions a license must still be kept."""
    p = tmp_dir / "ok.py"
    p.write_text(
        '"""Vendored subset of the upstream reader (MIT, CCL Forensics).\n'
        "\n"
        "Pure-python Chromium LevelDB reader used to extract the desktop app's\n"
        "session token from its on-disk LocalStorage store.\n"
        '"""\n'
        "def f(): pass\n"
    )
    chunks = extract_chunks_from_file(p, max_tokens=512)
    header = [c for c in chunks if c.metadata["entity_type"] == "module_doc"]
    assert len(header) == 1
    assert "Chromium LevelDB reader" in header[0].text
