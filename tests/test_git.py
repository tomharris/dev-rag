import subprocess
from pathlib import Path

import pytest

from devrag.utils.git import discover_files, infer_repo


@pytest.fixture
def git_repo(tmp_dir):
    subprocess.run(["git", "init", str(tmp_dir)], capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(tmp_dir), capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(tmp_dir), capture_output=True, check=True,
    )
    (tmp_dir / "src").mkdir()
    (tmp_dir / "src" / "main.py").write_text("print('hello')")
    (tmp_dir / "src" / "utils.py").write_text("x = 1")
    (tmp_dir / "src" / "data.min.js").write_text("minified")
    (tmp_dir / "node_modules").mkdir()
    (tmp_dir / "node_modules" / "pkg.js").write_text("module")
    (tmp_dir / "README.md").write_text("# Readme")
    (tmp_dir / ".gitignore").write_text("node_modules/\n")
    subprocess.run(["git", "add", "."], cwd=str(tmp_dir), capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_dir), capture_output=True, check=True)
    return tmp_dir


def test_discover_files_respects_gitignore(git_repo):
    files = discover_files(git_repo, exclude_patterns=[])
    rel_paths = {str(f.relative_to(git_repo)) for f in files}
    assert "src/main.py" in rel_paths
    assert "src/utils.py" in rel_paths
    assert "node_modules/pkg.js" not in rel_paths


def test_discover_files_applies_exclude_patterns(git_repo):
    files = discover_files(git_repo, exclude_patterns=["*.min.js", "*.md"])
    rel_paths = {str(f.relative_to(git_repo)) for f in files}
    assert "src/main.py" in rel_paths
    assert "src/data.min.js" not in rel_paths
    assert "README.md" not in rel_paths


def test_discover_files_nonexistent_dir():
    files = discover_files(Path("/nonexistent"), exclude_patterns=[])
    assert files == []


def test_discover_files_respects_devragignore(git_repo):
    (git_repo / ".devragignore").write_text("*.md\nsrc/data.min.js\n")
    import subprocess
    subprocess.run(["git", "add", ".devragignore"], cwd=str(git_repo), capture_output=True)
    subprocess.run(["git", "commit", "-m", "add devragignore"], cwd=str(git_repo), capture_output=True)
    files = discover_files(git_repo, exclude_patterns=[])
    rel_paths = {str(f.relative_to(git_repo)) for f in files}
    assert "src/main.py" in rel_paths
    assert "src/utils.py" in rel_paths
    assert "README.md" not in rel_paths
    assert "src/data.min.js" not in rel_paths


def test_devragignore_excludes_directory_pattern(git_repo):
    """A gitignore-style directory pattern excludes everything beneath it."""
    (git_repo / "docs").mkdir()
    (git_repo / "docs" / "internal").mkdir()
    (git_repo / "docs" / "internal" / "secret.md").write_text("# secret")
    (git_repo / "docs" / "public.md").write_text("# public")
    (git_repo / ".devragignore").write_text("docs/internal/\n")
    files = discover_files(git_repo, exclude_patterns=[])
    rel_paths = {str(f.relative_to(git_repo)) for f in files}
    assert "docs/public.md" in rel_paths
    assert "docs/internal/secret.md" not in rel_paths


def test_devragignore_excludes_bare_dir_name_at_any_depth(git_repo):
    """A bare directory name matches that directory anywhere in the tree."""
    (git_repo / "a" / "internal").mkdir(parents=True)
    (git_repo / "a" / "internal" / "x.py").write_text("x = 1")
    (git_repo / "a" / "keep.py").write_text("y = 2")
    (git_repo / ".devragignore").write_text("internal/\n")
    files = discover_files(git_repo, exclude_patterns=[])
    rel_paths = {str(f.relative_to(git_repo)) for f in files}
    assert "a/keep.py" in rel_paths
    assert "a/internal/x.py" not in rel_paths


def test_devragignore_supports_negation(git_repo):
    """A later `!pattern` re-includes a path excluded by an earlier pattern."""
    (git_repo / "notes.md").write_text("# notes")
    (git_repo / "KEEP.md").write_text("# keep")
    (git_repo / ".devragignore").write_text("*.md\n!KEEP.md\n")
    files = discover_files(git_repo, exclude_patterns=[])
    rel_paths = {str(f.relative_to(git_repo)) for f in files}
    assert "KEEP.md" in rel_paths
    assert "notes.md" not in rel_paths
    # README.md (committed by the fixture) is still excluded by `*.md`.
    assert "README.md" not in rel_paths


def test_infer_repo_matches_cwd_under_repo():
    repos = [("repo-a", "/home/u/Projects/repo-a"), ("repo-b", "/home/u/Projects/repo-b")]
    assert infer_repo(Path("/home/u/Projects/repo-b/src/app"), repos) == "repo-b"


def test_infer_repo_returns_empty_when_no_match():
    repos = [("repo-a", "/home/u/Projects/repo-a")]
    assert infer_repo(Path("/tmp/elsewhere"), repos) == ""


def test_infer_repo_prefers_most_specific_nested_repo():
    repos = [("outer", "/home/u/Projects"), ("inner", "/home/u/Projects/inner")]
    assert infer_repo(Path("/home/u/Projects/inner/src"), repos) == "inner"
