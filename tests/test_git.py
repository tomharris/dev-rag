import subprocess
from pathlib import Path

import pytest

from devrag.utils.git import discover_files


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
