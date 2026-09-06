from __future__ import annotations

import subprocess
from pathlib import Path

import pathspec


def infer_repo(cwd: Path, repos: list[tuple[str, str]]) -> str:
    """Return the registered repo name whose path contains ``cwd``.

    ``repos`` is a list of ``(name, path)`` pairs (see ``MetadataDB.get_all_repos``).
    When ``cwd`` sits inside nested repos, the most specific (deepest) path wins.
    Returns ``""`` when no registered repo contains ``cwd``.
    """
    cwd = cwd.resolve()
    best_name, best_depth = "", -1
    for name, path in repos:
        repo_path = Path(path).resolve()
        if cwd == repo_path or repo_path in cwd.parents:
            depth = len(repo_path.parts)
            if depth > best_depth:
                best_name, best_depth = name, depth
    return best_name


def relative_to_repo(file_path: Path, repo_path: Path | None) -> str:
    """Return *file_path* as a repo-relative POSIX string.

    Code and doc chunks store this rather than an absolute path so a chunk's
    `file_path` is the same string GitHub uses in a PR diff — which is what makes
    a PR joinable to the code it touched, and what makes `search --file-path`
    match both sources at once. Absolute paths also leaked the indexing machine's
    home directory into every payload.

    Falls back to `str(file_path)` when *repo_path* is None (a standalone doc
    directory, which has no repo root to be relative to) or when the file is
    somehow outside the repo.
    """
    if repo_path is None:
        return str(file_path)
    try:
        return file_path.resolve().relative_to(repo_path.resolve()).as_posix()
    except ValueError:
        return str(file_path)


def discover_files(
    repo_path: Path,
    exclude_patterns: list[str],
) -> list[Path]:
    if not repo_path.exists():
        return []

    # Read .devragignore patterns. These use real gitignore syntax (directories,
    # anchoring, `**`, and `!` negation) — evaluated below via pathspec.GitIgnoreSpec,
    # not fnmatch (which silently ignored directory patterns like `docs/internal/`).
    devragignore = repo_path / ".devragignore"
    extra_excludes: list[str] = []
    if devragignore.exists():
        for line in devragignore.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                extra_excludes.append(line)
    # Caller-supplied globs first, then .devragignore lines, so a user can re-include
    # a path with `!pattern` in .devragignore (later patterns win in gitignore).
    spec = pathspec.GitIgnoreSpec.from_lines(list(exclude_patterns) + extra_excludes)

    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=True,
        )
        rel_paths = [p for p in result.stdout.strip().split("\n") if p]
    except (subprocess.CalledProcessError, FileNotFoundError):
        rel_paths = [
            str(f.relative_to(repo_path))
            for f in repo_path.rglob("*")
            if f.is_file()
        ]

    filtered: list[Path] = []
    for rel in rel_paths:
        if spec.match_file(rel):
            continue
        full = repo_path / rel
        if full.is_file():
            filtered.append(full)

    return filtered
