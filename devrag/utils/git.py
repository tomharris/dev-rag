from __future__ import annotations

import fnmatch
import subprocess
from pathlib import Path


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


def discover_files(
    repo_path: Path,
    exclude_patterns: list[str],
) -> list[Path]:
    if not repo_path.exists():
        return []

    # Read .devragignore patterns
    devragignore = repo_path / ".devragignore"
    extra_excludes: list[str] = []
    if devragignore.exists():
        for line in devragignore.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                extra_excludes.append(line)
    all_excludes = list(exclude_patterns) + extra_excludes

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
        if any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(Path(rel).name, pat)
               for pat in all_excludes):
            continue
        full = repo_path / rel
        if full.is_file():
            filtered.append(full)

    return filtered
