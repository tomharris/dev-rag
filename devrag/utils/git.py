from __future__ import annotations

import fnmatch
import subprocess
from pathlib import Path


def discover_files(
    repo_path: Path,
    exclude_patterns: list[str],
) -> list[Path]:
    if not repo_path.exists():
        return []

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
               for pat in exclude_patterns):
            continue
        full = repo_path / rel
        if full.is_file():
            filtered.append(full)

    return filtered
