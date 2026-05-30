"""All-repos refresh orchestration.

Walks the registered-repo list and re-indexes each repo in place — code and,
optionally, docs — *without* the destructive full reset that ``reindex --all``
performs. External sync cursors (PRs/issues/Jira/Slite/Slack) are never touched.

The iteration is dependency-injected so it can be unit-tested without a real
embedder, vector store, or metadata DB. The CLI command and MCP tool are thin
wrappers that build the real dependencies and pass them in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


@dataclass
class RefreshSummary:
    """Outcome of a refresh sweep."""

    refreshed: int = 0
    skipped: int = 0
    skipped_repos: list[tuple[str, str]] = field(default_factory=list)


def refresh_all_repos(
    repos: list[tuple[str, str]],
    *,
    index_repo: Callable[[Path, str, bool], object],
    index_repo_docs: Optional[Callable[[Path, str, bool], object]] = None,
    remove_repo: Optional[Callable[[str], int]] = None,
    full: bool = False,
    log: Optional[Callable[[str], None]] = None,
) -> RefreshSummary:
    """Re-index every registered repo, skipping those whose directory is gone.

    Args:
        repos: ``(name, path)`` pairs, e.g. from ``MetadataDB.get_all_repos()``.
        index_repo: Called as ``index_repo(repo_dir, name, incremental)`` to
            (re-)index a repo's code.
        index_repo_docs: Optional; called the same way to index a repo's docs.
            Pass ``None`` to skip docs (e.g. when ``code.index_docs`` is off).
        remove_repo: Optional; called as ``remove_repo(name)`` before a ``full``
            re-index to clear the repo's existing chunks. Ignored when not
            ``full``. Returns the chunk count removed (used only for logging).
        full: When ``True``, remove each repo's chunks then re-index
            non-incrementally; otherwise incremental (file-hash skip) re-index.
        log: Optional progress sink, called with human-readable lines.

    Returns:
        A :class:`RefreshSummary` counting refreshed and skipped repos.

    A missing repo directory is skipped (and counted) rather than aborting the
    sweep — a missing dir may be a temporarily unmounted drive, so pruning the
    registry stays the explicit job of ``index remove-repo``.
    """
    emit = log or (lambda _msg: None)
    incremental = not full
    summary = RefreshSummary()

    for name, path in repos:
        repo_dir = Path(path)
        if not repo_dir.exists():
            emit(
                f"⚠ skipping {name}: directory not found at {path} "
                f"(run 'devrag index remove-repo {name}' to drop it)"
            )
            summary.skipped += 1
            summary.skipped_repos.append((name, path))
            continue

        if full and remove_repo is not None:
            removed = remove_repo(name)
            emit(f"Cleared {removed} chunks for '{name}'.")

        emit(f"{'Re-indexing' if full else 'Refreshing'} {name} ({path})...")
        index_repo(repo_dir, name, incremental)
        if index_repo_docs is not None:
            index_repo_docs(repo_dir, name, incremental)
        summary.refreshed += 1

    return summary
