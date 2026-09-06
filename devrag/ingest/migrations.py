"""One-shot payload migrations for already-indexed chunks.

These run automatically at the start of a sync and are idempotent: once a
migration has been applied, its filter matches nothing and the call is a no-op.
They rewrite metadata in place rather than re-embedding, so they cost one Qdrant
call per collection instead of a full re-sync.
"""
from __future__ import annotations

import logging

from devrag.utils.github import bare_repo_name

logger = logging.getLogger(__name__)


def backfill_bare_repo_name(store, collections: list[str], repo_slug: str) -> int:
    """Rewrite `repo` from an ``owner/name`` slug to the bare name on old chunks.

    PR and issue chunks used to store the full GitHub slug in `repo`, while code
    chunks store the bare directory name — so `search --repo dev-rag` silently
    excluded every PR, and PR chunks could never match the active-repo boost.
    New chunks are written with the bare name (see `_pr_base_metadata`), but
    cursor-based sync never revisits already-indexed PRs, so their payloads would
    otherwise stay wrong forever.

    Returns the number of points updated across *collections*.
    """
    if "/" not in repo_slug:
        return 0
    payload = {"repo": bare_repo_name(repo_slug), "repo_full": repo_slug}
    updated = 0
    for collection in collections:
        try:
            updated += store.set_payload(collection, {"repo": repo_slug}, payload)
        except Exception:  # pragma: no cover - a migration must not block a sync
            logger.warning("Could not backfill repo name in %s", collection, exc_info=True)
    if updated:
        logger.info("Backfilled bare repo name on %d chunk(s) for %s", updated, repo_slug)
    return updated
