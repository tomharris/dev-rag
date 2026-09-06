"""Related-chunk expansion: the file-path join between code and PR history.

Stage 2 of the graph-search investigation. Retrieval scores each chunk on its own
text, so a query that surfaces a function never surfaces the PR that changed it,
and vice versa — the two live in different collections and compete only on
wording. But they are joined by a fact the index already knows: they name the
same file.

This is one hop over the only edge the corpus actually has. Issue and Jira chunks
carry no ``file_path``, so they cannot be reached this way; the edge is code <-> PR
and nothing else. Expansion runs *before* reranking so added chunks compete on
merit rather than being appended as unranked extras — the cross-encoder decides
whether a PR that touched the file is worth a slot for this particular query.

The join only works because code and PR chunks agree on ``repo`` and
``file_path``; before that normalization it would have matched nothing at all.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from devrag.types import SearchResult

logger = logging.getLogger(__name__)

CODE_COLLECTION = "code_chunks"
PR_COLLECTIONS = ("pr_diffs", "pr_discussions")


def _anchor_targets(result: SearchResult) -> tuple[str, str, tuple[str, ...]] | None:
    """Return ``(repo, file_path, collections)`` to expand this result into.

    A PR chunk reaches for the current code; a code chunk reaches for the PRs
    that touched it. Anything without both a repo and a file path — an issue, a
    Slack thread, a standalone doc — has no edge to follow.
    """
    repo = result.metadata.get("repo")
    file_path = result.metadata.get("file_path")
    if not repo or not file_path:
        return None
    if "pr_number" in result.metadata:
        targets = (CODE_COLLECTION,)
    elif "issue_number" in result.metadata or "ticket_key" in result.metadata:
        return None
    else:
        targets = PR_COLLECTIONS
    return str(repo), str(file_path), targets


def expand_related(
    store,
    results: list[SearchResult],
    top_n: int = 5,
    per_anchor: int = 2,
    max_total: int = 10,
) -> list[SearchResult]:
    """Return extra candidates joined by file path to the top *top_n* results.

    Expanded candidates inherit their anchor's score and are placed directly
    after it, so a stable sort keeps the anchor ahead of the context it pulled
    in. They are *candidates*, not results: reranking (when on) re-scores them
    against the query, and dedupe still applies. A code chunk and a PR chunk for
    the same file have different dedupe source keys, so both can survive.

    Never returns a chunk already present in *results*. Store failures are
    logged and skipped — expansion is an enhancement, not a guarantee.
    """
    if not results or top_n <= 0 or per_anchor <= 0 or max_total <= 0:
        return []

    seen = {r.chunk_id for r in results}
    anchors = []
    for result in results[:top_n]:
        target = _anchor_targets(result)
        if target is not None:
            anchors.append((result, target))
    if not anchors:
        return []

    def _fetch(job):
        _, (repo, file_path, collections) = job
        hits = []
        for collection in collections:
            try:
                hits.append(store.fetch_by_filter(
                    collection, {"repo": repo, "file_path": file_path}, limit=per_anchor,
                ))
            except Exception:
                logger.debug("expansion lookup failed for %s/%s in %s",
                             repo, file_path, collection, exc_info=True)
        return hits

    with ThreadPoolExecutor(max_workers=min(len(anchors), 8)) as pool:
        per_anchor_hits = list(pool.map(_fetch, anchors))

    expanded: list[SearchResult] = []
    for (anchor, _), hit_sets in zip(anchors, per_anchor_hits):
        taken = 0
        for hits in hit_sets:
            for i, chunk_id in enumerate(hits.ids):
                if chunk_id in seen or taken >= per_anchor:
                    continue
                seen.add(chunk_id)
                taken += 1
                metadata = dict(hits.metadatas[i])
                metadata["expanded_from"] = anchor.chunk_id
                expanded.append(SearchResult(
                    chunk_id=chunk_id,
                    text=hits.documents[i],
                    score=anchor.score,
                    metadata=metadata,
                ))
                if len(expanded) >= max_total:
                    return expanded
    return expanded


def interleave_after_anchors(
    results: list[SearchResult], expanded: list[SearchResult]
) -> list[SearchResult]:
    """Place each expanded candidate immediately after the anchor it came from.

    Applied *after* ranking, this enforces the rule that makes expansion safe:
    context appears alongside the result that pulled it in, never instead of it.
    Without it, the reranker put two historical diffs of `jira_client.py` above
    the implementation itself for "how does the Jira client authenticate" — the
    query asked what the code does, and got what it used to do.

    An expanded chunk whose anchor did not survive ranking is dropped: it was
    only ever there as context for that anchor.
    """
    if not expanded:
        return results
    by_anchor: dict[str, list[SearchResult]] = {}
    for candidate in expanded:
        by_anchor.setdefault(str(candidate.metadata.get("expanded_from", "")), []).append(candidate)
    merged: list[SearchResult] = []
    for result in results:
        merged.append(result)
        merged.extend(by_anchor.pop(result.chunk_id, []))
    # Anything still pending lost its anchor during ranking, so it has nothing
    # left to be context for.
    return merged
