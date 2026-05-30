# Design: `devrag index refresh` — incremental all-repos refresh

**Date:** 2026-05-30
**Status:** Approved (implementation to be done by author)

## Problem

There is no cheap, non-destructive way to pick up code/doc changes across **all** registered
repos in one shot. The current command surface:

| Command | Scope | Mode | Side effects |
|---|---|---|---|
| `index repo <path>` | one repo | incremental (or `--full`) | none beyond that repo |
| `reindex --name <repo>` | one registered repo | full rebuild (remove + re-embed) | none beyond that repo |
| `reindex --all` | all registered repos | full rebuild after `reset_all()` | **wipes every external sync cursor** (PRs/issues/Jira/Slite/Slack) |

To refresh all repos today, the only one-shot option is `reindex --all`, which calls
`MetadataDB.reset_all()` — nuking all external sync cursors and forcing a from-scratch
re-embed of every chunk. That is a sledgehammer for what should be a delta update.

## Solution

A new subcommand **`devrag index refresh`** that walks the `code_repos` registry
(`MetadataDB.get_all_repos()` → `[(name, path), ...]`) and re-indexes each repo **in place**
— code *and* docs — **without ever calling `reset_all()`**. External sync cursors are never
touched.

This is a *composition* of existing, tested operations, not new indexing logic.

### Modes

| Invocation | Per-repo behavior | Reuses the body of |
|---|---|---|
| `devrag index refresh` | Incremental — file-hash check skips unchanged files; removed files reconciled | `index repo <path>` (cli.py `index_repo`, ~L101) |
| `devrag index refresh --full` | Remove the repo's chunks (code + docs), then re-embed non-incrementally | the `reindex --name` branch (cli.py `reindex`, ~L568–599) |

- **Incremental (default):** for each registered repo, run
  `CodeIndexer.index_repo(repo_dir, incremental=True, repo_name=name)` and, when
  `config.code.index_docs`, `DocIndexer.index_repo_docs(repo_dir, repo_name=name,
  incremental=True, exclude_patterns=config.code.exclude_patterns)`.
- **`--full`:** mirror the `reindex --name` branch per repo — remove the repo's chunk IDs from
  both `code_chunks` and `documents` (IDs are disjoint, absent-ID deletes are no-ops), clear
  its file hashes, then re-index with `incremental=False` for both code and docs.

`--full` here is a *clean per-repo rebuild*, distinct from `reindex --all`: it never resets
external cursors and never touches repos not in the loop.

## Resilience: missing directories (the one new design choice)

Registered repos may have moved or been deleted since indexing. The sweep must **not abort**
because one repo's directory is gone (cf. commit `160eda3` — "don't let a broken loader abort
the sweep").

Per repo, before indexing: if `Path(repo_path)` does not exist →
- print a warning that also hints at cleanup, e.g.:
  `⚠ skipping <name>: directory not found at <path> (run 'devrag index remove-repo <name>' to drop it)`
- `continue` to the next repo. **Never** auto-remove from the registry — a missing dir may be a
  temporarily unmounted drive; removal stays the explicit job of `index remove-repo`.

At the end, print a summary line:
`Refreshed N repos, skipped M (missing directories)`.

If the registry is empty, print the same guidance `reindex --all` uses
(`No code repos registered. Run 'devrag index repo .' to index code.`) and exit cleanly.

## Components to add

1. **CLI command** — `devrag/cli.py`, new `@index_app.command("refresh")` function
   `index_refresh(full: bool = typer.Option(False, "--full", help="Full re-embed of every repo (skip incremental)"))`.
   - Build `config`, `store`, `meta`, `embedder`, `sparse_encoder` the same way `index_repo`
     and `reindex` do.
   - Loop `meta.get_all_repos()`; per repo apply the missing-dir guard, then incremental or
     `--full` path; echo per-repo stats via the existing `format_index_stats` /
     `format_repo_doc_stats`.
   - Factor the shared per-repo doc step the way `reindex` already does with its
     `_reindex_repo_docs` helper (reuse or mirror it).

2. **MCP tool** — `devrag/mcp_server.py`, new `refresh()` tool alongside `index_repo`/`sync_*`,
   using the lazy singletons (`_get_config`, `_get_vector_store`, `_get_metadata_db`,
   `_get_embedder`, `_get_sparse_encoder`). Signature `refresh(full: bool = False) -> str`
   (`full` mirrors `--full`). Returns the concatenated per-repo stats + summary string,
   including any skipped-repo warnings.

## Out of scope

- Refreshing external sources (PRs/issues/Jira/Slite/Slack) — those keep their own
  `sync_*` commands and cursors.
- Per-repo selection / filtering (that is `reindex --name`).
- Auto-pruning stale registry entries.

## Testing notes

- Empty registry → clean exit with guidance, no crash.
- Mixed registry where one path is missing → healthy repos still refresh; missing one is
  warned + counted; exit code 0.
- Incremental run with no file changes → near-zero re-embeds (file-hash skip path exercised).
- `--full` → repo's stale chunks removed from both `code_chunks` and `documents` before
  re-embed; external sync cursors untouched (assert cursors survive).
