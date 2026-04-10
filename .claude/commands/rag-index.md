---
description: Index a codebase, documents, or PRs for DevRAG search
model: haiku
allowed-tools: mcp__devrag__index_repo, mcp__devrag__index_docs, mcp__devrag__sync_prs, mcp__devrag__sync_issues, mcp__devrag__sync_jira, mcp__devrag__sync_slite, mcp__devrag__status
---

Help the user index their codebase for search.

If $ARGUMENTS mentions a directory or repo path, use index_repo to index it.
If $ARGUMENTS mentions docs or documents, use index_docs.
If $ARGUMENTS mentions PRs or a GitHub repo, use sync_prs.
If $ARGUMENTS is empty or says "status", use status to show current index state.

After indexing, show the status.
