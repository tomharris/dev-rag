---
description: Search codebase knowledge (code, PRs, docs) using DevRAG
model: haiku
allowed-tools: mcp__devrag__search
---

Search DevRAG for: $ARGUMENTS

Present results grouped by source type (code, PR, doc).
For code results, show the file path and relevant snippet.
For PR results, show the PR title, number, and relevant excerpt.
For doc results, show the document title and section.
