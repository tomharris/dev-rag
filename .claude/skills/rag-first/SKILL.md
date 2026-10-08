---
name: rag-first
description: Use when the user asks ANY question about the codebase or asks to explore, investigate, or discover something in it - how code works, where something is defined, why something changed, what a module does, how components connect. MUST be invoked BEFORE any exploration tool, including Grep/Glob/Explore/Agent AND Bash grep/rg/find/git grep/git log. Skip only for a literal lookup of a known symbol or string.
---

# RAG-First Codebase Search

Before using Grep, Glob, Explore agents, Bash `grep`/`rg`/`find`/`git grep`, or other codebase exploration tools, ALWAYS search DevRAG first. The RAG index has semantic understanding of code structure, PR history, and documentation that keyword search misses.

## Process

1. **Formulate a search query** from the user's question. Use natural language — DevRAG uses semantic search, not keyword matching. Include key terms but phrase it as a question or description.

2. **Call `mcp__devrag__search`** with your query. If it is listed only as a deferred tool, load it first with ToolSearch `select:mcp__devrag__search`.

3. **Evaluate the results:**
   - If results are relevant and sufficient — present them grouped by source type (code, PR, doc). Show file paths, snippets, PR numbers, and document sections.
   - If results are partial — use them as a starting point, then supplement with targeted Grep/Glob/Read on specific files or patterns identified from the RAG results.
   - If results are empty or irrelevant — state "RAG results were limited for this query, falling back to direct codebase exploration" and proceed with Grep/Glob/Read/Explore as normal.

4. **Combine sources** — when RAG gives you file paths and context, use Read to pull in the full current code. RAG results may be from a previous index, so always verify against current files.

## Key Guidelines

- DevRAG searches across eleven collections: code chunks, PR diffs/discussions, issue descriptions/discussions, Jira descriptions/discussions, Slite pages, Slack messages, documents, and session logs. A single query routes to whichever are relevant.
- For "why did this change?" questions, RAG is especially powerful — it has PR history and review comments that Grep cannot find.
- For "where is X defined?" questions, RAG's AST-aware code chunks often give better results than grep patterns.
- For literal lookups — an exact symbol name, a config key, a string known to appear verbatim — Grep is the right tool and this skill does not apply. RAG's advantage is semantic and historical recall, not exact matching.
- If the DevRAG MCP server is not available (tool call fails), fall back to direct exploration without retrying.
