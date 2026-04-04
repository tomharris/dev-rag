---
name: rag-first
description: Use when the user asks ANY question about the codebase - how code works, where something is defined, why something changed, what a module does, how components connect, or any query that would normally trigger Grep/Glob/Explore/Agent exploration. This skill MUST be invoked BEFORE any codebase exploration tools.
---

# RAG-First Codebase Search

Before using Grep, Glob, Explore agents, or other codebase exploration tools, ALWAYS search DevRAG first. The RAG index has semantic understanding of code structure, PR history, and documentation that keyword search misses.

## Process

1. **Formulate a search query** from the user's question. Use natural language — DevRAG uses semantic search, not keyword matching. Include key terms but phrase it as a question or description.

2. **Call `mcp__devrag__search`** with your query.

3. **Evaluate the results:**
   - If results are relevant and sufficient — present them grouped by source type (code, PR, doc). Show file paths, snippets, PR numbers, and document sections.
   - If results are partial — use them as a starting point, then supplement with targeted Grep/Glob/Read on specific files or patterns identified from the RAG results.
   - If results are empty or irrelevant — state "RAG results were limited for this query, falling back to direct codebase exploration" and proceed with Grep/Glob/Read/Explore as normal.

4. **Combine sources** — when RAG gives you file paths and context, use Read to pull in the full current code. RAG results may be from a previous index, so always verify against current files.

## Key Guidelines

- DevRAG searches across four collections: code chunks, PR diffs, PR discussions, and documents. A single query searches all relevant collections.
- For "why did this change?" questions, RAG is especially powerful — it has PR history and review comments that Grep cannot find.
- For "where is X defined?" questions, RAG's AST-aware code chunks often give better results than grep patterns.
- Do NOT skip RAG even for seemingly simple lookups — the semantic search may surface related context you wouldn't have found with keywords.
- If the DevRAG MCP server is not available (tool call fails), fall back to direct exploration without retrying.
