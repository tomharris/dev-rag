---
name: rag-first-reminder-task
enabled: true
event: all
tool_matcher: "Task"
conditions:
  - field: subagent_type
    operator: regex_match
    pattern: ^(Explore|general-purpose)$
---

**DevRAG reminder**: Before dispatching an `Explore` or `general-purpose` subagent to search the codebase, consider calling `mcp__devrag__search` first. DevRAG has semantic understanding of code, PR history, and documentation that a fresh subagent would re-discover from scratch. If you already searched DevRAG, carry on.
