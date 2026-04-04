---
name: rag-first-reminder
enabled: true
event: all
tool_matcher: "Grep|Glob"
action: warn
conditions:
  - field: pattern
    operator: regex_match
    pattern: .+
---

**DevRAG reminder**: Consider using `/rag-search` before Grep/Glob. DevRAG has semantic code understanding, PR history, and documentation that keyword search misses. If you already searched DevRAG, carry on.
