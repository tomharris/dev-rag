#!/usr/bin/env python3
"""UserPromptSubmit nudge: remind the model to run rag-first on codebase questions.

Keys on prompt intent, never blocks. The skill listing alone loses to auto mode's
"search with grep and find" guidance and to devrag's tools being deferred behind
ToolSearch, so question-shaped prompts went straight to Bash grep.
"""
import json
import re
import sys

QUESTION = re.compile(
    r"\?|\b(how|why|where|what|which|when|who|explain|investigat\w*|explor\w*|"
    r"understand|find|figure out|look into|walk me through|trace|discover\w*)\b",
    re.I,
)

NUDGE = (
    "rag-first: this prompt looks like a codebase question/exploration. Before "
    "Grep/Glob/Explore or Bash grep/find/git grep, invoke the `rag-first` skill "
    "(load `mcp__devrag__search` via ToolSearch `select:mcp__devrag__search` if it is "
    "deferred). Skip only for literal lookups of a known symbol/string, or if the "
    "devrag MCP server failed to connect."
)


def main() -> None:
    try:
        prompt = json.load(sys.stdin).get("prompt", "")
    except (ValueError, AttributeError):
        return
    stripped = prompt.lstrip()
    if not stripped or stripped.startswith(("/", "!", "<")):
        return
    if not QUESTION.search(prompt):
        return
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": NUDGE,
        }
    }))


if __name__ == "__main__":
    main()
