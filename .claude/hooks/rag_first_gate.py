#!/usr/bin/env python3
"""Per-turn rag-first gate.

PreToolUse:       block Grep/Glob/Agent(Explore|general-purpose) unless marker exists.
PostToolUse:      touch marker when mcp__devrag__search has run.
UserPromptSubmit: delete marker (new turn re-requires a rag-search).
"""
import json
import os
import sys
import tempfile

MARKER_DIR = tempfile.gettempdir()
GATED_TOOLS = {"Grep", "Glob"}
GATED_AGENT_SUBTYPES = {"Explore", "general-purpose"}
SEARCH_TOOLS = {"mcp__devrag__search"}
BLOCK_MSG = (
    "rag-first gate: call `mcp__devrag__search` before using "
    "Grep/Glob/Agent for codebase exploration. After it runs, "
    "this gate releases for the rest of the turn. If rag returns "
    "nothing useful, run mcp__devrag__search with a different query "
    "to release the gate, then fall back to keyword search."
)


def marker_path(session_id: str) -> str:
    return os.path.join(MARKER_DIR, f"devrag-searched-{session_id}")


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    event = data.get("hook_event_name", "")
    session_id = data.get("session_id", "")
    if not session_id:
        sys.exit(0)
    marker = marker_path(session_id)

    if event == "UserPromptSubmit":
        try:
            os.remove(marker)
        except FileNotFoundError:
            pass
        sys.exit(0)

    if event == "PostToolUse":
        tool_name = data.get("tool_name", "")
        if tool_name in SEARCH_TOOLS:
            open(marker, "w").close()
        sys.exit(0)

    if event == "PreToolUse":
        tool_name = data.get("tool_name", "")
        tool_input = data.get("tool_input", {}) or {}
        is_gated = tool_name in GATED_TOOLS or (
            tool_name == "Agent"
            and tool_input.get("subagent_type") in GATED_AGENT_SUBTYPES
        )
        if is_gated and not os.path.exists(marker):
            print(
                json.dumps(
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                        },
                        "systemMessage": BLOCK_MSG,
                    }
                )
            )
            sys.exit(0)

    sys.exit(0)


if __name__ == "__main__":
    main()
