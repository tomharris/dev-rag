from __future__ import annotations
import re

ALL_COLLECTIONS = ["code_chunks", "pr_diffs", "pr_discussions", "issue_descriptions", "issue_discussions", "jira_descriptions", "jira_discussions", "slite_pages", "slack_messages", "documents", "session_logs"]
CODE_COLLECTIONS = ["code_chunks"]
PR_COLLECTIONS = ["pr_diffs", "pr_discussions"]
ISSUE_COLLECTIONS = ["issue_descriptions", "issue_discussions"]
JIRA_COLLECTIONS = ["jira_descriptions", "jira_discussions"]
SLITE_COLLECTIONS = ["slite_pages"]
SLACK_COLLECTIONS = ["slack_messages"]
DOC_COLLECTIONS = ["documents"]
SESSION_COLLECTIONS = ["session_logs"]

_PR_PATTERNS = [
    r"\bwhy\s+did\s+we\b", r"\bwhy\s+was\b", r"\bwhy\s+were\b",
    r"\bwhen\s+did\s+we\b", r"\bwho\s+changed\b", r"\bwho\s+added\b", r"\bwho\s+removed\b",
    r"\bswitch(?:ed)?\s+(?:from|to)\b", r"\bmigrat(?:e|ed|ion)\b",
    r"\bchange(?:d|s)?\s+(?:the|to|from)\b", r"\bremov(?:e|ed)\b.*\bwhy\b",
    r"\bwhy\b.*\bremov(?:e|ed)\b", r"\bintroduc(?:e|ed)\b", r"\brevert(?:ed)?\b", r"\bdeprecated?\b",
]

# Past-tense only for "filed"/"reported", plus an explicit "file a bug" form.
# The optional-d spellings (`\bfiled?\b`, `\breported?\b`) matched the bare
# words "file" and "report" — among the most common words in a code question —
# so "how does incremental indexing decide to skip a file" routed to issues.
_ISSUE_PATTERNS = [
    r"\bbug\b", r"\bissue[sd]?\b", r"\bfeature\s+request\b",
    r"\breported\b", r"\bfiled\b", r"\bticket\b",
    r"\b(?:file|report)\s+(?:a|an)\s+(?:bug|issue|ticket)\b",
]

_JIRA_PATTERNS = [
    r"\bjira\b", r"\bsprint\b", r"\bepic\b", r"\bstory\b",
    r"\bstory\s+points?\b",
]

_SLITE_PATTERNS = [
    r"\bslite\b", r"\bwiki\b", r"\bknowledge\s+base\b", r"\binternal\s+doc\b",
]

_SLACK_PATTERNS = [
    r"\bslack\b", r"\bdirect\s+message\b", r"\bdm(?:ed|s)?\b",
    r"#[a-z0-9][a-z0-9_-]*",  # #channel mentions
]

_SESSION_PATTERNS = [
    r"\blast\s+(?:time|session|week|conversation)\b",
    r"\bearlier\s+(?:session|conversation|we|I)\b",
    r"\bwe\s+discussed\b", r"\bwe\s+talked\s+about\b",
    r"\bprevious(?:ly)?\s+(?:session|chat|conversation)\b",
    r"\bi\s+asked\s+claude\b", r"\bclaude\s+(?:told|said|suggested)\b",
    r"\bchat\s+history\b", r"\bsession\s+log\b",
]

_DOC_PATTERNS = [
    r"\bpolicy\b", r"\bpolicies\b", r"\bspec(?:ification)?\b", r"\bdesign\s+doc\b",
    r"\barchitecture\b", r"\bdiagram\b", r"\bprocess\b", r"\bprocedure\b",
    r"\bguideline\b", r"\bstandard\b", r"\bconvention\b", r"\bdocument(?:ation)?\b",
    r"\bplaybook\b", r"\brunbook\b", r"\bonboarding\b", r"\btutorial\b",
    r"\bdescribe\s+the\b", r"\bwhat\s+does\s+the\s+(?:spec|doc|guide)\b",
]

_CODE_PATTERNS = [
    r"\bhow\s+does\b", r"\bhow\s+do\b", r"\bhow\s+is\b", r"\bwhat\s+does\b",
    r"\bimplement(?:s|ed|ation)?\b", r"\bdefin(?:e|ed|ition)\b",
]

_USAGE_PATTERNS = [
    r"\bwhere\s+is\b", r"\bwhere\s+are\b", r"\bwho\s+uses\b",
    r"\busage\s+of\b", r"\bcall(?:s|ed)\s+(?:from|by|in)\b",
]


# Every non-session intent includes `code_chunks`: these patterns fire on words
# that *name* a source ("slite", "ticket", "migrated") but usually describe code,
# and an intent that omits code can only return the wrong thing. Explicit
# `scope=` still narrows to one source. Session is the exception — a question
# about a past conversation is not answered by code.
#
# Whether a query is asking about *history* ("why did we…", "why is X like this")
# rather than current behaviour. Deliberately separate from the intent table and
# order-independent: intent is first-match-wins, so "why did we throttle Slack
# web API calls" is labelled `slack` and never reaches the `pr` rule — yet it is
# plainly a history question. A bare `\bwhy\b` is what closes the gap; on the
# eval sets this scores 16/16 on history questions with 1/40 false positives on
# code questions, and that one ("what do I need to change to add a new
# language") already routes as `pr`.
_HISTORY_PATTERNS = _PR_PATTERNS + [r"\bwhy\b"]
_COMPILED_HISTORY = [re.compile(p) for p in _HISTORY_PATTERNS]


# Ordered intent table. `route` and `classify` walk this same list, so the
# collections a query is sent to and the label it is logged under can never
# disagree. Order is significant — it reproduces the original if/elif chain
# exactly, first match wins (e.g. session before slack, so "what did we discuss
# last session" is not stolen by a later rule).
_INTENT_RULES: list[tuple[str, list[str], list[str]]] = [
    ("session", _SESSION_PATTERNS, SESSION_COLLECTIONS),
    ("slack", _SLACK_PATTERNS, SLACK_COLLECTIONS + CODE_COLLECTIONS),
    ("slite", _SLITE_PATTERNS, SLITE_COLLECTIONS + DOC_COLLECTIONS + CODE_COLLECTIONS),
    ("jira", _JIRA_PATTERNS, JIRA_COLLECTIONS + CODE_COLLECTIONS),
    ("issue", _ISSUE_PATTERNS, ISSUE_COLLECTIONS + JIRA_COLLECTIONS + CODE_COLLECTIONS),
    ("pr", _PR_PATTERNS, PR_COLLECTIONS + CODE_COLLECTIONS),
    ("doc", _DOC_PATTERNS, DOC_COLLECTIONS + CODE_COLLECTIONS),
    ("usage", _USAGE_PATTERNS, ["code_chunks", "pr_diffs"]),
    ("code", _CODE_PATTERNS, CODE_COLLECTIONS),
]

_COMPILED_RULES = [
    (name, [re.compile(p) for p in patterns], collections)
    for name, patterns, collections in _INTENT_RULES
]

_SCOPES: dict[str, list[str]] = {
    "code": CODE_COLLECTIONS,
    "prs": PR_COLLECTIONS,
    "issues": ISSUE_COLLECTIONS,
    "jira": JIRA_COLLECTIONS,
    "slite": SLITE_COLLECTIONS,
    "slack": SLACK_COLLECTIONS,
    "docs": DOC_COLLECTIONS,
    "sessions": SESSION_COLLECTIONS,
}

# Label used when no intent pattern matches and the query fans out to everything.
UNROUTED = "unrouted"


class QueryRouter:
    def route(self, query: str, scope: str = "all") -> list[str]:
        if scope in _SCOPES:
            return list(_SCOPES[scope])
        _, collections = self._match(query)
        return collections

    def classify(self, query: str, scope: str = "all") -> str:
        """Return the intent label a query routes under.

        Used for per-intent metric breakdowns: "usage" and "doc" queries are the
        multi-hop/architectural shapes that graph retrieval would target, while
        "code" is the single-hop shape it is not expected to help. An explicit
        scope short-circuits pattern matching, so it is reported as `scope:<name>`.
        """
        if scope in _SCOPES:
            return f"scope:{scope}"
        name, _ = self._match(query)
        return name

    def wants_history(self, query: str) -> bool:
        """True when the query asks why/when something changed.

        Used to decide whether related-chunk expansion should spend result slots
        on PR history. See `_HISTORY_PATTERNS` for why this is not `classify()`.
        """
        q = query.lower()
        return any(p.search(q) for p in _COMPILED_HISTORY)

    def _match(self, query: str) -> tuple[str, list[str]]:
        q = query.lower()
        for name, patterns, collections in _COMPILED_RULES:
            if any(p.search(q) for p in patterns):
                return name, list(collections)
        return UNROUTED, list(ALL_COLLECTIONS)
