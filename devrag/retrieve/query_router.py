from __future__ import annotations
import re

ALL_COLLECTIONS = ["code_chunks", "pr_diffs", "pr_discussions", "issue_descriptions", "issue_discussions", "jira_descriptions", "jira_discussions", "documents"]
CODE_COLLECTIONS = ["code_chunks"]
PR_COLLECTIONS = ["pr_diffs", "pr_discussions"]
ISSUE_COLLECTIONS = ["issue_descriptions", "issue_discussions"]
JIRA_COLLECTIONS = ["jira_descriptions", "jira_discussions"]
DOC_COLLECTIONS = ["documents"]

_PR_PATTERNS = [
    r"\bwhy\s+did\s+we\b", r"\bwhy\s+was\b", r"\bwhy\s+were\b",
    r"\bwhen\s+did\s+we\b", r"\bwho\s+changed\b", r"\bwho\s+added\b", r"\bwho\s+removed\b",
    r"\bswitch(?:ed)?\s+(?:from|to)\b", r"\bmigrat(?:e|ed|ion)\b",
    r"\bchange(?:d|s)?\s+(?:the|to|from)\b", r"\bremov(?:e|ed)\b.*\bwhy\b",
    r"\bwhy\b.*\bremov(?:e|ed)\b", r"\bintroduc(?:e|ed)\b", r"\brevert(?:ed)?\b", r"\bdeprecated?\b",
]

_ISSUE_PATTERNS = [
    r"\bbug\b", r"\bissue[sd]?\b", r"\bfeature\s+request\b",
    r"\breported?\b", r"\bfiled?\b", r"\bticket\b",
]

_JIRA_PATTERNS = [
    r"\bjira\b", r"\bsprint\b", r"\bepic\b", r"\bstory\b",
    r"\bstory\s+points?\b",
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


class QueryRouter:
    def route(self, query: str, scope: str = "all") -> list[str]:
        if scope == "code":
            return CODE_COLLECTIONS
        if scope == "prs":
            return PR_COLLECTIONS
        if scope == "issues":
            return ISSUE_COLLECTIONS
        if scope == "jira":
            return JIRA_COLLECTIONS
        if scope == "docs":
            return DOC_COLLECTIONS
        q = query.lower()
        for pattern in _JIRA_PATTERNS:
            if re.search(pattern, q):
                return JIRA_COLLECTIONS
        for pattern in _ISSUE_PATTERNS:
            if re.search(pattern, q):
                return ISSUE_COLLECTIONS + JIRA_COLLECTIONS
        for pattern in _PR_PATTERNS:
            if re.search(pattern, q):
                return PR_COLLECTIONS
        for pattern in _DOC_PATTERNS:
            if re.search(pattern, q):
                return DOC_COLLECTIONS
        for pattern in _USAGE_PATTERNS:
            if re.search(pattern, q):
                return ["code_chunks", "pr_diffs"]
        for pattern in _CODE_PATTERNS:
            if re.search(pattern, q):
                return CODE_COLLECTIONS
        return ALL_COLLECTIONS
