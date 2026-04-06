from __future__ import annotations

import base64
import time
from collections.abc import Iterator

import httpx

_BLOCK_TYPES = frozenset({
    "paragraph", "heading", "blockquote", "codeBlock",
    "bulletList", "orderedList", "listItem",
    "table", "tableRow", "tableCell", "tableHeader",
    "rule", "panel", "expand", "mediaGroup", "mediaSingle",
})


class JiraClient:
    def __init__(self, instance_url: str, email: str, api_token: str) -> None:
        credentials = base64.b64encode(f"{email}:{api_token}".encode()).decode()
        self._client = httpx.Client(
            base_url=f"{instance_url.rstrip('/')}/rest/api/3/",
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=30.0,
        )

    def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        resp = self._client.request(method, url, **kwargs)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", "5"))
            time.sleep(min(retry_after, 60))
            resp = self._client.request(method, url, **kwargs)
        resp.raise_for_status()
        return resp

    def search_issues(self, jql: str, fields: list[str], max_results: int = 100) -> Iterator[dict]:
        """Paginated JQL search. Yields individual issues."""
        next_page_token: str | None = None
        while True:
            body: dict = {
                "jql": jql,
                "fields": fields,
                "maxResults": max_results,
            }
            if next_page_token is not None:
                body["nextPageToken"] = next_page_token
            resp = self._request("POST", "search/jql", json=body)
            data = resp.json()
            issues = data.get("issues", [])
            if not issues:
                break
            yield from issues
            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                break

    @staticmethod
    def adf_to_text(adf: dict | str | None) -> str:
        """Recursively extract text from Atlassian Document Format JSON.

        Handles ADF objects (dicts), plain strings (older tickets), and None.
        Block-level nodes are joined with double newlines; inline nodes are concatenated.
        """
        if adf is None:
            return ""
        if isinstance(adf, str):
            return adf
        # TODO: This is a good place for you to implement the recursive extraction.
        # See the guidance below for the expected behavior.
        return _extract_adf_node(adf)

    def close(self) -> None:
        self._client.close()


def _extract_adf_node(node: dict) -> str:
    """Walk an ADF node tree depth-first, collecting text content."""
    if "text" in node:
        return node["text"]
    children = node.get("content", [])
    if not children:
        return ""
    child_texts: list[str] = []
    for child in children:
        text = _extract_adf_node(child)
        if text:
            child_texts.append(text)
    # Block-level children are joined with double newlines;
    # inline children (within a paragraph/heading) are concatenated.
    has_block_children = any(
        isinstance(c, dict) and c.get("type") in _BLOCK_TYPES
        for c in children
    )
    separator = "\n\n" if has_block_children or node.get("type") == "doc" else ""
    return separator.join(child_texts)
