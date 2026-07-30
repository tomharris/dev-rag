from __future__ import annotations
import re
import time
import httpx

from devrag.utils.http import (
    MAX_BACKOFF_SECONDS,
    backoff_delay,
    parse_retry_after,
    request_with_retries,
    resolve_verify,
    safe_int,
)

API_BASE = "https://api.github.com"


def _rate_limit_retry_delay(resp: httpx.Response, attempt: int) -> float | None:
    """Seconds to wait before retrying, or ``None`` to accept the response.

    GitHub signals throttling in three different shapes, and only handling one
    of them is what let long syncs die mid-run:

    * **Primary limit** — ``403`` with ``x-ratelimit-remaining: 0``; the quota
      refills at ``x-ratelimit-reset`` (a Unix timestamp).
    * **Secondary limit** — ``429`` (abuse detection) with ``Retry-After`` set
      and ``x-ratelimit-remaining`` *non-zero or absent*, so a
      ``remaining == 0`` test never fires for it.
    * **Server errors** — ordinary ``5xx``, retried with plain backoff.

    A ``403`` with quota still remaining is an authorization failure, not
    throttling, so it is returned as-is rather than retried — surfacing "token
    lacks this scope" immediately instead of stalling on it.
    """
    if resp.status_code >= 500:
        return backoff_delay(attempt, resp.headers.get("Retry-After"))
    if resp.status_code not in (403, 429):
        return None
    retry_after = parse_retry_after(resp.headers.get("Retry-After"))
    if retry_after is not None:
        return min(retry_after, MAX_BACKOFF_SECONDS)
    if safe_int(resp.headers.get("x-ratelimit-remaining"), 1) == 0:
        reset_at = safe_int(resp.headers.get("x-ratelimit-reset"), 0)
        # +1s of slack so we wake *after* the window rolls over, not on its edge.
        return min(max(reset_at - time.time(), 0) + 1, MAX_BACKOFF_SECONDS)
    return None


def parse_diff_hunks(patch: str, file_path: str) -> list[dict]:
    if not patch:
        return []
    hunks: list[dict] = []
    current_header = ""
    current_lines: list[str] = []
    for line in patch.split("\n"):
        if line.startswith("@@"):
            if current_lines:
                hunks.append({"file_path": file_path, "header": current_header, "content": "\n".join(current_lines)})
            current_header = line
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        hunks.append({"file_path": file_path, "header": current_header, "content": "\n".join(current_lines)})
    return hunks


def _get_next_url(response: httpx.Response) -> str | None:
    link = response.headers.get("link", "")
    match = re.search(r'<([^>]+)>;\s*rel="next"', link)
    return match.group(1) if match else None


class GitHubClient:
    def __init__(self, token: str | None = None, verify: str | bool | None = None,
                 max_retries: int = 5) -> None:
        self._headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            self._headers["Authorization"] = f"Bearer {token}"
        if verify is None:
            verify = resolve_verify()
        self._max_retries = max(max_retries, 0)
        self._client = httpx.Client(headers=self._headers, timeout=30.0, verify=verify)

    def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Issue a request, waiting out rate limits and retrying transient
        failures up to ``max_retries`` times (see ``_rate_limit_retry_delay``)."""
        resp = request_with_retries(
            lambda: self._client.request(method, url, **kwargs),
            max_retries=self._max_retries,
            retry_delay=_rate_limit_retry_delay,
        )
        resp.raise_for_status()
        return resp

    def paginate(self, url: str, params: dict | None = None) -> list[dict]:
        all_items: list[dict] = []
        current_url = url
        current_params = params
        while current_url:
            resp = self._request("GET", current_url, params=current_params)
            all_items.extend(resp.json())
            current_url = _get_next_url(resp)
            current_params = None
        return all_items

    def list_prs(self, repo: str, state: str = "all", sort: str = "updated",
                 direction: str = "desc", per_page: int = 100, since: str | None = None) -> list[dict]:
        params: dict = {"state": state, "sort": sort, "direction": direction, "per_page": per_page}
        url = f"{API_BASE}/repos/{repo}/pulls"
        if since is None:
            return self.paginate(url, params=params)
        # GitHub's /pulls endpoint has no server-side `since` filter. Since sort=updated,
        # direction=desc, we can stop paginating as soon as we see an item older than `since`.
        items: list[dict] = []
        current_url: str | None = url
        current_params: dict | None = params
        while current_url:
            resp = self._request("GET", current_url, params=current_params)
            for item in resp.json():
                if item.get("updated_at", "") < since:
                    return items
                items.append(item)
            current_url = _get_next_url(resp)
            current_params = None
        return items

    def get_pr_files(self, repo: str, pr_number: int) -> list[dict]:
        url = f"{API_BASE}/repos/{repo}/pulls/{pr_number}/files"
        return self.paginate(url, params={"per_page": 100})

    def get_pr_comments(self, repo: str, pr_number: int) -> list[dict]:
        url = f"{API_BASE}/repos/{repo}/pulls/{pr_number}/comments"
        return self.paginate(url, params={"per_page": 100})

    def list_issues(self, repo: str, state: str = "all", sort: str = "updated",
                    direction: str = "desc", per_page: int = 100, since: str | None = None) -> list[dict]:
        params: dict = {"state": state, "sort": sort, "direction": direction, "per_page": per_page}
        if since:
            params["since"] = since
        url = f"{API_BASE}/repos/{repo}/issues"
        return self.paginate(url, params=params)

    def get_issue_comments(self, repo: str, issue_number: int) -> list[dict]:
        url = f"{API_BASE}/repos/{repo}/issues/{issue_number}/comments"
        return self.paginate(url, params={"per_page": 100})

    def close(self) -> None:
        self._client.close()
