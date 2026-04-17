from __future__ import annotations
import re
import time
import httpx

API_BASE = "https://api.github.com"


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
    def __init__(self, token: str | None = None) -> None:
        self._headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            self._headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.Client(headers=self._headers, timeout=30.0)

    def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        resp = self._client.request(method, url, **kwargs)
        if resp.status_code in (403, 429):
            remaining = int(resp.headers.get("x-ratelimit-remaining", "1"))
            if remaining == 0:
                reset_at = int(resp.headers.get("x-ratelimit-reset", "0"))
                retry_after = int(resp.headers.get("retry-after", "0"))
                wait = retry_after if retry_after else max(reset_at - time.time(), 0) + 1
                time.sleep(min(wait, 60))
                resp = self._client.request(method, url, **kwargs)
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
