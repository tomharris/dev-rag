from __future__ import annotations

import time
from collections.abc import Iterator

import httpx


class SliteClient:
    def __init__(self, api_token: str) -> None:
        self._client = httpx.Client(
            base_url="https://api.slite.com/v1/",
            headers={
                "Authorization": f"Bearer {api_token}",
                "Accept": "application/json",
            },
            timeout=30.0,
        )

    def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        try:
            resp = self._client.request(method, url, **kwargs)
        except httpx.TimeoutException:
            time.sleep(2)
            resp = self._client.request(method, url, **kwargs)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", "5"))
            time.sleep(min(retry_after, 60))
            resp = self._client.request(method, url, **kwargs)
        resp.raise_for_status()
        return resp

    def list_notes(
        self,
        channel_ids: list[str] | None = None,
        since_days_ago: int | None = None,
        cursor: str | None = None,
    ) -> Iterator[dict]:
        """Paginated listing of notes via the knowledge-management endpoint.

        Yields individual note dicts (id, title, url, updatedAt).
        """
        while True:
            params: dict = {"first": 50}
            if channel_ids:
                params["channelIdList[]"] = channel_ids
            if since_days_ago is not None:
                params["sinceDaysAgo"] = since_days_ago
            if cursor:
                params["cursor"] = cursor
            resp = self._request("GET", "knowledge-management/notes", params=params)
            data = resp.json()
            notes = data.get("notes", [])
            if not notes:
                break
            yield from notes
            if not data.get("hasNextPage", False):
                break
            cursor = data.get("nextCursor")
            if not cursor:
                break

    def get_note(self, note_id: str, fmt: str = "md") -> dict:
        """Fetch a single note with full content."""
        resp = self._request("GET", f"notes/{note_id}", params={"format": fmt})
        return resp.json()

    def close(self) -> None:
        self._client.close()
