from __future__ import annotations

import random
import time
from collections.abc import Iterator

import httpx

from devrag.utils.http import resolve_verify


class SliteClient:
    def __init__(self, api_token: str, verify: str | bool | None = None,
                 max_retries: int = 5) -> None:
        if verify is None:
            verify = resolve_verify()
        self._max_retries = max(max_retries, 0)
        self._client = httpx.Client(
            base_url="https://api.slite.com/v1/",
            headers={
                "Authorization": f"Bearer {api_token}",
                "Accept": "application/json",
            },
            timeout=30.0,
            verify=verify,
        )

    @staticmethod
    def _backoff(attempt: int, retry_after: str | None = None) -> float:
        """Seconds to wait before the next attempt: honor ``Retry-After`` when
        present, else exponential backoff with jitter, capped at 60s."""
        if retry_after is not None:
            try:
                return min(float(retry_after), 60.0)
            except ValueError:
                pass
        return min(2.0 ** attempt, 60.0) + random.uniform(0, 0.5)

    def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Issue a request, retrying transient failures (429/5xx/timeout) with
        ``Retry-After``-aware exponential backoff up to ``max_retries`` times."""
        for attempt in range(self._max_retries + 1):
            last = attempt == self._max_retries
            try:
                resp = self._client.request(method, url, **kwargs)
            except httpx.TimeoutException:
                if last:
                    raise
                time.sleep(self._backoff(attempt))
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                if last:
                    resp.raise_for_status()
                time.sleep(self._backoff(attempt, resp.headers.get("Retry-After")))
                continue
            break
        resp.raise_for_status()
        return resp

    def list_notes(
        self,
        channel_ids: list[str] | None = None,
        since_days_ago: int | None = None,
        cursor: str | None = None,
    ) -> Iterator[dict]:
        """Paginated listing of notes via the knowledge-management endpoint.

        Yields individual note dicts. Each includes ``id``, ``title``, ``url``,
        ``updatedAt`` (a volatile metadata/popularity timestamp), and
        ``lastEditedAt`` (the actual content-edit time — use this for
        incremental sync).

        Note: ``since_days_ago`` maps to the API's ``sinceDaysAgo``, which is a
        popularity window rather than an edit filter, so it does not reliably
        narrow by modification time; callers doing incremental sync should
        filter client-side on ``lastEditedAt`` instead.
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
