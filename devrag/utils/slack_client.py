from __future__ import annotations

import time
from collections.abc import Iterator

import httpx

from devrag.utils.http import resolve_verify


class SlackError(Exception):
    """A Slack web-API call returned ``ok: false``."""


class SlackAuthError(SlackError):
    """The session token/cookie was rejected (expired or invalid)."""


# Slack errors that mean the browser credentials are no longer valid.
_AUTH_ERRORS = {"invalid_auth", "not_authed", "token_expired", "token_revoked"}

_AUTH_HINT = (
    "Slack rejected the session credentials ({error}). Re-extract a fresh "
    "xoxc token (web-client localStorage) and xoxd `d` cookie from your browser."
)


class SlackClient:
    """Thin wrapper over Slack's internal web API using browser session credentials.

    Authenticates as the logged-in user with an ``xoxc-…`` token plus the ``d``
    cookie (``xoxd-…``) — the same pair the web client uses — so no Slack App or
    workspace install is required. The token is sent as the ``token`` form field
    and the cookie as a ``d=…`` cookie, matching the web client's requests.
    """

    def __init__(self, token: str, cookie: str, page_size: int = 200,
                 ca_bundle: str | None = None) -> None:
        self._token = token
        self._page_size = page_size
        verify = resolve_verify(ca_bundle)
        self._client = httpx.Client(
            base_url="https://slack.com/api/",
            headers={"Accept": "application/json"},
            cookies={"d": cookie},
            timeout=30.0,
            verify=verify,
        )

    def _call(self, endpoint: str, params: dict) -> dict:
        """POST a form-encoded API call, retrying transient failures.

        Raises ``SlackAuthError`` for credential failures and ``SlackError`` for
        any other ``ok: false`` body, so callers never silently process empty
        results from an expired token.
        """
        data = {"token": self._token, **params}
        try:
            resp = self._client.post(endpoint, data=data)
        except httpx.TimeoutException:
            time.sleep(2)
            resp = self._client.post(endpoint, data=data)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", "5"))
            time.sleep(min(retry_after, 60))
            resp = self._client.post(endpoint, data=data)
        resp.raise_for_status()
        body = resp.json()
        if not body.get("ok", False):
            error = body.get("error", "unknown_error")
            if error in _AUTH_ERRORS:
                raise SlackAuthError(_AUTH_HINT.format(error=error))
            raise SlackError(f"Slack API {endpoint} failed: {error}")
        return body

    def _paginate(self, endpoint: str, key: str, params: dict) -> Iterator[dict]:
        """Yield items under ``key`` across cursor-paginated pages."""
        cursor = ""
        while True:
            page_params = {"limit": self._page_size, **params}
            if cursor:
                page_params["cursor"] = cursor
            body = self._call(endpoint, page_params)
            yield from body.get(key, [])
            cursor = body.get("response_metadata", {}).get("next_cursor", "")
            if not cursor:
                break

    def list_conversations(self, types: str = "public_channel") -> Iterator[dict]:
        """Yield channel dicts the user can see (id, name, is_member, …)."""
        yield from self._paginate("conversations.list", "channels", {"types": types})

    def conversations_history(self, channel: str, oldest: str | None = None) -> Iterator[dict]:
        """Yield messages in a channel, newest-first, optionally since ``oldest`` ts."""
        params: dict = {"channel": channel}
        if oldest:
            params["oldest"] = oldest
        yield from self._paginate("conversations.history", "messages", params)

    def conversations_replies(self, channel: str, thread_ts: str) -> list[dict]:
        """Return a thread's messages (root first), following pagination."""
        return list(
            self._paginate("conversations.replies", "messages",
                           {"channel": channel, "ts": thread_ts})
        )

    def users_list(self) -> Iterator[dict]:
        """Yield workspace member dicts for resolving user IDs to display names."""
        yield from self._paginate("users.list", "members", {})

    def auth_test(self) -> dict:
        """Return ``{user, team, url, …}`` for the session, validating the credentials.

        A cheap single call (no pagination) that confirms the token/cookie are
        live — it raises ``SlackAuthError`` via ``_call`` if they were rejected.
        """
        return self._call("auth.test", {})

    def close(self) -> None:
        self._client.close()
