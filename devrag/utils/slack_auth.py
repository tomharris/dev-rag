"""Obtain Slack session credentials (xoxc token + xoxd `d` cookie) automatically.

The web client holds two secrets: the ``d`` cookie (``xoxd-…``) lives in the
browser's cookie store, and the ``xoxc-…`` token is exposed in the boot data of
any workspace page. So we read the cookie from the local browser profile, then
*derive* the token over HTTP — no headless browser or localStorage parsing.

Used by ``devrag auth slack``. See ``devrag/utils/slack_client.py`` for how the
pair is consumed once obtained.
"""

from __future__ import annotations

import json
import re

import httpx

from devrag.utils.slack_client import SlackAuthError

_NO_COOKIE_HINT = (
    "Couldn't read the Slack `d` cookie from your browser ({detail}). Make sure "
    "you're logged in to Slack in {browser_label}, or fall back to the manual "
    "extraction steps in the README."
)

_NO_TOKEN_HINT = (
    "Fetched https://{workspace}.slack.com/ but found no xoxc token in the page. "
    "Check that --workspace is the correct subdomain and that the `d` cookie is "
    "still valid (it expires when your browser session rotates)."
)


def read_d_cookie(browser: str | None = None, domain: str = "slack.com") -> str:
    """Read the Slack ``d`` cookie value from the local browser profile.

    ``browser`` selects a specific browser (``chrome``/``firefox``/``brave``/
    ``edge``); ``None`` lets browser_cookie3 auto-detect across installed
    browsers. Decryption uses the OS keyring, so a logged-in session is enough.

    Raises ``SlackAuthError`` (with the manual-fallback hint) when the cookie is
    absent or can't be decrypted — never returns an empty/garbage value.
    """
    try:
        import browser_cookie3
    except ImportError as exc:  # optional dependency
        raise SlackAuthError(
            "browser_cookie3 is not installed. Install it with "
            "`uv sync --extra slack-auth` (or `pip install 'devrag[slack-auth]'`), "
            "or use the manual extraction steps in the README."
        ) from exc

    loaders = {
        "chrome": getattr(browser_cookie3, "chrome", None),
        "firefox": getattr(browser_cookie3, "firefox", None),
        "brave": getattr(browser_cookie3, "brave", None),
        "edge": getattr(browser_cookie3, "edge", None),
        "chromium": getattr(browser_cookie3, "chromium", None),
    }
    if browser is not None and browser not in loaders:
        raise SlackAuthError(
            f"Unknown browser '{browser}'. Choose one of: {', '.join(loaders)}."
        )
    loader = loaders[browser] if browser else browser_cookie3.load
    browser_label = f"{browser.capitalize()}" if browser else "your browser"

    try:
        jar = loader(domain_name=domain)
    except Exception as exc:  # browser_cookie3 raises various OS/decryption errors
        raise SlackAuthError(
            _NO_COOKIE_HINT.format(detail=exc, browser_label=browser_label)
        ) from exc

    for cookie in jar:
        if cookie.name == "d":
            return cookie.value
    raise SlackAuthError(
        _NO_COOKIE_HINT.format(detail="no `d` cookie found", browser_label=browser_label)
    )


def derive_xoxc_token(workspace: str, cookie: str, *, timeout: float = 30.0) -> str:
    """Derive the ``xoxc-…`` API token for ``workspace`` using the ``d`` cookie.

    Fetches the workspace's web page (which embeds the token in its boot data)
    authenticated by the cookie, then extracts the token from the response.

    Raises ``SlackAuthError`` if the page yields no token (wrong workspace or an
    expired cookie).
    """
    resp = httpx.get(
        f"https://{workspace}.slack.com/",
        cookies={"d": cookie},
        follow_redirects=True,
        timeout=timeout,
    )
    resp.raise_for_status()
    return _extract_token(resp.text, workspace)


def _extract_token(page_html: str, workspace: str) -> str:
    """Extract the xoxc token by JSON-parsing the embedded boot blob.

    Slack inlines its boot data as a JSON object (``boot_data = {…}``) whose
    ``api_token`` field holds the live ``xoxc-…`` token. Parsing the object and
    reading that *key* — rather than regex-grabbing the first ``xoxc-`` string in
    the page — avoids latching onto a placeholder/example token elsewhere in the
    markup. Falls back to scanning any embedded object so a renamed global still
    resolves; raises ``SlackAuthError`` if none yields a valid token.
    """
    for start in _object_starts(page_html):
        blob = _balanced_object(page_html, start)
        if blob is None:
            continue
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        token = data.get("api_token") if isinstance(data, dict) else None
        if isinstance(token, str) and token.startswith("xoxc-"):
            return token
    raise SlackAuthError(_NO_TOKEN_HINT.format(workspace=workspace))


def _object_starts(html: str):
    """Yield candidate ``{`` indices to parse, Slack's ``boot_data`` object first."""
    marker = re.search(r"boot_data\s*[:=]\s*\{", html)
    if marker:
        yield html.index("{", marker.start())
    yield from (m.start() for m in re.finditer(r"\{", html))


def _balanced_object(text: str, open_idx: int) -> str | None:
    """Return the balanced ``{…}`` substring starting at ``open_idx``.

    String-aware (ignores braces inside JSON string literals, honouring ``\\``
    escapes) so embedded ``{``/``}`` in values don't unbalance the scan.
    """
    depth = 0
    in_string = escape = False
    for i in range(open_idx, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx:i + 1]
    return None
