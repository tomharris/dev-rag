"""Obtain Slack session credentials (xoxc token + xoxd `d` cookie) automatically.

The web client holds two secrets: the ``d`` cookie (``xoxd-…``) lives in a
Chromium cookie store, and the ``xoxc-…`` token is exposed in the boot data of
any workspace page. So we read the cookie locally, then *derive* the token over
HTTP — no headless browser or localStorage parsing.

The cookie store we read is, by default, the **Slack desktop app's** (an
Electron/Chromium app — its ``d`` cookie lives in a ``Cookies`` SQLite file just
like a browser's). Browsers are tried as a fallback. Desktop-only users are
never logged in via a browser, so the app is the more reliable source. The token
still lives in the app's LevelDB store, but we never touch that — we derive it
over HTTP, consistent with this module's "no localStorage parsing" design.

Used by ``devrag auth slack``. See ``devrag/utils/slack_client.py`` for how the
pair is consumed once obtained.
"""

from __future__ import annotations

import json
import re

import httpx

from devrag.utils.http import resolve_verify
from devrag.utils.slack_client import SlackAuthError

_NO_COOKIE_HINT = (
    "Couldn't read the Slack `d` cookie from {source_label} ({detail}). Make sure "
    "you're logged in to Slack there, or fall back to the manual extraction steps "
    "in the README."
)

# Where the Slack desktop app keeps its Chromium cookie store, per OS. These are
# handed to ``ChromiumBased`` as candidate lists; it globs and picks the first
# one that exists, so order/extra entries are harmless.
_SLACK_APP_PATHS = {
    "linux_cookies": [
        "~/.config/Slack/Cookies",
        "~/.config/Slack/Network/Cookies",
        # Snap / Flatpak installs keep their own confined config tree.
        "~/snap/slack/current/.config/Slack/Cookies",
        "~/snap/slack/current/.config/Slack/Network/Cookies",
        "~/.var/app/com.slack.Slack/config/Slack/Cookies",
        "~/.var/app/com.slack.Slack/config/Slack/Network/Cookies",
    ],
    "osx_cookies": [
        "~/Library/Application Support/Slack/Cookies",
        "~/Library/Application Support/Slack/Network/Cookies",
    ],
    # Windows paths are resolved relative to %APPDATA% by browser_cookie3, so
    # they must NOT include the variable themselves.
    "windows_cookies": [
        "Slack\\Cookies",
        "Slack\\Network\\Cookies",
    ],
    "windows_keys": ["Slack\\Local State"],
}

_NO_TOKEN_HINT = (
    "Fetched https://{workspace}.slack.com/ but found no xoxc token in the page. "
    "Check that --workspace is the correct subdomain and that the `d` cookie is "
    "still valid (it expires when your browser session rotates)."
)


def read_d_cookie(browser: str | None = None, domain: str = "slack.com") -> str:
    """Read the Slack ``d`` cookie value from a local Chromium cookie store.

    ``browser`` selects a specific source: ``slack`` (alias ``desktop``) reads the
    Slack desktop app; ``chrome``/``firefox``/``brave``/``edge``/``chromium`` read
    that browser. ``None`` (auto) tries the **Slack desktop app first**, then
    sweeps installed browsers. Decryption uses the OS keyring, so a logged-in
    session is enough.

    Raises ``SlackAuthError`` (with the manual-fallback hint) when the cookie is
    absent or can't be decrypted — never returns an empty/garbage value.
    """
    # Lazy import: browser_cookie3 pulls in slow keyring/crypto libs, so we only
    # pay that cost when auth actually runs. It's a core dependency, so the
    # import itself won't fail.
    import browser_cookie3

    loaders = {
        "chrome": getattr(browser_cookie3, "chrome", None),
        "firefox": getattr(browser_cookie3, "firefox", None),
        "brave": getattr(browser_cookie3, "brave", None),
        "edge": getattr(browser_cookie3, "edge", None),
        "chromium": getattr(browser_cookie3, "chromium", None),
    }

    # Explicit Slack desktop app.
    if browser in ("slack", "desktop"):
        return read_d_cookie_from_slack_app(domain=domain)

    if browser is not None and browser not in loaders:
        raise SlackAuthError(
            f"Unknown source '{browser}'. Choose 'slack' (desktop app) or one of: "
            f"{', '.join(loaders)}."
        )

    # Explicit browser: try the one loader and surface its real failure (e.g. a
    # keyring-decryption error) verbatim — don't swallow it.
    if browser is not None:
        browser_label = browser.capitalize()
        try:
            jar = loaders[browser](domain_name=domain)
        except Exception as exc:  # browser_cookie3 raises various OS/decryption errors
            raise SlackAuthError(
                _NO_COOKIE_HINT.format(detail=exc, source_label=browser_label)
            ) from exc
        return _find_d_cookie(jar, browser_label)

    # Auto-detect: the Slack desktop app is the most reliable source (desktop-only
    # users have no browser session), so try it first and fall through to browsers
    # only if it yields nothing.
    try:
        return read_d_cookie_from_slack_app(domain=domain)
    except SlackAuthError:
        pass

    # Browser sweep: iterate every supported browser ourselves rather than via
    # browser_cookie3.load(), which only catches BrowserCookieError. A loader
    # that raises anything else (notably the Arc loader's TypeError on Linux,
    # where Arc has no cookie path) would otherwise abort load()'s whole sweep
    # and discard cookies already collected from Chrome/Firefox. We try each
    # independently and skip the ones that throw.
    skipped = 0
    for loader in browser_cookie3.all_browsers:
        try:
            jar = loader(domain_name=domain)
        except Exception:  # unsupported/locked profile for this browser — skip it
            skipped += 1
            continue
        for cookie in jar:
            if cookie.name == "d":
                return cookie.value
    detail = "no `d` cookie found"
    if skipped:
        detail += f"; {skipped} browser profile(s) couldn't be read and were skipped"
    raise SlackAuthError(
        _NO_COOKIE_HINT.format(detail=detail, source_label="the Slack desktop app or any browser")
    )


def read_d_cookie_from_slack_app(domain: str = "slack.com") -> str:
    """Read the Slack ``d`` cookie from the Slack **desktop app's** cookie store.

    The app is an Electron/Chromium app, so its ``Cookies`` SQLite file decrypts
    with the same machinery ``browser_cookie3`` uses for real browsers — we just
    point ``ChromiumBased`` at Slack's paths and OS-crypt identity ("Slack Safe
    Storage" on macOS, the ``slack`` Secret Service entry on Linux, the app's
    ``Local State`` DPAPI key on Windows). ``ChromiumBased`` copies the DB to a
    temp file before reading, so this works even while Slack is running.

    Raises ``SlackAuthError`` (app not installed / not logged in / cookie absent).
    """
    from browser_cookie3 import ChromiumBased

    try:
        jar = ChromiumBased(
            browser="Slack",
            domain_name=domain,
            os_crypt_name="slack",
            osx_key_service="Slack Safe Storage",
            osx_key_user="Slack",
            **_SLACK_APP_PATHS,
        ).load()
    except Exception as exc:  # no cookie file (app not installed) / decryption error
        raise SlackAuthError(
            _NO_COOKIE_HINT.format(detail=exc, source_label="the Slack desktop app")
        ) from exc
    return _find_d_cookie(jar, "the Slack desktop app")


def _find_d_cookie(jar, source_label: str) -> str:
    """Return the ``d`` cookie value from ``jar`` or raise the no-cookie hint."""
    for cookie in jar:
        if cookie.name == "d":
            return cookie.value
    raise SlackAuthError(
        _NO_COOKIE_HINT.format(detail="no `d` cookie found", source_label=source_label)
    )


def derive_xoxc_token(workspace: str, cookie: str, *, timeout: float = 30.0,
                      ca_bundle: str | None = None) -> str:
    """Derive the ``xoxc-…`` API token for ``workspace`` using the ``d`` cookie.

    Fetches the workspace's web page (which embeds the token in its boot data)
    authenticated by the cookie, then extracts the token from the response.

    Raises ``SlackAuthError`` if the page yields no token (wrong workspace or an
    expired cookie).
    """
    verify = resolve_verify(ca_bundle)
    resp = httpx.get(
        f"https://{workspace}.slack.com/",
        cookies={"d": cookie},
        follow_redirects=True,
        timeout=timeout,
        verify=verify,
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
