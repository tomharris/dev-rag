"""Obtain Slack session credentials (xoxc token + xoxd `d` cookie) automatically.

The web client holds two secrets, both living on disk in the **Slack desktop
app's** Chromium profile (an Electron/Chromium app):

- the ``d`` cookie (``xoxd-…``) in the app's ``Cookies`` SQLite store, and
- the ``xoxc-…`` API token in the web client's ``localConfig_v2`` *localStorage*
  value, persisted in the app's LevelDB store.

We used to *derive* the token over HTTP by scraping the workspace boot page, but
modern Slack no longer inlines the token in any server-rendered page (the page
HTML contains no ``xoxc`` token at all), so that approach is dead. We now read
the token directly from localStorage via a vendored pure-python Chromium-LevelDB
reader (see ``devrag/utils/_vendor/ccl``), which Snappy-decompresses the SSTable
blocks and honours LevelDB sequence numbers so we get the *live* token rather
than a stale/rotated one left behind in an older ``.ldb`` file. The store is read
copy-on-read (we copy it to a temp dir first) and never written to.

Used by ``devrag auth slack``. See ``devrag/utils/slack_client.py`` for how the
pair is consumed once obtained.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path

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

# Where the Slack desktop app keeps its Chromium localStorage LevelDB, per OS.
_SLACK_LOCAL_STORAGE_PATHS = [
    # macOS
    "~/Library/Application Support/Slack/Local Storage/leveldb",
    # Linux (native, snap, flatpak)
    "~/.config/Slack/Local Storage/leveldb",
    "~/snap/slack/current/.config/Slack/Local Storage/leveldb",
    "~/.var/app/com.slack.Slack/config/Slack/Local Storage/leveldb",
]

_NO_LOCAL_STORAGE_HINT = (
    "Couldn't find the Slack desktop app's local data (Local Storage/leveldb). "
    "Make sure the Slack desktop app is installed and you're signed in, or fall "
    "back to the manual extraction steps in the README."
)

_NO_TOKEN_HINT = (
    "Found the Slack desktop app's local data but no xoxc token for any workspace. "
    "Make sure you're signed in to Slack in the desktop app."
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


def slack_local_storage_dir() -> Path | None:
    """Return the Slack desktop app's localStorage LevelDB dir, or ``None``.

    Checks the per-OS install locations (incl. the Windows ``%APPDATA%`` tree)
    and returns the first that exists.
    """
    candidates = list(_SLACK_LOCAL_STORAGE_PATHS)
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(str(Path(appdata) / "Slack" / "Local Storage" / "leveldb"))
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.is_dir():
            return path
    return None


def read_xoxc_token_from_slack_app(workspace: str, *, leveldb_dir: Path | str | None = None) -> str:
    """Read the live ``xoxc-…`` token for ``workspace`` from the Slack desktop app.

    Reads the web client's ``localConfig_v2`` localStorage value (where Slack now
    keeps the token — it's no longer in any server-rendered page) from the desktop
    app's Chromium LevelDB store, picks the team matching ``workspace``, and
    returns its token. ``leveldb_dir`` overrides auto-detection (used in tests).

    Raises ``SlackAuthError`` if the store/token/workspace can't be found.
    """
    src = Path(leveldb_dir).expanduser() if leveldb_dir else slack_local_storage_dir()
    if src is None or not src.is_dir():
        raise SlackAuthError(_NO_LOCAL_STORAGE_HINT)

    config = _load_local_config(src)
    teams = config.get("teams") if isinstance(config, dict) else None
    if not isinstance(teams, dict) or not teams:
        raise SlackAuthError(_NO_TOKEN_HINT)

    token = _token_for_workspace(teams, workspace)
    if token is None:
        known = ", ".join(sorted(filter(None, (_team_label(t) for t in teams.values())))) or "none"
        raise SlackAuthError(
            f"No '{workspace}' workspace found in the Slack desktop app (signed in to: "
            f"{known}). Check --workspace (the <x> in https://<x>.slack.com) and that "
            f"you're signed in to that workspace in the Slack app."
        )
    return token


def _load_local_config(leveldb_dir: Path) -> dict:
    """Parse the live ``localConfig_v2`` localStorage value into a dict.

    Slack may be running and writing to the store, so we copy it to a temp dir
    first for a stable, lock-free read (the same approach ``browser_cookie3`` uses
    for the cookie DB). Among the matching records we take the one with the highest
    LevelDB sequence number, which is the current value — older ``.ldb`` files hold
    stale/rotated copies that must not be returned.
    """
    # Imported lazily: the vendored reader is only needed for `auth slack`.
    from devrag.utils._vendor.ccl import ccl_chromium_localstorage as ccl_ls

    with tempfile.TemporaryDirectory(prefix="devrag-slack-ls-") as tmp:
        tmp_dir = Path(tmp) / "leveldb"
        shutil.copytree(leveldb_dir, tmp_dir)
        with ccl_ls.LocalStoreDb(tmp_dir) as db:
            records = list(
                db.iter_records_for_script_key(
                    re.compile(r"slack\.com"),
                    re.compile(r"localConfig_v2"),
                    raise_on_no_result=False,
                )
            )

    live = [r for r in records if r.is_live] or records
    if not live:
        raise SlackAuthError(_NO_LOCAL_STORAGE_HINT)
    newest = max(live, key=lambda r: r.leveldb_seq_number)
    try:
        return json.loads(newest.value)
    except (json.JSONDecodeError, TypeError) as exc:
        raise SlackAuthError(
            "Slack's localConfig_v2 localStorage value wasn't valid JSON "
            f"({exc}). The desktop app's data may be mid-write — retry."
        ) from exc


def _token_for_workspace(teams: dict, workspace: str) -> str | None:
    """Return the ``xoxc-…`` token for the team matching ``workspace``, or ``None``.

    Matches on the team's ``domain`` first (the subdomain in
    https://<domain>.slack.com), then falls back to its ``url``. No single-team
    fallback: an unmatched ``workspace`` should surface as an error, not silently
    return some other workspace's token.
    """
    ws = workspace.lower()
    for team in teams.values():
        if not isinstance(team, dict):
            continue
        domain = str(team.get("domain") or "").lower()
        url = str(team.get("url") or "").lower()
        if domain == ws or f"{ws}.slack.com" in url:
            token = team.get("token")
            if isinstance(token, str) and token.startswith("xoxc-"):
                return token
    return None


def _team_label(team) -> str | None:
    """A human label for a team (for the 'signed in to: …' error), or ``None``."""
    if not isinstance(team, dict):
        return None
    return team.get("domain") or team.get("name") or team.get("url")
