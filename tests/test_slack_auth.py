import sys
import types

import pytest
from typer.testing import CliRunner

from devrag.cli import app
from devrag.utils import slack_auth
from devrag.utils.slack_auth import (
    read_d_cookie,
    read_d_cookie_from_slack_app,
    read_xoxc_token_from_slack_app,
)
from devrag.utils.slack_client import SlackAuthError

runner = CliRunner()


# --- token reading (localStorage / LevelDB) ---------------------------------
#
# Slack no longer inlines the xoxc token in any web page; it lives in the desktop
# app's localConfig_v2 localStorage value, read from its Chromium LevelDB store.

def test_token_for_workspace_matches_domain_case_insensitively():
    teams = {
        "T1": {"domain": "bamboohr", "url": "https://bamboohr.slack.com/", "token": "xoxc-AAA"},
        "T2": {"domain": "other", "url": "https://other.slack.com/", "token": "xoxc-BBB"},
    }
    assert slack_auth._token_for_workspace(teams, "bamboohr") == "xoxc-AAA"
    assert slack_auth._token_for_workspace(teams, "BambooHR") == "xoxc-AAA"
    assert slack_auth._token_for_workspace(teams, "other") == "xoxc-BBB"


def test_token_for_workspace_matches_url_when_domain_absent():
    teams = {"T1": {"url": "https://bamboohr.slack.com/", "token": "xoxc-URL"}}
    assert slack_auth._token_for_workspace(teams, "bamboohr") == "xoxc-URL"


def test_token_for_workspace_no_match_returns_none():
    teams = {"T1": {"domain": "bamboohr", "token": "xoxc-AAA"}}
    assert slack_auth._token_for_workspace(teams, "nope") is None


def test_token_for_workspace_ignores_non_xoxc_values():
    teams = {"T1": {"domain": "bamboohr", "token": "not-a-real-token"}}
    assert slack_auth._token_for_workspace(teams, "bamboohr") is None


def test_read_xoxc_token_raises_when_store_missing(tmp_path):
    with pytest.raises(SlackAuthError, match="Local Storage"):
        read_xoxc_token_from_slack_app("bamboohr", leveldb_dir=tmp_path / "nope")


def test_read_xoxc_token_returns_matched_token(monkeypatch, tmp_path):
    leveldb = tmp_path / "leveldb"
    leveldb.mkdir()
    monkeypatch.setattr(
        slack_auth, "_load_local_config",
        lambda d: {"teams": {"T1": {"domain": "bamboohr", "token": "xoxc-live"}}},
    )
    assert read_xoxc_token_from_slack_app("bamboohr", leveldb_dir=leveldb) == "xoxc-live"


def test_read_xoxc_token_lists_known_workspaces_on_mismatch(monkeypatch, tmp_path):
    leveldb = tmp_path / "leveldb"
    leveldb.mkdir()
    monkeypatch.setattr(
        slack_auth, "_load_local_config",
        lambda d: {"teams": {"T1": {"domain": "bamboohr", "token": "xoxc-live"}}},
    )
    with pytest.raises(SlackAuthError, match="signed in to: bamboohr"):
        read_xoxc_token_from_slack_app("wrongcorp", leveldb_dir=leveldb)


def test_load_local_config_picks_newest_live_record(monkeypatch, tmp_path):
    # The core anti-staleness guarantee: older .ldb files keep rotated tokens, so
    # we must return the live record with the highest LevelDB sequence number.
    import devrag.utils._vendor.ccl.ccl_chromium_localstorage as ccl_ls

    leveldb = tmp_path / "leveldb"
    leveldb.mkdir()

    class _Rec:
        def __init__(self, value, seq, is_live):
            self.value = value
            self.leveldb_seq_number = seq
            self.is_live = is_live

    class _FakeDb:
        def __init__(self, _dir):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def iter_records_for_script_key(self, _sk, _kk, *, raise_on_no_result=True):
            return [
                _Rec('{"teams":{"T1":{"domain":"bamboohr","token":"xoxc-STALE"}}}', 5, True),
                _Rec('{"teams":{"T1":{"domain":"bamboohr","token":"xoxc-LIVE"}}}', 9, True),
                _Rec("not-even-json-and-newest", 99, False),  # not live → ignored
            ]

    monkeypatch.setattr(ccl_ls, "LocalStoreDb", _FakeDb)
    cfg = slack_auth._load_local_config(leveldb)
    assert cfg["teams"]["T1"]["token"] == "xoxc-LIVE"


# --- cookie reading ---------------------------------------------------------

class _FakeCookie:
    def __init__(self, name, value):
        self.name = name
        self.value = value


@pytest.fixture
def fake_browser_cookie3(monkeypatch):
    """Inject a fake browser_cookie3 module.

    Tests set ``_jar`` (the cookies a loader returns); both the named loaders
    and the auto-detect ``all_browsers`` list resolve through it. Tests that
    exercise the multi-browser sweep override ``all_browsers`` directly.

    ``ChromiumBased`` (used by the Slack desktop-app reader) defaults to "app not
    installed" — its ``.load()`` raises — so auto-mode cleanly falls through to
    the browser sweep. Desktop-specific tests replace ``ChromiumBased``.
    """
    mod = types.ModuleType("browser_cookie3")
    mod._jar = []
    loader = lambda domain_name=None: mod._jar
    mod.load = loader
    for name in ("chrome", "chromium", "firefox", "brave", "edge"):
        setattr(mod, name, loader)
    mod.all_browsers = [loader]

    class _NoSlackApp:
        def __init__(self, **kwargs):
            pass

        def load(self):
            raise Exception("Failed to find cookies for Slack browser")

    mod.ChromiumBased = _NoSlackApp
    monkeypatch.setitem(sys.modules, "browser_cookie3", mod)
    return mod


def _slack_app_returning(value):
    """A fake ``ChromiumBased`` whose ``.load()`` yields a jar with a ``d`` cookie."""
    return _slack_app_by_account({}, default=value)


def _slack_app_by_account(behaviors, default=None):
    """A fake ``ChromiumBased`` driven by a per-keychain-account behaviour map.

    ``behaviors`` maps ``osx_key_user`` → the ``d`` cookie value that account
    decrypts to, or an ``Exception`` instance for accounts whose key lookup
    fails. ``default`` applies to accounts not named in the map; ``None`` means
    "this account has no keychain entry" and raises, mirroring browser_cookie3.
    ``accounts`` records the probe order so tests can assert on it.
    """
    class _FakeSlackApp:
        accounts: list = []
        last_kwargs: dict = {}

        def __init__(self, **kwargs):
            type(self).last_kwargs = kwargs
            self._account = kwargs.get("osx_key_user")
            type(self).accounts.append(self._account)

        def load(self):
            outcome = behaviors.get(self._account, default)
            if isinstance(outcome, Exception):
                raise outcome
            if outcome is None:
                raise Exception("Unable to get key for cookie decryption")
            return [_FakeCookie("d", outcome)]

    _FakeSlackApp.accounts = []
    return _FakeSlackApp


def test_read_d_cookie_returns_value(fake_browser_cookie3):
    fake_browser_cookie3._jar = [_FakeCookie("other", "x"), _FakeCookie("d", "xoxd-yes")]
    assert read_d_cookie() == "xoxd-yes"


def test_read_d_cookie_raises_when_absent(fake_browser_cookie3):
    fake_browser_cookie3._jar = [_FakeCookie("other", "x")]
    with pytest.raises(SlackAuthError, match="no `d` cookie"):
        read_d_cookie()


def test_read_d_cookie_unknown_browser(fake_browser_cookie3):
    with pytest.raises(SlackAuthError, match="Unknown source"):
        read_d_cookie(browser="netscape")


# --- install-location probing -----------------------------------------------
#
# The Mac App Store build is sandboxed: its profile lives under ~/Library/
# Containers, and such a machine has no ~/Library/Application Support/Slack at
# all. Both the cookie store and the localStorage LevelDB must probe it.

_APP_STORE_MARKER = "com.tinyspeck.slackmacgap"


def test_macos_paths_cover_both_install_locations():
    cookies = slack_auth._SLACK_APP_PATHS["osx_cookies"]
    leveldb = [p for p in slack_auth._SLACK_LOCAL_STORAGE_PATHS if p.startswith("~/Library")]

    for paths in (cookies, leveldb):
        assert any(_APP_STORE_MARKER in p for p in paths)
        # Direct download stays first so it wins when both installs exist.
        assert _APP_STORE_MARKER not in paths[0]


def test_slack_local_storage_dir_finds_app_store_container(monkeypatch, tmp_path):
    missing = tmp_path / "Application Support" / "Slack" / "Local Storage" / "leveldb"
    container = tmp_path / _APP_STORE_MARKER / "Slack" / "Local Storage" / "leveldb"
    container.mkdir(parents=True)

    monkeypatch.setattr(
        slack_auth, "_SLACK_LOCAL_STORAGE_PATHS", [str(missing), str(container)]
    )
    assert slack_auth.slack_local_storage_dir() == container


# --- Slack desktop app source ----------------------------------------------

def test_read_d_cookie_from_slack_app_reads_desktop_cookie(fake_browser_cookie3):
    fake_browser_cookie3.ChromiumBased = _slack_app_returning("xoxd-desktop")
    assert read_d_cookie_from_slack_app() == "xoxd-desktop"
    # Slack's Chromium identity must be passed through to browser_cookie3.
    kwargs = fake_browser_cookie3.ChromiumBased.last_kwargs
    assert kwargs["browser"] == "Slack"
    assert kwargs["os_crypt_name"] == "slack"
    assert kwargs["osx_key_service"] == "Slack Safe Storage"
    # The keychain account is probed from _MACOS_KEYCHAIN_ACCOUNTS; the direct
    # download's "Slack Key" comes first, and one success stops the probe.
    assert fake_browser_cookie3.ChromiumBased.accounts == ["Slack Key"]


def test_read_d_cookie_from_slack_app_tries_app_store_keychain_account(fake_browser_cookie3):
    # Mac App Store build: "Slack Key" doesn't exist, "Slack App Store Key" does.
    # A machine with only the sandboxed install must not fail on the first miss.
    fake_browser_cookie3.ChromiumBased = _slack_app_by_account(
        {"Slack App Store Key": "xoxd-appstore"}
    )
    assert read_d_cookie_from_slack_app() == "xoxd-appstore"
    assert fake_browser_cookie3.ChromiumBased.accounts == ["Slack Key", "Slack App Store Key"]


def test_read_d_cookie_from_slack_app_rejects_undecryptable_value(fake_browser_cookie3):
    # Cookie *names* aren't encrypted, so a wrong-but-existing keychain account
    # yields a `d` cookie holding garbage instead of raising. That must be
    # rejected on the xoxd- prefix and the next account tried.
    fake_browser_cookie3.ChromiumBased = _slack_app_by_account(
        {"Slack Key": "\x17\xa3garbage", "Slack App Store Key": "xoxd-real"}
    )
    assert read_d_cookie_from_slack_app() == "xoxd-real"


def test_read_d_cookie_from_slack_app_raises_when_no_account_works(fake_browser_cookie3):
    fake_browser_cookie3.ChromiumBased = _slack_app_by_account({}, default="not-a-token")
    with pytest.raises(SlackAuthError, match="Slack App Store Key"):
        read_d_cookie_from_slack_app()
    assert fake_browser_cookie3.ChromiumBased.accounts == list(
        slack_auth._MACOS_KEYCHAIN_ACCOUNTS
    )


def test_read_d_cookie_from_slack_app_raises_when_unavailable(fake_browser_cookie3):
    # Default fixture ChromiumBased.load() raises (app not installed).
    with pytest.raises(SlackAuthError, match="Slack desktop app"):
        read_d_cookie_from_slack_app()


def test_read_d_cookie_auto_prefers_desktop_app(fake_browser_cookie3):
    # Desktop app wins even when a browser also holds a `d` cookie.
    fake_browser_cookie3.ChromiumBased = _slack_app_returning("xoxd-desktop")
    fake_browser_cookie3._jar = [_FakeCookie("d", "xoxd-browser")]
    assert read_d_cookie() == "xoxd-desktop"


def test_read_d_cookie_auto_falls_back_to_browser(fake_browser_cookie3):
    # Desktop app unavailable (default fixture) → browser sweep supplies it.
    fake_browser_cookie3._jar = [_FakeCookie("d", "xoxd-browser")]
    assert read_d_cookie() == "xoxd-browser"


def test_read_d_cookie_explicit_slack_skips_browsers(fake_browser_cookie3):
    fake_browser_cookie3.ChromiumBased = _slack_app_returning("xoxd-desktop")

    def boom(domain_name=None):
        raise AssertionError("browsers must not be read when source is 'slack'")

    fake_browser_cookie3.all_browsers = [boom]
    assert read_d_cookie(browser="slack") == "xoxd-desktop"


def test_read_d_cookie_skips_failing_browser_loader(fake_browser_cookie3):
    # Reproduces the Arc-on-Linux bug: one loader raises a TypeError (not a
    # BrowserCookieError), which must not abort the auto-detect sweep. The
    # logged-in browser later in the list should still yield the `d` cookie.
    def broken(domain_name=None):
        raise TypeError("expected str, bytes or os.PathLike object, not NoneType")

    def good(domain_name=None):
        return [_FakeCookie("d", "xoxd-recovered")]

    fake_browser_cookie3.all_browsers = [broken, good]
    assert read_d_cookie() == "xoxd-recovered"


def test_read_d_cookie_explicit_browser_surfaces_real_error(fake_browser_cookie3):
    # When a specific browser is requested, a genuine loader failure (e.g.
    # keyring decryption) must be surfaced, not silently swallowed.
    def boom(domain_name=None):
        raise RuntimeError("keyring is locked")

    fake_browser_cookie3.chrome = boom
    with pytest.raises(SlackAuthError, match="keyring is locked"):
        read_d_cookie(browser="chrome")


# --- CLI --------------------------------------------------------------------

def test_auth_slack_emits_exports(monkeypatch):
    monkeypatch.setattr(slack_auth, "read_d_cookie", lambda browser=None: "xoxd-cookieval")
    monkeypatch.setattr(slack_auth, "read_xoxc_token_from_slack_app", lambda ws: "xoxc-tokenval")
    monkeypatch.setattr(
        "devrag.utils.slack_client.SlackClient.auth_test",
        lambda self: {"user": "tom", "team": "MyCorp"},
    )
    result = runner.invoke(app, ["auth", "slack", "--workspace", "mycorp"])
    assert result.exit_code == 0
    assert "export SLACK_XOXC_TOKEN=xoxc-tokenval" in result.stdout
    assert "export SLACK_XOXD_COOKIE=xoxd-cookieval" in result.stdout


def test_auth_slack_requires_workspace(monkeypatch):
    # Ensure no workspace leaks in from a project .devrag.yaml.
    from devrag.config import DevragConfig
    monkeypatch.setattr("devrag.config.load_config", lambda **kw: DevragConfig())
    result = runner.invoke(app, ["auth", "slack"])
    assert result.exit_code == 1
    assert "no workspace" in result.output.lower()
