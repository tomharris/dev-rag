import sys
import types

import httpx
import pytest
import respx
from typer.testing import CliRunner

from devrag.cli import app
from devrag.utils import slack_auth
from devrag.utils.slack_auth import derive_xoxc_token, read_d_cookie
from devrag.utils.slack_client import SlackAuthError

runner = CliRunner()

# A boot page with a decoy `xoxc-` placeholder *before* the real boot_data, so a
# naive first-match regex would grab the wrong one — the JSON-parse path must read
# the keyed api_token instead.
BOOT_HTML = (
    '<html><script>var example_token = "xoxc-000-decoy-placeholder";</script>'
    '<script>var boot_data = {"team_id":"T1","api_token":"xoxc-111-222-realtoken"};'
    '</script></html>'
)


# --- token derivation -------------------------------------------------------

@respx.mock
def test_derive_xoxc_token_extracts_from_boot_page():
    respx.get("https://mycorp.slack.com/").mock(
        return_value=httpx.Response(200, html=BOOT_HTML)
    )
    token = derive_xoxc_token("mycorp", "xoxd-cookie")
    assert token == "xoxc-111-222-realtoken"


@respx.mock
def test_derive_xoxc_token_sends_d_cookie():
    route = respx.get("https://mycorp.slack.com/").mock(
        return_value=httpx.Response(200, html=BOOT_HTML)
    )
    derive_xoxc_token("mycorp", "xoxd-secret")
    assert "d=xoxd-secret" in route.calls.last.request.headers.get("cookie", "")


@respx.mock
def test_derive_xoxc_token_raises_when_no_token():
    respx.get("https://mycorp.slack.com/").mock(
        return_value=httpx.Response(200, html="<html>no token here</html>")
    )
    with pytest.raises(SlackAuthError, match="no xoxc token"):
        derive_xoxc_token("mycorp", "xoxd-cookie")


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
    """
    mod = types.ModuleType("browser_cookie3")
    mod._jar = []
    loader = lambda domain_name=None: mod._jar
    mod.load = loader
    for name in ("chrome", "chromium", "firefox", "brave", "edge"):
        setattr(mod, name, loader)
    mod.all_browsers = [loader]
    monkeypatch.setitem(sys.modules, "browser_cookie3", mod)
    return mod


def test_read_d_cookie_returns_value(fake_browser_cookie3):
    fake_browser_cookie3._jar = [_FakeCookie("other", "x"), _FakeCookie("d", "xoxd-yes")]
    assert read_d_cookie() == "xoxd-yes"


def test_read_d_cookie_raises_when_absent(fake_browser_cookie3):
    fake_browser_cookie3._jar = [_FakeCookie("other", "x")]
    with pytest.raises(SlackAuthError, match="no `d` cookie"):
        read_d_cookie()


def test_read_d_cookie_unknown_browser(fake_browser_cookie3):
    with pytest.raises(SlackAuthError, match="Unknown browser"):
        read_d_cookie(browser="netscape")


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
    monkeypatch.setattr(slack_auth, "derive_xoxc_token", lambda ws, cookie: "xoxc-tokenval")
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
