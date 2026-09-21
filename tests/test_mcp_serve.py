"""Network-serving policy for the MCP server.

`configure_server` mutates a module-level singleton, so every test reloads
`devrag.mcp_server` to get a fresh one — otherwise a `--read-only` test would
strip the write tools for the rest of the session.
"""
from __future__ import annotations

import asyncio
import importlib
from unittest import mock

import pytest
from fastmcp.exceptions import AuthorizationError
from typer.testing import CliRunner

from devrag.cli import app

runner = CliRunner()


@pytest.fixture
def server_module():
    import devrag.mcp_server as mod
    yield importlib.reload(mod)
    importlib.reload(mod)


def _tool_names(mcp) -> set[str]:
    return {t.name for t in asyncio.run(mcp.list_tools())}


def test_read_only_removes_write_tools(server_module):
    before = _tool_names(server_module.mcp)
    assert {"search", "status"} <= before
    assert set(server_module.WRITE_TOOLS) <= before

    server_module.configure_server(read_only=True)

    after = _tool_names(server_module.mcp)
    assert after == {"search", "status"}


def test_read_write_is_the_default(server_module):
    server_module.configure_server()
    assert set(server_module.WRITE_TOOLS) <= _tool_names(server_module.mcp)


def test_auth_token_installs_a_verifier(server_module):
    assert server_module.mcp.auth is None
    server_module.configure_server(auth_token="s3cret")
    assert server_module.mcp.auth is not None


def test_empty_auth_token_leaves_the_server_open(server_module):
    server_module.configure_server(auth_token="")
    assert server_module.mcp.auth is None


def test_prefer_repo_override_is_applied(server_module):
    assert server_module._default_prefer_repo is None
    server_module.configure_server(prefer_repo="")
    # "" is a real value, not "unset": it disables cwd inference outright.
    assert server_module._default_prefer_repo == ""


def test_prefer_repo_none_keeps_cwd_inference(server_module):
    server_module.configure_server(prefer_repo=None)
    assert server_module._default_prefer_repo is None


def test_serve_rejects_an_unknown_transport():
    result = runner.invoke(app, ["serve", "--transport", "websocket"])
    assert result.exit_code == 2
    assert "stdio" in result.output


# --- GitHub OAuth path (public endpoint / Claude connectors) ---------------


def _run_allowlist(allowlist, token):
    """Drive the middleware with a stubbed access token."""
    import devrag.mcp_server as mod

    async def call_next(_ctx):
        return "allowed"

    with mock.patch.object(mod, "get_access_token", lambda: token):
        return asyncio.run(allowlist.on_message(object(), call_next))


class _Token:
    def __init__(self, login):
        self.claims = {"login": login} if login is not None else {}


def test_allowlist_admits_a_listed_login(server_module):
    allowlist = server_module.GitHubUserAllowlist(["TomHarris"])
    assert _run_allowlist(allowlist, _Token("tomharris")) == "allowed"


def test_allowlist_rejects_an_unlisted_login(server_module):
    allowlist = server_module.GitHubUserAllowlist(["tomharris"])
    with pytest.raises(AuthorizationError):
        _run_allowlist(allowlist, _Token("someone-else"))


def test_allowlist_rejects_a_missing_token(server_module):
    allowlist = server_module.GitHubUserAllowlist(["tomharris"])
    with pytest.raises(AuthorizationError):
        _run_allowlist(allowlist, None)


def test_allowlist_rejects_a_token_without_a_login(server_module):
    allowlist = server_module.GitHubUserAllowlist(["tomharris"])
    with pytest.raises(AuthorizationError):
        _run_allowlist(allowlist, _Token(None))


def test_empty_allowlist_is_refused(server_module):
    # An empty allowlist on a public endpoint would admit every GitHub account.
    with pytest.raises(ValueError):
        server_module.GitHubUserAllowlist([])


def test_build_github_auth_pins_the_claude_callback(server_module, tmp_path):
    auth, allowlist = server_module.build_github_auth(
        "cid", "secret", "https://box.tailnet.ts.net/", ["tomharris"],
        storage_dir=tmp_path / "clients",
    )
    assert auth is not None
    assert isinstance(allowlist, server_module.GitHubUserAllowlist)


@pytest.mark.parametrize("args,missing", [
    ([], "$DEVRAG_GITHUB_CLIENT_ID"),
    (["--public-url", "https://x.ts.net", "--allow-user", "me"], "$DEVRAG_GITHUB_CLIENT_ID"),
])
def test_github_auth_requires_its_inputs(monkeypatch, args, missing):
    monkeypatch.delenv("DEVRAG_GITHUB_CLIENT_ID", raising=False)
    monkeypatch.delenv("DEVRAG_GITHUB_CLIENT_SECRET", raising=False)
    result = runner.invoke(app, ["serve", "--transport", "http", "--auth", "github", *args])
    assert result.exit_code == 2
    assert missing in result.output


def test_github_auth_rejects_a_plain_http_public_url(monkeypatch):
    monkeypatch.setenv("DEVRAG_GITHUB_CLIENT_ID", "cid")
    monkeypatch.setenv("DEVRAG_GITHUB_CLIENT_SECRET", "secret")
    result = runner.invoke(app, [
        "serve", "--transport", "http", "--auth", "github",
        "--public-url", "http://box.tailnet.ts.net", "--allow-user", "me",
    ])
    assert result.exit_code == 2
    assert "https://" in result.output


def test_serve_rejects_an_unknown_auth_mode():
    result = runner.invoke(app, ["serve", "--transport", "http", "--auth", "basic"])
    assert result.exit_code == 2
