import httpx
import pytest
import respx

from devrag.utils.slack_client import SlackAuthError, SlackError, SlackClient

API = "https://slack.com/api"


@respx.mock
def test_list_conversations_pagination():
    respx.post(f"{API}/conversations.list").mock(side_effect=[
        httpx.Response(200, json={
            "ok": True,
            "channels": [{"id": "C1", "name": "general", "is_member": True}],
            "response_metadata": {"next_cursor": "page2"},
        }),
        httpx.Response(200, json={
            "ok": True,
            "channels": [{"id": "C2", "name": "random", "is_member": True}],
            "response_metadata": {"next_cursor": ""},
        }),
    ])
    client = SlackClient(token="xoxc-test", cookie="xoxd-test")
    channels = list(client.list_conversations())
    assert [c["id"] for c in channels] == ["C1", "C2"]


@respx.mock
def test_conversations_history_pagination():
    respx.post(f"{API}/conversations.history").mock(side_effect=[
        httpx.Response(200, json={
            "ok": True,
            "messages": [{"ts": "100.0", "text": "hello", "user": "U1"}],
            "has_more": True,
            "response_metadata": {"next_cursor": "next"},
        }),
        httpx.Response(200, json={
            "ok": True,
            "messages": [{"ts": "101.0", "text": "world", "user": "U2"}],
            "has_more": False,
        }),
    ])
    client = SlackClient(token="xoxc-test", cookie="xoxd-test")
    messages = list(client.conversations_history("C1", oldest="50.0"))
    assert [m["ts"] for m in messages] == ["100.0", "101.0"]


@respx.mock
def test_conversations_replies():
    respx.post(f"{API}/conversations.replies").respond(json={
        "ok": True,
        "messages": [
            {"ts": "100.0", "text": "root", "user": "U1", "thread_ts": "100.0"},
            {"ts": "100.5", "text": "reply", "user": "U2", "thread_ts": "100.0"},
        ],
    })
    client = SlackClient(token="xoxc-test", cookie="xoxd-test")
    replies = client.conversations_replies("C1", "100.0")
    assert [m["ts"] for m in replies] == ["100.0", "100.5"]


@respx.mock
def test_users_list():
    respx.post(f"{API}/users.list").respond(json={
        "ok": True,
        "members": [
            {"id": "U1", "name": "alice", "profile": {"display_name": "Alice"}},
            {"id": "U2", "name": "bob", "profile": {"display_name": ""}},
        ],
        "response_metadata": {"next_cursor": ""},
    })
    client = SlackClient(token="xoxc-test", cookie="xoxd-test")
    members = list(client.users_list())
    assert [m["id"] for m in members] == ["U1", "U2"]


@respx.mock
def test_invalid_auth_raises_clear_error():
    respx.post(f"{API}/conversations.list").respond(json={"ok": False, "error": "invalid_auth"})
    client = SlackClient(token="xoxc-expired", cookie="xoxd-expired")
    with pytest.raises(SlackAuthError) as exc:
        list(client.list_conversations())
    # Message should guide the user to re-extract credentials from the browser
    assert "re-extract" in str(exc.value).lower()


@respx.mock
def test_generic_ok_false_raises():
    respx.post(f"{API}/conversations.history").respond(json={"ok": False, "error": "channel_not_found"})
    client = SlackClient(token="xoxc-test", cookie="xoxd-test")
    with pytest.raises(SlackError) as exc:
        list(client.conversations_history("CBAD"))
    assert "channel_not_found" in str(exc.value)


@respx.mock
def test_cookie_and_token_sent():
    route = respx.post(f"{API}/conversations.list").respond(json={
        "ok": True, "channels": [], "response_metadata": {"next_cursor": ""},
    })
    client = SlackClient(token="xoxc-abc", cookie="xoxd-xyz")
    list(client.list_conversations())
    request = route.calls.last.request
    assert "d=xoxd-xyz" in request.headers.get("cookie", "")
    assert b"token=xoxc-abc" in request.content
