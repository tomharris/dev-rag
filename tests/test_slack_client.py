import time

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
def test_429_retries_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] <= 2:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"ok": False})
        return httpx.Response(200, json={
            "ok": True, "channels": [{"id": "C1"}], "response_metadata": {"next_cursor": ""},
        })

    respx.post(f"{API}/conversations.list").mock(side_effect=handler)
    client = SlackClient(token="xoxc-test", cookie="xoxd-test", max_retries=5)
    channels = list(client.list_conversations())
    assert [c["id"] for c in channels] == ["C1"]
    assert calls["n"] == 3  # two 429s + one success


@respx.mock
def test_429_exhausts_retries_and_raises():
    route = respx.post(f"{API}/conversations.list").mock(
        return_value=httpx.Response(429, headers={"Retry-After": "0"}, json={"ok": False})
    )
    client = SlackClient(token="xoxc-test", cookie="xoxd-test", max_retries=2)
    with pytest.raises(httpx.HTTPStatusError):
        list(client.list_conversations())
    assert route.call_count == 3  # max_retries + 1 attempts


@respx.mock
def test_throttle_paces_request_starts():
    respx.post(f"{API}/auth.test").respond(json={"ok": True, "user": "u"})
    client = SlackClient(token="xoxc-test", cookie="xoxd-test",
                         min_request_interval=0.05, max_retries=0)
    start = time.monotonic()
    client.auth_test()
    client.auth_test()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.05  # second request start spaced by the interval


@respx.mock
def test_auth_error_short_circuits_without_retry():
    route = respx.post(f"{API}/conversations.list").respond(
        json={"ok": False, "error": "invalid_auth"}
    )
    client = SlackClient(token="xoxc-expired", cookie="xoxd-expired", max_retries=5)
    with pytest.raises(SlackAuthError):
        list(client.list_conversations())
    assert route.call_count == 1  # ok:false auth error is not retried


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
