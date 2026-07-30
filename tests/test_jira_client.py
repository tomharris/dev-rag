import httpx
import pytest
import respx

from devrag.utils.jira_client import JiraClient

BASE = "https://acme.atlassian.net/rest/api/3"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("devrag.utils.http.time.sleep", lambda _: None)


def _client(**kwargs) -> JiraClient:
    return JiraClient(
        instance_url="https://acme.atlassian.net",
        email="user@acme.test",
        api_token="token",
        **kwargs,
    )


def _page(issues: list[dict]) -> dict:
    return {"issues": issues, "nextPageToken": None}


@respx.mock
def test_consecutive_429s_are_all_retried():
    # A single-retry client aborted the whole sync on the second 429.
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] <= 3:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json=_page([{"key": "ENG-1"}]))

    respx.post(f"{BASE}/search/jql").mock(side_effect=handler)
    issues = list(_client(max_retries=5).search_issues("project = ENG", ["summary"]))
    assert [i["key"] for i in issues] == ["ENG-1"]
    assert calls["n"] == 4


@respx.mock
def test_non_numeric_retry_after_does_not_crash():
    # RFC 9110 allows an HTTP-date; bare int() raised ValueError on it.
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}
            )
        return httpx.Response(200, json=_page([{"key": "ENG-2"}]))

    respx.post(f"{BASE}/search/jql").mock(side_effect=handler)
    issues = list(_client(max_retries=3).search_issues("project = ENG", ["summary"]))
    assert [i["key"] for i in issues] == ["ENG-2"]


@respx.mock
def test_5xx_is_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=_page([{"key": "ENG-3"}]))

    respx.post(f"{BASE}/search/jql").mock(side_effect=handler)
    issues = list(_client(max_retries=3).search_issues("project = ENG", ["summary"]))
    assert [i["key"] for i in issues] == ["ENG-3"]
    assert calls["n"] == 2


@respx.mock
def test_transport_error_is_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, json=_page([{"key": "ENG-4"}]))

    respx.post(f"{BASE}/search/jql").mock(side_effect=handler)
    issues = list(_client(max_retries=3).search_issues("project = ENG", ["summary"]))
    assert [i["key"] for i in issues] == ["ENG-4"]
    assert calls["n"] == 2


@respx.mock
def test_exhausted_retries_raise_http_status_error():
    route = respx.post(f"{BASE}/search/jql").mock(
        return_value=httpx.Response(429, headers={"Retry-After": "0"})
    )
    with pytest.raises(httpx.HTTPStatusError):
        list(_client(max_retries=2).search_issues("project = ENG", ["summary"]))
    assert route.call_count == 3


@respx.mock
def test_client_error_is_not_retried():
    # A 400 is a caller bug (bad JQL); retrying it just delays the failure.
    route = respx.post(f"{BASE}/search/jql").mock(return_value=httpx.Response(400))
    with pytest.raises(httpx.HTTPStatusError):
        list(_client(max_retries=3).search_issues("bad jql", ["summary"]))
    assert route.call_count == 1
