import json
import time
import httpx
import pytest
import respx
from devrag.utils.github import GitHubClient, parse_diff_hunks


@respx.mock
def test_list_prs():
    respx.get("https://api.github.com/repos/acme/backend/pulls").respond(json=[
        {"number": 1, "title": "Add auth", "body": "Adds authentication module",
         "state": "closed", "user": {"login": "alice"}, "labels": [{"name": "feature"}],
         "created_at": "2026-03-01T00:00:00Z", "updated_at": "2026-03-02T00:00:00Z",
         "merged_at": "2026-03-02T12:00:00Z", "draft": False},
    ])
    client = GitHubClient(token="test-token")
    prs = client.list_prs("acme/backend", state="all", per_page=100)
    assert len(prs) == 1
    assert prs[0]["number"] == 1
    assert prs[0]["title"] == "Add auth"


@respx.mock
def test_get_pr_files():
    respx.get("https://api.github.com/repos/acme/backend/pulls/1/files").respond(json=[
        {"filename": "src/auth.py", "status": "added", "additions": 50, "deletions": 0,
         "patch": "@@ -0,0 +1,50 @@\n+def authenticate():\n+    pass"},
    ])
    client = GitHubClient(token="test-token")
    files = client.get_pr_files("acme/backend", 1)
    assert len(files) == 1
    assert files[0]["filename"] == "src/auth.py"
    assert "patch" in files[0]


@respx.mock
def test_get_pr_comments():
    respx.get("https://api.github.com/repos/acme/backend/pulls/1/comments").respond(json=[
        {"id": 101, "body": "Consider using bcrypt here", "user": {"login": "bob"},
         "path": "src/auth.py", "line": 10, "created_at": "2026-03-02T10:00:00Z"},
    ])
    client = GitHubClient(token="test-token")
    comments = client.get_pr_comments("acme/backend", 1)
    assert len(comments) == 1
    assert comments[0]["body"] == "Consider using bcrypt here"


@respx.mock
def test_pagination():
    page1_url = "https://api.github.com/repos/acme/backend/pulls"
    page2_url = "https://api.github.com/repos/acme/backend/pulls?page=2"
    # Register page2_url first so the more-specific route takes priority in respx matching
    respx.get(page2_url).respond(json=[{"number": 2, "title": "PR 2"}], headers={})
    respx.get(page1_url).respond(json=[{"number": 1, "title": "PR 1"}],
        headers={"link": f'<{page2_url}>; rel="next"'})
    client = GitHubClient(token="test-token")
    all_items = client.paginate(page1_url, params={"per_page": 1})
    assert len(all_items) == 2
    assert all_items[0]["number"] == 1
    assert all_items[1]["number"] == 2


@respx.mock
def test_list_prs_since_short_circuits_pagination():
    # Page 1 has a PR newer than `since`, and an older one that should trigger the early return.
    # Page 2 (reachable via Link header) should NEVER be fetched.
    page1_url = "https://api.github.com/repos/acme/backend/pulls"
    page2_url = "https://api.github.com/repos/acme/backend/pulls?page=2"
    page2_route = respx.get(page2_url).respond(json=[
        {"number": 3, "title": "Very old", "updated_at": "2025-01-01T00:00:00Z"},
    ])
    respx.get(page1_url).respond(
        json=[
            {"number": 1, "title": "Newer",   "updated_at": "2026-03-10T00:00:00Z"},
            {"number": 2, "title": "Too old", "updated_at": "2026-02-01T00:00:00Z"},
        ],
        headers={"link": f'<{page2_url}>; rel="next"'},
    )
    client = GitHubClient(token="test-token")
    prs = client.list_prs("acme/backend", since="2026-03-01T00:00:00Z")
    assert [p["number"] for p in prs] == [1]
    assert page2_route.call_count == 0  # pagination short-circuited before next page


@respx.mock
def test_list_prs_since_none_uses_full_pagination():
    # Regression: since=None should behave like before (walk all pages).
    page1_url = "https://api.github.com/repos/acme/backend/pulls"
    page2_url = "https://api.github.com/repos/acme/backend/pulls?page=2"
    respx.get(page2_url).respond(json=[{"number": 2, "title": "PR 2", "updated_at": "2025-01-01T00:00:00Z"}])
    respx.get(page1_url).respond(
        json=[{"number": 1, "title": "PR 1", "updated_at": "2026-03-10T00:00:00Z"}],
        headers={"link": f'<{page2_url}>; rel="next"'},
    )
    client = GitHubClient(token="test-token")
    prs = client.list_prs("acme/backend")
    assert [p["number"] for p in prs] == [1, 2]


@respx.mock
def test_rate_limit_backoff():
    reset_time = str(int(time.time()) + 1)
    call_count = 0
    def handler(request):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(403, json={"message": "API rate limit exceeded"},
                headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": reset_time})
        return httpx.Response(200, json=[{"number": 1}])
    respx.get("https://api.github.com/repos/acme/backend/pulls").mock(side_effect=handler)
    client = GitHubClient(token="test-token")
    result = client.list_prs("acme/backend")
    assert len(result) == 1
    assert call_count == 2


def test_parse_diff_hunks():
    patch = """@@ -10,4 +10,6 @@ class Auth:
     def login(self):
         pass
+    def logout(self):
+        pass
@@ -20,3 +22,5 @@ class Auth:
     def refresh(self):
         pass
+    def revoke(self):
+        pass"""
    hunks = parse_diff_hunks(patch, "src/auth.py")
    assert len(hunks) == 2
    assert hunks[0]["file_path"] == "src/auth.py"
    assert "+    def logout" in hunks[0]["content"]
    assert "+    def revoke" in hunks[1]["content"]


def test_github_client_no_token():
    client = GitHubClient(token=None)
    assert "Authorization" not in client._headers


# --- rate-limit / retry handling ---


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr("devrag.utils.http.time.sleep", lambda *_: None)


@respx.mock
def test_non_numeric_rate_limit_headers_do_not_crash(no_sleep):
    # A proxy or error page can put anything in these headers; bare int() turned
    # that into a ValueError that aborted the whole sync.
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(403, json={"message": "rate limited"}, headers={
                "x-ratelimit-remaining": "unknown",
                "x-ratelimit-reset": "not-a-timestamp",
                "retry-after": "later",
            })
        return httpx.Response(200, json=[{"number": 1}])

    respx.get("https://api.github.com/repos/acme/backend/pulls").mock(side_effect=handler)
    client = GitHubClient(token="t")
    # Unparseable headers must not raise. With no usable signal the 403 is
    # treated as a genuine permission error and surfaced as an HTTP error.
    with pytest.raises(httpx.HTTPStatusError):
        client.list_prs("acme/backend")
    assert calls["n"] == 1


@respx.mock
def test_non_numeric_reset_still_retries_when_remaining_is_zero(no_sleep):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(403, json={"message": "rate limited"}, headers={
                "x-ratelimit-remaining": "0",
                "x-ratelimit-reset": "garbage",
            })
        return httpx.Response(200, json=[{"number": 1}])

    respx.get("https://api.github.com/repos/acme/backend/pulls").mock(side_effect=handler)
    result = GitHubClient(token="t").list_prs("acme/backend")
    assert [p["number"] for p in result] == [1]
    assert calls["n"] == 2


@respx.mock
def test_secondary_rate_limit_429_is_retried(no_sleep):
    # GitHub's secondary limits return 429 + Retry-After with a NON-zero
    # x-ratelimit-remaining, so a remaining==0 check alone never fired.
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"message": "secondary rate limit"},
                                  headers={"retry-after": "1", "x-ratelimit-remaining": "4999"})
        return httpx.Response(200, json=[{"number": 7}])

    respx.get("https://api.github.com/repos/acme/backend/pulls").mock(side_effect=handler)
    result = GitHubClient(token="t").list_prs("acme/backend")
    assert [p["number"] for p in result] == [7]
    assert calls["n"] == 2


@respx.mock
def test_consecutive_429s_are_all_retried(no_sleep):
    # The old single-shot retry gave up on the second 429.
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] <= 3:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, json=[{"number": 9}])

    respx.get("https://api.github.com/repos/acme/backend/pulls").mock(side_effect=handler)
    result = GitHubClient(token="t", max_retries=5).list_prs("acme/backend")
    assert [p["number"] for p in result] == [9]
    assert calls["n"] == 4


@respx.mock
def test_5xx_is_retried(no_sleep):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(502)
        return httpx.Response(200, json=[{"number": 11}])

    respx.get("https://api.github.com/repos/acme/backend/pulls").mock(side_effect=handler)
    result = GitHubClient(token="t").list_prs("acme/backend")
    assert [p["number"] for p in result] == [11]
    assert calls["n"] == 2


@respx.mock
def test_transport_error_is_retried(no_sleep):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, json=[{"number": 13}])

    respx.get("https://api.github.com/repos/acme/backend/pulls").mock(side_effect=handler)
    result = GitHubClient(token="t").list_prs("acme/backend")
    assert [p["number"] for p in result] == [13]
    assert calls["n"] == 2


@respx.mock
def test_permission_403_is_not_retried(no_sleep):
    # A 403 with quota remaining is an authorization failure, not a rate limit —
    # retrying it just delays an error the caller needs to see.
    route = respx.get("https://api.github.com/repos/acme/backend/pulls").mock(
        return_value=httpx.Response(403, json={"message": "Resource not accessible"},
                                    headers={"x-ratelimit-remaining": "4999"})
    )
    with pytest.raises(httpx.HTTPStatusError):
        GitHubClient(token="t", max_retries=3).list_prs("acme/backend")
    assert route.call_count == 1


@respx.mock
def test_404_is_not_retried(no_sleep):
    route = respx.get("https://api.github.com/repos/acme/backend/pulls").mock(
        return_value=httpx.Response(404, json={"message": "Not Found"})
    )
    with pytest.raises(httpx.HTTPStatusError):
        GitHubClient(token="t", max_retries=3).list_prs("acme/backend")
    assert route.call_count == 1


@respx.mock
def test_exhausted_rate_limit_retries_raise(no_sleep):
    route = respx.get("https://api.github.com/repos/acme/backend/pulls").mock(
        return_value=httpx.Response(429, headers={"retry-after": "0"})
    )
    with pytest.raises(httpx.HTTPStatusError):
        GitHubClient(token="t", max_retries=2).list_prs("acme/backend")
    assert route.call_count == 3
