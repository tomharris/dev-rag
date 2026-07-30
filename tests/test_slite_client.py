import httpx
import pytest
import respx

from devrag.utils.slite_client import SliteClient

API = "https://api.slite.com/v1"


@respx.mock
def test_request_retries_transient_429(monkeypatch):
    """A 429 that clears on retry should not surface as an error."""
    # Don't actually sleep during the backoff.
    monkeypatch.setattr("devrag.utils.http.time.sleep", lambda *_: None)

    route = respx.get(f"{API}/notes/page-1").mock(side_effect=[
        httpx.Response(429, headers={"Retry-After": "0"}, json={}),
        httpx.Response(200, json={"id": "page-1", "content": "# Hi"}),
    ])
    client = SliteClient(api_token="tok")
    note = client.get_note("page-1")
    assert note["content"] == "# Hi"
    assert route.call_count == 2


@respx.mock
def test_request_raises_after_exhausting_retries(monkeypatch):
    """A persistent 429 raises HTTPStatusError once retries are exhausted."""
    monkeypatch.setattr("devrag.utils.http.time.sleep", lambda *_: None)

    respx.get(f"{API}/notes/page-1").mock(
        return_value=httpx.Response(429, headers={"Retry-After": "0"}, json={})
    )
    client = SliteClient(api_token="tok", max_retries=2)
    with pytest.raises(httpx.HTTPStatusError) as exc:
        client.get_note("page-1")
    assert exc.value.response.status_code == 429


@respx.mock
def test_request_retries_5xx(monkeypatch):
    monkeypatch.setattr("devrag.utils.http.time.sleep", lambda *_: None)

    route = respx.get(f"{API}/notes/page-1").mock(side_effect=[
        httpx.Response(503, json={}),
        httpx.Response(200, json={"id": "page-1", "content": "# Hi"}),
    ])
    client = SliteClient(api_token="tok")
    assert client.get_note("page-1")["content"] == "# Hi"
    assert route.call_count == 2
