import ssl
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path

import httpx
import pytest

from devrag.utils.http import (
    MAX_BACKOFF_SECONDS,
    _resolve_ca_path,
    backoff_delay,
    parse_retry_after,
    request_with_retries,
    resolve_verify,
    safe_int,
    transient_retry_delay,
)


# --- path resolution / precedence (no real files needed) ---


def test_empty_path_returns_none():
    assert _resolve_ca_path("") is None
    assert _resolve_ca_path(None) is None


def test_explicit_ca_bundle_is_expanded():
    result = _resolve_ca_path("~/corp-ca.pem")
    assert result == str(Path("~/corp-ca.pem").expanduser())
    assert "~" not in result


def test_requests_ca_bundle_env_fallback(monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/etc/ssl/corp.pem")
    assert _resolve_ca_path() == "/etc/ssl/corp.pem"


def test_ssl_cert_file_env_fallback(monkeypatch):
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.setenv("SSL_CERT_FILE", "/etc/ssl/cert-file.pem")
    assert _resolve_ca_path() == "/etc/ssl/cert-file.pem"


def test_explicit_arg_wins_over_env(monkeypatch):
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/etc/ssl/env.pem")
    assert _resolve_ca_path("/explicit/ca.pem") == "/explicit/ca.pem"


def test_requests_ca_bundle_preferred_over_ssl_cert_file(monkeypatch):
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/etc/ssl/requests.pem")
    monkeypatch.setenv("SSL_CERT_FILE", "/etc/ssl/cert-file.pem")
    assert _resolve_ca_path() == "/etc/ssl/requests.pem"


# --- resolve_verify (httpx verify= value) ---


def test_empty_returns_true(monkeypatch):
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    assert resolve_verify("") is True
    assert resolve_verify(None) is True


@pytest.fixture
def ca_pem(tmp_path) -> Path:
    """A self-signed CA PEM whose basicConstraints is NOT marked critical.

    Mirrors the Netskope corporate-proxy cert that trips Python 3.13's strict
    X.509 verification, so the relaxed context this module builds is exercised
    against a cert that the strict default would reject.
    """
    pytest.importorskip("cryptography")
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test-corp-ca")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        # critical=False is the whole point — strict X.509 rejects this.
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=False)
        .sign(key, hashes.SHA256())
    )
    pem = tmp_path / "corp-ca.pem"
    pem.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return pem


def test_custom_bundle_returns_relaxed_context(ca_pem):
    ctx = resolve_verify(str(ca_pem))
    assert isinstance(ctx, ssl.SSLContext)
    # VERIFY_X509_STRICT must be cleared so non-critical-basicConstraints corp
    # CAs (Netskope et al.) don't fail under Python 3.13's strict default.
    assert ctx.verify_flags & ssl.VERIFY_X509_STRICT == 0
    # The CA from the file was actually loaded into the trust store.
    assert ctx.get_ca_certs()


def test_custom_bundle_still_verifies(ca_pem):
    ctx = resolve_verify(str(ca_pem))
    assert ctx.verify_mode == ssl.CERT_REQUIRED


def test_custom_bundle_also_trusts_public_roots(ca_pem):
    # The corporate CA is ADDED on top of certifi's public roots, not a
    # replacement — TLS-intercepting proxies bypass some hosts (which keep their
    # real public certs), so dropping the public roots would break those hosts.
    import certifi

    ctx = resolve_verify(str(ca_pem))
    loaded = {(c["serialNumber"], c["subject"]) for c in ctx.get_ca_certs()}
    # The temp corporate CA is present...
    assert len(loaded) > 1
    # ...alongside a large public-root set (certifi ships hundreds).
    certifi_ctx = ssl.create_default_context(cafile=certifi.where())
    assert len(ctx.get_ca_certs()) >= len(certifi_ctx.get_ca_certs())


def test_tilde_expansion_loads_context(ca_pem, monkeypatch):
    # Point HOME at the temp dir and reference the cert via ~ to prove
    # expansion feeds a real, loadable path into the context builder.
    monkeypatch.setenv("HOME", str(ca_pem.parent))
    ctx = resolve_verify(f"~/{ca_pem.name}")
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.get_ca_certs()


def test_env_fallback_returns_context(ca_pem, monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(ca_pem))
    ctx = resolve_verify()
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.verify_flags & ssl.VERIFY_X509_STRICT == 0


# --- retry helpers (shared by the GitHub/Jira/Slack/Slite clients) ---


@pytest.fixture
def no_sleep(monkeypatch):
    """Record sleep durations instead of actually sleeping."""
    slept: list[float] = []
    monkeypatch.setattr("devrag.utils.http.time.sleep", slept.append)
    return slept


def test_safe_int_parses_and_falls_back():
    assert safe_int("7", 1) == 7
    assert safe_int(None, 1) == 1
    assert safe_int("", 1) == 1
    # A non-numeric header must not raise — that is the crash this guards.
    assert safe_int("not-a-number", 1) == 1
    assert safe_int("1.5", 1) == 1


def test_parse_retry_after_delta_seconds():
    assert parse_retry_after("30") == 30.0
    assert parse_retry_after(" 30 ") == 30.0


def test_parse_retry_after_accepts_http_date():
    # RFC 9110 permits an HTTP-date here, so bare int() is a reachable crash.
    future = datetime.now(timezone.utc) + timedelta(seconds=120)
    seconds = parse_retry_after(format_datetime(future, usegmt=True))
    assert seconds is not None
    assert 100 <= seconds <= 130


def test_parse_retry_after_past_date_is_zero():
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    assert parse_retry_after(format_datetime(past, usegmt=True)) == 0.0


def test_parse_retry_after_garbage_is_none():
    assert parse_retry_after("soon") is None
    assert parse_retry_after(None) is None
    assert parse_retry_after("") is None


def test_backoff_delay_honors_retry_after():
    assert backoff_delay(0, "5") == 5.0


def test_backoff_delay_caps_retry_after():
    assert backoff_delay(0, "99999") == MAX_BACKOFF_SECONDS


def test_backoff_delay_falls_back_to_exponential_on_bad_header():
    # A garbage Retry-After must degrade to exponential backoff, not crash.
    delay = backoff_delay(2, "not-a-number")
    assert 4.0 <= delay <= 4.5


def test_backoff_delay_is_exponential_and_capped():
    assert 1.0 <= backoff_delay(0) <= 1.5
    assert 2.0 <= backoff_delay(1) <= 2.5
    assert backoff_delay(100) <= MAX_BACKOFF_SECONDS + 0.5


def test_transient_retry_delay_flags_429_and_5xx():
    assert transient_retry_delay(httpx.Response(429, headers={"Retry-After": "3"}), 0) == 3.0
    assert transient_retry_delay(httpx.Response(503), 0) is not None
    assert transient_retry_delay(httpx.Response(200), 0) is None
    assert transient_retry_delay(httpx.Response(404), 0) is None


def test_request_with_retries_returns_first_success():
    calls = {"n": 0}

    def send():
        calls["n"] += 1
        return httpx.Response(200)

    resp = request_with_retries(send, max_retries=3)
    assert resp.status_code == 200
    assert calls["n"] == 1


def test_request_with_retries_retries_transport_errors(no_sleep):
    # ConnectError is a sibling of TimeoutException under TransportError — the
    # exact class that used to bypass the retry and abort a whole sync.
    calls = {"n": 0}

    def send():
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200)

    resp = request_with_retries(send, max_retries=3)
    assert resp.status_code == 200
    assert calls["n"] == 2
    assert len(no_sleep) == 1


def test_request_with_retries_reraises_transport_error_after_exhaustion(no_sleep):
    def send():
        raise httpx.ConnectError("connection refused")

    with pytest.raises(httpx.ConnectError):
        request_with_retries(send, max_retries=2)
    assert len(no_sleep) == 2  # slept between attempts, not after the last


def test_request_with_retries_loops_on_repeated_429(no_sleep):
    # A *second* consecutive 429 must also be retried — single-retry clients
    # aborted here.
    calls = {"n": 0}

    def send():
        calls["n"] += 1
        if calls["n"] <= 3:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200)

    resp = request_with_retries(send, max_retries=5)
    assert resp.status_code == 200
    assert calls["n"] == 4


def test_request_with_retries_returns_last_response_when_exhausted(no_sleep):
    def send():
        return httpx.Response(
            429,
            headers={"Retry-After": "0"},
            request=httpx.Request("GET", "https://example.test/x"),
        )

    resp = request_with_retries(send, max_retries=2)
    # The helper hands back the final response; callers decide to raise.
    assert resp.status_code == 429
    with pytest.raises(httpx.HTTPStatusError):
        resp.raise_for_status()


def test_request_with_retries_runs_before_attempt_hook(no_sleep):
    ticks = {"n": 0}
    calls = {"n": 0}

    def before():
        ticks["n"] += 1

    def send():
        calls["n"] += 1
        return httpx.Response(500 if calls["n"] == 1 else 200)

    request_with_retries(send, max_retries=3, before_attempt=before)
    assert ticks["n"] == calls["n"] == 2  # hook fires before every attempt


def test_request_with_retries_accepts_custom_policy(no_sleep):
    # A caller-supplied policy can retry statuses the default ignores.
    calls = {"n": 0}

    def send():
        calls["n"] += 1
        return httpx.Response(403 if calls["n"] == 1 else 200)

    def policy(resp, attempt):
        return 0.0 if resp.status_code == 403 else None

    resp = request_with_retries(send, max_retries=3, retry_delay=policy)
    assert resp.status_code == 200
    assert calls["n"] == 2


def test_request_with_retries_no_retries_when_max_is_zero():
    calls = {"n": 0}

    def send():
        calls["n"] += 1
        return httpx.Response(429, headers={"Retry-After": "0"})

    resp = request_with_retries(send, max_retries=0)
    assert resp.status_code == 429
    assert calls["n"] == 1
