"""Shared HTTP/TLS helpers for DevRAG's httpx clients.

Two concerns live here, both shared by every outbound client (GitHub, Jira,
Slack, Slite):

**TLS trust.** httpx verifies certificates against the bundled ``certifi`` CA
list and does **not** consult the OS keychain or honor ``SSL_CERT_FILE`` on its
own. Behind a TLS-intercepting corporate proxy (which injects a self-signed root
CA), that makes every outbound HTTPS call fail with
``CERTIFICATE_VERIFY_FAILED``. ``resolve_verify`` lets callers point httpx at a
CA bundle that includes the proxy's root.

**Retries.** Each client previously carried its own retry loop, and the copies
drifted: one retried a single time, one caught only timeouts, one parsed
rate-limit headers with a bare ``int()``. ``request_with_retries`` is the single
implementation they now share, with per-service policy supplied as a callback.
"""

from __future__ import annotations

import os
import random
import ssl
import time
from collections.abc import Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import certifi
import httpx


def _resolve_ca_path(ca_bundle: str | None = None) -> str | None:
    """Resolve the CA bundle path to use, or ``None`` for certifi default.

    Precedence: an explicit ``ca_bundle`` (e.g. ``config.network.ca_bundle``),
    then the standard ``REQUESTS_CA_BUNDLE`` / ``SSL_CERT_FILE`` env vars, else
    ``None``. A returned path is ``~``-expanded so config values like
    ``~/corp-ca.pem`` work.
    """
    candidate = ca_bundle or os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")
    if candidate:
        return str(Path(candidate).expanduser())
    return None


def resolve_verify(ca_bundle: str | None = None) -> ssl.SSLContext | bool:
    """Resolve the value to pass as httpx's ``verify=``.

    With no custom CA bundle (and no ``REQUESTS_CA_BUNDLE`` / ``SSL_CERT_FILE``
    env var), returns ``True`` — httpx's default certifi verification, which is
    correct for RFC-compliant public CAs.

    When a custom bundle IS configured, returns an ``ssl.SSLContext`` that trusts
    **both** certifi's public roots and the bundle's corporate CA, with
    ``VERIFY_X509_STRICT`` cleared.

    Two corporate-proxy realities drive this:

    1. The context is seeded from ``certifi.where()`` and the bundle is *added*
       via ``load_verify_locations`` — it does not replace the public roots.
       TLS-intercepting proxies (e.g. Netskope) typically intercept most hosts
       (which then present the corporate CA) but *bypass* some (e.g.
       ``app.slack.com``, which keeps its real public cert). A bundle-only
       context would verify the intercepted hosts yet fail the bypassed ones with
       ``CERTIFICATE_VERIFY_FAILED``; trusting both roots covers both.

    2. ``VERIFY_X509_STRICT`` is cleared. Python 3.13's default context enables
       strict X.509 verification, which rejects many corporate MITM-proxy root
       CAs whose ``basicConstraints`` extension isn't marked critical — failing
       with "Basic Constraints of CA cert not marked critical" even though the
       chain is otherwise valid (and the openssl CLI accepts it). Clearing the
       flag restores pre-3.13 leniency for the corporate-proxy case only; the
       no-bundle path keeps full strict verification against certifi.
    """
    path = _resolve_ca_path(ca_bundle)
    if path is None:
        return True
    ctx = ssl.create_default_context(cafile=certifi.where())
    ctx.load_verify_locations(cafile=path)
    ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return ctx


# --- retry helpers -----------------------------------------------------------

# No single backoff ever parks a sync for more than a minute, however large the
# server's Retry-After or rate-limit reset window.
MAX_BACKOFF_SECONDS = 60.0

#: Perform one HTTP attempt. Raises ``httpx.TransportError`` on network failure.
Send = Callable[[], httpx.Response]

#: Decide whether a response warrants another attempt. Returns the seconds to
#: wait, or ``None`` to accept the response as final.
RetryPolicy = Callable[[httpx.Response, int], "float | None"]


def safe_int(value: str | None, default: int) -> int:
    """Parse an integer header value, falling back to ``default``.

    Rate-limit headers are server-controlled strings, not guaranteed integers —
    a proxy or an error page can put anything there. A bare ``int()`` turns that
    into a ``ValueError`` that propagates up through pagination and aborts the
    entire sync, so every header read goes through here.
    """
    if value is None:
        return default
    try:
        return int(value.strip())
    except (ValueError, AttributeError):
        return default


def parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header into seconds, or ``None`` if unparseable.

    RFC 9110 permits **two** forms: delta-seconds (``120``) and an HTTP-date
    (``Wed, 21 Oct 2026 07:28:00 GMT``). Both are handled; a date in the past
    yields ``0.0`` rather than a negative sleep.
    """
    if not value:
        return None
    text = value.strip()
    try:
        return max(float(text), 0.0)
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max((when - datetime.now(timezone.utc)).total_seconds(), 0.0)


def backoff_delay(attempt: int, retry_after: str | None = None) -> float:
    """Seconds to wait before the next attempt.

    Honors a parseable ``Retry-After``; otherwise exponential backoff with
    jitter. The jitter matters when several threads retry together (the Slack
    indexer fans replies out over a thread pool) — without it they would all
    wake at the same instant and re-trip the rate limit.
    """
    seconds = parse_retry_after(retry_after)
    if seconds is not None:
        return min(seconds, MAX_BACKOFF_SECONDS)
    return min(2.0**attempt, MAX_BACKOFF_SECONDS) + random.uniform(0, 0.5)


def transient_retry_delay(resp: httpx.Response, attempt: int) -> float | None:
    """Default policy: retry 429 and 5xx, accept everything else."""
    if resp.status_code == 429 or resp.status_code >= 500:
        return backoff_delay(attempt, resp.headers.get("Retry-After"))
    return None


def request_with_retries(
    send: Send,
    *,
    max_retries: int = 5,
    before_attempt: Callable[[], None] | None = None,
    retry_delay: RetryPolicy = transient_retry_delay,
) -> httpx.Response:
    """Run ``send`` with bounded retries, returning the last response.

    Retries every ``httpx.TransportError`` — not just ``TimeoutException``.
    ``ConnectError`` (DNS failure, connection refused), ``ReadError`` /
    ``WriteError`` (connection reset) and ``RemoteProtocolError`` are *siblings*
    of ``TimeoutException`` under ``TransportError``, so catching the narrower
    class lets the most common real-world network faults abort a sync on first
    occurrence.

    Which HTTP responses are retryable is left to ``retry_delay`` so services
    with idiosyncratic signalling (GitHub encodes rate limits as ``403`` plus
    ``x-ratelimit-remaining``) share this loop instead of forking it.

    ``before_attempt`` runs ahead of every attempt, including retries — the
    Slack client uses it to hold retries inside its global rate cap.

    The final response is **returned, not raised on**, so callers keep their own
    ``raise_for_status()`` and any body-level error handling. A transport error
    on the last attempt propagates, since there is no response to hand back.
    """
    attempts = max(max_retries, 0) + 1
    for attempt in range(attempts):
        is_last = attempt == attempts - 1
        if before_attempt is not None:
            before_attempt()
        try:
            resp = send()
        except httpx.TransportError:
            if is_last:
                raise
            time.sleep(backoff_delay(attempt))
            continue
        if is_last:
            return resp
        delay = retry_delay(resp, attempt)
        if delay is None:
            return resp
        time.sleep(delay)
    raise AssertionError("unreachable: retry loop always returns or raises")
