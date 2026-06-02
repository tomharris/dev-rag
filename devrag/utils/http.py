"""Shared HTTP/TLS helpers for DevRAG's httpx clients.

httpx verifies certificates against the bundled ``certifi`` CA list and does
**not** consult the OS keychain or honor ``SSL_CERT_FILE`` on its own. Behind a
TLS-intercepting corporate proxy (which injects a self-signed root CA), that
makes every outbound HTTPS call fail with ``CERTIFICATE_VERIFY_FAILED``. This
resolver lets callers point httpx at a CA bundle that includes the proxy's root.
"""

from __future__ import annotations

import os
import ssl
from pathlib import Path

import certifi


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
