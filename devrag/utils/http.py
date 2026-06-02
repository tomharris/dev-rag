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

    When a custom bundle IS configured, returns an ``ssl.SSLContext`` built from
    it with ``VERIFY_X509_STRICT`` cleared. Python 3.13's default context enables
    strict X.509 verification, which rejects many corporate MITM-proxy root CAs
    (e.g. Netskope) whose ``basicConstraints`` extension isn't marked critical —
    failing with "Basic Constraints of CA cert not marked critical" even though
    the chain is otherwise valid (and the openssl CLI accepts it). Clearing the
    flag restores pre-3.13 leniency for the corporate-proxy case only; the
    no-bundle path keeps full strict verification against certifi.
    """
    path = _resolve_ca_path(ca_bundle)
    if path is None:
        return True
    ctx = ssl.create_default_context(cafile=path)
    ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return ctx
