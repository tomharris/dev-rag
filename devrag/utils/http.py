"""Shared HTTP/TLS helpers for DevRAG's httpx clients.

httpx verifies certificates against the bundled ``certifi`` CA list and does
**not** consult the OS keychain or honor ``SSL_CERT_FILE`` on its own. Behind a
TLS-intercepting corporate proxy (which injects a self-signed root CA), that
makes every outbound HTTPS call fail with ``CERTIFICATE_VERIFY_FAILED``. This
resolver lets callers point httpx at a CA bundle that includes the proxy's root.
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_verify(ca_bundle: str | None = None) -> str | bool:
    """Resolve the value to pass as httpx's ``verify=``.

    Precedence: an explicit ``ca_bundle`` (e.g. ``config.network.ca_bundle``),
    then the standard ``REQUESTS_CA_BUNDLE`` / ``SSL_CERT_FILE`` env vars, else
    ``True`` (httpx's default certifi verification). A returned path is
    ``~``-expanded so config values like ``~/corp-ca.pem`` work.
    """
    candidate = ca_bundle or os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")
    if candidate:
        return str(Path(candidate).expanduser())
    return True
