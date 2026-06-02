import ssl
from pathlib import Path

import pytest

from devrag.utils.http import _resolve_ca_path, resolve_verify


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
