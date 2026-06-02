from pathlib import Path

from devrag.utils.http import resolve_verify


def test_empty_returns_true():
    assert resolve_verify("") is True
    assert resolve_verify(None) is True


def test_explicit_ca_bundle_is_expanded():
    result = resolve_verify("~/corp-ca.pem")
    assert result == str(Path("~/corp-ca.pem").expanduser())
    assert "~" not in result


def test_requests_ca_bundle_env_fallback(monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/etc/ssl/corp.pem")
    assert resolve_verify() == "/etc/ssl/corp.pem"


def test_ssl_cert_file_env_fallback(monkeypatch):
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.setenv("SSL_CERT_FILE", "/etc/ssl/cert-file.pem")
    assert resolve_verify() == "/etc/ssl/cert-file.pem"


def test_explicit_arg_wins_over_env(monkeypatch):
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/etc/ssl/env.pem")
    assert resolve_verify("/explicit/ca.pem") == "/explicit/ca.pem"


def test_requests_ca_bundle_preferred_over_ssl_cert_file(monkeypatch):
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/etc/ssl/requests.pem")
    monkeypatch.setenv("SSL_CERT_FILE", "/etc/ssl/cert-file.pem")
    assert resolve_verify() == "/etc/ssl/requests.pem"
