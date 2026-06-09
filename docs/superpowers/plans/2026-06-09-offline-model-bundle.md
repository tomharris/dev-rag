# Offline Model Bundle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let DevRAG bootstrap its reranker and BM25 models on a network that blocks huggingface.co, by fetching a model bundle from a dev-rag GitHub release and unpacking it into the local caches — automatically on first use and via an explicit command.

**Architecture:** A new `devrag/ingest/model_bundle.py` owns the bundle URL/checksum constants, cache-dir resolution, a filesystem presence check, the CA-aware download+safe-extract, and the `ensure_models()` auto path. FastEmbed is pointed at a stable cache dir (it defaults to a volatile temp dir). The CLI gains a `download-models` command; CLI and MCP both call `ensure_models()` at their sparse-encoder chokepoints, so first use self-heals. The offline-first loaders from PR #62 then load both models with zero HF access.

**Tech Stack:** Python 3.12+, httpx (with `resolve_verify` for the corporate CA), tarfile (`filter='data'` safe extraction), huggingface_hub (cache path only), fastembed, Typer (CLI), pytest + respx (HTTP mocking).

---

## File Structure

- **Create** `devrag/ingest/model_bundle.py` — bundle constants, `resolve_fastembed_cache_dir`, `bundle_target_dirs`, `models_present`, `download_bundle`, `ensure_models`.
- **Modify** `devrag/config.py` — `SparseEmbeddingConfig.cache_dir`; `NetworkConfig.model_bundle_url` / `model_bundle_sha256` / `auto_download_models`.
- **Modify** `devrag/ingest/sparse_encoder.py` — accept `cache_dir`, pass to FastEmbed on both load paths, add a clear error hint.
- **Modify** `devrag/retrieve/reranker.py` — extend the existing error hint to mention `devrag download-models`.
- **Modify** `devrag/cli.py` — pass `cache_dir` in `_make_sparse_encoder`; call `ensure_models`; add `download-models` command.
- **Modify** `devrag/mcp_server.py` — pass `cache_dir` in `_get_sparse_encoder`; call `ensure_models`.
- **Create** `scripts/build_model_bundle.py` — maintainer script that warms caches, tars them, prints sha256.
- **Modify** `README.md` — document the fully-blocked first-run flow and overrides.
- **Tests:** `tests/test_model_bundle.py` (new), `tests/test_sparse_encoder.py`, `tests/test_reranker.py`, `tests/test_cli.py`.

---

## Task 1: Config fields

**Files:**
- Modify: `devrag/config.py:18-21` (SparseEmbeddingConfig), `devrag/config.py:126-131` (NetworkConfig)
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_model_bundle_and_cache_dir_defaults():
    from devrag.config import DevragConfig
    c = DevragConfig()
    assert c.sparse_embedding.cache_dir == ""
    assert c.network.model_bundle_url == ""
    assert c.network.model_bundle_sha256 == ""
    assert c.network.auto_download_models is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_model_bundle_and_cache_dir_defaults -v`
Expected: FAIL with `AttributeError: 'SparseEmbeddingConfig' object has no attribute 'cache_dir'`

- [ ] **Step 3: Add the fields**

In `devrag/config.py`, change `SparseEmbeddingConfig`:

```python
@dataclass
class SparseEmbeddingConfig:
    model: str = "Qdrant/bm25"
    batch_size: int = 64
    # FastEmbed model cache dir. "" resolves to ~/.cache/devrag/fastembed.
    # FastEmbed defaults to a volatile temp dir that is wiped on reboot, so we
    # pin a persistent one (also the extraction target for `download-models`).
    cache_dir: str = ""
```

And change `NetworkConfig` (keep the existing `ca_bundle` field and its comment):

```python
@dataclass
class NetworkConfig:
    # Path to a PEM CA bundle for httpx's `verify`; "" = certifi default.
    # Point this at your corporate proxy's root CA when behind a TLS-intercepting
    # proxy. Falls back to REQUESTS_CA_BUNDLE / SSL_CERT_FILE when left empty.
    ca_bundle: str = ""
    # Override the model-bundle download URL (e.g. an internal/air-gapped mirror).
    # "" uses the built-in dev-rag GitHub release asset.
    model_bundle_url: str = ""
    # Expected sha256 of the bundle at model_bundle_url. "" uses the built-in
    # checksum when model_bundle_url is also "", otherwise skips verification.
    model_bundle_sha256: str = ""
    # Auto-download the model bundle on first use when models aren't cached.
    # Set false (CI / air-gapped) to fail fast with the explicit-command hint.
    auto_download_models: bool = True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py::test_model_bundle_and_cache_dir_defaults -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add devrag/config.py tests/test_config.py
git commit -m "feat: config fields for model bundle and fastembed cache dir"
```

---

## Task 2: Cache-dir resolution and target dirs

**Files:**
- Create: `devrag/ingest/model_bundle.py`
- Test: `tests/test_model_bundle.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_model_bundle.py`:

```python
from pathlib import Path

import huggingface_hub.constants as hf_constants

from devrag.config import DevragConfig
from devrag.ingest import model_bundle


def test_resolve_fastembed_cache_dir_default():
    cfg = DevragConfig()
    expected = Path("~/.cache/devrag/fastembed").expanduser()
    assert model_bundle.resolve_fastembed_cache_dir(cfg) == expected


def test_resolve_fastembed_cache_dir_override(tmp_path):
    cfg = DevragConfig()
    cfg.sparse_embedding.cache_dir = str(tmp_path / "fe")
    assert model_bundle.resolve_fastembed_cache_dir(cfg) == tmp_path / "fe"


def test_bundle_target_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    cfg = DevragConfig()
    cfg.sparse_embedding.cache_dir = str(tmp_path / "fe")
    hf_dir, fe_dir = model_bundle.bundle_target_dirs(cfg)
    assert hf_dir == tmp_path / "hub"
    assert fe_dir == tmp_path / "fe"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model_bundle.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'devrag.ingest.model_bundle'`

- [ ] **Step 3: Create the module with these functions**

Create `devrag/ingest/model_bundle.py`:

```python
"""Download and place DevRAG's HF models from a self-hosted bundle.

On networks that block huggingface.co, the reranker and BM25 models cannot be
downloaded from HF. This module fetches a pre-built bundle of both models from a
dev-rag GitHub release (reachable behind the corporate proxy) and unpacks it
into the HF hub cache and the FastEmbed cache, so the offline-first loaders find
them with no HF access.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Pinned per release. The release task (Task 9) creates the `models-v1` tag,
# uploads the bundle, and fills DEFAULT_BUNDLE_SHA256 with the printed digest.
DEFAULT_BUNDLE_URL = "https://github.com/tomharris/dev-rag/releases/download/models-v1/devrag-models.tar.gz"
DEFAULT_BUNDLE_SHA256 = ""


def resolve_fastembed_cache_dir(config) -> Path:
    """The persistent FastEmbed cache dir; default ~/.cache/devrag/fastembed."""
    raw = config.sparse_embedding.cache_dir
    if raw:
        return Path(raw).expanduser()
    return Path("~/.cache/devrag/fastembed").expanduser()


def bundle_target_dirs(config) -> tuple[Path, Path]:
    """(HF hub cache dir, FastEmbed cache dir) — the bundle's extraction targets."""
    from huggingface_hub import constants
    return Path(constants.HF_HUB_CACHE), resolve_fastembed_cache_dir(config)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model_bundle.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add devrag/ingest/model_bundle.py tests/test_model_bundle.py
git commit -m "feat: model_bundle cache-dir resolution and target dirs"
```

---

## Task 3: `models_present` filesystem check

**Files:**
- Modify: `devrag/ingest/model_bundle.py`
- Test: `tests/test_model_bundle.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_model_bundle.py`:

```python
def _hf_repo_dir(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def test_models_present_false_when_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    cfg = DevragConfig()
    cfg.sparse_embedding.cache_dir = str(tmp_path / "fe")
    assert model_bundle.models_present(cfg) is False


def test_models_present_true_when_both_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    cfg = DevragConfig()
    cfg.sparse_embedding.cache_dir = str(tmp_path / "fe")
    # reranker: HF cache snapshot with a file
    snap = tmp_path / "hub" / _hf_repo_dir(cfg.retrieval.reranker_model) / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    # bm25: fastembed cache dir with the model dir non-empty
    bm = tmp_path / "fe" / _hf_repo_dir(cfg.sparse_embedding.model)
    bm.mkdir(parents=True)
    (bm / "config.json").write_text("{}")
    assert model_bundle.models_present(cfg) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model_bundle.py -k models_present -v`
Expected: FAIL with `AttributeError: module 'devrag.ingest.model_bundle' has no attribute 'models_present'`

- [ ] **Step 3: Implement**

Add to `devrag/ingest/model_bundle.py`:

```python
def _hf_repo_dir(repo_id: str) -> str:
    """HF/FastEmbed on-disk cache dir name for a repo id (org/name)."""
    return "models--" + repo_id.replace("/", "--")


def _has_files(directory: Path) -> bool:
    return directory.is_dir() and any(p.is_file() for p in directory.rglob("*"))


def models_present(config) -> bool:
    """True when both the reranker and BM25 models are already on disk."""
    hf_dir, fe_dir = bundle_target_dirs(config)
    reranker_ok = _has_files(hf_dir / _hf_repo_dir(config.retrieval.reranker_model) / "snapshots")
    bm25_ok = _has_files(fe_dir / _hf_repo_dir(config.sparse_embedding.model))
    return reranker_ok and bm25_ok
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model_bundle.py -k models_present -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add devrag/ingest/model_bundle.py tests/test_model_bundle.py
git commit -m "feat: models_present filesystem presence check"
```

---

## Task 4: `download_bundle` — fetch, verify, safe-extract

**Files:**
- Modify: `devrag/ingest/model_bundle.py`
- Test: `tests/test_model_bundle.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_model_bundle.py` (add imports `import hashlib, io, tarfile, pytest, respx, httpx` at the top):

```python
def _make_bundle_bytes() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in [
            ("hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2/snapshots/abc/config.json", b"{}"),
            ("fastembed/models--Qdrant--bm25/config.json", b"{}"),
        ]:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _make_evil_bundle_bytes() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = b"pwned"
        info = tarfile.TarInfo(name="../evil.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


@respx.mock
def test_download_bundle_fetches_verifies_and_extracts(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    cfg = DevragConfig()
    cfg.sparse_embedding.cache_dir = str(tmp_path / "fe")
    payload = _make_bundle_bytes()
    cfg.network.model_bundle_url = "https://example.test/bundle.tar.gz"
    cfg.network.model_bundle_sha256 = hashlib.sha256(payload).hexdigest()
    respx.get(cfg.network.model_bundle_url).mock(return_value=httpx.Response(200, content=payload))

    model_bundle.download_bundle(cfg)

    assert (tmp_path / "hub" / "models--cross-encoder--ms-marco-MiniLM-L-6-v2"
            / "snapshots" / "abc" / "config.json").exists()
    assert (tmp_path / "fe" / "models--Qdrant--bm25" / "config.json").exists()


@respx.mock
def test_download_bundle_checksum_mismatch_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    cfg = DevragConfig()
    cfg.sparse_embedding.cache_dir = str(tmp_path / "fe")
    cfg.network.model_bundle_url = "https://example.test/bundle.tar.gz"
    cfg.network.model_bundle_sha256 = "0" * 64
    respx.get(cfg.network.model_bundle_url).mock(return_value=httpx.Response(200, content=_make_bundle_bytes()))
    with pytest.raises(RuntimeError, match="checksum"):
        model_bundle.download_bundle(cfg)


@respx.mock
def test_download_bundle_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    cfg = DevragConfig()
    cfg.sparse_embedding.cache_dir = str(tmp_path / "fe")
    payload = _make_evil_bundle_bytes()
    cfg.network.model_bundle_url = "https://example.test/evil.tar.gz"
    cfg.network.model_bundle_sha256 = hashlib.sha256(payload).hexdigest()
    respx.get(cfg.network.model_bundle_url).mock(return_value=httpx.Response(200, content=payload))
    with pytest.raises(Exception):
        model_bundle.download_bundle(cfg)
    assert not (tmp_path / "evil.txt").exists()


@respx.mock
def test_download_bundle_skips_when_present_without_force(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    cfg = DevragConfig()
    cfg.sparse_embedding.cache_dir = str(tmp_path / "fe")
    # Pre-create both models so models_present() is True.
    snap = tmp_path / "hub" / "models--cross-encoder--ms-marco-MiniLM-L-6-v2" / "snapshots" / "abc"
    snap.mkdir(parents=True); (snap / "config.json").write_text("{}")
    bm = tmp_path / "fe" / "models--Qdrant--bm25"; bm.mkdir(parents=True); (bm / "x").write_text("{}")
    route = respx.get("https://example.test/bundle.tar.gz").mock(return_value=httpx.Response(200, content=b""))
    cfg.network.model_bundle_url = "https://example.test/bundle.tar.gz"
    model_bundle.download_bundle(cfg)  # no force
    assert route.call_count == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model_bundle.py -k download_bundle -v`
Expected: FAIL with `AttributeError: ... has no attribute 'download_bundle'`

- [ ] **Step 3: Implement**

Add to `devrag/ingest/model_bundle.py` (add `import hashlib, shutil, tarfile, tempfile` and `import httpx` to the imports):

```python
def _resolve_url_and_sha(config) -> tuple[str, str]:
    url = config.network.model_bundle_url or DEFAULT_BUNDLE_URL
    sha = config.network.model_bundle_sha256
    if not sha and url == DEFAULT_BUNDLE_URL:
        sha = DEFAULT_BUNDLE_SHA256
    return url, sha


def download_bundle(config, *, force: bool = False) -> None:
    """Download the model bundle and extract it into the HF + FastEmbed caches."""
    from devrag.utils.http import resolve_verify

    if not force and models_present(config):
        return

    url, expected_sha = _resolve_url_and_sha(config)
    hf_dir, fe_dir = bundle_target_dirs(config)
    verify = resolve_verify(config.network.ca_bundle)

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "bundle.tar.gz"
        hasher = hashlib.sha256()
        with httpx.Client(verify=verify, follow_redirects=True, timeout=120.0) as client:
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                with archive.open("wb") as fh:
                    for chunk in resp.iter_bytes():
                        hasher.update(chunk)
                        fh.write(chunk)

        if expected_sha:
            actual = hasher.hexdigest()
            if actual != expected_sha:
                raise RuntimeError(
                    f"Model bundle checksum mismatch: expected {expected_sha}, got {actual}"
                )
        else:
            logger.warning("No expected sha256 for model bundle %s; skipping verification", url)

        staging = Path(tmp) / "staging"
        staging.mkdir()
        with tarfile.open(archive, mode="r:gz") as tar:
            tar.extractall(path=staging, filter="data")  # filter='data' rejects traversal/absolute paths

        _merge_tree(staging / "hub", hf_dir)
        _merge_tree(staging / "fastembed", fe_dir)


def _merge_tree(src: Path, dst: Path) -> None:
    if not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
```

Note: `tarfile.extractall(filter="data")` (Python 3.12+) raises on members with absolute paths or `..` traversal, so the evil-bundle test raises before anything is written outside `staging`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model_bundle.py -k download_bundle -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add devrag/ingest/model_bundle.py tests/test_model_bundle.py
git commit -m "feat: download_bundle fetch/verify/safe-extract"
```

---

## Task 5: `ensure_models` auto path

**Files:**
- Modify: `devrag/ingest/model_bundle.py`
- Test: `tests/test_model_bundle.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_model_bundle.py`:

```python
def test_ensure_models_noop_when_present(tmp_path, monkeypatch):
    cfg = DevragConfig()
    monkeypatch.setattr(model_bundle, "models_present", lambda c: True)
    called = {"n": 0}
    monkeypatch.setattr(model_bundle, "download_bundle", lambda c, **k: called.__setitem__("n", called["n"] + 1))
    model_bundle.ensure_models(cfg)
    assert called["n"] == 0


def test_ensure_models_downloads_when_absent_and_auto_on(tmp_path, monkeypatch):
    cfg = DevragConfig()
    monkeypatch.setattr(model_bundle, "models_present", lambda c: False)
    called = {"n": 0}
    monkeypatch.setattr(model_bundle, "download_bundle", lambda c, **k: called.__setitem__("n", called["n"] + 1))
    model_bundle.ensure_models(cfg)
    assert called["n"] == 1


def test_ensure_models_noop_when_auto_off(tmp_path, monkeypatch):
    cfg = DevragConfig()
    cfg.network.auto_download_models = False
    monkeypatch.setattr(model_bundle, "models_present", lambda c: False)
    called = {"n": 0}
    monkeypatch.setattr(model_bundle, "download_bundle", lambda c, **k: called.__setitem__("n", called["n"] + 1))
    model_bundle.ensure_models(cfg)
    assert called["n"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model_bundle.py -k ensure_models -v`
Expected: FAIL with `AttributeError: ... has no attribute 'ensure_models'`

- [ ] **Step 3: Implement**

Add to `devrag/ingest/model_bundle.py`:

```python
def ensure_models(config) -> None:
    """Auto-download the bundle on first use when models aren't cached.

    No-op when the models are present or when auto-download is disabled (the
    offline-first loaders then surface a clear error pointing at
    `devrag download-models`).
    """
    if models_present(config):
        return
    if not config.network.auto_download_models:
        return
    url, _ = _resolve_url_and_sha(config)
    print(f"DevRAG models not found locally; downloading bundle (~88 MB) from {url} ...", file=sys.stderr)
    download_bundle(config)
    print("DevRAG models ready.", file=sys.stderr)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model_bundle.py -k ensure_models -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add devrag/ingest/model_bundle.py tests/test_model_bundle.py
git commit -m "feat: ensure_models auto-download on first use"
```

---

## Task 6: FastEmbed stable cache dir + error hint

**Files:**
- Modify: `devrag/ingest/sparse_encoder.py:10-20`
- Test: `tests/test_sparse_encoder.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sparse_encoder.py`:

```python
def test_sparse_encoder_passes_cache_dir_on_both_paths():
    online_model = MagicMock()
    with patch("fastembed.SparseTextEmbedding", side_effect=[OSError("not cached"), online_model]) as mock_ste:
        enc = BM25SparseEncoder(model_name="Qdrant/bm25", cache_dir="/tmp/devrag-fe")
        enc._get_model()
    assert mock_ste.call_count == 2
    assert mock_ste.call_args_list[0].kwargs.get("cache_dir") == "/tmp/devrag-fe"
    assert mock_ste.call_args_list[0].kwargs.get("local_files_only") is True
    assert mock_ste.call_args_list[1].kwargs.get("cache_dir") == "/tmp/devrag-fe"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sparse_encoder.py::test_sparse_encoder_passes_cache_dir_on_both_paths -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'cache_dir'`

- [ ] **Step 3: Implement**

Replace `BM25SparseEncoder.__init__` and `_get_model` in `devrag/ingest/sparse_encoder.py`:

```python
class BM25SparseEncoder:
    def __init__(self, model_name: str = "Qdrant/bm25", batch_size: int = 64, cache_dir: str | None = None) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self._model = None

    def _get_model(self):
        if self._model is None:
            from fastembed import SparseTextEmbedding
            # Prefer the local cache so a network that blocks huggingface.co
            # (or the huggingface_hub closed-client bug on a metadata refresh)
            # can't break loading an already-downloaded model.
            try:
                self._model = SparseTextEmbedding(
                    model_name=self.model_name, cache_dir=self.cache_dir, local_files_only=True
                )
            except Exception:
                # Not cached — allow a network download (first run, HF reachable).
                try:
                    self._model = SparseTextEmbedding(model_name=self.model_name, cache_dir=self.cache_dir)
                except Exception as exc:
                    raise RuntimeError(
                        f"BM25 model '{self.model_name}' is not cached locally and could not be "
                        f"downloaded. Run `devrag download-models` to fetch the model bundle."
                    ) from exc
        return self._model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sparse_encoder.py -v`
Expected: PASS (all, including the existing tests which patch `_get_model` or don't pass `cache_dir`)

- [ ] **Step 5: Commit**

```bash
git add devrag/ingest/sparse_encoder.py tests/test_sparse_encoder.py
git commit -m "feat: fastembed stable cache_dir + download-models error hint"
```

---

## Task 7: Reranker error hint

**Files:**
- Modify: `devrag/retrieve/reranker.py:30-35`
- Test: `tests/test_reranker.py:71-78` (the `test_reranker_clear_error_when_uncached_and_offline` added in PR #62)

- [ ] **Step 1: Update the test to assert the hint**

Replace the body of `test_reranker_clear_error_when_uncached_and_offline` in `tests/test_reranker.py`:

```python
@patch("devrag.retrieve.reranker.CrossEncoder")
def test_reranker_clear_error_when_uncached_and_offline(mock_ce_class):
    mock_ce_class.side_effect = [
        OSError("not cached"),
        RuntimeError("Cannot send a request, as the client has been closed."),
    ]
    with pytest.raises(RuntimeError, match="devrag download-models"):
        Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_reranker.py::test_reranker_clear_error_when_uncached_and_offline -v`
Expected: FAIL (current message says "Pre-download it on a network…", not "devrag download-models")

- [ ] **Step 3: Update the error message**

In `devrag/retrieve/reranker.py`, change the `raise RuntimeError(...)` block to:

```python
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"Reranker model '{model_name}' is not cached locally and huggingface.co "
                        f"could not be reached. Run `devrag download-models` to fetch the model "
                        f"bundle, or set retrieval.rerank: false to disable reranking."
                    ) from exc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_reranker.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add devrag/retrieve/reranker.py tests/test_reranker.py
git commit -m "feat: point reranker error at devrag download-models"
```

---

## Task 8: Wire CLI + MCP (cache_dir, ensure_models, download-models command)

**Files:**
- Modify: `devrag/cli.py:25-30` (`_make_sparse_encoder`), `devrag/cli.py` (new command)
- Modify: `devrag/mcp_server.py:78-86` (`_get_sparse_encoder`)
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli.py`. The file already has `from unittest.mock import MagicMock, patch`, `from devrag.cli import app`, and a module-level `runner = CliRunner()` — reuse them:

```python
def test_download_models_command_invokes_download_bundle():
    from devrag.config import DevragConfig
    with patch("devrag.ingest.model_bundle.download_bundle") as mock_dl, \
         patch("devrag.config.load_config", return_value=DevragConfig()):
        result = runner.invoke(app, ["download-models", "--force"])
    assert result.exit_code == 0, result.output
    assert mock_dl.call_count == 1
    assert mock_dl.call_args.kwargs.get("force") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py::test_download_models_command_invokes_download_bundle -v`
Expected: FAIL with a non-zero exit code (no such command `download-models`)

- [ ] **Step 3a: Pass cache_dir + ensure_models in CLI**

In `devrag/cli.py`, replace `_make_sparse_encoder`:

```python
def _make_sparse_encoder(config):
    from devrag.ingest.model_bundle import ensure_models, resolve_fastembed_cache_dir
    from devrag.ingest.sparse_encoder import BM25SparseEncoder
    ensure_models(config)
    return BM25SparseEncoder(
        model_name=config.sparse_embedding.model,
        batch_size=config.sparse_embedding.batch_size,
        cache_dir=str(resolve_fastembed_cache_dir(config)),
    )
```

(Every `search`/`index` path already calls `_make_sparse_encoder(config)`, so this is the single CLI chokepoint for both auto-download and the persistent cache dir.)

- [ ] **Step 3b: Add the `download-models` command**

In `devrag/cli.py`, add (near the other `@app.command()` definitions):

```python
@app.command("download-models")
def download_models(
    force: bool = typer.Option(False, "--force", help="Re-download even if models are already cached"),
    url: str = typer.Option("", "--url", help="One-off override of the bundle URL"),
):
    """Download the reranker + BM25 model bundle into the local caches."""
    from devrag.config import load_config
    from devrag.ingest.model_bundle import download_bundle
    config = load_config(project_dir=Path.cwd())
    if url:
        config.network.model_bundle_url = url
    typer.echo("Downloading DevRAG model bundle...")
    download_bundle(config, force=force)
    typer.echo("Models ready.")
```

- [ ] **Step 3c: Pass cache_dir + ensure_models in MCP**

In `devrag/mcp_server.py`, replace `_get_sparse_encoder`:

```python
def _get_sparse_encoder() -> BM25SparseEncoder:
    global _sparse_encoder
    if _sparse_encoder is None:
        from devrag.ingest.model_bundle import ensure_models, resolve_fastembed_cache_dir
        config = _get_config()
        ensure_models(config)
        _sparse_encoder = BM25SparseEncoder(
            model_name=config.sparse_embedding.model,
            batch_size=config.sparse_embedding.batch_size,
            cache_dir=str(resolve_fastembed_cache_dir(config)),
        )
    return _sparse_encoder
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (including the new command test)

- [ ] **Step 5: Commit**

```bash
git add devrag/cli.py devrag/mcp_server.py tests/test_cli.py
git commit -m "feat: wire ensure_models + cache_dir into CLI/MCP and add download-models command"
```

---

## Task 9: Maintainer build script + release (produces the real URL/sha)

**Files:**
- Create: `scripts/build_model_bundle.py`
- Modify: `devrag/ingest/model_bundle.py` (set `DEFAULT_BUNDLE_SHA256`)

- [ ] **Step 1: Create the build script**

Create `scripts/build_model_bundle.py`:

```python
"""Build the DevRAG model bundle on an HF-reachable machine.

Warms the default reranker + BM25 models into temp caches, tars them into the
hub/ + fastembed/ layout the extractor expects, and prints the sha256. Upload
the result to a dev-rag GitHub release, then update DEFAULT_BUNDLE_URL /
DEFAULT_BUNDLE_SHA256 in devrag/ingest/model_bundle.py.

Usage: uv run python scripts/build_model_bundle.py
"""
from __future__ import annotations

import hashlib
import tarfile
import tempfile
from pathlib import Path

RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BM25 = "Qdrant/bm25"
OUT = Path("dist/devrag-models.tar.gz")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        hub = Path(tmp) / "hub"
        fe = Path(tmp) / "fastembed"
        hub.mkdir(); fe.mkdir()

        from sentence_transformers import CrossEncoder
        CrossEncoder(RERANKER, cache_folder=str(hub))

        from fastembed import SparseTextEmbedding
        m = SparseTextEmbedding(model_name=BM25, cache_dir=str(fe))
        next(m.query_embed("warm"))

        with tarfile.open(OUT, "w:gz") as tar:
            tar.add(hub, arcname="hub")
            tar.add(fe, arcname="fastembed")

    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")
    print(f"sha256: {digest}")
    print("Next: gh release create models-v1 dist/devrag-models.tar.gz  (or `gh release upload models-v1 ...`)")
    print("Then set DEFAULT_BUNDLE_SHA256 in devrag/ingest/model_bundle.py to the sha256 above.")


if __name__ == "__main__":
    main()
```

Note: `CrossEncoder(..., cache_folder=str(hub))` writes the HF hub cache layout under `hub/`, matching `models--cross-encoder--...`. Confirm the produced top-level dir is `models--...` (HF default); if `cache_folder` nests differently, adjust the `arcname` so the archive contains `hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2/...`.

- [ ] **Step 2: Run the script on an HF-reachable network**

Run: `uv run python scripts/build_model_bundle.py`
Expected: prints `dist/devrag-models.tar.gz`, a byte size (~88 MB), and a `sha256:` line.

- [ ] **Step 3: Create the release and upload the asset**

Run:
```bash
gh release create models-v1 dist/devrag-models.tar.gz \
  --title "DevRAG model bundle v1" \
  --notes "Reranker (cross-encoder/ms-marco-MiniLM-L-6-v2) + BM25 (Qdrant/bm25) for offline/blocked-network bootstrap."
```
Expected: release `models-v1` created with the asset attached at the URL matching `DEFAULT_BUNDLE_URL`.

- [ ] **Step 4: Pin the checksum**

In `devrag/ingest/model_bundle.py`, set `DEFAULT_BUNDLE_SHA256` to the sha256 printed in Step 2.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_model_bundle.py devrag/ingest/model_bundle.py
git commit -m "feat: model-bundle build script and pinned release checksum"
```

---

## Task 10: Docs + full verification

**Files:**
- Modify: `README.md` (Requirements / Quick Start area, around the Ollama requirement at `README.md:53-58`)

- [ ] **Step 1: Document the offline flow**

Add a subsection under Requirements/Quick Start in `README.md`:

```markdown
### Models on a network that blocks Hugging Face

DevRAG's reranker and BM25 models come from Hugging Face. On first use it
auto-downloads a self-hosted bundle of both models from a dev-rag GitHub release
(reachable even when huggingface.co is blocked) and caches them locally; later
runs load offline. To fetch them explicitly:

```bash
devrag download-models          # fetch/refresh the model bundle
devrag download-models --force  # re-download even if cached
```

Overrides (in `~/.config/devrag/devrag.yaml` or `.devrag.yaml`):

```yaml
network:
  auto_download_models: true            # set false to disable auto-fetch (CI/air-gapped)
  model_bundle_url: ""                  # internal/air-gapped mirror of the bundle
  model_bundle_sha256: ""               # expected checksum for a custom model_bundle_url
sparse_embedding:
  cache_dir: ""                         # FastEmbed cache; "" = ~/.cache/devrag/fastembed
```

Note: the bundle carries the **default** model names. If you override
`retrieval.reranker_model` or `sparse_embedding.model`, you need Hugging Face (or
a faithful mirror) access for your chosen model.
```

- [ ] **Step 2: Run the full unit suite**

Run: `uv run pytest tests/ -q`
Expected: all pass (no new skips).

- [ ] **Step 3: Verify the explicit command end-to-end**

Run (on the blocked network, after deleting caches to simulate first run — back them up first):
```bash
mv ~/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2 /tmp/_bk_rr 2>/dev/null || true
rm -rf ~/.cache/devrag/fastembed
uv run devrag download-models
uv run devrag search "reranker" --scope code
```
Expected: `download-models` reports "Models ready."; the search returns results with no crash.

- [ ] **Step 4: Verify auto-download on first use**

Run:
```bash
rm -rf ~/.cache/devrag/fastembed
mv ~/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2 /tmp/_bk_rr2 2>/dev/null || true
uv run devrag search "auto download" --scope code
```
Expected: stderr shows "DevRAG models not found locally; downloading bundle…", then "DevRAG models ready.", then results.

- [ ] **Step 5: Reinstall the uv tool and commit docs**

```bash
uv tool install --reinstall .
git add README.md
git commit -m "docs: offline model-bundle first-run flow and overrides"
```

---

## Self-Review notes

- **Spec coverage:** stable FastEmbed cache (Task 6 + Task 1), bundle module with `bundle_target_dirs`/`models_present`/`download_bundle`/`ensure_models` (Tasks 2–5), CLI `download-models` + auto-wire in CLI & MCP (Task 8), config additions (Task 1), error-message hints (Tasks 6–7), maintainer build script + release (Task 9), docs (Task 10). All spec sections map to a task.
- **Checksum two-phase:** `DEFAULT_BUNDLE_SHA256` starts `""` (verification skipped with a logged warning) and is hardened to the real digest in Task 9 Step 4 — a concrete step with the exact value source, not a placeholder.
- **Type/name consistency:** `resolve_fastembed_cache_dir`, `bundle_target_dirs`, `models_present`, `download_bundle(config, *, force=False)`, `ensure_models(config)`, `_hf_repo_dir`, `_merge_tree` are used identically across tasks and tests; `BM25SparseEncoder(..., cache_dir=...)` matches all call sites (CLI, MCP, tests).
- **Safe extraction** relies on `tarfile.extractall(filter="data")` (Python 3.12+, which the project targets).
