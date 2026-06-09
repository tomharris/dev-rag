# Offline model bundle — design

## Context

DevRAG loads two models from Hugging Face: the cross-encoder reranker
(`cross-encoder/ms-marco-MiniLM-L-6-v2`) and the FastEmbed BM25 sparse model
(`Qdrant/bm25`). A prior fix (PR #62) made loading **offline-first** — both
loaders try `local_files_only=True` and fall back to an online download only
when the model isn't cached. That fixed the steady state but left a
**first-run hole**: a user whose machine has never cached the models, on a
network that blocks `huggingface.co`, cannot bootstrap them at all.

Findings from probing the target environment:

- `huggingface.co` is **blocked at the TLS layer** behind the corporate
  (Netskope) proxy — intermittently for the origin, and the blocking applies
  to the whole HF domain set. There is **no corporate HF-API mirror** and one
  is unlikely to be provisioned.
- The public mirror `hf-mirror.com` is reachable but is **not a
  protocol-faithful HF mirror** — `huggingface_hub`'s downloader rejects it
  (`FileMetadataError: Distant resource does not seem to be on huggingface.co`),
  so pointing `HF_ENDPOINT` at it does not work.
- **`github.com` / `objects.githubusercontent.com`, `raw.githubusercontent.com`,
  and PyPI are reachable** through the corporate CA even when HF is blocked.
- The corporate proxy intercepts TLS, so any client we use must trust the
  corporate CA. DevRAG already centralizes this via `resolve_verify()` /
  `network.ca_bundle`; `huggingface_hub`'s own client does not, which is why we
  avoid relying on it here.
- The reranker uses the persistent HF cache (`~/.cache/huggingface/hub`), but
  **FastEmbed caches BM25 to a volatile temp dir**
  (`$TMPDIR/fastembed_cache`) that is wiped on reboot — so even a successful
  download re-fetches later and re-hits the block.

Model sizes: reranker ~88 MB, BM25 small. Both licenses permit redistribution.

### Intended outcome

A user on a fully HF-blocked network can run DevRAG with zero manual model
setup: the models are fetched from a host that *is* reachable (a dev-rag
GitHub release), placed into persistent caches, and loaded offline thereafter.

## Approach

Distribute both models as one **bundle attached to a dev-rag GitHub release**.
A `devrag download-models` command — and an automatic on-first-use path that
shares its implementation — fetches the bundle through DevRAG's CA-aware httpx
client and unpacks it into the model caches. The offline-first loaders from
PR #62 then find the models with no HF access.

Rejected alternatives: configurable `HF_ENDPOINT` mirror (no faithful mirror
available, ruled out above); a manual local-directory-only path (doesn't solve
*how* files arrive).

## Components

### 1. Stable FastEmbed cache — `config.py`, `devrag/ingest/sparse_encoder.py`

- Add `SparseEmbeddingConfig.cache_dir: str = ""`. Empty resolves to
  `~/.cache/devrag/fastembed` via a shared helper
  `resolve_fastembed_cache_dir(config)`.
- `BM25SparseEncoder` accepts/derives this dir and passes `cache_dir=` to
  `SparseTextEmbedding` on **both** the `local_files_only=True` and the online
  fallback paths (verified: a custom `cache_dir` warms and loads offline).
- Fixes the reboot-wipe and gives the bundle a deterministic extraction target.

### 2. Shared bundle module — `devrag/ingest/model_bundle.py` (new)

- Constants pinned per release: `DEFAULT_BUNDLE_URL` (a dev-rag GitHub release
  asset), `DEFAULT_BUNDLE_SHA256`.
- `bundle_target_dirs(config) -> (hf_hub_cache, fastembed_cache)` — resolves
  `huggingface_hub.constants.HF_HUB_CACHE` and the FastEmbed cache dir.
- `models_present(config) -> bool` — lightweight **filesystem** presence check
  (reranker snapshot dir with key files in HF cache; `models--Qdrant--bm25`
  in the FastEmbed cache). Avoids a heavy model load just to probe.
- `download_bundle(config, *, force=False) -> None`:
  - Resolve URL: `config.network.model_bundle_url` override else
    `DEFAULT_BUNDLE_URL`.
  - GET via `httpx` with `verify=resolve_verify(config.network.ca_bundle)`
    (works behind the proxy; github verified reachable with the CA), streamed
    to a temp file with a timeout.
  - Verify sha256 against the configured/expected checksum; mismatch → raise.
  - **Safe-extract** (reject absolute paths / `..` traversal) the tar: entries
    under `hub/` → HF hub cache, under `fastembed/` → FastEmbed cache dir.
    Extract to a temp dir then move into place; clean up partials on failure.
- `ensure_models(config) -> None`: if `not models_present()` and
  `config.network.auto_download_models`, print a one-line stderr notice
  ("DevRAG models not found; downloading ~88 MB from <url>…") and call
  `download_bundle()`. No-op when present or auto-download disabled.

### 3. CLI command — `devrag download-models` (`cli.py`)

- Options: `--force` (re-download even if present), `--url` (one-off override).
- Calls `download_bundle(config, force=...)`; prints source, size, destination,
  and a final "ready" confirmation.

### 4. Auto-on-first-use wiring

- `_get_search_components()` calls `ensure_models(config)` before building the
  reranker / sparse encoder.
- The `index` paths that build a BM25 encoder call `ensure_models(config)` too.
- Mirror the same calls in the MCP server's component initialization
  (`mcp_server.py`), so CLI and MCP behave identically.
- Because both the command and the auto path call `download_bundle` /
  `ensure_models`, they never drift.

### 5. Config additions — `config.py`

- `NetworkConfig.model_bundle_url: str = ""` — override host (internal/air-gapped
  mirror); empty → `DEFAULT_BUNDLE_URL`.
- `NetworkConfig.auto_download_models: bool = True` — disable surprise fetches
  (CI, air-gapped) to fail fast with the explicit-command hint instead.

### 6. Error-message hints

- The shipped reranker "not cached … and huggingface.co could not be reached"
  error, and a matching new one for the sparse encoder, gain a line:
  "run `devrag download-models`".

### 7. Maintainer build script — `scripts/build_model_bundle.py` (new)

- Run on an HF-reachable machine: warm the reranker into a temp HF cache and
  BM25 into a temp FastEmbed cache (default model names), tar them into the
  `hub/` + `fastembed/` layout with a `manifest.json` (model names, versions),
  and print the resulting **sha256**.
- Maintainer uploads via `gh release upload <tag> dist/devrag-models.tar.gz`,
  then updates `DEFAULT_BUNDLE_URL` + `DEFAULT_BUNDLE_SHA256`.
- Documented coupling: the bundle carries the **default** model names only.

## Data flow

`devrag search` (or `index`, or first MCP tool call) → `ensure_models(config)`
→ if absent & auto-download on: CA-aware GET from the dev-rag GitHub release →
sha256 verify → safe-extract into HF + FastEmbed caches → offline-first loaders
(PR #62) load both models with no HF access.

## Error handling

- Network failure / non-200 / timeout, checksum mismatch, or a malformed/unsafe
  archive each raise a clear, actionable error; partial downloads/extracts are
  removed so a retry is clean.
- Auto-download disabled or bundle host unreachable → the loaders' existing
  clear error fires, now pointing at `devrag download-models`.
- Tar extraction rejects entries with absolute paths or `..` components.

## Testing

- `sparse_encoder`: `cache_dir` is resolved and passed on both load paths.
- `model_bundle`: `models_present` true/false detection; `download_bundle`
  against a mocked HTTP server serving a tiny fixture tar — success, checksum
  mismatch, idempotent skip, `--force`, and path-traversal rejection; partial
  cleanup on failure.
- `ensure_models`: downloads when absent + auto on; no-op when present; no-op
  (and no network) when auto off.
- CLI: `download-models` invokes `download_bundle` with the right args; error
  messages include the hint.

## Out of scope (YAGNI)

- The bundle covers only the **default** model names; users who override
  `reranker_model` / the sparse model need HF or mirror access for their choice
  (documented).
- No streaming/partial model loading, no delta updates — re-download the whole
  (small) bundle when stale.
