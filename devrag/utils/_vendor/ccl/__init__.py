"""Vendored subset of ccl_chromium_reader / ccl_simplesnappy (MIT, CCL Forensics).

Pure-python Chromium LevelDB + LocalStorage reader used to extract the Slack
desktop app's ``xoxc`` session token from its on-disk LocalStorage store. Vendored
(rather than depended on) because the upstream packages are GitHub-only — not on
PyPI — and every PyPI alternative pulls in C extensions (``python-snappy``,
``zstd``) that would break ``uv tool install`` on machines without those system
libraries. See ``LICENSE`` in this directory for the upstream license/attribution.

Sources:
  - https://github.com/cclgroupltd/ccl_chromium_reader  (ccl_leveldb, ccl_chromium_localstorage, common)
  - https://github.com/cclgroupltd/ccl_simplesnappy      (ccl_simplesnappy)

Local modifications: import paths flattened to this package (the upstream
``storage_formats`` subpackage and top-level ``ccl_simplesnappy`` become
intra-package imports). No logic changes.
"""
