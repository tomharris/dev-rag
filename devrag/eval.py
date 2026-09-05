from __future__ import annotations
import json
from pathlib import Path


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return sum(1 for r in top_k if r in relevant) / len(top_k)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the relevant set covered by the top k results.

    Counts *distinct* relevant items: dedupe allows up to `max_per_source`
    chunks from the same file, so the same expected path can occupy two rank
    positions. Summing hits instead of intersecting sets scored those twice and
    produced recall above 1.0.
    """
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for i, item in enumerate(retrieved):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def _normalize(path: str) -> str:
    return path.replace("\\", "/").strip("/")


def match_expected_file(actual: str, expected_files: set[str]) -> str | None:
    """Return the expected path ``actual`` refers to, or None.

    Matching is path-suffix tolerant in both directions because the index and the
    eval set disagree about path form today: ``code_indexer`` stores an absolute
    path (``/home/tom/Projects/dev-rag/devrag/eval.py``) while PR diffs and
    hand-written expectations use repo-relative ones (``devrag/eval.py``).
    Comparison is on whole path segments, so ``eval.py`` never matches
    ``my_eval.py``. Keeping expectations repo-relative means this eval set stays
    valid after paths are normalized in the index.
    """
    a = _normalize(actual)
    for expected in expected_files:
        e = _normalize(expected)
        if a == e or a.endswith("/" + e) or e.endswith("/" + a):
            return expected
    return None


def _identify(result: dict, expected_files: set[str], expected_prs: set[str], idx: int) -> str:
    """Map one result to the expected identifier it satisfies, else a unique miss token.

    Every result consumes a rank position whether or not it is relevant — a miss
    token rather than a skip. Dropping non-matching results (the previous
    behaviour) shrank the precision denominator, so a query answered with four
    irrelevant Slack chunks and one correct file scored 1.0 instead of 0.2.

    PRs are checked before files because a PR *diff* chunk carries both
    ``pr_number`` and a ``file_path``; for a case that expects PRs, the PR number
    is the identifier under test.
    """
    pr = result.get("pr_number")
    if pr is not None and str(pr) in expected_prs:
        return str(pr)
    file_path = result.get("file_path")
    if file_path:
        matched = match_expected_file(str(file_path), expected_files)
        if matched is not None:
            return matched
    return f"__miss_{idx}__"


def compute_metrics(test_cases: list[dict], search_results: dict[str, list[dict]], k: int = 5) -> dict[str, float]:
    all_precision, all_recall, all_mrr = [], [], []
    for case in test_cases:
        query = case["query"]
        results = search_results.get(query, [])
        expected_files = set(case.get("expected_files", []))
        expected_prs = {str(p) for p in case.get("expected_prs", [])}
        retrieved = [_identify(r, expected_files, expected_prs, i) for i, r in enumerate(results)]
        relevant = expected_files | expected_prs
        all_precision.append(precision_at_k(retrieved, relevant, k))
        all_recall.append(recall_at_k(retrieved, relevant, k))
        all_mrr.append(mrr(retrieved, relevant))
    n = len(test_cases) or 1
    return {f"precision_at_{k}": sum(all_precision) / n,
            f"recall_at_{k}": sum(all_recall) / n,
            "mrr": sum(all_mrr) / n, "num_queries": len(test_cases)}


def compute_grouped_metrics(test_cases: list[dict], search_results: dict[str, list[dict]],
                            key: str, k: int = 5) -> dict[str, dict[str, float]]:
    """Compute metrics per value of a label on each test case.

    The point of Stage 0: `key="hop_type"` splits single-hop lookups from
    multi-hop questions, which is the split the graph-retrieval literature says
    determines whether a graph layer would help at all. Cases missing the label
    are grouped under "unlabeled".
    """
    groups: dict[str, list[dict]] = {}
    for case in test_cases:
        groups.setdefault(str(case.get(key, "unlabeled")), []).append(case)
    return {name: compute_metrics(cases, search_results, k=k)
            for name, cases in sorted(groups.items())}


def load_test_queries(path: Path) -> list[dict]:
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def save_results(results: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


def load_results(path: Path) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results
