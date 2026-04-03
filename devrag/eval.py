from __future__ import annotations
import json
from pathlib import Path


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return sum(1 for r in top_k if r in relevant) / len(top_k)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    return sum(1 for r in top_k if r in relevant) / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for i, item in enumerate(retrieved):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def compute_metrics(test_cases: list[dict], search_results: dict[str, list[dict]], k: int = 5) -> dict[str, float]:
    all_precision, all_recall, all_mrr = [], [], []
    for case in test_cases:
        query = case["query"]
        results = search_results.get(query, [])
        expected_files = set(case.get("expected_files", []))
        expected_prs = set(case.get("expected_prs", []))
        retrieved: list[str] = []
        for r in results:
            if "file_path" in r and expected_files:
                retrieved.append(r["file_path"])
            elif "pr_number" in r and expected_prs:
                retrieved.append(str(r["pr_number"]))
        relevant = expected_files | {str(p) for p in expected_prs}
        all_precision.append(precision_at_k(retrieved, relevant, k))
        all_recall.append(recall_at_k(retrieved, relevant, k))
        all_mrr.append(mrr(retrieved, relevant))
    n = len(test_cases) or 1
    return {f"precision_at_{k}": sum(all_precision) / n,
            f"recall_at_{k}": sum(all_recall) / n,
            "mrr": sum(all_mrr) / n, "num_queries": len(test_cases)}


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
