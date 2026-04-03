import json
import pytest
from devrag.eval import compute_metrics, precision_at_k, recall_at_k, mrr


def test_precision_at_k():
    retrieved = ["a.py", "b.py", "c.py", "d.py", "e.py"]
    relevant = {"a.py", "c.py", "e.py"}
    assert precision_at_k(retrieved, relevant, k=5) == 3 / 5


def test_precision_at_k_partial():
    retrieved = ["a.py", "b.py", "c.py"]
    relevant = {"a.py"}
    assert precision_at_k(retrieved, relevant, k=2) == 1 / 2


def test_recall_at_k():
    retrieved = ["a.py", "b.py", "c.py"]
    relevant = {"a.py", "c.py", "d.py"}
    assert recall_at_k(retrieved, relevant, k=3) == 2 / 3


def test_mrr():
    retrieved = ["b.py", "a.py", "c.py"]
    relevant = {"a.py"}
    assert mrr(retrieved, relevant) == 1 / 2


def test_mrr_first_position():
    retrieved = ["a.py", "b.py"]
    relevant = {"a.py"}
    assert mrr(retrieved, relevant) == 1.0


def test_mrr_no_relevant():
    retrieved = ["b.py", "c.py"]
    relevant = {"a.py"}
    assert mrr(retrieved, relevant) == 0.0


def test_compute_metrics():
    test_cases = [{"query": "how does auth work", "expected_files": ["src/auth.py", "src/login.py"]}]
    search_results = {
        "how does auth work": [
            {"file_path": "src/auth.py"}, {"file_path": "src/other.py"}, {"file_path": "src/login.py"},
        ],
    }
    metrics = compute_metrics(test_cases, search_results, k=5)
    assert metrics["precision_at_5"] == pytest.approx(2 / 3, abs=0.01)
    assert metrics["recall_at_5"] == pytest.approx(2 / 2, abs=0.01)
    assert metrics["mrr"] == pytest.approx(1.0, abs=0.01)


def test_compute_metrics_with_prs():
    test_cases = [{"query": "why did we change auth", "expected_prs": [42, 56]}]
    search_results = {
        "why did we change auth": [{"pr_number": 42}, {"pr_number": 99}],
    }
    metrics = compute_metrics(test_cases, search_results, k=5)
    assert metrics["precision_at_5"] == pytest.approx(1 / 2, abs=0.01)
    assert metrics["recall_at_5"] == pytest.approx(1 / 2, abs=0.01)
