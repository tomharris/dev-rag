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


def test_match_expected_file_suffix():
    """Index stores absolute paths; expectations are repo-relative."""
    from devrag.eval import match_expected_file
    expected = {"devrag/eval.py"}
    assert match_expected_file("/home/tom/Projects/dev-rag/devrag/eval.py", expected) == "devrag/eval.py"
    assert match_expected_file("devrag/eval.py", expected) == "devrag/eval.py"


def test_match_expected_file_respects_segment_boundaries():
    from devrag.eval import match_expected_file
    assert match_expected_file("/repo/src/my_eval.py", {"eval.py"}) is None
    assert match_expected_file("/repo/src/eval.py", {"eval.py"}) == "eval.py"


def test_compute_metrics_counts_irrelevant_results_in_denominator():
    """Results with no file_path/pr_number still consume a rank position.

    Previously they were skipped, so 4 Slack chunks + 1 correct file scored
    precision 1.0 instead of 0.2.
    """
    test_cases = [{"query": "q", "expected_files": ["src/auth.py"]}]
    search_results = {"q": [
        {"channel_id": "C1"}, {"channel_id": "C1"}, {"channel_id": "C1"},
        {"channel_id": "C1"}, {"file_path": "src/auth.py"},
    ]}
    metrics = compute_metrics(test_cases, search_results, k=5)
    assert metrics["precision_at_5"] == pytest.approx(1 / 5, abs=0.01)
    assert metrics["mrr"] == pytest.approx(1 / 5, abs=0.01)


def test_compute_metrics_matches_absolute_indexed_paths():
    test_cases = [{"query": "q", "expected_files": ["devrag/retrieve/hybrid_search.py"]}]
    search_results = {"q": [{"file_path": "/home/tom/Projects/dev-rag/devrag/retrieve/hybrid_search.py"}]}
    assert compute_metrics(test_cases, search_results, k=5)["mrr"] == pytest.approx(1.0)


def test_compute_metrics_prefers_pr_number_on_pr_diff_chunks():
    """A PR diff chunk carries both pr_number and file_path; expected_prs wins."""
    test_cases = [{"query": "q", "expected_prs": [42]}]
    search_results = {"q": [{"pr_number": 42, "file_path": "internal/ingest/roster.go"}]}
    assert compute_metrics(test_cases, search_results, k=5)["mrr"] == pytest.approx(1.0)


def test_compute_grouped_metrics_splits_by_label():
    from devrag.eval import compute_grouped_metrics
    test_cases = [
        {"query": "a", "hop_type": "single", "expected_files": ["a.py"]},
        {"query": "b", "hop_type": "multi", "expected_files": ["b.py"]},
    ]
    search_results = {"a": [{"file_path": "a.py"}], "b": [{"file_path": "zzz.py"}]}
    grouped = compute_grouped_metrics(test_cases, search_results, key="hop_type", k=5)
    assert grouped["single"]["mrr"] == pytest.approx(1.0)
    assert grouped["multi"]["mrr"] == pytest.approx(0.0)
    assert grouped["single"]["num_queries"] == 1


def test_compute_grouped_metrics_unlabeled_bucket():
    from devrag.eval import compute_grouped_metrics
    grouped = compute_grouped_metrics([{"query": "a", "expected_files": ["a.py"]}], {"a": []},
                                      key="hop_type", k=5)
    assert list(grouped) == ["unlabeled"]


def test_recall_at_k_counts_distinct_relevant_items():
    """Two chunks from the same expected file must not score recall above 1.0."""
    assert recall_at_k(["a.py", "a.py"], {"a.py"}, k=5) == 1.0


def test_compute_metrics_recall_never_exceeds_one():
    test_cases = [{"query": "q", "expected_files": ["src/auth.py"]}]
    search_results = {"q": [
        {"file_path": "/abs/src/auth.py"}, {"file_path": "/abs/src/auth.py"},
    ]}
    metrics = compute_metrics(test_cases, search_results, k=5)
    assert metrics["recall_at_5"] == pytest.approx(1.0)
    # Both results really are relevant, so precision counts both.
    assert metrics["precision_at_5"] == pytest.approx(1.0)
