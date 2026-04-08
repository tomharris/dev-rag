from unittest.mock import MagicMock, patch
from typer.testing import CliRunner
from devrag.cli import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "search" in result.stdout.lower() or "Search" in result.stdout


def test_cli_search_help():
    result = runner.invoke(app, ["search", "--help"])
    assert result.exit_code == 0
    assert "query" in result.stdout.lower()
    assert "scope" in result.stdout.lower()


@patch("devrag.cli._get_search_components")
def test_cli_search(mock_get):
    mock_hybrid = MagicMock()
    mock_hybrid.search.return_value = []
    mock_reranker = MagicMock()
    mock_get.return_value = (mock_hybrid, mock_reranker, MagicMock())
    result = runner.invoke(app, ["search", "how does auth work"])
    assert result.exit_code == 0


def test_cli_status_help():
    result = runner.invoke(app, ["status", "--help"])
    assert result.exit_code == 0


def test_cli_index_repo_help():
    result = runner.invoke(app, ["index", "repo", "--help"])
    assert result.exit_code == 0
    assert "path" in result.stdout.lower()


def test_cli_index_docs_help():
    result = runner.invoke(app, ["index", "docs", "--help"])
    assert result.exit_code == 0


def test_cli_index_prs_help():
    result = runner.invoke(app, ["index", "prs", "--help"])
    assert result.exit_code == 0


def test_cli_config_help():
    result = runner.invoke(app, ["config", "--help"])
    assert result.exit_code == 0


def test_cli_serve_help():
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0


def test_cli_reindex_help():
    result = runner.invoke(app, ["reindex", "--help"])
    assert result.exit_code == 0
    assert "all" in result.stdout.lower()
    assert "--name" in result.stdout


def test_cli_reindex_no_args():
    result = runner.invoke(app, ["reindex"])
    assert result.exit_code == 1


def test_cli_eval_run_help():
    result = runner.invoke(app, ["eval", "run", "--help"])
    assert result.exit_code == 0


def test_cli_eval_compare_help():
    result = runner.invoke(app, ["eval", "compare", "--help"])
    assert result.exit_code == 0
