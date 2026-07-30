from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from devrag.cli import _parse_since, app

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


def test_cli_index_refresh_help():
    result = runner.invoke(app, ["index", "refresh", "--help"])
    assert result.exit_code == 0
    assert "--full" in result.stdout


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


def test_download_models_command_invokes_download_bundle():
    from devrag.config import DevragConfig
    with patch("devrag.ingest.model_bundle.download_bundle") as mock_dl, \
         patch("devrag.config.load_config", return_value=DevragConfig()):
        result = runner.invoke(app, ["download-models", "--force"])
    assert result.exit_code == 0, result.output
    assert mock_dl.call_count == 1
    assert mock_dl.call_args.kwargs.get("force") is True


# --- --since parsing ---


def test_parse_since_none_and_empty():
    assert _parse_since(None) is None
    assert _parse_since("") is None


def test_parse_since_valid_day_count():
    assert _parse_since("90d") == 90
    assert _parse_since("1d") == 1
    assert _parse_since("0d") == 0


def test_parse_since_tolerates_surrounding_whitespace():
    assert _parse_since("  30d  ") == 30


@pytest.mark.parametrize("bad", ["90days", "3w", "6m", "abc", "d", "90", "-5d", "9 0d", "90dd"])
def test_parse_since_rejects_malformed_input(bad):
    # A bare ValueError traceback is not a usable CLI error; BadParameter renders
    # a proper usage message and exits 2.
    with pytest.raises(typer.BadParameter):
        _parse_since(bad)


def test_parse_since_error_message_names_the_value_and_format():
    with pytest.raises(typer.BadParameter) as exc:
        _parse_since("3w")
    message = str(exc.value)
    assert "3w" in message
    assert "90d" in message


@pytest.mark.parametrize(
    "argv",
    [
        ["index", "prs", "acme/backend", "--since", "3w"],
        ["index", "issues", "acme/backend", "--since", "90days"],
        ["index", "jira", "--since", "abc"],
        ["index", "slite", "--since", "d"],
        ["index", "slack", "--since", "6m"],
        ["index", "sessions", "--since", "-5d"],
    ],
)
def test_cli_rejects_malformed_since_before_doing_work(argv):
    # Validation must happen before the store/embedder are built, so a typo
    # fails instantly instead of after loading models.
    result = runner.invoke(app, argv)
    assert result.exit_code == 2
    assert "since" in result.output.lower()


# --- blank query guard (embed_query contract) ---


@patch("devrag.cli._get_search_components")
def test_cli_search_rejects_blank_query(mock_get):
    # embed_query() now raises on blank input; the CLI must turn that into a
    # usage error rather than surfacing a ValueError traceback.
    result = runner.invoke(app, ["search", "   "])
    assert result.exit_code == 2
    assert "query" in result.output.lower()
    assert mock_get.call_count == 0  # rejected before loading models
