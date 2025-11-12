from pathlib import Path
from unittest.mock import patch
from click.testing import CliRunner
from claude_lint.cli import main
from claude_lint.metrics import AnalysisMetrics


@patch("claude_lint.cli.run_compliance_check")
def test_cli_full_scan(mock_run_check):
    """Test CLI with full scan."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        # Create config
        Path(".agent-lint.json").write_text('{"include": ["**/*.py"]}')
        Path("CLAUDE.md").write_text("# Guidelines")

        # Mock compliance check
        mock_metrics = AnalysisMetrics()
        mock_metrics.finish()
        mock_run_check.return_value = ([{"file": "test.py", "violations": []}], mock_metrics)

        # Run CLI
        result = runner.invoke(main, ["--full"])

        assert result.exit_code == 0
        assert "test.py" in result.output


@patch("claude_lint.cli.run_compliance_check")
def test_cli_diff_mode(mock_run_check):
    """Test CLI with diff mode."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path(".agent-lint.json").write_text("{}")
        Path("CLAUDE.md").write_text("# Guidelines")

        mock_metrics = AnalysisMetrics()
        mock_metrics.finish()
        mock_run_check.return_value = ([], mock_metrics)

        runner.invoke(main, ["--diff", "main"])

        assert mock_run_check.called
        call_args = mock_run_check.call_args
        assert call_args[1]["mode"] == "diff"
        assert call_args[1]["base_branch"] == "main"


@patch("claude_lint.cli.run_compliance_check")
def test_cli_json_output(mock_run_check):
    """Test CLI with JSON output format."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path(".agent-lint.json").write_text("{}")
        Path("CLAUDE.md").write_text("# Guidelines")

        mock_metrics = AnalysisMetrics()
        mock_metrics.finish()
        mock_run_check.return_value = ([{"file": "test.py", "violations": []}], mock_metrics)

        result = runner.invoke(main, ["--full", "--json"])

        assert result.exit_code == 0
        assert '"results"' in result.output
        assert '"summary"' in result.output
        assert '"metrics"' in result.output


@patch("claude_lint.cli.run_compliance_check")
def test_cli_exit_code_on_violations(mock_run_check):
    """Test CLI returns exit code 1 when violations found."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path(".agent-lint.json").write_text("{}")
        Path("CLAUDE.md").write_text("# Guidelines")

        mock_metrics = AnalysisMetrics()
        mock_metrics.finish()
        mock_run_check.return_value = (
            [
                {
                    "file": "test.py",
                    "violations": [{"type": "error", "message": "bad", "line": None}],
                }
            ],
            mock_metrics,
        )

        result = runner.invoke(main, ["--full"])

        assert result.exit_code == 1
