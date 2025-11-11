import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from click.testing import CliRunner
import pytest
from claude_lint.cli import main


@patch("claude_lint.cli.Orchestrator")
def test_cli_full_scan(mock_orchestrator):
    """Test CLI with full scan."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        # Create config
        Path(".agent-lint.json").write_text('{"include": ["**/*.py"]}')
        Path("CLAUDE.md").write_text("# Guidelines")

        # Mock orchestrator
        mock_orch = Mock()
        mock_orch.run.return_value = [
            {"file": "test.py", "violations": []}
        ]
        mock_orchestrator.return_value = mock_orch

        # Run CLI
        result = runner.invoke(main, ["--full"])

        assert result.exit_code == 0
        assert "test.py" in result.output


@patch("claude_lint.cli.Orchestrator")
def test_cli_diff_mode(mock_orchestrator):
    """Test CLI with diff mode."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path(".agent-lint.json").write_text('{}')
        Path("CLAUDE.md").write_text("# Guidelines")

        mock_orch = Mock()
        mock_orch.run.return_value = []
        mock_orchestrator.return_value = mock_orch

        result = runner.invoke(main, ["--diff", "main"])

        assert mock_orch.run.called
        call_args = mock_orch.run.call_args
        assert call_args[1]["mode"] == "diff"
        assert call_args[1]["base_branch"] == "main"


@patch("claude_lint.cli.Orchestrator")
def test_cli_json_output(mock_orchestrator):
    """Test CLI with JSON output format."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path(".agent-lint.json").write_text('{}')
        Path("CLAUDE.md").write_text("# Guidelines")

        mock_orch = Mock()
        mock_orch.run.return_value = [
            {"file": "test.py", "violations": []}
        ]
        mock_orchestrator.return_value = mock_orch

        result = runner.invoke(main, ["--full", "--json"])

        assert result.exit_code == 0
        assert '"results"' in result.output
        assert '"summary"' in result.output


@patch("claude_lint.cli.Orchestrator")
def test_cli_exit_code_on_violations(mock_orchestrator):
    """Test CLI returns exit code 1 when violations found."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path(".agent-lint.json").write_text('{}')
        Path("CLAUDE.md").write_text("# Guidelines")

        mock_orch = Mock()
        mock_orch.run.return_value = [
            {
                "file": "test.py",
                "violations": [{"type": "error", "message": "bad", "line": None}]
            }
        ]
        mock_orchestrator.return_value = mock_orch

        result = runner.invoke(main, ["--full"])

        assert result.exit_code == 1
