"""Tests for CLI exception handling."""
from click.testing import CliRunner
from unittest.mock import patch
from claude_lint.cli import main


def test_keyboard_interrupt_exits_with_130():
    """Test that KeyboardInterrupt exits with code 130 (SIGINT)."""
    runner = CliRunner()

    with patch("claude_lint.cli.run_compliance_check") as mock_run:
        mock_run.side_effect = KeyboardInterrupt()

        result = runner.invoke(main, ["--full"])

        assert result.exit_code == 130
        assert "cancelled" in result.output.lower()


def test_value_error_shows_error_message():
    """Test that ValueError shows helpful error message."""
    runner = CliRunner()

    with patch("claude_lint.cli.run_compliance_check") as mock_run:
        mock_run.side_effect = ValueError("Invalid configuration")

        result = runner.invoke(main, ["--full"])

        assert result.exit_code == 2
        assert "Invalid configuration" in result.output


def test_generic_exception_shows_helpful_message():
    """Test that unexpected exceptions show helpful message."""
    runner = CliRunner()

    with patch("claude_lint.cli.run_compliance_check") as mock_run:
        mock_run.side_effect = RuntimeError("Unexpected internal error")

        result = runner.invoke(main, ["--full", "--verbose"])

        assert result.exit_code == 2
        # Should see the actual error in verbose mode
