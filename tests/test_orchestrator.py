import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
from claude_lint.orchestrator import Orchestrator
from claude_lint.config import Config


@patch("claude_lint.orchestrator.ClaudeClient")
@patch("claude_lint.orchestrator.is_git_repo")
def test_orchestrator_full_scan(mock_is_git, mock_claude_client):
    """Test full project scan mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test files
        (tmpdir / "file1.py").write_text("print('hello')")
        (tmpdir / "file2.py").write_text("print('world')")
        (tmpdir / "CLAUDE.md").write_text("# Guidelines\n\nUse TDD.")

        # Mock git check
        mock_is_git.return_value = False

        # Mock Claude API
        mock_client = Mock()
        mock_client.analyze_files.return_value = """
        ```json
        {
          "results": [
            {"file": "file1.py", "violations": []},
            {"file": "file2.py", "violations": []}
          ]
        }
        ```
        """
        mock_claude_client.return_value = mock_client

        # Run orchestrator
        config = Config(
            include=["**/*.py"],
            exclude=["tests/**"],
            batch_size=10,
            api_key="test-key"
        )

        orchestrator = Orchestrator(tmpdir, config)
        results = orchestrator.run(mode="full")

        # Verify
        assert len(results) == 2
        assert mock_client.analyze_files.called


@patch("claude_lint.orchestrator.ClaudeClient")
@patch("claude_lint.orchestrator.get_changed_files_from_branch")
@patch("claude_lint.orchestrator.is_git_repo")
def test_orchestrator_diff_mode(mock_is_git, mock_git_diff, mock_claude_client):
    """Test diff mode with git."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test files
        (tmpdir / "file1.py").write_text("code")
        (tmpdir / "file2.py").write_text("code")
        (tmpdir / "CLAUDE.md").write_text("# Guidelines")

        # Mock git
        mock_is_git.return_value = True
        mock_git_diff.return_value = ["file1.py"]  # Only file1 changed

        # Mock Claude API
        mock_client = Mock()
        mock_client.analyze_files.return_value = """
        ```json
        {"results": [{"file": "file1.py", "violations": []}]}
        ```
        """
        mock_claude_client.return_value = mock_client

        # Run orchestrator
        config = Config(
            include=["**/*.py"],
            exclude=[],
            batch_size=10,
            api_key="test-key"
        )

        orchestrator = Orchestrator(tmpdir, config)
        results = orchestrator.run(mode="diff", base_branch="main")

        # Verify only file1 was checked
        assert len(results) == 1
        assert results[0]["file"] == "file1.py"
