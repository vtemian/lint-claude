import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest
from claude_lint.git_utils import is_git_repo, get_changed_files_from_branch, GIT_TIMEOUT


def test_is_git_repo_timeout():
    """Test that is_git_repo has timeout."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("git", GIT_TIMEOUT)

            result = is_git_repo(Path(tmpdir))

            assert result is False
            assert mock_run.call_args[1]["timeout"] == GIT_TIMEOUT


def test_get_changed_files_timeout():
    """Test that get_changed_files_from_branch raises on timeout."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("git", GIT_TIMEOUT)

            with pytest.raises(RuntimeError, match="Git command timed out"):
                get_changed_files_from_branch(Path(tmpdir), "main")


def test_git_timeout_constant_exists():
    """Test that GIT_TIMEOUT constant is defined."""
    from claude_lint import git_utils

    assert hasattr(git_utils, "GIT_TIMEOUT")
    assert isinstance(git_utils.GIT_TIMEOUT, (int, float))
    assert git_utils.GIT_TIMEOUT > 0
