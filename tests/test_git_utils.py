import tempfile
from pathlib import Path
import subprocess
import pytest
from claude_lint.git_utils import (
    get_changed_files_from_branch,
    get_working_directory_files,
    get_staged_files,
    is_git_repo
)


def setup_git_repo(tmpdir: Path) -> Path:
    """Helper to set up a git repo for testing."""
    repo_dir = tmpdir / "repo"
    repo_dir.mkdir()

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_dir,
        check=True,
        capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_dir,
        check=True,
        capture_output=True
    )

    # Create initial commit
    (repo_dir / "file1.py").write_text("print('hello')")
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=repo_dir,
        check=True,
        capture_output=True
    )

    return repo_dir


def test_is_git_repo():
    """Test checking if directory is a git repo."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        repo_dir = setup_git_repo(tmpdir)

        assert is_git_repo(repo_dir) is True
        assert is_git_repo(tmpdir / "nonexistent") is False


def test_get_changed_files_from_branch():
    """Test getting files changed from a branch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        repo_dir = setup_git_repo(tmpdir)

        # Create a new file
        (repo_dir / "file2.py").write_text("print('world')")
        subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "add file2"],
            cwd=repo_dir,
            check=True,
            capture_output=True
        )

        # Get files changed from HEAD~1
        files = get_changed_files_from_branch(repo_dir, "HEAD~1")

        assert "file2.py" in files
        assert "file1.py" not in files


def test_get_working_directory_files():
    """Test getting modified and untracked files in working directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        repo_dir = setup_git_repo(tmpdir)

        # Modify existing file
        (repo_dir / "file1.py").write_text("print('modified')")

        # Add untracked file
        (repo_dir / "file3.py").write_text("print('new')")

        files = get_working_directory_files(repo_dir)

        assert "file1.py" in files  # modified
        assert "file3.py" in files  # untracked


def test_get_staged_files():
    """Test getting staged files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        repo_dir = setup_git_repo(tmpdir)

        # Modify and stage file
        (repo_dir / "file1.py").write_text("print('staged')")
        subprocess.run(["git", "add", "file1.py"], cwd=repo_dir, check=True, capture_output=True)

        # Create unstaged file
        (repo_dir / "file3.py").write_text("print('unstaged')")

        files = get_staged_files(repo_dir)

        assert "file1.py" in files
        assert "file3.py" not in files
