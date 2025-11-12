"""Git integration utilities."""
import subprocess
from pathlib import Path

# Timeout for git operations in seconds
GIT_TIMEOUT = 30


def is_git_repo(path: Path) -> bool:
    """Check if path is inside a git repository.

    Args:
        path: Directory path to check

    Returns:
        True if path is in a git repo, False otherwise
    """
    try:
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_changed_files_from_branch(repo_path: Path, base_branch: str) -> list[str]:
    """Get files changed from a base branch.

    Args:
        repo_path: Path to git repository
        base_branch: Base branch to compare against (e.g., 'main', 'HEAD~1')

    Returns:
        List of changed file paths relative to repo root

    Raises:
        RuntimeError: If git command times out or fails
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", base_branch],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Git command timed out after {GIT_TIMEOUT}s") from e

    files = [f.strip() for f in result.stdout.split("\n") if f.strip()]
    return files


def get_working_directory_files(repo_path: Path) -> list[str]:
    """Get modified and untracked files in working directory.

    Args:
        repo_path: Path to git repository

    Returns:
        List of modified/untracked file paths relative to repo root

    Raises:
        RuntimeError: If git command times out or fails
    """
    try:
        # Get modified files
        result = subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
        )
        modified = [f.strip() for f in result.stdout.split("\n") if f.strip()]

        # Get untracked files
        result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
        )
        untracked = [f.strip() for f in result.stdout.split("\n") if f.strip()]
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Git command timed out after {GIT_TIMEOUT}s") from e

    return list(set(modified + untracked))


def get_staged_files(repo_path: Path) -> list[str]:
    """Get staged files.

    Args:
        repo_path: Path to git repository

    Returns:
        List of staged file paths relative to repo root

    Raises:
        RuntimeError: If git command times out or fails
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--cached"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Git command timed out after {GIT_TIMEOUT}s") from e

    files = [f.strip() for f in result.stdout.split("\n") if f.strip()]
    return files
