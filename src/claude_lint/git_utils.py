"""Git integration utilities."""
from pathlib import Path
import subprocess
from typing import Optional


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
            text=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_changed_files_from_branch(repo_path: Path, base_branch: str) -> list[str]:
    """Get files changed from a base branch.

    Args:
        repo_path: Path to git repository
        base_branch: Base branch to compare against (e.g., 'main', 'HEAD~1')

    Returns:
        List of changed file paths relative to repo root
    """
    result = subprocess.run(
        ["git", "diff", "--name-only", base_branch],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True
    )

    files = [f.strip() for f in result.stdout.split("\n") if f.strip()]
    return files


def get_working_directory_files(repo_path: Path) -> list[str]:
    """Get modified and untracked files in working directory.

    Args:
        repo_path: Path to git repository

    Returns:
        List of modified/untracked file paths relative to repo root
    """
    # Get modified files
    result = subprocess.run(
        ["git", "diff", "--name-only"],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True
    )
    modified = [f.strip() for f in result.stdout.split("\n") if f.strip()]

    # Get untracked files
    result = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True
    )
    untracked = [f.strip() for f in result.stdout.split("\n") if f.strip()]

    return list(set(modified + untracked))


def get_staged_files(repo_path: Path) -> list[str]:
    """Get staged files.

    Args:
        repo_path: Path to git repository

    Returns:
        List of staged file paths relative to repo root
    """
    result = subprocess.run(
        ["git", "diff", "--name-only", "--cached"],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True
    )

    files = [f.strip() for f in result.stdout.split("\n") if f.strip()]
    return files
