# Code Review Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all critical, major, and minor issues identified in senior developer code review to make lint-claude production-ready.

**Architecture:** Refactor existing functional codebase to add proper logging, error handling, atomic file operations, consistent pattern matching, and improved CLI experience while maintaining the no-classes constraint.

**Tech Stack:** Python 3.11+, anthropic SDK, click, pytest, logging module, atomic file operations

---

## Task 1: Add Logging Framework (Critical #2)

**Priority:** Critical
**Files:**
- Modify: `src/claude_lint/orchestrator.py:101,104,107`
- Create: `src/claude_lint/logging_config.py`
- Modify: `src/claude_lint/cli.py:18`
- Create: `tests/test_logging_config.py`

**Step 1: Write test for logging configuration**

Create `tests/test_logging_config.py`:

```python
import logging
from claude_lint.logging_config import setup_logging, get_logger


def test_setup_logging_default():
    """Test default logging setup."""
    setup_logging()
    logger = get_logger(__name__)
    assert logger.level == logging.WARNING


def test_setup_logging_verbose():
    """Test verbose logging setup."""
    setup_logging(verbose=True)
    logger = get_logger(__name__)
    assert logger.level == logging.INFO


def test_setup_logging_quiet():
    """Test quiet logging setup."""
    setup_logging(quiet=True)
    logger = get_logger(__name__)
    assert logger.level == logging.ERROR


def test_get_logger():
    """Test getting named logger."""
    logger = get_logger("test.module")
    assert logger.name == "claude_lint.test.module"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_logging_config.py -v`
Expected: FAIL with "No module named 'claude_lint.logging_config'"

**Step 3: Create logging configuration module**

Create `src/claude_lint/logging_config.py`:

```python
"""Logging configuration for lint-claude."""
import logging
import sys


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging for lint-claude.

    Args:
        verbose: Enable verbose (INFO level) logging
        quiet: Enable quiet (ERROR only) logging
    """
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    # Configure root logger for claude_lint namespace
    logger = logging.getLogger("claude_lint")
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Create console handler
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        "%(levelname)s: %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get logger for module.

    Args:
        name: Module name (will be prefixed with 'claude_lint.')

    Returns:
        Logger instance
    """
    if not name.startswith("claude_lint."):
        name = f"claude_lint.{name}"
    return logging.getLogger(name)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_logging_config.py -v`
Expected: PASS (4 tests)

**Step 5: Replace print statements in orchestrator**

Modify `src/claude_lint/orchestrator.py`:

Add import at top:
```python
from claude_lint.logging_config import get_logger

logger = get_logger(__name__)
```

Replace lines 101, 104, 107:
```python
# OLD:
print(f"Warning: File not found, skipping: {rel_path}")
print(f"Warning: Unable to decode file, skipping: {rel_path}")
print(f"Warning: Error reading file {rel_path}, skipping: {e}")

# NEW:
logger.warning(f"File not found, skipping: {rel_path}")
logger.warning(f"Unable to decode file, skipping: {rel_path}")
logger.warning(f"Error reading file {rel_path}, skipping: {e}")
```

**Step 6: Add logging flags to CLI**

Modify `src/claude_lint/cli.py`:

Add imports:
```python
from claude_lint.logging_config import setup_logging
```

Add CLI options after line 16:
```python
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
@click.option("--quiet", is_flag=True, help="Suppress warnings (errors only)")
@click.option("--config", type=click.Path(exists=True), help="Path to config file")
def main(full, diff, working, staged, output_json, verbose, quiet, config):
```

Add setup_logging call after line 18:
```python
def main(full, diff, working, staged, output_json, verbose, quiet, config):
    """Claude-lint: CLAUDE.md compliance checker."""
    # Setup logging
    setup_logging(verbose=verbose, quiet=quiet)
```

**Step 7: Run existing tests to verify no breakage**

Run: `pytest tests/ -v`
Expected: All existing tests still pass

**Step 8: Commit**

```bash
git add src/claude_lint/logging_config.py tests/test_logging_config.py src/claude_lint/orchestrator.py src/claude_lint/cli.py
git commit -m "feat: add logging framework with verbose/quiet flags

- Replace print() statements with proper logging
- Add --verbose and --quiet CLI flags
- Setup logging configuration module
- Log to stderr instead of stdout"
```

---

## Task 2: Add Subprocess Timeouts (Critical #1)

**Priority:** Critical
**Files:**
- Modify: `src/claude_lint/git_utils.py:16-25,38-47,60-79,91-100`
- Create: `tests/test_git_utils_timeout.py`

**Step 1: Write test for subprocess timeout**

Create `tests/test_git_utils_timeout.py`:

```python
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from claude_lint.git_utils import (
    is_git_repo,
    get_changed_files_from_branch,
    get_working_directory_files,
    get_staged_files,
    GIT_TIMEOUT
)


def test_is_git_repo_timeout():
    """Test that is_git_repo has timeout."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired('git', GIT_TIMEOUT)

            result = is_git_repo(Path(tmpdir))

            assert result is False
            assert mock_run.call_args[1]['timeout'] == GIT_TIMEOUT


def test_get_changed_files_timeout():
    """Test that get_changed_files_from_branch raises on timeout."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired('git', GIT_TIMEOUT)

            with pytest.raises(RuntimeError, match="Git command timed out"):
                get_changed_files_from_branch(Path(tmpdir), "main")


def test_git_timeout_constant_exists():
    """Test that GIT_TIMEOUT constant is defined."""
    from claude_lint import git_utils
    assert hasattr(git_utils, 'GIT_TIMEOUT')
    assert isinstance(git_utils.GIT_TIMEOUT, (int, float))
    assert git_utils.GIT_TIMEOUT > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_git_utils_timeout.py -v`
Expected: FAIL with "ImportError: cannot import name 'GIT_TIMEOUT'"

**Step 3: Add timeout to all subprocess calls**

Modify `src/claude_lint/git_utils.py`:

Add constant at top after imports:
```python
import subprocess

# Timeout for git operations in seconds
GIT_TIMEOUT = 30
```

Modify `is_git_repo()`:
```python
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
            timeout=GIT_TIMEOUT
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False
```

Modify `get_changed_files_from_branch()`:
```python
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
            timeout=GIT_TIMEOUT
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Git command timed out after {GIT_TIMEOUT}s") from e

    files = [f.strip() for f in result.stdout.split("\n") if f.strip()]
    return files
```

Modify `get_working_directory_files()`:
```python
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
            timeout=GIT_TIMEOUT
        )
        modified = [f.strip() for f in result.stdout.split("\n") if f.strip()]

        # Get untracked files
        result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT
        )
        untracked = [f.strip() for f in result.stdout.split("\n") if f.strip()]
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Git command timed out after {GIT_TIMEOUT}s") from e

    return list(set(modified + untracked))
```

Modify `get_staged_files()`:
```python
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
            timeout=GIT_TIMEOUT
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Git command timed out after {GIT_TIMEOUT}s") from e

    files = [f.strip() for f in result.stdout.split("\n") if f.strip()]
    return files
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_git_utils_timeout.py -v`
Expected: PASS (3 tests)

**Step 5: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/claude_lint/git_utils.py tests/test_git_utils_timeout.py
git commit -m "fix: add 30s timeout to all git subprocess calls

- Prevent indefinite hangs on git operations
- Add GIT_TIMEOUT constant (30 seconds)
- Raise RuntimeError on timeout with clear message
- Add timeout tests"
```

---

## Task 3: Fix Atomic File Writes (Critical #3)

**Priority:** Critical
**Files:**
- Create: `src/claude_lint/file_utils.py`
- Modify: `src/claude_lint/cache.py:52-69`
- Modify: `src/claude_lint/progress.py:50-59`
- Create: `tests/test_file_utils.py`

**Step 1: Write test for atomic write**

Create `tests/test_file_utils.py`:

```python
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from claude_lint.file_utils import atomic_write_json


def test_atomic_write_json_success():
    """Test successful atomic write."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        target = tmpdir / "test.json"
        data = {"key": "value", "list": [1, 2, 3]}

        atomic_write_json(data, target)

        # Verify file exists and contains correct data
        assert target.exists()
        with open(target) as f:
            loaded = json.load(f)
        assert loaded == data


def test_atomic_write_json_no_corruption_on_failure():
    """Test that existing file is not corrupted on write failure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        target = tmpdir / "test.json"

        # Write initial data
        original_data = {"original": "data"}
        target.write_text(json.dumps(original_data))

        # Mock json.dump to fail
        with patch('json.dump', side_effect=RuntimeError("Write failed")):
            with pytest.raises(RuntimeError, match="Write failed"):
                atomic_write_json({"new": "data"}, target)

        # Verify original file is unchanged
        with open(target) as f:
            loaded = json.load(f)
        assert loaded == original_data


def test_atomic_write_json_tmp_cleaned_up():
    """Test that temporary file is cleaned up on failure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        target = tmpdir / "test.json"

        # Mock json.dump to fail
        with patch('json.dump', side_effect=RuntimeError("Write failed")):
            with pytest.raises(RuntimeError):
                atomic_write_json({"data": "value"}, target)

        # Verify no .tmp files left behind
        tmp_files = list(tmpdir.glob("*.tmp"))
        assert len(tmp_files) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_file_utils.py -v`
Expected: FAIL with "No module named 'claude_lint.file_utils'"

**Step 3: Create atomic write utility**

Create `src/claude_lint/file_utils.py`:

```python
"""File operation utilities."""
import json
from pathlib import Path
from typing import Any


def atomic_write_json(data: Any, target_path: Path) -> None:
    """Write JSON data atomically to prevent corruption.

    Writes to a temporary file first, then atomically replaces the target.
    This ensures the target file is never in a partially-written state.

    Args:
        data: Data to serialize as JSON
        target_path: Target file path

    Raises:
        IOError: If write fails
        ValueError: If data cannot be serialized
    """
    # Create temp file in same directory as target for atomic replace
    tmp_path = target_path.with_suffix('.tmp')

    try:
        # Write to temporary file
        with open(tmp_path, 'w') as f:
            json.dump(data, f, indent=2)

        # Atomic replace (POSIX guarantees atomicity)
        tmp_path.replace(target_path)
    except Exception as e:
        # Clean up temp file on failure
        if tmp_path.exists():
            tmp_path.unlink()
        raise
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_file_utils.py -v`
Expected: PASS (3 tests)

**Step 5: Update cache.py to use atomic writes**

Modify `src/claude_lint/cache.py`:

Add import:
```python
from claude_lint.file_utils import atomic_write_json
```

Replace `save_cache()` function:
```python
def save_cache(cache: Cache, cache_path: Path) -> None:
    """Save cache to file atomically.

    Args:
        cache: Cache object to save
        cache_path: Path to cache file
    """
    entries_dict = {}
    for file_path, entry in cache.entries.items():
        entries_dict[file_path] = asdict(entry)

    data = {
        "claudeMdHash": cache.claude_md_hash,
        "entries": entries_dict
    }

    atomic_write_json(data, cache_path)
```

**Step 6: Update progress.py to use atomic writes**

Modify `src/claude_lint/progress.py`:

Add import:
```python
from claude_lint.file_utils import atomic_write_json
```

Replace `save_progress()` function:
```python
def save_progress(state: ProgressState, progress_file: Path) -> None:
    """Save progress to file atomically.

    Args:
        state: Progress state to save
        progress_file: Path to progress file
    """
    data = asdict(state)
    atomic_write_json(data, progress_file)
```

**Step 7: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 8: Commit**

```bash
git add src/claude_lint/file_utils.py tests/test_file_utils.py src/claude_lint/cache.py src/claude_lint/progress.py
git commit -m "fix: use atomic writes for cache and progress files

- Prevent file corruption on crashes or interruptions
- Create atomic_write_json utility function
- Write to .tmp then atomically replace target
- Clean up temp files on write failure"
```

---

## Task 4: Fix Exception Handling (Critical #4)

**Priority:** Critical
**Files:**
- Modify: `src/claude_lint/retry.py:34`
- Modify: `src/claude_lint/api_client.py:48`
- Create: `tests/test_retry_exceptions.py`
- Modify: `tests/test_api_client.py`

**Step 1: Write test for specific exception handling**

Create `tests/test_retry_exceptions.py`:

```python
import pytest
from claude_lint.retry import retry_with_backoff


def test_retry_does_not_catch_keyboard_interrupt():
    """Test that KeyboardInterrupt is not caught by retry."""
    def failing_func():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        retry_with_backoff(failing_func, max_retries=3)


def test_retry_does_not_catch_system_exit():
    """Test that SystemExit is not caught by retry."""
    def failing_func():
        raise SystemExit(1)

    with pytest.raises(SystemExit):
        retry_with_backoff(failing_func, max_retries=3)


def test_retry_catches_runtime_errors():
    """Test that runtime errors are caught and retried."""
    attempts = []

    def failing_func():
        attempts.append(1)
        raise RuntimeError("API failed")

    with pytest.raises(RuntimeError, match="API failed"):
        retry_with_backoff(failing_func, max_retries=3)

    assert len(attempts) == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_retry_exceptions.py -v`
Expected: FAIL - KeyboardInterrupt and SystemExit tests fail

**Step 3: Fix retry exception handling**

Modify `src/claude_lint/retry.py`:

```python
def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
) -> T:
    """Retry a function with exponential backoff.

    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry

    Returns:
        Result from successful function call

    Raises:
        Exception: The last exception if all retries exhausted
        KeyboardInterrupt: Immediately re-raised without retry
        SystemExit: Immediately re-raised without retry
    """
    last_exception = None
    delay = initial_delay

    for attempt in range(max_retries):
        try:
            return func()
        except (KeyboardInterrupt, SystemExit):
            # Never catch these - re-raise immediately
            raise
        except Exception as e:
            last_exception = e

            if attempt < max_retries - 1:  # Don't sleep on last attempt
                time.sleep(delay)
                delay *= backoff_factor

    # All retries exhausted
    raise last_exception
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_retry_exceptions.py -v`
Expected: PASS (3 tests)

**Step 5: Fix API client exception handling**

Modify `src/claude_lint/api_client.py`:

Add import:
```python
from anthropic import Anthropic, APIError, APIConnectionError, RateLimitError
```

Update `analyze_files()`:
```python
def analyze_files(api_key: str, guidelines: str, prompt: str) -> tuple[str, Any]:
    """Analyze files using Claude API with cached guidelines.

    Args:
        api_key: Anthropic API key
        guidelines: CLAUDE.md content (will be cached)
        prompt: Prompt with files to analyze

    Returns:
        Tuple of (response text, response object)

    Raises:
        ValueError: If guidelines or prompt are empty or invalid
        APIError: If API call fails (includes connection, rate limit, etc.)
    """
    # Input validation
    if not guidelines or not isinstance(guidelines, str) or not guidelines.strip():
        raise ValueError("guidelines must be a non-empty string")
    if not prompt or not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    client = Anthropic(api_key=api_key)

    # Use prompt caching for guidelines
    try:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": guidelines,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
    except (APIError, APIConnectionError, RateLimitError) as e:
        # Re-raise API errors with context
        raise APIError(f"Claude API call failed: {e}") from e
    except (KeyboardInterrupt, SystemExit):
        # Never catch these
        raise

    # Validate response
    if not response.content:
        raise ValueError("API returned empty response content")

    return response.content[0].text, response
```

**Step 6: Add test for API exception handling**

Modify `tests/test_api_client.py`, add test:

```python
from anthropic import APIError
from unittest.mock import patch, MagicMock
import pytest


def test_analyze_files_does_not_catch_keyboard_interrupt():
    """Test that KeyboardInterrupt is not caught."""
    with patch('anthropic.Anthropic') as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = KeyboardInterrupt()
        mock_anthropic.return_value = mock_client

        with pytest.raises(KeyboardInterrupt):
            analyze_files("key", "guidelines", "prompt")
```

**Step 7: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 8: Commit**

```bash
git add src/claude_lint/retry.py src/claude_lint/api_client.py tests/test_retry_exceptions.py tests/test_api_client.py
git commit -m "fix: never catch KeyboardInterrupt or SystemExit

- Catch specific exceptions in retry logic
- Use specific Anthropic API exceptions
- Allow ctrl-C to work properly
- Add tests for exception handling"
```

---

## Task 5: Remove Unused Dependency (Critical #5)

**Priority:** Critical
**Files:**
- Modify: `pyproject.toml:8`

**Step 1: Verify gitpython is unused**

Run: `grep -r "import git" src/ tests/`
Expected: No matches (or only comments)

Run: `grep -r "from git" src/ tests/`
Expected: No matches

**Step 2: Remove gitpython from dependencies**

Modify `pyproject.toml`:

```toml
dependencies = [
    "anthropic>=0.18.0",
    "click>=8.1.0",
]
```

**Step 3: Run tests to verify nothing breaks**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 4: Test installation in clean environment**

Run: `uv sync --reinstall`
Expected: Successful sync without gitpython

**Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "chore: remove unused gitpython dependency

- Using subprocess for git operations
- Reduces dependency footprint"
```

---

## Task 6: Unify Pattern Matching (Major #6)

**Priority:** Major
**Files:**
- Modify: `src/claude_lint/collector.py:4,9-37,97-110`
- Create: `tests/test_collector_patterns.py`

**Step 1: Write tests for pattern matching edge cases**

Create `tests/test_collector_patterns.py`:

```python
import tempfile
from pathlib import Path
from claude_lint.collector import collect_all_files, is_excluded
from claude_lint.config import Config


def test_pattern_matching_nested_double_star():
    """Test pattern with multiple ** segments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create nested structure
        (tmpdir / "src" / "tests" / "unit").mkdir(parents=True)
        (tmpdir / "src" / "tests" / "unit" / "test.py").write_text("code")
        (tmpdir / "lib" / "vendor" / "package").mkdir(parents=True)
        (tmpdir / "lib" / "vendor" / "package" / "file.py").write_text("code")

        config = Config(
            include=["**/*.py"],
            exclude=["**/tests/**"],
            batch_size=10
        )

        files = collect_all_files(tmpdir, config)
        file_names = [f.name for f in files]

        # Should exclude anything under tests directory
        assert "test.py" not in file_names
        assert "file.py" in file_names


def test_is_excluded_consistent_with_include():
    """Test that exclusion uses same matching as inclusion."""
    # Both should use pathlib.PurePath.match() for consistency
    test_cases = [
        ("src/tests/test.py", ["**/tests/**"], True),
        ("tests/test.py", ["**/tests/**"], True),
        ("src/test.py", ["**/tests/**"], False),
        ("node_modules/lib/file.js", ["node_modules/**"], True),
        ("src/node_modules/file.js", ["node_modules/**"], True),
    ]

    for path, patterns, expected in test_cases:
        result = is_excluded(Path(path), patterns)
        assert result == expected, f"Path {path} with {patterns} should be {expected}"


def test_pattern_matching_consistency():
    """Test that include and exclude use same pattern matching."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        (tmpdir / "src").mkdir()
        (tmpdir / "src" / "main.py").write_text("code")
        (tmpdir / "tests").mkdir()
        (tmpdir / "tests" / "test.py").write_text("code")

        # Include all .py, exclude tests - should use same matching
        config = Config(
            include=["**/*.py"],
            exclude=["tests/**"],
            batch_size=10
        )

        files = collect_all_files(tmpdir, config)
        file_names = [f.name for f in files]

        assert "main.py" in file_names
        assert "test.py" not in file_names
```

**Step 2: Run test to verify current implementation is inconsistent**

Run: `pytest tests/test_collector_patterns.py -v`
Expected: Some tests may fail due to fnmatch/PurePath inconsistency

**Step 3: Refactor to use pathlib.PurePath.match() consistently**

Modify `src/claude_lint/collector.py`:

Remove fnmatch import:
```python
import hashlib
from pathlib import Path, PurePath
# REMOVED: import fnmatch

from claude_lint.config import Config
```

Replace `collect_all_files()`:
```python
def collect_all_files(root_path: Path, config: Config) -> list[Path]:
    """Collect all files matching patterns.

    Args:
        root_path: Root directory to search from
        config: Configuration with include/exclude patterns

    Returns:
        List of matching file paths
    """
    all_files = []

    for pattern in config.include:
        # Use rglob for ** patterns, glob otherwise
        if "**" in pattern:
            # For ** patterns, extract the file pattern
            # e.g., "**/*.py" -> "*.py", "src/**/*.py" -> walk and match
            parts = pattern.split("/")
            if pattern.startswith("**/"):
                # Simple case: **/*.ext
                glob_pattern = "/".join(parts[1:])
                matching = root_path.rglob(glob_pattern)
            else:
                # Complex case: use rglob with ** and filter
                glob_pattern = "/".join(parts)
                matching = root_path.glob(glob_pattern)
        else:
            matching = root_path.glob(pattern)

        for file_path in matching:
            if file_path.is_file():
                # Check if excluded
                relative = file_path.relative_to(root_path)
                if not is_excluded(relative, config.exclude):
                    all_files.append(file_path)

    return list(set(all_files))  # Remove duplicates
```

Replace `is_excluded()`:
```python
def is_excluded(relative_path: Path, exclude_patterns: list[str]) -> bool:
    """Check if file is excluded by patterns.

    Uses pathlib.PurePath.match() for consistent glob-style matching
    with include patterns. Supports ** for recursive matching.

    Args:
        relative_path: File path relative to root
        exclude_patterns: List of exclude patterns

    Returns:
        True if file should be excluded
    """
    path_obj = PurePath(relative_path)

    for pattern in exclude_patterns:
        # PurePath.match() handles ** correctly for nested paths
        if path_obj.match(pattern):
            return True

        # Also check if pattern starts with ** (for patterns like **/tests/**)
        if pattern.startswith("**/"):
            # Match without ** prefix for any nesting level
            if path_obj.match(pattern[3:]) or path_obj.match(pattern):
                return True

    return False
```

**Step 4: Run tests to verify consistency**

Run: `pytest tests/test_collector_patterns.py -v`
Expected: PASS (3 tests)

**Step 5: Run all existing tests**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/claude_lint/collector.py tests/test_collector_patterns.py
git commit -m "fix: unify pattern matching to use PurePath.match()

- Remove fnmatch in favor of consistent PurePath.match()
- Fix handling of complex ** patterns
- Add tests for nested patterns and consistency
- Both include and exclude now use same matching logic"
```

---

## Task 7: Make Model Configurable (Major #7)

**Priority:** Major
**Files:**
- Modify: `src/claude_lint/config.py:8-28,48-53`
- Modify: `src/claude_lint/api_client.py:6,32`
- Modify: `src/claude_lint/orchestrator.py:49`
- Modify: `tests/test_config.py`
- Modify: `tests/test_api_client.py`

**Step 1: Write test for model configuration**

Modify `tests/test_config.py`, add test:

```python
def test_config_model_default():
    """Test default model configuration."""
    config = get_default_config()
    assert config.model == "claude-sonnet-4-5-20250929"


def test_load_config_with_model():
    """Test loading config with custom model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / ".lint-claude.json"
        config_file.write_text(json.dumps({
            "model": "claude-opus-4-5-20250929"
        }))

        config = load_config(config_file)
        assert config.model == "claude-opus-4-5-20250929"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_config_model_default -v`
Expected: FAIL - Config has no model attribute

**Step 3: Add model to Config dataclass**

Modify `src/claude_lint/config.py`:

```python
@dataclass
class Config:
    """Configuration for lint-claude."""
    include: list[str]
    exclude: list[str]
    batch_size: int
    model: str = "claude-sonnet-4-5-20250929"
    api_key: Optional[str] = None
```

Update `get_default_config()`:
```python
def get_default_config() -> Config:
    """Return default configuration.

    Returns:
        Config with default values
    """
    return Config(
        include=["**/*.py", "**/*.js", "**/*.ts"],
        exclude=["node_modules/**", "dist/**", ".git/**"],
        batch_size=10,
        model="claude-sonnet-4-5-20250929",
        api_key=None
    )
```

Update `load_config()`:
```python
def load_config(config_path: Path) -> Config:
    """Load configuration from file or return defaults.

    Args:
        config_path: Path to .lint-claude.json file

    Returns:
        Config object with loaded or default values
    """
    if not config_path.exists():
        return get_default_config()

    with open(config_path) as f:
        data = json.load(f)

    defaults = get_default_config()

    return Config(
        include=data.get("include", defaults.include),
        exclude=data.get("exclude", defaults.exclude),
        batch_size=data.get("batchSize", defaults.batch_size),
        model=data.get("model", defaults.model),
        api_key=data.get("apiKey")
    )
```

**Step 4: Update API client to accept model parameter**

Modify `src/claude_lint/api_client.py`:

```python
def analyze_files(
    api_key: str,
    guidelines: str,
    prompt: str,
    model: str = "claude-sonnet-4-5-20250929"
) -> tuple[str, Any]:
    """Analyze files using Claude API with cached guidelines.

    Args:
        api_key: Anthropic API key
        guidelines: CLAUDE.md content (will be cached)
        prompt: Prompt with files to analyze
        model: Claude model to use

    Returns:
        Tuple of (response text, response object)

    Raises:
        ValueError: If guidelines or prompt are empty or invalid
        APIError: If API call fails (includes connection, rate limit, etc.)
    """
    # Input validation
    if not guidelines or not isinstance(guidelines, str) or not guidelines.strip():
        raise ValueError("guidelines must be a non-empty string")
    if not prompt or not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    client = Anthropic(api_key=api_key)

    # Use prompt caching for guidelines
    try:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": guidelines,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
    except (APIError, APIConnectionError, RateLimitError) as e:
        raise APIError(f"Claude API call failed: {e}") from e
    except (KeyboardInterrupt, SystemExit):
        raise

    # Validate response
    if not response.content:
        raise ValueError("API returned empty response content")

    return response.content[0].text, response
```

**Step 5: Update orchestrator to pass model**

Modify `src/claude_lint/orchestrator.py`:

In the API call section (around line 115):
```python
# Make API call with retry
def api_call():
    response_text, response_obj = analyze_files(
        api_key, guidelines, prompt, model=config.model
    )
    return response_text

response = retry_with_backoff(api_call)
```

**Step 6: Update API client tests**

Modify `tests/test_api_client.py`, update tests to pass model parameter where needed.

**Step 7: Run tests to verify it passes**

Run: `pytest tests/test_config.py tests/test_api_client.py -v`
Expected: PASS

**Step 8: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 9: Commit**

```bash
git add src/claude_lint/config.py src/claude_lint/api_client.py src/claude_lint/orchestrator.py tests/test_config.py tests/test_api_client.py
git commit -m "feat: make Claude model configurable

- Add model field to Config with default
- Pass model to API client from config
- Support 'model' in .lint-claude.json
- Allows using different Claude models (opus, sonnet, etc.)"
```

---

## Task 8: Fix File Reading Error Handling (Major #8)

**Priority:** Major
**Files:**
- Modify: `src/claude_lint/orchestrator.py:98-108`
- Create: `tests/test_orchestrator_file_errors.py`

**Step 1: Write test for file encoding issues**

Create `tests/test_orchestrator_file_errors.py`:

```python
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
from claude_lint.orchestrator import run_compliance_check
from claude_lint.config import Config


def test_file_with_invalid_utf8():
    """Test handling of files with invalid UTF-8."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create file with invalid UTF-8
        binary_file = tmpdir / "binary.py"
        binary_file.write_bytes(b"print('hello')\n\x80\x81\x82")

        (tmpdir / "CLAUDE.md").write_text("# Guidelines")

        config = Config(
            include=["**/*.py"],
            exclude=[],
            batch_size=10,
            api_key="test-key"
        )

        # Should skip binary file with warning (check logs)
        with patch("claude_lint.orchestrator.analyze_files") as mock_api:
            mock_api.return_value = ('{"results": []}', Mock())

            # File should be skipped, not crash
            results = run_compliance_check(tmpdir, config, mode="full")

            # Should return empty or handle gracefully
            assert isinstance(results, list)


def test_file_reading_fallback_encoding():
    """Test that files are attempted with fallback encoding."""
    # This tests the new behavior where we try UTF-8, then latin-1
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create file with latin-1 encoding
        latin_file = tmpdir / "latin.py"
        content = "# Café"
        latin_file.write_bytes(content.encode('latin-1'))

        (tmpdir / "CLAUDE.md").write_text("# Guidelines")

        config = Config(
            include=["**/*.py"],
            exclude=[],
            batch_size=10,
            api_key="test-key"
        )

        with patch("claude_lint.orchestrator.analyze_files") as mock_api:
            mock_api.return_value = (
                '{"results": [{"file": "latin.py", "violations": []}]}',
                Mock()
            )

            results = run_compliance_check(tmpdir, config, mode="full")

            # Should successfully read with fallback
            assert mock_api.called
```

**Step 2: Run test to verify current behavior**

Run: `pytest tests/test_orchestrator_file_errors.py -v`
Expected: May pass or fail depending on current behavior

**Step 3: Improve file reading with encoding fallback**

Modify `src/claude_lint/orchestrator.py`:

Add import for logger:
```python
from claude_lint.logging_config import get_logger

logger = get_logger(__name__)
```

Replace file reading section (lines 93-108):
```python
        # Read file contents
        file_contents = {}
        for file_path in batch:
            rel_path = file_path.relative_to(project_root)
            try:
                # Try UTF-8 first
                content = file_path.read_text(encoding='utf-8')
                file_contents[str(rel_path)] = content
            except UnicodeDecodeError:
                # Fall back to latin-1 which accepts all byte sequences
                try:
                    logger.warning(
                        f"File {rel_path} is not valid UTF-8, trying latin-1"
                    )
                    content = file_path.read_text(encoding='latin-1')
                    file_contents[str(rel_path)] = content
                except Exception as e:
                    logger.warning(
                        f"Unable to decode file {rel_path}, skipping: {e}"
                    )
                    continue
            except FileNotFoundError:
                logger.warning(f"File not found, skipping: {rel_path}")
                continue
            except Exception as e:
                logger.warning(f"Error reading file {rel_path}, skipping: {e}")
                continue
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_orchestrator_file_errors.py -v`
Expected: PASS (2 tests)

**Step 5: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/claude_lint/orchestrator.py tests/test_orchestrator_file_errors.py
git commit -m "fix: improve file encoding handling

- Try UTF-8 first, fall back to latin-1
- Warn on encoding issues instead of silent replacement
- Skip files that cannot be decoded
- Better error messages for debugging"
```

---

## Task 9: Add Input Validation (Major #9)

**Priority:** Major
**Files:**
- Create: `src/claude_lint/validation.py`
- Modify: `src/claude_lint/orchestrator.py:31-51`
- Modify: `src/claude_lint/cli.py:18-41`
- Create: `tests/test_validation.py`

**Step 1: Write validation tests**

Create `tests/test_validation.py`:

```python
import tempfile
from pathlib import Path
import pytest
from claude_lint.validation import (
    validate_project_root,
    validate_mode,
    validate_batch_size,
    validate_api_key
)


def test_validate_project_root_exists():
    """Test validation of existing directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        validate_project_root(Path(tmpdir))  # Should not raise


def test_validate_project_root_not_exists():
    """Test validation of non-existent directory."""
    with pytest.raises(ValueError, match="does not exist"):
        validate_project_root(Path("/nonexistent/path"))


def test_validate_project_root_is_file():
    """Test validation fails for file instead of directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "file.txt"
        file_path.write_text("content")

        with pytest.raises(ValueError, match="not a directory"):
            validate_project_root(file_path)


def test_validate_mode_valid():
    """Test validation of valid modes."""
    for mode in ["full", "diff", "working", "staged"]:
        validate_mode(mode)  # Should not raise


def test_validate_mode_invalid():
    """Test validation of invalid mode."""
    with pytest.raises(ValueError, match="Invalid mode"):
        validate_mode("invalid")


def test_validate_batch_size_valid():
    """Test validation of valid batch sizes."""
    validate_batch_size(1)
    validate_batch_size(10)
    validate_batch_size(100)


def test_validate_batch_size_zero():
    """Test validation fails for zero."""
    with pytest.raises(ValueError, match="must be positive"):
        validate_batch_size(0)


def test_validate_batch_size_negative():
    """Test validation fails for negative."""
    with pytest.raises(ValueError, match="must be positive"):
        validate_batch_size(-5)


def test_validate_api_key_valid():
    """Test validation of valid API key."""
    validate_api_key("sk-ant-1234567890")  # Should not raise


def test_validate_api_key_empty():
    """Test validation fails for empty key."""
    with pytest.raises(ValueError, match="API key is required"):
        validate_api_key("")


def test_validate_api_key_none():
    """Test validation fails for None."""
    with pytest.raises(ValueError, match="API key is required"):
        validate_api_key(None)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_validation.py -v`
Expected: FAIL - module does not exist

**Step 3: Create validation module**

Create `src/claude_lint/validation.py`:

```python
"""Input validation functions."""
from pathlib import Path
from typing import Optional


VALID_MODES = {"full", "diff", "working", "staged"}


def validate_project_root(project_root: Path) -> None:
    """Validate project root directory exists.

    Args:
        project_root: Path to validate

    Raises:
        ValueError: If path does not exist or is not a directory
    """
    if not project_root.exists():
        raise ValueError(f"Project root does not exist: {project_root}")

    if not project_root.is_dir():
        raise ValueError(f"Project root is not a directory: {project_root}")


def validate_mode(mode: str) -> None:
    """Validate check mode.

    Args:
        mode: Mode to validate

    Raises:
        ValueError: If mode is not valid
    """
    if mode not in VALID_MODES:
        raise ValueError(
            f"Invalid mode: {mode}. Must be one of: {', '.join(sorted(VALID_MODES))}"
        )


def validate_batch_size(batch_size: int) -> None:
    """Validate batch size.

    Args:
        batch_size: Batch size to validate

    Raises:
        ValueError: If batch size is not positive
    """
    if batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got: {batch_size}")


def validate_api_key(api_key: Optional[str]) -> None:
    """Validate API key.

    Args:
        api_key: API key to validate

    Raises:
        ValueError: If API key is missing or empty
    """
    if not api_key:
        raise ValueError(
            "API key is required. Set ANTHROPIC_API_KEY environment variable "
            "or provide in config file."
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_validation.py -v`
Expected: PASS (11 tests)

**Step 5: Add validation to orchestrator**

Modify `src/claude_lint/orchestrator.py`:

Add import:
```python
from claude_lint.validation import (
    validate_project_root,
    validate_mode,
    validate_batch_size,
    validate_api_key
)
```

Add validation at start of `run_compliance_check()`:
```python
def run_compliance_check(
    project_root: Path,
    config: Config,
    mode: str = "full",
    base_branch: Optional[str] = None
) -> list[dict[str, Any]]:
    """Run compliance check.

    Args:
        project_root: Project root directory
        config: Configuration
        mode: Check mode - 'full', 'diff', 'working', 'staged'
        base_branch: Base branch for diff mode

    Returns:
        List of results for all checked files

    Raises:
        ValueError: If inputs are invalid
    """
    # Input validation
    validate_project_root(project_root)
    validate_mode(mode)
    validate_batch_size(config.batch_size)

    # Get API key
    api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
    validate_api_key(api_key)
```

**Step 6: Add validation to CLI**

Modify `src/claude_lint/cli.py`:

Update error messages to use click.echo with err=True:
```python
    # Determine mode
    mode_count = sum([full, bool(diff), working, staged])
    if mode_count == 0:
        click.echo("Error: Must specify one mode: --full, --diff, --working, or --staged", err=True)
        sys.exit(2)
    elif mode_count > 1:
        click.echo("Error: Only one mode can be specified", err=True)
        sys.exit(2)
```

**Step 7: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 8: Commit**

```bash
git add src/claude_lint/validation.py tests/test_validation.py src/claude_lint/orchestrator.py src/claude_lint/cli.py
git commit -m "feat: add comprehensive input validation

- Validate project root exists and is directory
- Validate mode is in allowed set
- Validate batch size is positive
- Validate API key is present
- Better error messages to stderr"
```

---

## Task 10: Reuse Anthropic Client (Major #10)

**Priority:** Major
**Files:**
- Modify: `src/claude_lint/api_client.py:6,27`
- Modify: `src/claude_lint/orchestrator.py:49,114-116`
- Modify: `tests/test_api_client.py`

**Step 1: Write test for client reuse**

Modify `tests/test_api_client.py`, add test:

```python
from unittest.mock import patch, MagicMock, call


def test_client_reuse():
    """Test that same client instance is reused."""
    with patch('anthropic.Anthropic') as mock_anthropic_class:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="response")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        client = create_client("test-key")

        # Make multiple calls with same client
        analyze_files_with_client(client, "guidelines", "prompt1")
        analyze_files_with_client(client, "guidelines", "prompt2")

        # Anthropic() should only be called once
        assert mock_anthropic_class.call_count == 1

        # But messages.create should be called twice
        assert mock_client.messages.create.call_count == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_client.py::test_client_reuse -v`
Expected: FAIL - functions don't exist

**Step 3: Refactor API client for reuse**

Modify `src/claude_lint/api_client.py`:

```python
"""Claude API client with prompt caching support."""
from typing import Optional, Any
from anthropic import Anthropic, APIError, APIConnectionError, RateLimitError


def create_client(api_key: str) -> Anthropic:
    """Create Anthropic client.

    Args:
        api_key: Anthropic API key

    Returns:
        Anthropic client instance
    """
    return Anthropic(api_key=api_key)


def analyze_files_with_client(
    client: Anthropic,
    guidelines: str,
    prompt: str,
    model: str = "claude-sonnet-4-5-20250929"
) -> tuple[str, Any]:
    """Analyze files using existing Claude API client.

    Args:
        client: Anthropic client instance
        guidelines: CLAUDE.md content (will be cached)
        prompt: Prompt with files to analyze
        model: Claude model to use

    Returns:
        Tuple of (response text, response object)

    Raises:
        ValueError: If guidelines or prompt are empty or invalid
        APIError: If API call fails (includes connection, rate limit, etc.)
    """
    # Input validation
    if not guidelines or not isinstance(guidelines, str) or not guidelines.strip():
        raise ValueError("guidelines must be a non-empty string")
    if not prompt or not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    # Use prompt caching for guidelines
    try:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": guidelines,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
    except (APIError, APIConnectionError, RateLimitError) as e:
        raise APIError(f"Claude API call failed: {e}") from e
    except (KeyboardInterrupt, SystemExit):
        raise

    # Validate response
    if not response.content:
        raise ValueError("API returned empty response content")

    return response.content[0].text, response


def analyze_files(
    api_key: str,
    guidelines: str,
    prompt: str,
    model: str = "claude-sonnet-4-5-20250929"
) -> tuple[str, Any]:
    """Analyze files using Claude API with cached guidelines.

    Convenience wrapper that creates client and makes call.
    For multiple calls, use create_client() and analyze_files_with_client().

    Args:
        api_key: Anthropic API key
        guidelines: CLAUDE.md content (will be cached)
        prompt: Prompt with files to analyze
        model: Claude model to use

    Returns:
        Tuple of (response text, response object)

    Raises:
        ValueError: If guidelines or prompt are empty or invalid
        APIError: If API call fails (includes connection, rate limit, etc.)
    """
    client = create_client(api_key)
    return analyze_files_with_client(client, guidelines, prompt, model)
```

**Step 4: Update orchestrator to reuse client**

Modify `src/claude_lint/orchestrator.py`:

Add import:
```python
from claude_lint.api_client import create_client, analyze_files_with_client
```

Create client once before batch loop:
```python
    # Create API client once for all batches
    client = create_client(api_key)

    # Process batches
    all_results = list(progress_state.results)  # Start with resumed results

    for batch_idx in get_remaining_batch_indices(progress_state):
```

Update API call to use client:
```python
        # Make API call with retry
        def api_call():
            response_text, response_obj = analyze_files_with_client(
                client, guidelines, prompt, model=config.model
            )
            return response_text

        response = retry_with_backoff(api_call)
```

**Step 5: Run tests**

Run: `pytest tests/test_api_client.py -v`
Expected: PASS

**Step 6: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 7: Commit**

```bash
git add src/claude_lint/api_client.py src/claude_lint/orchestrator.py tests/test_api_client.py
git commit -m "perf: reuse Anthropic client across batches

- Create client once, reuse for all API calls
- Add create_client() and analyze_files_with_client()
- Keep analyze_files() as convenience wrapper
- Reduces overhead and improves performance"
```

---

## Task 11: Add Version and CLI Flags (Major #11)

**Priority:** Major
**Files:**
- Modify: `pyproject.toml:2`
- Modify: `src/claude_lint/cli.py:11-18`
- Create: `src/claude_lint/__version__.py`
- Modify: `tests/test_cli.py`

**Step 1: Write tests for new CLI flags**

Modify `tests/test_cli.py` (or create if doesn't exist):

```python
from click.testing import CliRunner
from claude_lint.cli import main


def test_version_flag():
    """Test --version flag."""
    runner = CliRunner()
    result = runner.invoke(main, ['--version'])

    assert result.exit_code == 0
    assert 'lint-claude' in result.output
    assert '0.1.0' in result.output


def test_help_flag():
    """Test --help flag."""
    runner = CliRunner()
    result = runner.invoke(main, ['--help'])

    assert result.exit_code == 0
    assert '--version' in result.output
    assert '--verbose' in result.output
    assert '--quiet' in result.output
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py::test_version_flag -v`
Expected: FAIL - no version flag

**Step 3: Create version module**

Create `src/claude_lint/__version__.py`:

```python
"""Version information for lint-claude."""

__version__ = "0.1.0"
```

**Step 4: Add version flag to CLI**

Modify `src/claude_lint/cli.py`:

Add import:
```python
from claude_lint.__version__ import __version__
```

Add version option:
```python
@click.command()
@click.version_option(version=__version__, prog_name='lint-claude')
@click.option("--full", is_flag=True, help="Full project scan")
```

**Step 5: Update pyproject.toml to use dynamic version**

Modify `pyproject.toml`:

```toml
[project]
name = "lint-claude"
version = "0.1.0"
description = "CLAUDE.md compliance checker using Claude API"
```

**Step 6: Run tests**

Run: `pytest tests/test_cli.py -v`
Expected: PASS

**Step 7: Test CLI manually**

Run: `uv run lint-claude --version`
Expected: Output showing version

Run: `uv run lint-claude --help`
Expected: Help output showing all flags

**Step 8: Commit**

```bash
git add src/claude_lint/__version__.py src/claude_lint/cli.py pyproject.toml tests/test_cli.py
git commit -m "feat: add --version flag and improve CLI help

- Add __version__ module for version tracking
- Add --version flag to CLI
- All flags now visible in --help
- Follows click best practices"
```

---

## Task 12: Add File Size Limits (Major #12)

**Priority:** Major
**Files:**
- Modify: `src/claude_lint/config.py:8-28`
- Modify: `src/claude_lint/orchestrator.py:93-125`
- Create: `tests/test_file_size_limits.py`

**Step 1: Write test for file size limits**

Create `tests/test_file_size_limits.py`:

```python
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
from claude_lint.orchestrator import run_compliance_check
from claude_lint.config import Config


def test_skip_large_files():
    """Test that files over size limit are skipped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create small file
        small_file = tmpdir / "small.py"
        small_file.write_text("print('hello')")

        # Create large file (over 1MB)
        large_file = tmpdir / "large.py"
        large_file.write_text("x" * (2 * 1024 * 1024))  # 2MB

        (tmpdir / "CLAUDE.md").write_text("# Guidelines")

        config = Config(
            include=["**/*.py"],
            exclude=[],
            batch_size=10,
            max_file_size_mb=1.0,  # 1MB limit
            api_key="test-key"
        )

        with patch("claude_lint.orchestrator.analyze_files_with_client") as mock_api:
            with patch("claude_lint.orchestrator.create_client") as mock_create:
                mock_create.return_value = Mock()
                mock_api.return_value = (
                    '{"results": [{"file": "small.py", "violations": []}]}',
                    Mock()
                )

                results = run_compliance_check(tmpdir, config, mode="full")

                # Only small file should be analyzed
                call_args = mock_api.call_args
                prompt = call_args[0][2]  # Third argument is prompt

                assert "small.py" in prompt
                assert "large.py" not in prompt


def test_default_file_size_limit():
    """Test default file size limit."""
    from claude_lint.config import get_default_config
    config = get_default_config()
    assert config.max_file_size_mb == 1.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_file_size_limits.py -v`
Expected: FAIL - Config has no max_file_size_mb

**Step 3: Add file size limit to config**

Modify `src/claude_lint/config.py`:

```python
@dataclass
class Config:
    """Configuration for lint-claude."""
    include: list[str]
    exclude: list[str]
    batch_size: int
    model: str = "claude-sonnet-4-5-20250929"
    max_file_size_mb: float = 1.0
    api_key: Optional[str] = None
```

Update `get_default_config()`:
```python
def get_default_config() -> Config:
    """Return default configuration.

    Returns:
        Config with default values
    """
    return Config(
        include=["**/*.py", "**/*.js", "**/*.ts"],
        exclude=["node_modules/**", "dist/**", ".git/**"],
        batch_size=10,
        model="claude-sonnet-4-5-20250929",
        max_file_size_mb=1.0,
        api_key=None
    )
```

Update `load_config()`:
```python
    return Config(
        include=data.get("include", defaults.include),
        exclude=data.get("exclude", defaults.exclude),
        batch_size=data.get("batchSize", defaults.batch_size),
        model=data.get("model", defaults.model),
        max_file_size_mb=data.get("maxFileSizeMb", defaults.max_file_size_mb),
        api_key=data.get("apiKey")
    )
```

**Step 4: Add file size check to orchestrator**

Modify `src/claude_lint/orchestrator.py`:

Update file reading loop:
```python
        # Read file contents
        file_contents = {}
        max_size_bytes = int(config.max_file_size_mb * 1024 * 1024)

        for file_path in batch:
            rel_path = file_path.relative_to(project_root)

            # Check file size
            try:
                file_size = file_path.stat().st_size
                if file_size > max_size_bytes:
                    logger.warning(
                        f"File {rel_path} exceeds size limit "
                        f"({file_size / 1024 / 1024:.2f}MB > "
                        f"{config.max_file_size_mb}MB), skipping"
                    )
                    continue
            except OSError as e:
                logger.warning(f"Cannot stat file {rel_path}, skipping: {e}")
                continue

            try:
                # Try UTF-8 first
                content = file_path.read_text(encoding='utf-8')
                file_contents[str(rel_path)] = content
            except UnicodeDecodeError:
                # Fall back to latin-1
                try:
                    logger.warning(
                        f"File {rel_path} is not valid UTF-8, trying latin-1"
                    )
                    content = file_path.read_text(encoding='latin-1')
                    file_contents[str(rel_path)] = content
                except Exception as e:
                    logger.warning(
                        f"Unable to decode file {rel_path}, skipping: {e}"
                    )
                    continue
            except FileNotFoundError:
                logger.warning(f"File not found, skipping: {rel_path}")
                continue
            except Exception as e:
                logger.warning(f"Error reading file {rel_path}, skipping: {e}")
                continue
```

**Step 5: Run tests**

Run: `pytest tests/test_file_size_limits.py -v`
Expected: PASS (2 tests)

**Step 6: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 7: Commit**

```bash
git add src/claude_lint/config.py src/claude_lint/orchestrator.py tests/test_file_size_limits.py
git commit -m "feat: add file size limits to prevent API overload

- Add max_file_size_mb config option (default 1MB)
- Skip files over limit with warning
- Configurable via maxFileSizeMb in config
- Prevents sending huge files to API"
```

---

## Task 13: Add TypedDict for Violations (Minor #13)

**Priority:** Minor
**Files:**
- Create: `src/claude_lint/types.py`
- Modify: `src/claude_lint/processor.py:1,73`
- Modify: `src/claude_lint/cache.py:1,13`
- Modify: `src/claude_lint/reporter.py:1,6,47,73,86`

**Step 1: Create types module**

Create `src/claude_lint/types.py`:

```python
"""Type definitions for lint-claude."""
from typing import TypedDict, Optional


class Violation(TypedDict):
    """Single violation structure."""
    type: str
    message: str
    line: Optional[int]


class FileResult(TypedDict):
    """Result for a single file."""
    file: str
    violations: list[Violation]
```

**Step 2: Update processor to use TypedDict**

Modify `src/claude_lint/processor.py`:

Add import:
```python
from claude_lint.types import FileResult
```

Update return type:
```python
def parse_response(response: str) -> list[FileResult]:
    """Parse Claude API response into results.

    Args:
        response: API response text

    Returns:
        List of file results with violations
    """
```

**Step 3: Update cache to use TypedDict**

Modify `src/claude_lint/cache.py`:

Add import:
```python
from claude_lint.types import Violation
```

Update CacheEntry:
```python
@dataclass
class CacheEntry:
    """Cache entry for a single file."""
    file_hash: str
    claude_md_hash: str
    violations: list[Violation]
    timestamp: int
```

**Step 4: Update reporter to use TypedDict**

Modify `src/claude_lint/reporter.py`:

Add import:
```python
from claude_lint.types import FileResult
```

Update function signatures:
```python
def format_detailed_report(results: list[FileResult]) -> str:
    """Format results as detailed human-readable report.

    Args:
        results: List of file results with violations

    Returns:
        Formatted report string
    """


def format_json_report(results: list[FileResult]) -> str:
    """Format results as JSON.

    Args:
        results: List of file results with violations

    Returns:
        JSON string
    """


def get_exit_code(results: list[FileResult]) -> int:
    """Get exit code based on results.

    Args:
        results: List of file results

    Returns:
        0 if no violations, 1 if violations found
    """


def get_summary(results: list[FileResult]) -> dict[str, int]:
    """Get summary statistics.

    Args:
        results: List of file results

    Returns:
        Dict with summary counts
    """
```

**Step 5: Run type checker**

Run: `uv run mypy src/claude_lint/ --strict`
Expected: Should pass or show only minor issues

**Step 6: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 7: Commit**

```bash
git add src/claude_lint/types.py src/claude_lint/processor.py src/claude_lint/cache.py src/claude_lint/reporter.py
git commit -m "refactor: add TypedDict for violation structures

- Define Violation and FileResult types
- Improve type safety throughout codebase
- Better IDE autocomplete
- Clearer API contracts"
```

---

## Task 14: Add Module Exports (Minor #14)

**Priority:** Minor
**Files:**
- Modify: `src/claude_lint/__init__.py`
- Create if doesn't exist

**Step 1: Create __init__ with exports**

Create/modify `src/claude_lint/__init__.py`:

```python
"""Claude-lint: CLAUDE.md compliance checker."""

from claude_lint.__version__ import __version__
from claude_lint.config import Config, load_config, get_default_config
from claude_lint.orchestrator import run_compliance_check
from claude_lint.types import Violation, FileResult

__all__ = [
    "__version__",
    "Config",
    "load_config",
    "get_default_config",
    "run_compliance_check",
    "Violation",
    "FileResult",
]
```

**Step 2: Test imports**

Run: `uv run python -c "from claude_lint import __version__, Config, run_compliance_check; print(__version__)"`
Expected: Prints version number

**Step 3: Commit**

```bash
git add src/claude_lint/__init__.py
git commit -m "refactor: add __all__ exports for public API

- Define public API surface
- Enable 'from claude_lint import X'
- Better for library usage"
```

---

## Task 15: Normalize Config Keys to Snake Case (Minor #15)

**Priority:** Minor
**Files:**
- Modify: `src/claude_lint/config.py:48-57`
- Modify: `README.md` (example config)
- Create: `tests/test_config_snake_case.py`

**Step 1: Write test for snake_case config**

Create `tests/test_config_snake_case.py`:

```python
import tempfile
import json
from pathlib import Path
from claude_lint.config import load_config


def test_load_config_snake_case():
    """Test loading config with snake_case keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / ".lint-claude.json"
        config_file.write_text(json.dumps({
            "batch_size": 20,
            "max_file_size_mb": 2.0,
            "api_key": "test-key"
        }))

        config = load_config(config_file)

        assert config.batch_size == 20
        assert config.max_file_size_mb == 2.0
        assert config.api_key == "test-key"


def test_load_config_backwards_compat_camel_case():
    """Test that camelCase is still supported for backwards compatibility."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / ".lint-claude.json"
        config_file.write_text(json.dumps({
            "batchSize": 20,
            "maxFileSizeMb": 2.0,
            "apiKey": "test-key"
        }))

        config = load_config(config_file)

        assert config.batch_size == 20
        assert config.max_file_size_mb == 2.0
        assert config.api_key == "test-key"
```

**Step 2: Run test to verify snake_case fails**

Run: `pytest tests/test_config_snake_case.py::test_load_config_snake_case -v`
Expected: FAIL - snake_case not supported yet

**Step 3: Update load_config to support both**

Modify `src/claude_lint/config.py`:

```python
def load_config(config_path: Path) -> Config:
    """Load configuration from file or return defaults.

    Supports both snake_case (preferred) and camelCase (backwards compat) keys.

    Args:
        config_path: Path to .lint-claude.json file

    Returns:
        Config object with loaded or default values
    """
    if not config_path.exists():
        return get_default_config()

    with open(config_path) as f:
        data = json.load(f)

    defaults = get_default_config()

    return Config(
        include=data.get("include", defaults.include),
        exclude=data.get("exclude", defaults.exclude),
        # Support both snake_case and camelCase
        batch_size=data.get("batch_size", data.get("batchSize", defaults.batch_size)),
        model=data.get("model", defaults.model),
        max_file_size_mb=data.get(
            "max_file_size_mb",
            data.get("maxFileSizeMb", defaults.max_file_size_mb)
        ),
        api_key=data.get("api_key", data.get("apiKey"))
    )
```

**Step 4: Run tests**

Run: `pytest tests/test_config_snake_case.py -v`
Expected: PASS (2 tests)

**Step 5: Update README with snake_case example**

Modify `README.md`:

Find the config example and update to:
```json
{
  "include": ["**/*.py", "**/*.js"],
  "exclude": ["tests/**", "node_modules/**"],
  "batch_size": 15,
  "max_file_size_mb": 1.0,
  "model": "claude-sonnet-4-5-20250929",
  "api_key": "sk-ant-..."
}
```

**Step 6: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 7: Commit**

```bash
git add src/claude_lint/config.py tests/test_config_snake_case.py README.md
git commit -m "refactor: prefer snake_case in config files

- Support snake_case config keys (Python convention)
- Maintain backwards compatibility with camelCase
- Update README with snake_case examples
- Tests for both formats"
```

---

## Task 16: Update Documentation (Minor)

**Priority:** Minor
**Files:**
- Modify: `README.md`
- Create: `docs/TROUBLESHOOTING.md`
- Create: `docs/ARCHITECTURE.md`

**Step 1: Update README with new features**

Modify `README.md`:

Add to Features section:
```markdown
## Features

- Smart caching based on file and CLAUDE.md hashes
- Multiple scan modes: full project, git diff, working directory, staged files
- Batch processing for large projects
- Progress tracking with resume capability
- Prompt caching for cost efficiency
- Configurable Claude model (Sonnet, Opus, etc.)
- File size limits to prevent API overload
- Atomic file writes prevent cache corruption
- Comprehensive logging with --verbose and --quiet flags
- Input validation for robust operation
```

Add Configuration section:
```markdown
## Configuration

Create `.lint-claude.json` in your project root:

```json
{
  "include": ["**/*.py", "**/*.js", "**/*.ts"],
  "exclude": ["node_modules/**", "dist/**", ".git/**"],
  "batch_size": 10,
  "max_file_size_mb": 1.0,
  "model": "claude-sonnet-4-5-20250929",
  "api_key": "sk-ant-..."
}
```

All keys are optional. Defaults shown above.
```

**Step 2: Create troubleshooting guide**

Create `docs/TROUBLESHOOTING.md`:

```markdown
# Troubleshooting Guide

## Common Issues

### API Key Not Found

**Error:** `ValueError: API key is required`

**Solution:** Set the `ANTHROPIC_API_KEY` environment variable or add `api_key` to `.lint-claude.json`:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Git Command Timeout

**Error:** `RuntimeError: Git command timed out after 30s`

**Solution:**
- Check if git is hanging (network issues with remote)
- Large repositories may need optimization
- Check git config for hanging processes

### Files Skipped Due to Encoding

**Warning:** `File X is not valid UTF-8, trying latin-1`

**Info:** Claude-lint tries UTF-8 first, falls back to latin-1. If both fail, the file is skipped.

**Solution:**
- Check file encoding: `file -I <filename>`
- Convert to UTF-8 if possible
- Binary files will be skipped (expected)

### File Size Limit

**Warning:** `File X exceeds size limit (2.5MB > 1.0MB), skipping`

**Solution:** Increase `max_file_size_mb` in config:

```json
{
  "max_file_size_mb": 5.0
}
```

### Cache Corruption

**Error:** Issues loading cache file

**Solution:** Delete cache and progress files:

```bash
rm .lint-claude-cache.json .lint-claude-progress.json
```

Claude-lint will rebuild the cache on next run.

### Pattern Matching Not Working

**Issue:** Files not being included/excluded as expected

**Solution:**
- Use `**` for recursive patterns: `**/*.py`
- Patterns are matched against relative paths from project root
- Test with `--verbose` to see which files are collected
- Both include and exclude use same glob pattern matching

### Git Not Found

**Error:** `FileNotFoundError: git`

**Solution:** Install git and ensure it's in your PATH.

## Debug Mode

Run with `--verbose` for detailed logging:

```bash
lint-claude --full --verbose
```

This shows:
- Which files are being collected
- Which files are skipped and why
- Cache hit/miss information
- API call details

## Getting Help

1. Check this troubleshooting guide
2. Review [Architecture docs](ARCHITECTURE.md)
3. Run with `--verbose` for detailed output
4. File an issue at https://github.com/vtemian/lint-claude/issues
```

**Step 3: Create architecture documentation**

Create `docs/ARCHITECTURE.md`:

```markdown
# Architecture

## Overview

Claude-lint is designed as a functional Python CLI that checks code compliance with CLAUDE.md guidelines using the Claude API.

## Design Principles

1. **Functional Programming**: No classes (except pure dataclasses), only functions
2. **Separation of Concerns**: Each module has single responsibility
3. **Testability**: Pure functions, dependency injection
4. **Robustness**: Comprehensive validation, atomic operations
5. **Performance**: Caching, batching, client reuse

## Module Structure

```
src/claude_lint/
├── __init__.py          # Public API exports
├── __version__.py       # Version tracking
├── types.py             # TypedDict definitions
├── cli.py               # Click CLI interface
├── config.py            # Configuration loading
├── validation.py        # Input validation
├── orchestrator.py      # Main coordination logic
├── collector.py         # File collection & filtering
├── processor.py         # Batch creation & XML prompts
├── api_client.py        # Claude API integration
├── cache.py             # Result caching
├── progress.py          # Progress tracking
├── git_utils.py         # Git integration
├── guidelines.py        # CLAUDE.md reading
├── retry.py             # Exponential backoff
├── reporter.py          # Output formatting
├── logging_config.py    # Logging setup
└── file_utils.py        # Atomic file operations
```

## Data Flow

1. **CLI Entry** (`cli.py`)
   - Parse command line arguments
   - Setup logging
   - Call orchestrator

2. **Validation** (`validation.py`)
   - Validate project root, mode, batch size, API key
   - Early failure on invalid inputs

3. **File Collection** (`collector.py`)
   - Collect files based on mode (full/diff/working/staged)
   - Apply include/exclude patterns
   - Filter by file size limits

4. **Caching** (`cache.py`)
   - Load existing cache
   - Filter out unchanged files (file hash + CLAUDE.md hash)
   - Return cached results for unchanged files

5. **Batch Processing** (`processor.py`)
   - Create batches of files
   - Build XML prompts for each batch
   - Escape file paths and contents

6. **API Calls** (`api_client.py`)
   - Create client once, reuse across batches
   - Use prompt caching for CLAUDE.md
   - Retry on transient failures

7. **Progress Tracking** (`progress.py`)
   - Save progress after each batch
   - Enable resume on interruption
   - Clean up on completion

8. **Reporting** (`reporter.py`)
   - Format results (detailed or JSON)
   - Calculate exit code
   - Generate summary statistics

## Key Design Decisions

### Why No Classes?

User requirement: "NEVER USE CLASSES". This forces:
- Functional decomposition
- Pure functions
- Explicit dependencies
- Testability without mocking complex state

### Why Atomic Writes?

Cache and progress files must never be partially written. Atomic writes ensure:
- Write to `.tmp` file
- Atomically replace original
- No corruption on crashes

### Why Client Reuse?

Creating Anthropic client has overhead. Reusing across batches:
- Reduces connection overhead
- Maintains connection pooling
- Improves performance

### Why Pattern Matching Unification?

Originally used fnmatch for excludes, PurePath.match() for includes. This caused:
- Different matching semantics
- Confusing behavior
- Bugs with `**` patterns

Unified to `PurePath.match()` for consistency.

### Why Fallback Encoding?

Files may not be UTF-8. Strategy:
1. Try UTF-8 (most common)
2. Fall back to latin-1 (accepts all bytes)
3. Skip if both fail

This handles most files while warning on issues.

## Performance Characteristics

- **File Collection**: O(n) where n = files in project
- **Cache Lookup**: O(n) where n = files to check
- **Batching**: O(n/b) API calls where b = batch size
- **API Calls**: ~1-3 seconds per batch (depends on model)

For 100 files with batch size 10:
- ~10 API calls
- ~10-30 seconds total (with caching)
- Subsequent runs: <1 second (cache hits)

## Extension Points

### Adding New Modes

1. Add mode to `VALID_MODES` in `validation.py`
2. Add git function to `git_utils.py` if needed
3. Add case to `collect_files_for_mode()` in `orchestrator.py`
4. Add CLI option in `cli.py`

### Adding New Config Options

1. Add field to `Config` dataclass in `config.py`
2. Update `get_default_config()` and `load_config()`
3. Use config value in relevant module
4. Add tests

### Custom Output Formats

1. Add format function to `reporter.py`
2. Add CLI option in `cli.py`
3. Call format function based on option

## Testing Strategy

- **Unit Tests**: Each module tested in isolation
- **Integration Tests**: Orchestrator with mocked API
- **Property Tests**: Pattern matching, encoding handling
- **Error Tests**: Timeouts, exceptions, edge cases

## Security Considerations

- Input validation prevents path traversal
- Subprocess calls use list form (no shell injection)
- Timeouts prevent hanging
- File size limits prevent DoS
- API key not logged or exposed
```

**Step 4: Commit**

```bash
git add README.md docs/TROUBLESHOOTING.md docs/ARCHITECTURE.md
git commit -m "docs: comprehensive documentation updates

- Update README with all new features
- Add troubleshooting guide
- Add architecture documentation
- Document design decisions and extension points"
```

---

## Task 17: Run Full Test Suite and Verify

**Priority:** Critical
**Files:**
- All test files

**Step 1: Run all tests with coverage**

Run: `pytest tests/ -v --cov=src/claude_lint --cov-report=term-missing`
Expected: All tests pass, high coverage (>85%)

**Step 2: Run type checking**

Run: `uv run mypy src/claude_lint/ --ignore-missing-imports`
Expected: No type errors

**Step 3: Run linter**

Run: `uv run ruff check src/claude_lint/`
Expected: No linting errors

**Step 4: Test CLI commands manually**

Run: `uv run lint-claude --version`
Expected: Version output

Run: `uv run lint-claude --help`
Expected: Help with all options

**Step 5: Create summary of changes**

Create `docs/CHANGELOG.md`:

```markdown
# Changelog

## [0.2.0] - 2025-11-11

### Critical Fixes
- Added 30-second timeout to all git subprocess calls to prevent hanging
- Replaced print() statements with proper logging framework
- Implemented atomic file writes for cache and progress files
- Fixed exception handling to not catch KeyboardInterrupt/SystemExit
- Removed unused gitpython dependency

### Major Improvements
- Unified pattern matching to use PurePath.match() consistently
- Made Claude model configurable via config file
- Improved file encoding handling with UTF-8 and latin-1 fallback
- Added comprehensive input validation for all parameters
- Implemented Anthropic client reuse for better performance
- Added --verbose, --quiet, and --version CLI flags
- Added file size limits (configurable, default 1MB)

### Minor Improvements
- Added TypedDict for violation structures
- Added __all__ exports for public API
- Normalized config keys to snake_case (with camelCase backwards compat)
- Comprehensive documentation (architecture, troubleshooting)
- Improved error messages to stderr
- Better type safety throughout

### Documentation
- Added ARCHITECTURE.md with design decisions
- Added TROUBLESHOOTING.md for common issues
- Updated README with all new features
- Added inline documentation improvements

## [0.1.0] - 2025-11-11

Initial release with core functionality.
```

**Step 6: Commit changelog**

```bash
git add docs/CHANGELOG.md
git commit -m "docs: add changelog for v0.2.0"
```

---

## Task 18: Tag Release

**Priority:** Minor
**Files:**
- Git tags

**Step 1: Update version**

Modify `src/claude_lint/__version__.py`:
```python
"""Version information for lint-claude."""

__version__ = "0.2.0"
```

Modify `pyproject.toml`:
```toml
version = "0.2.0"
```

**Step 2: Commit version bump**

```bash
git add src/claude_lint/__version__.py pyproject.toml
git commit -m "chore: bump version to 0.2.0"
```

**Step 3: Create git tag**

```bash
git tag -a v0.2.0 -m "Release v0.2.0: Production-ready with comprehensive fixes

Critical fixes:
- Subprocess timeouts
- Atomic file writes
- Proper exception handling
- Logging framework

Major improvements:
- Configurable model
- Input validation
- Better error handling
- Performance improvements

See CHANGELOG.md for full details"
```

**Step 4: Push commits and tag**

```bash
git push origin main
git push origin v0.2.0
```

**Step 5: Verify**

Run: `git tag -l`
Expected: Shows v0.1.0 and v0.2.0

Run: `git log --oneline`
Expected: Shows all commits from plan

---

## Summary

This plan addresses all issues from the code review:

**Critical Issues (Fixed):**
1. Subprocess timeouts (Task 2)
2. Logging framework (Task 1)
3. Atomic file writes (Task 3)
4. Exception handling (Task 4)
5. Unused dependency (Task 5)

**Major Issues (Fixed):**
6. Pattern matching unification (Task 6)
7. Configurable model (Task 7)
8. File encoding handling (Task 8)
9. Input validation (Task 9)
10. Client reuse (Task 10)
11. CLI flags (Task 11)
12. File size limits (Task 12)

**Minor Issues (Fixed):**
13. TypedDict for violations (Task 13)
14. Module exports (Task 14)
15. Config snake_case (Task 15)
16. Documentation (Task 16)

**Total Tasks:** 18
**Estimated Time:** 4-6 hours for sequential execution
**Test Coverage Expected:** >90%
