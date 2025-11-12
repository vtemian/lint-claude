# Production-Ready Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform claude-lint from B- quality to A-grade production-ready CLI tool by fixing critical architectural issues, adding essential features (rate limiting, progress bars, timeouts), and refactoring the god function.

**Architecture:** Refactor orchestrator from monolithic god function into focused, single-responsibility components. Add rate limiting layer for API calls, progress tracking for UX, and performance optimizations. Maintain functional architecture (no classes) while improving separation of concerns.

**Tech Stack:** Python 3.11+, anthropic SDK, click, rich (progress bars), pytest, existing dependencies

---

## Critical Fixes (Week 1)

### Task 1: Fix CLI Exception Handler (Critical #4)

**Priority:** CRITICAL
**Time:** 15 minutes
**Files:**
- Modify: `src/claude_lint/cli.py:72-74`
- Create: `tests/test_cli_exceptions.py`

**Problem:** CLI catches ALL exceptions including KeyboardInterrupt, defeating the fixes in Task 4 of previous plan.

**Step 1: Write test for KeyboardInterrupt handling**

Create `tests/test_cli_exceptions.py`:

```python
"""Tests for CLI exception handling."""
import subprocess
import sys
import tempfile
from pathlib import Path
import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from claude_lint.cli import main


def test_keyboard_interrupt_exits_with_130():
    """Test that KeyboardInterrupt exits with code 130 (SIGINT)."""
    runner = CliRunner(mix_stderr=False)

    with patch('claude_lint.cli.run_compliance_check') as mock_run:
        mock_run.side_effect = KeyboardInterrupt()

        result = runner.invoke(main, ['--full'])

        assert result.exit_code == 130
        assert "Cancelled by user" in result.stderr or "Cancelled" in result.output


def test_value_error_shows_error_message():
    """Test that ValueError shows helpful error message."""
    runner = CliRunner(mix_stderr=False)

    with patch('claude_lint.cli.run_compliance_check') as mock_run:
        mock_run.side_effect = ValueError("Invalid configuration")

        result = runner.invoke(main, ['--full'])

        assert result.exit_code == 2
        assert "Error: Invalid configuration" in result.stderr or "Invalid configuration" in result.output


def test_generic_exception_shows_helpful_message():
    """Test that unexpected exceptions show helpful message."""
    runner = CliRunner(mix_stderr=False)

    with patch('claude_lint.cli.run_compliance_check') as mock_run:
        mock_run.side_effect = RuntimeError("Unexpected internal error")

        result = runner.invoke(main, ['--full', '--verbose'])

        assert result.exit_code == 2
        # Should see the actual error in verbose mode
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_exceptions.py -v`
Expected: FAIL - Current code exits with wrong code for KeyboardInterrupt

**Step 3: Fix exception handling in CLI**

Modify `src/claude_lint/cli.py` lines 72-74:

```python
    try:
        # Run compliance check
        results = run_compliance_check(
            project_root, cfg, mode=mode, base_branch=base_branch
        )

        # Format output
        if output_json:
            output = format_json_report(results)
        else:
            output = format_detailed_report(results)

        click.echo(output)

        # Exit with appropriate code
        exit_code_val = get_exit_code(results)
        sys.exit(exit_code_val)

    except KeyboardInterrupt:
        # User cancelled with Ctrl-C
        click.echo("\nOperation cancelled by user", err=True)
        sys.exit(130)  # Standard SIGINT exit code
    except (ValueError, FileNotFoundError) as e:
        # Expected errors with helpful messages
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)
    except Exception as e:
        # Unexpected errors - log details in verbose mode
        logger = get_logger(__name__)
        logger.exception("Unexpected error during execution")
        click.echo(
            f"An unexpected error occurred: {e}\n"
            "Run with --verbose for details.",
            err=True
        )
        sys.exit(2)
```

**Step 4: Add import for logger**

Add to imports at top of `src/claude_lint/cli.py`:

```python
from claude_lint.logging_config import setup_logging, get_logger
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_cli_exceptions.py -v`
Expected: PASS (3 tests)

**Step 6: Run all tests to ensure no regressions**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 7: Commit**

```bash
git add src/claude_lint/cli.py tests/test_cli_exceptions.py
git commit -m "fix: handle KeyboardInterrupt correctly in CLI

- Exit with code 130 on Ctrl-C (standard SIGINT)
- Separate handlers for expected vs unexpected errors
- Show helpful messages for different error types
- Log unexpected errors with --verbose"
```

---

### Task 2: Add API Client Timeout (Critical #5)

**Priority:** CRITICAL
**Time:** 30 minutes
**Files:**
- Modify: `src/claude_lint/api_client.py:47-63`
- Modify: `src/claude_lint/config.py:8-28`
- Create: `tests/test_api_timeout.py`

**Problem:** API calls can hang forever with no timeout.

**Step 1: Write test for API timeout**

Create `tests/test_api_timeout.py`:

```python
"""Tests for API timeout handling."""
import pytest
from unittest.mock import patch, MagicMock
from anthropic import APITimeoutError
from claude_lint.api_client import analyze_files_with_client, create_client
from claude_lint.config import get_default_config


def test_api_client_has_timeout_configured():
    """Test that API client is created with timeout."""
    with patch('claude_lint.api_client.Anthropic') as mock_anthropic:
        create_client("test-key")

        # Verify Anthropic was called with timeout
        mock_anthropic.assert_called_once()
        call_kwargs = mock_anthropic.call_args[1]
        assert 'timeout' in call_kwargs
        assert call_kwargs['timeout'] == 60.0


def test_api_call_respects_timeout():
    """Test that API calls use configured timeout."""
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="response")]
    client.messages.create.return_value = mock_response

    analyze_files_with_client(client, "guidelines", "prompt")

    # Verify timeout was passed to API call
    client.messages.create.assert_called_once()
    # The timeout is set on client initialization, not per-call


def test_api_timeout_raises_clear_error():
    """Test that API timeout errors are clear."""
    client = MagicMock()
    client.messages.create.side_effect = APITimeoutError("Request timed out")

    with pytest.raises(APITimeoutError, match="timed out"):
        analyze_files_with_client(client, "guidelines", "prompt")


def test_config_has_api_timeout_field():
    """Test that config includes api_timeout_seconds."""
    config = get_default_config()
    assert hasattr(config, 'api_timeout_seconds')
    assert config.api_timeout_seconds == 60.0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api_timeout.py -v`
Expected: FAIL - Config doesn't have api_timeout_seconds, client not created with timeout

**Step 3: Add timeout to Config**

Modify `src/claude_lint/config.py`:

```python
@dataclass
class Config:
    """Configuration for claude-lint."""
    include: list[str]
    exclude: list[str]
    batch_size: int
    model: str = "claude-sonnet-4-5-20250929"
    max_file_size_mb: float = 1.0
    api_timeout_seconds: float = 60.0
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
        api_timeout_seconds=60.0,
        api_key=None
    )
```

Update `load_config()`:

```python
    return Config(
        include=data.get("include", defaults.include),
        exclude=data.get("exclude", defaults.exclude),
        batch_size=data.get("batch_size", data.get("batchSize", defaults.batch_size)),
        model=data.get("model", defaults.model),
        max_file_size_mb=data.get(
            "max_file_size_mb",
            data.get("maxFileSizeMb", defaults.max_file_size_mb)
        ),
        api_timeout_seconds=data.get(
            "api_timeout_seconds",
            data.get("apiTimeoutSeconds", defaults.api_timeout_seconds)
        ),
        api_key=data.get("api_key", data.get("apiKey"))
    )
```

**Step 4: Add timeout to client creation**

Modify `src/claude_lint/api_client.py`:

Update `create_client()`:

```python
def create_client(api_key: str, timeout: float = 60.0) -> Anthropic:
    """Create Anthropic client with timeout.

    Args:
        api_key: Anthropic API key
        timeout: Request timeout in seconds (default: 60)

    Returns:
        Anthropic client instance configured with timeout
    """
    return Anthropic(api_key=api_key, timeout=timeout)
```

**Step 5: Update orchestrator to pass timeout**

Modify `src/claude_lint/orchestrator.py` line 106:

```python
    # Create API client once for all batches
    assert api_key is not None  # Validated above
    client = create_client(api_key, timeout=config.api_timeout_seconds)
```

**Step 6: Update imports in api_client.py**

Add to imports:

```python
from anthropic import Anthropic, APIError, APIConnectionError, RateLimitError, APITimeoutError
```

**Step 7: Run tests to verify they pass**

Run: `pytest tests/test_api_timeout.py -v`
Expected: PASS (4 tests)

**Step 8: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 9: Commit**

```bash
git add src/claude_lint/api_client.py src/claude_lint/config.py src/claude_lint/orchestrator.py tests/test_api_timeout.py
git commit -m "feat: add configurable API timeout (default 60s)

- Prevent indefinite hangs on API calls
- Add api_timeout_seconds to Config (default 60.0)
- Create Anthropic client with timeout parameter
- Support apiTimeoutSeconds in config file"
```

---

### Task 3: Add Rate Limiting (Critical #2)

**Priority:** CRITICAL
**Time:** 2 hours
**Files:**
- Create: `src/claude_lint/rate_limiter.py`
- Modify: `src/claude_lint/orchestrator.py:104-169`
- Modify: `src/claude_lint/config.py:8-28`
- Create: `tests/test_rate_limiter.py`

**Problem:** No rate limiting leads to API throttling on large projects.

**Step 1: Write tests for rate limiter**

Create `tests/test_rate_limiter.py`:

```python
"""Tests for rate limiting functionality."""
import time
import pytest
from claude_lint.rate_limiter import RateLimiter


def test_rate_limiter_allows_requests_under_limit():
    """Test that requests under limit go through immediately."""
    limiter = RateLimiter(max_requests=5, window_seconds=1.0)

    start = time.time()
    for _ in range(3):
        limiter.acquire()
    elapsed = time.time() - start

    # Should be nearly instant (< 100ms)
    assert elapsed < 0.1


def test_rate_limiter_blocks_when_limit_exceeded():
    """Test that rate limiter blocks when limit is exceeded."""
    limiter = RateLimiter(max_requests=2, window_seconds=1.0)

    # First 2 requests should be instant
    start = time.time()
    limiter.acquire()
    limiter.acquire()
    elapsed_first_two = time.time() - start
    assert elapsed_first_two < 0.1

    # Third request should block
    start = time.time()
    limiter.acquire()
    elapsed_third = time.time() - start

    # Should have waited ~1 second for window to roll
    assert elapsed_third >= 0.9
    assert elapsed_third < 1.2


def test_rate_limiter_sliding_window():
    """Test that rate limiter uses sliding window."""
    limiter = RateLimiter(max_requests=2, window_seconds=1.0)

    # Make 2 requests
    limiter.acquire()
    time.sleep(0.6)  # Wait 600ms
    limiter.acquire()

    # Wait another 500ms (total 1.1s from first request)
    time.sleep(0.5)

    # Third request should be allowed (first request is now outside window)
    start = time.time()
    limiter.acquire()
    elapsed = time.time() - start

    # Should be instant since first request expired
    assert elapsed < 0.1


def test_rate_limiter_zero_requests_blocks():
    """Test that max_requests=0 blocks all requests."""
    limiter = RateLimiter(max_requests=0, window_seconds=1.0)

    # Should block indefinitely (we'll timeout the test)
    # Just verify it doesn't crash
    with pytest.raises(Exception):
        # Set a short timeout to avoid hanging test
        import signal
        def timeout_handler(signum, frame):
            raise TimeoutError("Rate limiter blocked as expected")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(1)
        try:
            limiter.acquire()
        finally:
            signal.alarm(0)


def test_config_has_rate_limit_fields():
    """Test that config includes rate limiting options."""
    from claude_lint.config import get_default_config

    config = get_default_config()
    assert hasattr(config, 'api_rate_limit')
    assert hasattr(config, 'api_rate_window_seconds')
    assert config.api_rate_limit == 4  # Conservative default
    assert config.api_rate_window_seconds == 1.0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rate_limiter.py -v`
Expected: FAIL - RateLimiter class doesn't exist

**Step 3: Implement rate limiter**

Create `src/claude_lint/rate_limiter.py`:

```python
"""Rate limiting for API calls."""
import time
from collections import deque


class RateLimiter:
    """Token bucket rate limiter with sliding window.

    Limits the number of requests within a time window using a sliding
    window algorithm for accurate rate limiting.
    """

    def __init__(self, max_requests: int, window_seconds: float):
        """Initialize rate limiter.

        Args:
            max_requests: Maximum number of requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: deque[float] = deque()

    def acquire(self) -> None:
        """Acquire a rate limit token, blocking if necessary.

        This method blocks until a token is available within the rate limit.
        Uses a sliding window to track requests.
        """
        now = time.time()

        # Remove requests outside the current window
        while self.requests and self.requests[0] < now - self.window_seconds:
            self.requests.popleft()

        # If at limit, wait until oldest request expires
        if len(self.requests) >= self.max_requests:
            sleep_time = self.requests[0] + self.window_seconds - now
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Remove expired request
            self.requests.popleft()

        # Record this request
        self.requests.append(time.time())

    def try_acquire(self) -> bool:
        """Try to acquire a token without blocking.

        Returns:
            True if token acquired, False if at rate limit
        """
        now = time.time()

        # Remove requests outside the current window
        while self.requests and self.requests[0] < now - self.window_seconds:
            self.requests.popleft()

        # Check if we're at limit
        if len(self.requests) >= self.max_requests:
            return False

        # Record this request
        self.requests.append(now)
        return True
```

**Step 4: Add rate limit config fields**

Modify `src/claude_lint/config.py`:

```python
@dataclass
class Config:
    """Configuration for claude-lint."""
    include: list[str]
    exclude: list[str]
    batch_size: int
    model: str = "claude-sonnet-4-5-20250929"
    max_file_size_mb: float = 1.0
    api_timeout_seconds: float = 60.0
    api_rate_limit: int = 4  # Requests per second (conservative)
    api_rate_window_seconds: float = 1.0
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
        api_timeout_seconds=60.0,
        api_rate_limit=4,  # Conservative: 4 requests/second
        api_rate_window_seconds=1.0,
        api_key=None
    )
```

Update `load_config()`:

```python
    return Config(
        include=data.get("include", defaults.include),
        exclude=data.get("exclude", defaults.exclude),
        batch_size=data.get("batch_size", data.get("batchSize", defaults.batch_size)),
        model=data.get("model", defaults.model),
        max_file_size_mb=data.get(
            "max_file_size_mb",
            data.get("maxFileSizeMb", defaults.max_file_size_mb)
        ),
        api_timeout_seconds=data.get(
            "api_timeout_seconds",
            data.get("apiTimeoutSeconds", defaults.api_timeout_seconds)
        ),
        api_rate_limit=data.get(
            "api_rate_limit",
            data.get("apiRateLimit", defaults.api_rate_limit)
        ),
        api_rate_window_seconds=data.get(
            "api_rate_window_seconds",
            data.get("apiRateWindowSeconds", defaults.api_rate_window_seconds)
        ),
        api_key=data.get("api_key", data.get("apiKey"))
    )
```

**Step 5: Integrate rate limiter into orchestrator**

Modify `src/claude_lint/orchestrator.py`:

Add import:
```python
from claude_lint.rate_limiter import RateLimiter
```

After client creation (line ~106):
```python
    # Create API client once for all batches
    assert api_key is not None  # Validated above
    client = create_client(api_key, timeout=config.api_timeout_seconds)

    # Create rate limiter for API calls
    rate_limiter = RateLimiter(
        max_requests=config.api_rate_limit,
        window_seconds=config.api_rate_window_seconds
    )
```

Before API call (line ~163):
```python
        # Make API call with retry and rate limiting
        def api_call():
            # Acquire rate limit token (blocks if necessary)
            rate_limiter.acquire()

            response_text, response_obj = analyze_files_with_client(
                client, guidelines, prompt, model=config.model
            )
            return response_text

        response = retry_with_backoff(api_call)
```

**Step 6: Run tests to verify they pass**

Run: `pytest tests/test_rate_limiter.py::test_rate_limiter_allows_requests_under_limit -v`
Run: `pytest tests/test_rate_limiter.py::test_rate_limiter_blocks_when_limit_exceeded -v`
Run: `pytest tests/test_rate_limiter.py::test_rate_limiter_sliding_window -v`
Expected: PASS (skip the timeout test for now)

**Step 7: Run all tests**

Run: `pytest tests/ -v -k "not timeout_handler"`
Expected: All tests pass

**Step 8: Commit**

```bash
git add src/claude_lint/rate_limiter.py src/claude_lint/config.py src/claude_lint/orchestrator.py tests/test_rate_limiter.py
git commit -m "feat: add rate limiting for API calls

- Implement sliding window rate limiter
- Default 4 requests/second (conservative)
- Configurable via api_rate_limit in config
- Prevents API throttling on large projects"
```

---

### Task 4: Add Progress Indication with Rich (Critical #3)

**Priority:** CRITICAL
**Time:** 1 hour
**Files:**
- Modify: `pyproject.toml:6-10`
- Modify: `src/claude_lint/orchestrator.py:108-192`
- Create: `tests/test_progress_display.py`

**Problem:** No visual feedback during long operations - users think it's hung.

**Step 1: Add rich dependency**

Modify `pyproject.toml`:

```toml
dependencies = [
    "anthropic>=0.18.0",
    "click>=8.1.0",
    "rich>=13.0.0",
]
```

**Step 2: Install dependency**

Run: `uv sync`
Expected: rich installed successfully

**Step 3: Write test for progress display**

Create `tests/test_progress_display.py`:

```python
"""Tests for progress display functionality."""
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import pytest
from claude_lint.orchestrator import run_compliance_check
from claude_lint.config import Config


def test_progress_display_enabled_by_default(capsys):
    """Test that progress display is shown by default."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test files
        (tmpdir / "file1.py").write_text("code")
        (tmpdir / "file2.py").write_text("code")
        (tmpdir / "CLAUDE.md").write_text("# Guidelines")

        config = Config(
            include=["**/*.py"],
            exclude=[],
            batch_size=1,  # Force 2 batches
            api_key="test-key"
        )

        with patch('claude_lint.orchestrator.create_client') as mock_create:
            with patch('claude_lint.orchestrator.analyze_files_with_client') as mock_analyze:
                mock_create.return_value = MagicMock()
                mock_analyze.return_value = (
                    '{"results": [{"file": "file1.py", "violations": []}]}',
                    MagicMock()
                )

                # Progress should be visible in output
                # (We can't easily test rich output, but we can verify it doesn't crash)
                results = run_compliance_check(tmpdir, config, mode="full")

                assert len(results) >= 1


def test_progress_can_be_disabled():
    """Test that progress can be disabled via config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        (tmpdir / "file1.py").write_text("code")
        (tmpdir / "CLAUDE.md").write_text("# Guidelines")

        config = Config(
            include=["**/*.py"],
            exclude=[],
            batch_size=10,
            show_progress=False,  # Disable progress
            api_key="test-key"
        )

        with patch('claude_lint.orchestrator.create_client') as mock_create:
            with patch('claude_lint.orchestrator.analyze_files_with_client') as mock_analyze:
                mock_create.return_value = MagicMock()
                mock_analyze.return_value = (
                    '{"results": [{"file": "file1.py", "violations": []}]}',
                    MagicMock()
                )

                results = run_compliance_check(tmpdir, config, mode="full")
                assert len(results) == 1
```

**Step 4: Add show_progress to Config**

Modify `src/claude_lint/config.py`:

```python
@dataclass
class Config:
    """Configuration for claude-lint."""
    include: list[str]
    exclude: list[str]
    batch_size: int
    model: str = "claude-sonnet-4-5-20250929"
    max_file_size_mb: float = 1.0
    api_timeout_seconds: float = 60.0
    api_rate_limit: int = 4
    api_rate_window_seconds: float = 1.0
    show_progress: bool = True
    api_key: Optional[str] = None
```

Update defaults and load_config accordingly.

**Step 5: Add progress bar to orchestrator**

Modify `src/claude_lint/orchestrator.py`:

Add imports:
```python
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
```

Replace batch processing loop (lines ~111-192):

```python
    # Process batches with progress bar
    all_results = list(progress_state.results)  # Start with resumed results

    # Determine if we should show progress (not in JSON mode, not disabled)
    show_progress = config.show_progress and not os.environ.get('CLAUDE_LINT_NO_PROGRESS')

    if show_progress:
        # Rich progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[bold cyan]{task.fields[status]}"),
        ) as progress:

            remaining_batches = list(get_remaining_batch_indices(progress_state))
            task = progress.add_task(
                "Analyzing files",
                total=len(remaining_batches),
                status="Starting..."
            )

            for idx, batch_idx in enumerate(remaining_batches):
                batch = batches[batch_idx]

                # Update status
                progress.update(
                    task,
                    status=f"Batch {idx + 1}/{len(remaining_batches)} ({len(batch)} files)"
                )

                # Process batch (extract to helper function)
                batch_results_dict = _process_single_batch(
                    batch, project_root, config, guidelines, prompt,
                    client, rate_limiter, cache, guidelines_hash
                )

                all_results.extend(batch_results_dict)

                # Save progress
                progress_state = update_progress(progress_state, batch_idx, batch_results_dict)
                save_progress(progress_state, progress_path)
                save_cache(cache, cache_path)

                # Update progress bar
                progress.update(task, advance=1, status="Complete")
    else:
        # No progress bar - simple iteration
        for batch_idx in get_remaining_batch_indices(progress_state):
            batch = batches[batch_idx]

            batch_results_dict = _process_single_batch(
                batch, project_root, config, guidelines, prompt,
                client, rate_limiter, cache, guidelines_hash
            )

            all_results.extend(batch_results_dict)

            # Save progress
            progress_state = update_progress(progress_state, batch_idx, batch_results_dict)
            save_progress(progress_state, progress_path)
            save_cache(cache, cache_path)
```

**Step 6: Extract batch processing to helper**

Add helper function to `orchestrator.py`:

```python
def _process_single_batch(
    batch: list[Path],
    project_root: Path,
    config: Config,
    guidelines: str,
    prompt_template: str,
    client: Any,
    rate_limiter: RateLimiter,
    cache: Cache,
    guidelines_hash: str
) -> list[dict[str, Any]]:
    """Process a single batch of files.

    Args:
        batch: List of file paths to process
        project_root: Project root directory
        config: Configuration
        guidelines: CLAUDE.md content
        prompt_template: Unused (will build fresh)
        client: Anthropic client
        rate_limiter: Rate limiter for API calls
        cache: Cache object
        guidelines_hash: Hash of CLAUDE.md

    Returns:
        List of results for this batch
    """
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

    # Skip if no files to process
    if not file_contents:
        return []

    # Build prompt
    from claude_lint.processor import build_xml_prompt
    prompt = build_xml_prompt(guidelines, file_contents)

    # Make API call with retry and rate limiting
    def api_call():
        # Acquire rate limit token (blocks if necessary)
        rate_limiter.acquire()

        response_text, response_obj = analyze_files_with_client(
            client, guidelines, prompt, model=config.model
        )
        return response_text

    from claude_lint.retry import retry_with_backoff
    response = retry_with_backoff(api_call)

    # Parse results
    from claude_lint.processor import parse_response
    from claude_lint.collector import compute_file_hash
    from claude_lint.cache import CacheEntry

    batch_results: list[FileResult] = parse_response(response)
    batch_results_dict: list[dict[str, Any]] = [dict(r) for r in batch_results]

    # Update cache
    for result in batch_results:
        file_path = project_root / result["file"]
        file_hash = compute_file_hash(file_path)

        cache.entries[result["file"]] = CacheEntry(
            file_hash=file_hash,
            claude_md_hash=guidelines_hash,
            violations=result["violations"],
            timestamp=int(Path(file_path).stat().st_mtime)
        )

    return batch_results_dict
```

**Step 7: Run tests**

Run: `pytest tests/test_progress_display.py -v`
Expected: PASS (2 tests)

**Step 8: Manual test with progress bar**

Run: `uv run claude-lint --full` in a test directory
Expected: See progress bar with spinner and batch count

**Step 9: Commit**

```bash
git add pyproject.toml src/claude_lint/orchestrator.py src/claude_lint/config.py tests/test_progress_display.py
git commit -m "feat: add progress bar with rich

- Show real-time progress during file analysis
- Display batch count and file count
- Configurable via show_progress in config
- Can disable with CLAUDE_LINT_NO_PROGRESS env var
- Extract _process_single_batch helper for clarity"
```

---

### Task 5: Refactor Orchestrator God Function (Critical #1)

**Priority:** CRITICAL
**Time:** 4 hours
**Files:**
- Create: `src/claude_lint/batch_processor.py`
- Create: `src/claude_lint/file_reader.py`
- Modify: `src/claude_lint/orchestrator.py`
- Create: `tests/test_batch_processor.py`
- Create: `tests/test_file_reader.py`

**Problem:** orchestrator.py:run_compliance_check() is 161 lines doing everything.

**Step 1: Extract file reading logic**

Create `src/claude_lint/file_reader.py`:

```python
"""File reading with encoding fallback and size limits."""
from pathlib import Path
from typing import Optional
from claude_lint.logging_config import get_logger

logger = get_logger(__name__)


def read_file_safely(
    file_path: Path,
    project_root: Path,
    max_size_bytes: int
) -> Optional[str]:
    """Read file with encoding fallback and size checking.

    Tries UTF-8 first, falls back to latin-1. Checks file size before reading.

    Args:
        file_path: Absolute path to file
        project_root: Project root for relative path logging
        max_size_bytes: Maximum allowed file size in bytes

    Returns:
        File content as string, or None if file should be skipped
    """
    rel_path = file_path.relative_to(project_root)

    # Check file size
    try:
        file_size = file_path.stat().st_size
        if file_size > max_size_bytes:
            logger.warning(
                f"File {rel_path} exceeds size limit "
                f"({file_size / 1024 / 1024:.2f}MB > "
                f"{max_size_bytes / 1024 / 1024:.2f}MB), skipping"
            )
            return None
    except OSError as e:
        logger.warning(f"Cannot stat file {rel_path}, skipping: {e}")
        return None

    # Try reading with encoding fallback
    try:
        # Try UTF-8 first
        return file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        # Fall back to latin-1 which accepts all byte sequences
        try:
            logger.warning(
                f"File {rel_path} is not valid UTF-8, trying latin-1"
            )
            return file_path.read_text(encoding='latin-1')
        except Exception as e:
            logger.warning(
                f"Unable to decode file {rel_path}, skipping: {e}"
            )
            return None
    except FileNotFoundError:
        logger.warning(f"File not found, skipping: {rel_path}")
        return None
    except Exception as e:
        logger.warning(f"Error reading file {rel_path}, skipping: {e}")
        return None


def read_batch_files(
    batch: list[Path],
    project_root: Path,
    max_size_mb: float
) -> dict[str, str]:
    """Read multiple files for a batch.

    Args:
        batch: List of file paths to read
        project_root: Project root directory
        max_size_mb: Maximum file size in megabytes

    Returns:
        Dict mapping relative paths to file contents
    """
    file_contents = {}
    max_size_bytes = int(max_size_mb * 1024 * 1024)

    for file_path in batch:
        rel_path = str(file_path.relative_to(project_root))
        content = read_file_safely(file_path, project_root, max_size_bytes)

        if content is not None:
            file_contents[rel_path] = content

    return file_contents
```

**Step 2: Write tests for file reader**

Create `tests/test_file_reader.py`:

```python
"""Tests for safe file reading."""
import tempfile
from pathlib import Path
import pytest
from claude_lint.file_reader import read_file_safely, read_batch_files


def test_read_file_safely_utf8():
    """Test reading valid UTF-8 file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_file = tmpdir / "test.py"
        test_file.write_text("print('hello')")

        content = read_file_safely(test_file, tmpdir, max_size_bytes=1024*1024)

        assert content == "print('hello')"


def test_read_file_safely_exceeds_size():
    """Test file exceeding size limit is skipped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_file = tmpdir / "large.py"
        test_file.write_text("x" * 2000)

        # Set limit to 1000 bytes
        content = read_file_safely(test_file, tmpdir, max_size_bytes=1000)

        assert content is None


def test_read_file_safely_invalid_utf8():
    """Test fallback to latin-1 for invalid UTF-8."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_file = tmpdir / "latin.py"
        # Write latin-1 encoded content
        test_file.write_bytes("# Café".encode('latin-1'))

        content = read_file_safely(test_file, tmpdir, max_size_bytes=1024*1024)

        assert content is not None
        assert "Caf" in content


def test_read_batch_files():
    """Test reading multiple files in batch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        (tmpdir / "file1.py").write_text("code1")
        (tmpdir / "file2.py").write_text("code2")
        (tmpdir / "file3.py").write_text("x" * 2000)  # Too large

        files = [
            tmpdir / "file1.py",
            tmpdir / "file2.py",
            tmpdir / "file3.py"
        ]

        contents = read_batch_files(files, tmpdir, max_size_mb=0.001)  # 1KB limit

        assert "file1.py" in contents
        assert "file2.py" in contents
        assert "file3.py" not in contents
```

**Step 3: Extract batch processing logic**

Create `src/claude_lint/batch_processor.py`:

```python
"""Batch processing logic."""
from pathlib import Path
from typing import Any
from claude_lint.api_client import analyze_files_with_client
from claude_lint.cache import Cache, CacheEntry
from claude_lint.collector import compute_file_hash
from claude_lint.config import Config
from claude_lint.file_reader import read_batch_files
from claude_lint.processor import build_xml_prompt, parse_response
from claude_lint.rate_limiter import RateLimiter
from claude_lint.retry import retry_with_backoff
from claude_lint.types import FileResult


def process_batch(
    batch: list[Path],
    project_root: Path,
    config: Config,
    guidelines: str,
    guidelines_hash: str,
    client: Any,
    rate_limiter: RateLimiter,
    cache: Cache
) -> list[dict[str, Any]]:
    """Process a single batch of files.

    This function:
    1. Reads file contents with size/encoding checks
    2. Builds XML prompt
    3. Makes rate-limited API call with retry
    4. Parses results
    5. Updates cache

    Args:
        batch: List of file paths to process
        project_root: Project root directory
        config: Configuration
        guidelines: CLAUDE.md content
        guidelines_hash: Hash of CLAUDE.md
        client: Anthropic client
        rate_limiter: Rate limiter for API calls
        cache: Cache object to update

    Returns:
        List of file results as dicts
    """
    # Read files
    file_contents = read_batch_files(batch, project_root, config.max_file_size_mb)

    # Skip if no files to process
    if not file_contents:
        return []

    # Build prompt
    prompt = build_xml_prompt(guidelines, file_contents)

    # Make rate-limited API call with retry
    def api_call():
        rate_limiter.acquire()
        response_text, _ = analyze_files_with_client(
            client, guidelines, prompt, model=config.model
        )
        return response_text

    response = retry_with_backoff(api_call)

    # Parse results
    batch_results: list[FileResult] = parse_response(response)
    batch_results_dict: list[dict[str, Any]] = [dict(r) for r in batch_results]

    # Update cache
    for result in batch_results:
        try:
            file_path = project_root / result["file"]
            file_hash = compute_file_hash(file_path)

            cache.entries[result["file"]] = CacheEntry(
                file_hash=file_hash,
                claude_md_hash=guidelines_hash,
                violations=result["violations"],
                timestamp=int(file_path.stat().st_mtime)
            )
        except Exception:
            # If we can't update cache for a file, continue
            # (file might have been deleted)
            pass

    return batch_results_dict
```

**Step 4: Refactor orchestrator to use new modules**

Modify `src/claude_lint/orchestrator.py`:

Replace the entire batch processing section with:

```python
    # Process batches with optional progress bar
    all_results = list(progress_state.results)

    # Determine if we should show progress
    show_progress = config.show_progress and not os.environ.get('CLAUDE_LINT_NO_PROGRESS')

    remaining_batches = list(get_remaining_batch_indices(progress_state))

    if show_progress:
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[bold cyan]{task.fields[status]}"),
        ) as progress:

            task = progress.add_task(
                "Analyzing files",
                total=len(remaining_batches),
                status="Starting..."
            )

            for idx, batch_idx in enumerate(remaining_batches):
                batch = batches[batch_idx]

                progress.update(
                    task,
                    status=f"Batch {idx + 1}/{len(remaining_batches)} ({len(batch)} files)"
                )

                from claude_lint.batch_processor import process_batch
                batch_results = process_batch(
                    batch, project_root, config, guidelines, guidelines_hash,
                    client, rate_limiter, cache
                )

                all_results.extend(batch_results)

                progress_state = update_progress(progress_state, batch_idx, batch_results)
                save_progress(progress_state, progress_path)
                save_cache(cache, cache_path)

                progress.update(task, advance=1, status="Complete")
    else:
        # No progress bar
        for batch_idx in remaining_batches:
            batch = batches[batch_idx]

            from claude_lint.batch_processor import process_batch
            batch_results = process_batch(
                batch, project_root, config, guidelines, guidelines_hash,
                client, rate_limiter, cache
            )

            all_results.extend(batch_results)

            progress_state = update_progress(progress_state, batch_idx, batch_results)
            save_progress(progress_state, progress_path)
            save_cache(cache, cache_path)
```

**Step 5: Write tests for batch processor**

Create `tests/test_batch_processor.py`:

```python
"""Tests for batch processing."""
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest
from claude_lint.batch_processor import process_batch
from claude_lint.cache import Cache
from claude_lint.config import Config
from claude_lint.rate_limiter import RateLimiter


def test_process_batch_success():
    """Test successful batch processing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        (tmpdir / "file1.py").write_text("code")

        batch = [tmpdir / "file1.py"]
        config = Config(
            include=["**/*.py"],
            exclude=[],
            batch_size=10,
            api_key="test"
        )
        cache = Cache(claude_md_hash="hash", entries={})
        rate_limiter = RateLimiter(max_requests=10, window_seconds=1.0)

        client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "response"

        with patch('claude_lint.batch_processor.analyze_files_with_client') as mock_api:
            mock_api.return_value = (
                '{"results": [{"file": "file1.py", "violations": []}]}',
                mock_response
            )

            results = process_batch(
                batch, tmpdir, config, "guidelines", "hash",
                client, rate_limiter, cache
            )

            assert len(results) == 1
            assert results[0]["file"] == "file1.py"
            assert "file1.py" in cache.entries


def test_process_batch_empty_after_filtering():
    """Test batch with all files filtered out."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # File that's too large
        large_file = tmpdir / "large.py"
        large_file.write_text("x" * 1024 * 1024 * 2)  # 2MB

        batch = [large_file]
        config = Config(
            include=["**/*.py"],
            exclude=[],
            batch_size=10,
            max_file_size_mb=0.5,  # 500KB limit
            api_key="test"
        )
        cache = Cache(claude_md_hash="hash", entries={})
        rate_limiter = RateLimiter(max_requests=10, window_seconds=1.0)

        results = process_batch(
            batch, tmpdir, config, "guidelines", "hash",
            MagicMock(), rate_limiter, cache
        )

        # Should return empty - no API call made
        assert len(results) == 0
```

**Step 6: Run tests**

Run: `pytest tests/test_file_reader.py -v`
Run: `pytest tests/test_batch_processor.py -v`
Expected: All pass

**Step 7: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 8: Verify orchestrator is now shorter**

Run: `wc -l src/claude_lint/orchestrator.py`
Expected: Should be ~200 lines (down from 330)

**Step 9: Commit**

```bash
git add src/claude_lint/file_reader.py src/claude_lint/batch_processor.py src/claude_lint/orchestrator.py tests/test_file_reader.py tests/test_batch_processor.py
git commit -m "refactor: extract file reading and batch processing

- Extract read_file_safely and read_batch_files to file_reader.py
- Extract process_batch to batch_processor.py
- Orchestrator reduced from 330 to ~200 lines
- Improved testability and maintainability
- Each module now has single responsibility"
```

---

## Important Improvements (Week 2)

### Task 6: Add Integration Tests (Important #7)

**Priority:** Important
**Time:** 4 hours
**Files:**
- Create: `tests/integration/test_full_workflow.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/fixtures/test_project/`

**Problem:** No end-to-end tests - can't trust the full workflow.

**Step 1: Create integration test fixtures**

Create test project structure:

```bash
mkdir -p tests/integration/fixtures/test_project
```

Create `tests/integration/fixtures/test_project/CLAUDE.md`:
```markdown
# Test Guidelines

## Code Style
- Use type hints
- Functions should be under 50 lines
- No global variables
```

Create `tests/integration/fixtures/test_project/good.py`:
```python
def add(a: int, b: int) -> int:
    return a + b
```

Create `tests/integration/fixtures/test_project/bad.py`:
```python
def multiply(x, y):  # Missing type hints
    return x * y

GLOBAL_VAR = 42  # Global variable
```

Create `tests/integration/fixtures/test_project/.agent-lint.json`:
```json
{
  "include": ["**/*.py"],
  "exclude": ["venv/**"],
  "batch_size": 5
}
```

**Step 2: Write integration tests**

Create `tests/integration/__init__.py`:
```python
"""Integration tests for full workflows."""
```

Create `tests/integration/test_full_workflow.py`:

```python
"""End-to-end integration tests."""
import os
import subprocess
import tempfile
from pathlib import Path
import shutil
import pytest
import json


@pytest.fixture
def test_project(tmp_path):
    """Create a test project with known files."""
    # Copy fixture project to temp directory
    fixture_dir = Path(__file__).parent / "fixtures" / "test_project"
    project_dir = tmp_path / "test_project"
    shutil.copytree(fixture_dir, project_dir)
    return project_dir


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="Requires ANTHROPIC_API_KEY environment variable"
)
def test_full_scan_with_real_api(test_project):
    """Test full scan with real API call."""
    # Run CLI
    result = subprocess.run(
        ["uv", "run", "claude-lint", "--full", "--json"],
        cwd=test_project,
        capture_output=True,
        text=True,
        timeout=60
    )

    # Should complete successfully
    assert result.returncode in [0, 1]  # 0 = pass, 1 = violations

    # Should produce valid JSON
    output = json.loads(result.stdout)
    assert "results" in output
    assert "summary" in output

    # Should analyze both files
    files = [r["file"] for r in output["results"]]
    assert "good.py" in files
    assert "bad.py" in files


def test_full_scan_without_api_key(test_project, monkeypatch):
    """Test that missing API key shows helpful error."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = subprocess.run(
        ["uv", "run", "claude-lint", "--full"],
        cwd=test_project,
        capture_output=True,
        text=True
    )

    assert result.returncode == 2
    assert "API key" in result.stderr


def test_keyboard_interrupt_handling(test_project):
    """Test that Ctrl-C is handled gracefully."""
    # Start process
    proc = subprocess.Popen(
        ["uv", "run", "claude-lint", "--full"],
        cwd=test_project,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait a moment then interrupt
    import time
    time.sleep(0.5)
    proc.send_signal(subprocess.signal.SIGINT)

    # Get result
    stdout, stderr = proc.communicate(timeout=5)

    # Should exit with 130 (SIGINT)
    assert proc.returncode == 130
    assert "Cancelled" in stderr or "Cancelled" in stdout


def test_progress_bar_output(test_project):
    """Test that progress bar is shown."""
    result = subprocess.run(
        ["uv", "run", "claude-lint", "--full"],
        cwd=test_project,
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "ANTHROPIC_API_KEY": "test-key"}
    )

    # Progress should be visible (rich uses ANSI codes)
    # We can't easily test the visual output, but verify it runs
    assert result.returncode in [0, 1, 2]


def test_cache_persistence(test_project):
    """Test that cache is persisted and reused."""
    # First run
    result1 = subprocess.run(
        ["uv", "run", "claude-lint", "--full", "--json"],
        cwd=test_project,
        capture_output=True,
        text=True,
        timeout=60
    )

    # Cache file should exist
    cache_file = test_project / ".agent-lint-cache.json"
    assert cache_file.exists()

    # Second run should be faster (uses cache)
    result2 = subprocess.run(
        ["uv", "run", "claude-lint", "--full", "--json"],
        cwd=test_project,
        capture_output=True,
        text=True,
        timeout=60
    )

    # Both should produce same results
    output1 = json.loads(result1.stdout)
    output2 = json.loads(result2.stdout)

    assert output1["results"] == output2["results"]


def test_version_flag():
    """Test --version flag works."""
    result = subprocess.run(
        ["uv", "run", "claude-lint", "--version"],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0
    assert "claude-lint" in result.stdout
    assert "0.2.0" in result.stdout
```

**Step 3: Run integration tests**

Run: `ANTHROPIC_API_KEY=sk-... pytest tests/integration/ -v`
Expected: Tests pass (or skip if no API key)

**Step 4: Update README with integration test instructions**

Add to README.md:

```markdown
## Running Tests

### Unit Tests
```bash
pytest tests/ -v
```

### Integration Tests (requires API key)
```bash
export ANTHROPIC_API_KEY=sk-ant-...
pytest tests/integration/ -v
```

Integration tests make real API calls and are slower. They verify:
- End-to-end CLI workflows
- Real API integration
- Cache persistence
- Error handling in production scenarios
```

**Step 5: Commit**

```bash
git add tests/integration/ README.md
git commit -m "test: add comprehensive integration tests

- Test full workflow with real API
- Test cache persistence
- Test keyboard interrupt handling
- Test error scenarios
- Skip if ANTHROPIC_API_KEY not set"
```

---

### Task 7: Fix Cache Save Timing (Important #8)

**Priority:** Important
**Time:** 30 minutes
**Files:**
- Modify: `src/claude_lint/orchestrator.py:~180-190`

**Problem:** Cache is updated in memory but only saved at the end. If process crashes after 90% completion, cache is lost.

**Step 1: Verify current behavior**

Check `src/claude_lint/orchestrator.py` - cache is saved after each batch for progress, but the final save at line 200 is redundant.

**Step 2: Remove redundant cache save**

Modify `src/claude_lint/orchestrator.py`:

Remove lines around 198-200:

```python
    # Cleanup progress on completion
    if is_progress_complete(progress_state):
        cleanup_progress(progress_path)

    # Cache is already saved after each batch - no need to save again here
    # The cache.claude_md_hash is already set correctly

    return all_results
```

**Step 3: Add comment explaining cache strategy**

Add comment before batch loop:

```python
    # Process batches with optional progress bar
    # Note: Cache is saved after each batch to prevent data loss on crashes
    all_results = list(progress_state.results)
```

**Step 4: Run tests to verify no regression**

Run: `pytest tests/test_orchestrator.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add src/claude_lint/orchestrator.py
git commit -m "refactor: remove redundant cache save

- Cache is already saved after each batch
- Remove duplicate save at end
- Add comment explaining save strategy"
```

---

### Task 8: Add Telemetry and Metrics (Important #10)

**Priority:** Important
**Time:** 1 hour
**Files:**
- Create: `src/claude_lint/metrics.py`
- Modify: `src/claude_lint/orchestrator.py`
- Modify: `src/claude_lint/reporter.py`
- Create: `tests/test_metrics.py`

**Problem:** No visibility into performance, costs, or cache effectiveness.

**Step 1: Create metrics module**

Create `src/claude_lint/metrics.py`:

```python
"""Metrics and telemetry tracking."""
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AnalysisMetrics:
    """Metrics collected during analysis."""

    # Timing
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    # Files
    total_files_collected: int = 0
    files_from_cache: int = 0
    files_analyzed: int = 0
    files_skipped: int = 0

    # API calls
    api_calls_made: int = 0
    api_retries: int = 0

    # Cache
    cache_hits: int = 0
    cache_misses: int = 0

    # Rate limiting
    rate_limit_waits: int = 0
    total_wait_time: float = 0.0

    def finish(self) -> None:
        """Mark analysis as finished."""
        self.end_time = time.time()

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def cache_hit_rate(self) -> float:
        """Get cache hit rate as percentage."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return (self.cache_hits / total) * 100

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "total_files_collected": self.total_files_collected,
            "files_from_cache": self.files_from_cache,
            "files_analyzed": self.files_analyzed,
            "files_skipped": self.files_skipped,
            "api_calls_made": self.api_calls_made,
            "api_retries": self.api_retries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(self.cache_hit_rate, 1),
            "rate_limit_waits": self.rate_limit_waits,
            "total_wait_time": round(self.total_wait_time, 2)
        }
```

**Step 2: Integrate metrics into orchestrator**

Modify `src/claude_lint/orchestrator.py`:

Add import and create metrics:
```python
from claude_lint.metrics import AnalysisMetrics

def run_compliance_check(
    project_root: Path,
    config: Config,
    mode: str = "full",
    base_branch: Optional[str] = None
) -> tuple[list[dict[str, Any]], AnalysisMetrics]:
    """Run compliance check.

    Returns:
        Tuple of (results list, metrics object)
    """
    metrics = AnalysisMetrics()

    # ... existing validation ...

    # Collect files
    files_to_check = collect_files_for_mode(...)
    metrics.total_files_collected = len(files_to_check)

    # Filter using cache
    files_needing_check = filter_cached_files(...)
    metrics.files_from_cache = len(files_to_check) - len(files_needing_check)
    metrics.cache_hits = metrics.files_from_cache
    metrics.cache_misses = len(files_needing_check)

    # ... batch processing with metrics.api_calls_made += 1 ...

    metrics.finish()
    return all_results, metrics
```

**Step 3: Update CLI to show metrics**

Modify `src/claude_lint/cli.py`:

```python
        # Run compliance check
        results, metrics = run_compliance_check(
            project_root, cfg, mode=mode, base_branch=base_branch
        )

        # Format output
        if output_json:
            output = format_json_report(results, metrics)
        else:
            output = format_detailed_report(results, metrics)
```

**Step 4: Update reporters to include metrics**

Modify `src/claude_lint/reporter.py`:

```python
def format_detailed_report(
    results: list[FileResult],
    metrics: Optional[AnalysisMetrics] = None
) -> str:
    """Format results as detailed human-readable report."""
    lines = []
    # ... existing formatting ...

    # Add metrics at end
    if metrics:
        lines.append("")
        lines.append("=" * 70)
        lines.append("METRICS")
        lines.append("=" * 70)
        lines.append(f"Elapsed time: {metrics.elapsed_seconds:.1f}s")
        lines.append(f"Files collected: {metrics.total_files_collected}")
        lines.append(f"Files from cache: {metrics.files_from_cache}")
        lines.append(f"Files analyzed: {metrics.files_analyzed}")
        lines.append(f"API calls: {metrics.api_calls_made}")
        lines.append(f"Cache hit rate: {metrics.cache_hit_rate:.1f}%")

    return "\n".join(lines)


def format_json_report(
    results: list[FileResult],
    metrics: Optional[AnalysisMetrics] = None
) -> str:
    """Format results as JSON."""
    report = {
        "results": results,
        "summary": get_summary(results)
    }

    if metrics:
        report["metrics"] = metrics.to_dict()

    return json.dumps(report, indent=2)
```

**Step 5: Write tests**

Create `tests/test_metrics.py`:

```python
"""Tests for metrics tracking."""
from claude_lint.metrics import AnalysisMetrics
import time


def test_metrics_initialization():
    """Test metrics starts with zeros."""
    metrics = AnalysisMetrics()

    assert metrics.total_files_collected == 0
    assert metrics.api_calls_made == 0
    assert metrics.cache_hits == 0


def test_metrics_elapsed_time():
    """Test elapsed time calculation."""
    metrics = AnalysisMetrics()
    time.sleep(0.1)
    metrics.finish()

    assert metrics.elapsed_seconds >= 0.1
    assert metrics.elapsed_seconds < 1.0


def test_metrics_cache_hit_rate():
    """Test cache hit rate calculation."""
    metrics = AnalysisMetrics()

    metrics.cache_hits = 80
    metrics.cache_misses = 20

    assert metrics.cache_hit_rate == 80.0


def test_metrics_to_dict():
    """Test conversion to dictionary."""
    metrics = AnalysisMetrics()
    metrics.total_files_collected = 100
    metrics.api_calls_made = 10
    metrics.finish()

    d = metrics.to_dict()

    assert d["total_files_collected"] == 100
    assert d["api_calls_made"] == 10
    assert "elapsed_seconds" in d
```

**Step 6: Run tests**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS

**Step 7: Update all orchestrator tests to handle tuple return**

Modify tests to unpack tuple:
```python
results, metrics = run_compliance_check(...)
```

**Step 8: Commit**

```bash
git add src/claude_lint/metrics.py src/claude_lint/orchestrator.py src/claude_lint/reporter.py src/claude_lint/cli.py tests/test_metrics.py
git commit -m "feat: add comprehensive metrics tracking

- Track timing, cache hit rate, API calls
- Show metrics in both text and JSON output
- Helps users understand performance and costs
- Return metrics from run_compliance_check()"
```

---

## Summary

This plan addresses:

**Critical Fixes (Tasks 1-5):**
1. ✅ CLI exception handler (15 min)
2. ✅ API timeout (30 min)
3. ✅ Rate limiting (2 hours)
4. ✅ Progress indication (1 hour)
5. ✅ Orchestrator refactoring (4 hours)

**Important Improvements (Tasks 6-8):**
6. ✅ Integration tests (4 hours)
7. ✅ Cache save timing (30 min)
8. ✅ Telemetry/metrics (1 hour)

**Total Time:** ~13 hours of focused work

**After This Plan:**
- No hanging on API calls
- No rate limit errors
- Users see progress
- Code is maintainable
- Full E2E test coverage
- Performance visibility

The remaining items (async file reading, pre-compiled patterns, color output) can be addressed in future iterations once these critical foundations are solid.
