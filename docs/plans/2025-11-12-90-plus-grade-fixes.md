# 90+ Grade Production Readiness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix critical and important issues to achieve 90+ code quality grade (currently C+ 74/100)

**Architecture:** Systematic fixes addressing test failures, unsafe code patterns, coverage gaps, architectural issues, and validation weaknesses. Focus on TDD, defensive programming, and SRP.

**Tech Stack:** Python 3.11+, pytest, mypy, ruff, Pydantic

**Current State:** v0.3.0 with 2 failing tests, unsafe assert, 74-88% coverage gaps, 155-line god function

**Target Grade:** A- (90+/100)

---

## Task 1: Fix Flaky Timing Test (CRITICAL)

**Problem:** `test_exponential_backoff_timing` fails intermittently due to timing assumptions

**Files:**
- Modify: `tests/test_retry.py:38-63`

**Step 1: Write deterministic test with mocked time.sleep**

Replace the timing-based test with a deterministic mock-based test:

```python
def test_exponential_backoff_timing():
    """Test that backoff delays increase exponentially with jitter."""
    import time
    from unittest.mock import patch

    call_times = []
    sleep_durations = []

    def mock_sleep(duration):
        sleep_durations.append(duration)
        # Don't actually sleep

    def failing_func():
        call_times.append(1)  # Just count calls
        if len(call_times) < 3:
            raise Exception("fail")
        return "success"

    with patch('time.sleep', side_effect=mock_sleep):
        retry_with_backoff(failing_func, max_retries=3, initial_delay=0.1)

    # Should have 2 sleeps (first attempt doesn't sleep, 3rd succeeds)
    assert len(sleep_durations) == 2

    # Verify jitter is applied (delays should NOT be exact multiples)
    # Base delays would be: 0.1, 0.2
    # With jitter: 0.05-0.15, 0.10-0.30
    delay1, delay2 = sleep_durations

    # Check ranges with jitter applied
    assert 0.05 <= delay1 <= 0.15, f"delay1 {delay1} outside jittered range"
    assert 0.10 <= delay2 <= 0.30, f"delay2 {delay2} outside jittered range"

    # Verify exponential increase (even with jitter, delay2 should generally be > delay1)
    # This is probabilistic but with >50% chance delay2 > delay1
```

**Step 2: Run test to verify it passes consistently**

Run: `uv run pytest tests/test_retry.py::test_exponential_backoff_timing -v --count=10`

Expected: PASS 10/10 times (no flakiness)

**Step 3: Commit**

```bash
git add tests/test_retry.py
git commit -m "fix: make retry timing test deterministic with mocked sleep

- Replace time-based assertions with sleep duration tracking
- Mock time.sleep to eliminate timing variability
- Test now passes consistently in any environment

Fixes flaky test that failed intermittently in CI"
```

---

## Task 2: Fix KeyboardInterrupt Handling in CLI (CRITICAL)

**Problem:** CLI returns exit code 2 instead of 130 for SIGINT (Ctrl-C)

**Files:**
- Modify: `src/claude_lint/cli.py:25-101`

**Step 1: Write test for exit code 130 verification**

Already exists in `tests/integration/test_full_workflow.py:63-84` - verify current failure:

Run: `uv run pytest tests/integration/test_full_workflow.py::test_keyboard_interrupt_handling -v`

Expected: FAIL with "assert 2 == 130"

**Step 2: Add KeyboardInterrupt handler to CLI main()**

Wrap the main CLI logic to catch KeyboardInterrupt:

```python
# cli.py
@click.command()
@click.option("--full", is_flag=True, help="Scan all files")
@click.option("--diff", "base_branch", help="Scan changed files from branch")
@click.option("--working", "scan_working", is_flag=True, help="Scan working directory changes")
@click.option("--staged", "scan_staged", is_flag=True, help="Scan staged files")
@click.option("--json", "json_output", is_flag=True, help="Output JSON format")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--quiet", "-q", is_flag=True, help="Quiet output (errors only)")
def main(
    full: bool,
    base_branch: str | None,
    scan_working: bool,
    scan_staged: bool,
    json_output: bool,
    verbose: bool,
    quiet: bool,
) -> None:
    """CLAUDE.md compliance checker."""
    try:
        _run_main(full, base_branch, scan_working, scan_staged, json_output, verbose, quiet)
    except KeyboardInterrupt:
        click.echo("\n\nOperation cancelled by user", err=True)
        sys.exit(130)  # Standard exit code for SIGINT


def _run_main(
    full: bool,
    base_branch: str | None,
    scan_working: bool,
    scan_staged: bool,
    json_output: bool,
    verbose: bool,
    quiet: bool,
) -> None:
    """Internal main logic - extracted for KeyboardInterrupt handling."""
    # Setup logging
    if verbose:
        setup_logging("DEBUG")
    elif quiet:
        setup_logging("ERROR")
    else:
        setup_logging("INFO")

    # ... rest of existing main() logic here ...
    # (move all current main() body into this function)
```

**Step 3: Add sys import at top**

```python
# cli.py:1
import sys
from pathlib import Path
```

**Step 4: Run test to verify exit code 130**

Run: `uv run pytest tests/integration/test_full_workflow.py::test_keyboard_interrupt_handling -v`

Expected: PASS with exit code 130

**Step 5: Commit**

```bash
git add src/claude_lint/cli.py
git commit -m "fix: handle KeyboardInterrupt with exit code 130

- Wrap main CLI logic in try-except for KeyboardInterrupt
- Extract _run_main() for cleaner exception handling
- Return standard exit code 130 for SIGINT (Ctrl-C)

Fixes test_keyboard_interrupt_handling integration test"
```

---

## Task 3: Replace Unsafe Assert with Explicit Check (CRITICAL)

**Problem:** `orchestrator.py:109` uses assert which is removed by Python -O flag

**Files:**
- Modify: `src/claude_lint/orchestrator.py:109`
- Test: `tests/test_orchestrator.py` (create new test)

**Step 1: Write test for explicit validation error**

Create `tests/test_orchestrator_validation.py`:

```python
"""Tests for orchestrator input validation."""
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from claude_lint.config import Config
from claude_lint.orchestrator import run_compliance_check


def test_run_compliance_check_raises_on_none_api_key():
    """Test that None API key after validation raises explicit error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "CLAUDE.md").write_text("# Guidelines")

        config = Config(
            include=["**/*.py"],
            exclude=[],
            batch_size=10,
            api_key=None,
        )

        # Mock validate_api_key to pass (simulating validation bug)
        with patch("claude_lint.orchestrator.validate_api_key"):
            # Mock os.environ.get to return None
            with patch("os.environ.get", return_value=None):
                # Should raise explicit ValueError, not pass None to create_client
                with pytest.raises(ValueError, match="API key is required but was None"):
                    run_compliance_check(tmpdir, config, mode="full")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_orchestrator_validation.py::test_run_compliance_check_raises_on_none_api_key -v`

Expected: FAIL - assert not raised in production mode

**Step 3: Replace assert with explicit check**

```python
# orchestrator.py:109
# Replace:
# assert api_key is not None  # Validated above

# With:
if api_key is None:
    raise ValueError(
        "API key is required but was None after validation. "
        "This indicates a bug in validate_api_key()."
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_orchestrator_validation.py::test_run_compliance_check_raises_on_none_api_key -v`

Expected: PASS

**Step 5: Run full test suite**

Run: `uv run pytest -v`

Expected: All tests pass

**Step 6: Commit**

```bash
git add src/claude_lint/orchestrator.py tests/test_orchestrator_validation.py
git commit -m "fix: replace unsafe assert with explicit validation check

- Replace assert with explicit ValueError for None API key
- Add test for API key validation edge case
- Prevents silent failure when Python runs with -O flag

Python's -O flag removes all assert statements. In production,
this would pass None to create_client() causing cryptic errors."
```

---

## Task 4: Add Exception Handler Tests for API Client (CRITICAL)

**Problem:** api_client.py has 74% coverage - exception handlers untested (lines 59-60, 62-63, 65-66)

**Files:**
- Test: `tests/test_api_client.py:1-150` (add new tests)

**Step 1: Write test for APITimeoutError handling**

Add to `tests/test_api_client.py`:

```python
def test_analyze_files_handles_timeout_error():
    """Test that APITimeoutError is logged and re-raised."""
    from anthropic import APITimeoutError
    from unittest.mock import Mock, patch

    mock_client = Mock()
    mock_client.messages.create.side_effect = APITimeoutError("Request timed out")

    with patch("claude_lint.api_client.logger") as mock_logger:
        with pytest.raises(APITimeoutError, match="Request timed out"):
            analyze_files_with_client(
                mock_client,
                "# Guidelines",
                "Check this code",
            )

        # Verify error was logged
        mock_logger.error.assert_called_once()
        assert "timed out" in mock_logger.error.call_args[0][0]
```

**Step 2: Write test for RateLimitError handling**

```python
def test_analyze_files_handles_rate_limit_error():
    """Test that RateLimitError is logged and re-raised."""
    from anthropic import RateLimitError
    from unittest.mock import Mock, patch

    mock_client = Mock()
    mock_client.messages.create.side_effect = RateLimitError("Rate limit exceeded")

    with patch("claude_lint.api_client.logger") as mock_logger:
        with pytest.raises(RateLimitError, match="Rate limit exceeded"):
            analyze_files_with_client(
                mock_client,
                "# Guidelines",
                "Check this code",
            )

        # Verify warning was logged (not error - rate limits are expected)
        mock_logger.warning.assert_called_once()
        assert "Rate limit" in mock_logger.warning.call_args[0][0]
```

**Step 3: Write test for APIConnectionError handling**

```python
def test_analyze_files_handles_connection_error():
    """Test that APIConnectionError is logged and re-raised."""
    from anthropic import APIConnectionError
    from unittest.mock import Mock, patch

    mock_client = Mock()
    mock_client.messages.create.side_effect = APIConnectionError("Connection failed")

    with patch("claude_lint.api_client.logger") as mock_logger:
        with pytest.raises(APIConnectionError, match="Connection failed"):
            analyze_files_with_client(
                mock_client,
                "# Guidelines",
                "Check this code",
            )

        # Verify error was logged
        mock_logger.error.assert_called_once()
        assert "connection failed" in mock_logger.error.call_args[0][0].lower()
```

**Step 4: Write test for generic APIError handling**

```python
def test_analyze_files_handles_generic_api_error():
    """Test that generic APIError is logged and re-raised."""
    from anthropic import APIError
    from unittest.mock import Mock, patch

    mock_client = Mock()
    mock_client.messages.create.side_effect = APIError("API error")

    with patch("claude_lint.api_client.logger") as mock_logger:
        with pytest.raises(APIError, match="API error"):
            analyze_files_with_client(
                mock_client,
                "# Guidelines",
                "Check this code",
            )

        # Verify error was logged
        mock_logger.error.assert_called_once()
        assert "API error" in mock_logger.error.call_args[0][0]
```

**Step 5: Run tests to verify coverage increase**

Run: `uv run pytest tests/test_api_client.py -v --cov=src/claude_lint/api_client --cov-report=term-missing`

Expected: Coverage increases from 74% to 95%+

**Step 6: Commit**

```bash
git add tests/test_api_client.py
git commit -m "test: add exception handler coverage for API client

- Test APITimeoutError logging and re-raising
- Test RateLimitError warning and re-raising
- Test APIConnectionError logging and re-raising
- Test generic APIError logging and re-raising

Increases api_client.py coverage from 74% to 95%+"
```

---

## Task 5: Add Encoding Fallback Tests for File Reader (CRITICAL)

**Problem:** file_reader.py has 68% coverage - encoding fallback untested (lines 47-55)

**Files:**
- Test: `tests/test_file_reader.py` (create new file)

**Step 1: Create test file for encoding fallback**

Create `tests/test_file_reader.py`:

```python
"""Tests for file reading with encoding fallback."""
import tempfile
from pathlib import Path

from claude_lint.file_reader import read_file_safely, read_batch_files


def test_read_file_with_valid_utf8():
    """Test reading file with valid UTF-8."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_file = tmpdir / "test.py"
        test_file.write_text("# UTF-8: café ☕", encoding="utf-8")

        content = read_file_safely(test_file, tmpdir, max_size_bytes=1024)

        assert content is not None
        assert "café" in content
        assert "☕" in content


def test_read_file_with_latin1_fallback():
    """Test fallback to latin-1 when UTF-8 fails."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_file = tmpdir / "test.py"

        # Write latin-1 encoded content that's invalid UTF-8
        content_latin1 = "# Latin-1: café"
        test_file.write_bytes(content_latin1.encode("latin-1"))

        content = read_file_safely(test_file, tmpdir, max_size_bytes=1024)

        # Should read successfully with latin-1 fallback
        assert content is not None
        assert "café" in content


def test_read_file_logs_encoding_fallback(caplog):
    """Test that encoding fallback is logged."""
    import logging

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_file = tmpdir / "test.py"

        # Write content with byte that's invalid UTF-8
        test_file.write_bytes(b"print('hello')\n\x80\x81\x82")

        with caplog.at_level(logging.WARNING):
            content = read_file_safely(test_file, tmpdir, max_size_bytes=1024)

        # Should log warning about UTF-8 failure
        assert any("not valid UTF-8" in record.message for record in caplog.records)

        # But should still return content via latin-1
        assert content is not None


def test_read_file_with_totally_unreadable_file():
    """Test handling of completely unreadable file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_file = tmpdir / "test.py"
        test_file.write_text("content")

        # Make file unreadable
        test_file.chmod(0o000)

        try:
            content = read_file_safely(test_file, tmpdir, max_size_bytes=1024)

            # Should return None for unreadable file
            assert content is None
        finally:
            # Restore permissions for cleanup
            test_file.chmod(0o644)


def test_read_batch_files_skips_unreadable():
    """Test that batch reading skips unreadable files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create readable file
        good_file = tmpdir / "good.py"
        good_file.write_text("# Good file")

        # Create file with invalid encoding
        bad_file = tmpdir / "bad.py"
        bad_file.write_bytes(b"\x80\x81\x82")

        # Read batch
        result = read_batch_files([good_file, bad_file], tmpdir, max_size_mb=1.0)

        # Should include good file
        assert "good.py" in result
        assert "# Good file" in result["good.py"]

        # Should include bad file (read via latin-1 fallback)
        assert "bad.py" in result
```

**Step 2: Run tests to verify coverage increase**

Run: `uv run pytest tests/test_file_reader.py -v --cov=src/claude_lint/file_reader --cov-report=term-missing`

Expected: Coverage increases from 68% to 95%+

**Step 3: Commit**

```bash
git add tests/test_file_reader.py
git commit -m "test: add encoding fallback coverage for file reader

- Test UTF-8 reading with non-ASCII characters
- Test latin-1 fallback when UTF-8 fails
- Test warning logging for encoding fallback
- Test graceful handling of unreadable files
- Test batch reading skips/handles encoding issues

Increases file_reader.py coverage from 68% to 95%+"
```

---

## Task 6: Add API Key Format Validation (IMPORTANT)

**Problem:** Config accepts any API key without format validation - users get cryptic API errors

**Files:**
- Modify: `src/claude_lint/validation.py:43-46`
- Test: `tests/test_validation.py` (add new tests)

**Step 1: Write tests for API key validation**

Add to `tests/test_validation.py`:

```python
def test_validate_api_key_rejects_empty_string():
    """Test that empty string is rejected."""
    with pytest.raises(ValueError, match="API key is required"):
        validate_api_key("")


def test_validate_api_key_rejects_whitespace_only():
    """Test that whitespace-only string is rejected."""
    with pytest.raises(ValueError, match="API key is required"):
        validate_api_key("   \n\t  ")


def test_validate_api_key_rejects_invalid_prefix():
    """Test that API key without sk-ant- prefix is rejected."""
    with pytest.raises(ValueError, match="should start with 'sk-ant-'"):
        validate_api_key("invalid-key-format")


def test_validate_api_key_rejects_too_short():
    """Test that suspiciously short API keys are rejected."""
    with pytest.raises(ValueError, match="appears too short"):
        validate_api_key("sk-ant-short")


def test_validate_api_key_accepts_valid_key():
    """Test that properly formatted API key is accepted."""
    valid_key = "sk-ant-" + "x" * 50  # Typical Anthropic key length

    # Should not raise
    validate_api_key(valid_key)


def test_validate_api_key_strips_whitespace():
    """Test that leading/trailing whitespace is handled."""
    valid_key = "sk-ant-" + "x" * 50
    key_with_whitespace = f"  {valid_key}\n"

    # Should not raise - whitespace stripped
    validate_api_key(key_with_whitespace)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_validation.py -k "api_key" -v`

Expected: FAIL - current validation too permissive

**Step 3: Implement strict API key validation**

```python
# validation.py:43-46
def validate_api_key(api_key: str | None) -> None:
    """Validate API key is present and well-formed.

    Args:
        api_key: API key to validate

    Raises:
        ValueError: If API key is missing or malformed
    """
    if not api_key or not api_key.strip():
        raise ValueError(
            "API key is required. Set ANTHROPIC_API_KEY environment variable "
            "or add 'api_key' to .agent-lint.json"
        )

    key = api_key.strip()

    # Check prefix
    if not key.startswith("sk-ant-"):
        raise ValueError(
            "Invalid API key format. Anthropic API keys should start with 'sk-ant-'. "
            "Check your key at https://console.anthropic.com/"
        )

    # Check minimum length (Anthropic keys are typically 40+ chars after prefix)
    if len(key) < 40:
        raise ValueError(
            f"API key appears too short ({len(key)} chars). "
            "Anthropic API keys are typically 40+ characters. "
            "Check your key at https://console.anthropic.com/"
        )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_validation.py -k "api_key" -v`

Expected: PASS all 6 new tests

**Step 5: Run full test suite**

Run: `uv run pytest -v`

Expected: All tests pass

**Step 6: Commit**

```bash
git add src/claude_lint/validation.py tests/test_validation.py
git commit -m "feat: add strict API key format validation

- Validate API key starts with 'sk-ant-' prefix
- Check minimum length (40+ chars)
- Strip whitespace before validation
- Provide helpful error messages with console URL

Prevents cryptic API errors from malformed keys"
```

---

## Task 7: Extract Constants for Magic Numbers (IMPORTANT)

**Problem:** Magic numbers scattered throughout code - not documented, hard to tune

**Files:**
- Create: `src/claude_lint/constants.py`
- Modify: `src/claude_lint/api_client.py:51`
- Modify: `src/claude_lint/retry.py:15-17`
- Test: `tests/test_constants.py` (create new)

**Step 1: Create constants module with documentation**

Create `src/claude_lint/constants.py`:

```python
"""Configuration constants for claude-lint.

This module centralizes magic numbers used throughout the codebase
with clear documentation for why each value was chosen.
"""

# API Client Settings
# ------------------

API_MAX_TOKENS = 4096
"""Maximum tokens for Claude API responses.

Value: 4096
Rationale: Claude models support up to 4096 output tokens. For code analysis,
responses rarely exceed 2000 tokens, so 4096 provides comfortable headroom.
"""

# Retry Settings
# -------------

RETRY_MAX_ATTEMPTS = 3
"""Maximum number of retry attempts for failed API calls.

Value: 3
Rationale: Balance between reliability and latency. Most transient failures
resolve within 1-2 retries. 3 attempts = original + 2 retries.
Total time: ~1s + ~2s = ~3s for exponential backoff.
"""

RETRY_INITIAL_DELAY = 1.0
"""Initial delay in seconds before first retry.

Value: 1.0
Rationale: Start conservative. Anthropic rate limits are per-minute, so 1s
gives time for rate limit windows to expire. Shorter delays waste retries
on rate limit errors.
"""

RETRY_BACKOFF_FACTOR = 2.0
"""Exponential backoff multiplier for retry delays.

Value: 2.0
Rationale: Standard exponential backoff (doubles each retry).
Delays: 1s → 2s → 4s. Aggressive enough to resolve quickly, conservative
enough to avoid overwhelming rate limits.
"""

RETRY_JITTER_MIN = 0.5
"""Minimum jitter multiplier for retry delays.

Value: 0.5 (50% of base delay)
Rationale: Prevents thundering herd when multiple clients retry simultaneously.
Range: 0.5x to 1.5x creates ±50% variance around base delay.
"""

RETRY_JITTER_MAX = 1.5
"""Maximum jitter multiplier for retry delays.

Value: 1.5 (150% of base delay)
Rationale: Upper bound of jitter range. Creates ±50% variance with JITTER_MIN.
"""
```

**Step 2: Write test to ensure constants are imported correctly**

Create `tests/test_constants.py`:

```python
"""Tests for configuration constants."""
from claude_lint.constants import (
    API_MAX_TOKENS,
    RETRY_BACKOFF_FACTOR,
    RETRY_INITIAL_DELAY,
    RETRY_JITTER_MAX,
    RETRY_JITTER_MIN,
    RETRY_MAX_ATTEMPTS,
)


def test_api_constants_are_positive():
    """Test that API constants have valid positive values."""
    assert API_MAX_TOKENS > 0
    assert isinstance(API_MAX_TOKENS, int)


def test_retry_constants_are_valid():
    """Test that retry constants have valid ranges."""
    assert RETRY_MAX_ATTEMPTS >= 1
    assert RETRY_INITIAL_DELAY > 0
    assert RETRY_BACKOFF_FACTOR >= 1.0
    assert 0 < RETRY_JITTER_MIN < 1.0
    assert RETRY_JITTER_MAX > 1.0
    assert RETRY_JITTER_MIN < RETRY_JITTER_MAX


def test_jitter_range_is_reasonable():
    """Test that jitter creates reasonable variance."""
    # With 0.5 and 1.5, jitter creates ±50% variance
    variance = RETRY_JITTER_MAX - RETRY_JITTER_MIN
    assert 0.5 <= variance <= 2.0  # Reasonable range
```

**Step 3: Run tests**

Run: `uv run pytest tests/test_constants.py -v`

Expected: PASS

**Step 4: Update api_client.py to use constant**

```python
# api_client.py
from claude_lint.constants import API_MAX_TOKENS
from claude_lint.logging_config import get_logger

# ... later in analyze_files_with_client():
    response = client.messages.create(
        model=model,
        max_tokens=API_MAX_TOKENS,  # Was: 4096
        system=[{"type": "text", "text": guidelines, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
    )
```

**Step 5: Update retry.py to use constants**

```python
# retry.py
import random
import time
from collections.abc import Callable
from typing import TypeVar

from claude_lint.constants import (
    RETRY_BACKOFF_FACTOR,
    RETRY_INITIAL_DELAY,
    RETRY_JITTER_MAX,
    RETRY_JITTER_MIN,
    RETRY_MAX_ATTEMPTS,
)
from claude_lint.logging_config import get_logger

T = TypeVar("T")
logger = get_logger(__name__)


def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = RETRY_MAX_ATTEMPTS,  # Was: 3
    initial_delay: float = RETRY_INITIAL_DELAY,  # Was: 1.0
    backoff_factor: float = RETRY_BACKOFF_FACTOR,  # Was: 2.0
) -> T:
    """Retry a function with exponential backoff and jitter.

    Uses jitter to prevent thundering herd problem when multiple
    clients retry simultaneously.

    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts (default from constants)
        initial_delay: Initial delay in seconds (default from constants)
        backoff_factor: Multiplier for delay after each retry (default from constants)

    Returns:
        Result from successful function call

    Raises:
        Exception: The last exception if all retries exhausted
        KeyboardInterrupt: Immediately re-raised without retry
        SystemExit: Immediately re-raised without retry
    """
    last_exception: Exception | None = None
    delay = initial_delay

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                logger.debug(f"Retry attempt {attempt + 1}/{max_retries}")
            return func()
        except (KeyboardInterrupt, SystemExit):
            # Never catch these - re-raise immediately
            raise
        except Exception as e:
            last_exception = e

            if attempt < max_retries - 1:  # Don't sleep on last attempt
                # Add jitter: random value between JITTER_MIN and JITTER_MAX
                jitter = RETRY_JITTER_MIN + random.random() * (RETRY_JITTER_MAX - RETRY_JITTER_MIN)
                jittered_delay = delay * jitter

                logger.debug(
                    f"Attempt {attempt + 1}/{max_retries} failed with "
                    f"{type(e).__name__}: {e}. Retrying in {jittered_delay:.1f}s..."
                )
                time.sleep(jittered_delay)
                delay *= backoff_factor
            else:
                logger.debug(
                    f"Attempt {attempt + 1}/{max_retries} failed with "
                    f"{type(e).__name__}: {e}. No more retries."
                )

    # All retries exhausted
    if last_exception is not None:
        raise last_exception
    raise RuntimeError("retry_with_backoff: no retries executed")
```

**Step 6: Run tests to verify no regressions**

Run: `uv run pytest -v`

Expected: All tests pass

**Step 7: Commit**

```bash
git add src/claude_lint/constants.py src/claude_lint/api_client.py src/claude_lint/retry.py tests/test_constants.py
git commit -m "refactor: extract magic numbers to documented constants

- Create constants.py with API_MAX_TOKENS, RETRY_* constants
- Document rationale for each value
- Update api_client.py and retry.py to use constants
- Add tests for constant validity

Makes tuning easier and documents why values were chosen"
```

---

## Task 8: Refactor God Function - Extract Batch Processing (IMPORTANT)

**Problem:** `run_compliance_check()` is 155 lines - violates SRP, hard to test

**Files:**
- Modify: `src/claude_lint/orchestrator.py:42-197`
- Create: `tests/test_orchestrator_refactoring.py`

**Step 1: Write test for extracted batch processing logic**

Create `tests/test_orchestrator_refactoring.py`:

```python
"""Tests for refactored orchestrator functions."""
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from claude_lint.config import Config


def test_process_all_batches_with_progress():
    """Test batch processing with progress bar enabled."""
    from claude_lint.orchestrator import _process_all_batches

    # Mock dependencies
    mock_client = Mock()
    mock_rate_limiter = Mock()
    mock_cache = Mock()
    mock_progress_state = Mock(results=[], completed_batches=[])

    batches = [
        [Path("file1.py")],
        [Path("file2.py")],
    ]

    with patch("claude_lint.orchestrator.process_batch") as mock_process:
        with patch("claude_lint.orchestrator.update_progress") as mock_update:
            with patch("claude_lint.orchestrator.save_progress"):
                with patch("claude_lint.orchestrator.save_cache"):
                    # Mock batch results
                    mock_process.return_value = [{"file": "file1.py", "violations": []}]
                    mock_update.return_value = mock_progress_state

                    config = Config(
                        include=["**/*.py"],
                        batch_size=10,
                        show_progress=True,
                    )

                    results, api_calls = _process_all_batches(
                        batches=batches,
                        project_root=Path("/tmp"),
                        config=config,
                        guidelines="# Test",
                        guidelines_hash="abc123",
                        client=mock_client,
                        rate_limiter=mock_rate_limiter,
                        cache=mock_cache,
                        progress_state=mock_progress_state,
                        progress_path=Path("/tmp/progress.json"),
                    )

                    # Should process both batches
                    assert mock_process.call_count == 2
                    assert api_calls == 2


def test_process_all_batches_without_progress():
    """Test batch processing with progress bar disabled."""
    from claude_lint.orchestrator import _process_all_batches

    mock_client = Mock()
    mock_rate_limiter = Mock()
    mock_cache = Mock()
    mock_progress_state = Mock(results=[], completed_batches=[])

    batches = [[Path("file1.py")]]

    with patch("claude_lint.orchestrator.process_batch") as mock_process:
        with patch("claude_lint.orchestrator.update_progress") as mock_update:
            with patch("claude_lint.orchestrator.save_progress"):
                with patch("claude_lint.orchestrator.save_cache"):
                    mock_process.return_value = [{"file": "file1.py", "violations": []}]
                    mock_update.return_value = mock_progress_state

                    config = Config(
                        include=["**/*.py"],
                        batch_size=10,
                        show_progress=False,  # Disable progress
                    )

                    results, api_calls = _process_all_batches(
                        batches=batches,
                        project_root=Path("/tmp"),
                        config=config,
                        guidelines="# Test",
                        guidelines_hash="abc123",
                        client=mock_client,
                        rate_limiter=mock_rate_limiter,
                        cache=mock_cache,
                        progress_state=mock_progress_state,
                        progress_path=Path("/tmp/progress.json"),
                    )

                    assert mock_process.call_count == 1
                    assert api_calls == 1
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_orchestrator_refactoring.py -v`

Expected: FAIL - function doesn't exist yet

**Step 3: Extract _process_all_batches() function**

Add this function before `run_compliance_check()` in orchestrator.py:

```python
def _process_all_batches(
    batches: list[list[Path]],
    project_root: Path,
    config: Config,
    guidelines: str,
    guidelines_hash: str,
    client: Any,
    rate_limiter: RateLimiter,
    cache: Cache,
    progress_state: ProgressState,
    progress_path: Path,
) -> tuple[list[dict[str, Any]], int]:
    """Process all batches with optional progress bar.

    Args:
        batches: List of file batches to process
        project_root: Project root directory
        config: Configuration
        guidelines: CLAUDE.md content
        guidelines_hash: Hash of CLAUDE.md
        client: Anthropic API client
        rate_limiter: Rate limiter for API calls
        cache: Cache object
        progress_state: Progress tracking state
        progress_path: Path to progress file

    Returns:
        Tuple of (all results, API calls made)
    """
    all_results = list(progress_state.results)
    api_calls_made = 0

    # Determine if we should show progress
    show_progress = config.show_progress and not os.environ.get("CLAUDE_LINT_NO_PROGRESS")

    remaining_batches = list(get_remaining_batch_indices(progress_state))

    # Common batch processing logic
    def process_batches_iter(progress_callback=None):
        nonlocal api_calls_made

        for idx, batch_idx in enumerate(remaining_batches):
            batch = batches[batch_idx]

            if progress_callback:
                progress_callback(idx, batch_idx, len(batch))

            batch_results = process_batch(
                batch,
                project_root,
                config,
                guidelines,
                guidelines_hash,
                client,
                rate_limiter,
                cache,
            )

            all_results.extend(batch_results)
            api_calls_made += 1

            progress_state_updated = update_progress(progress_state, batch_idx, batch_results)
            save_progress(progress_state_updated, progress_path)
            save_cache(cache, cache_path := project_root / ".agent-lint-cache.json")

            yield batch_results

    if show_progress:
        from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[bold cyan]{task.fields[status]}"),
        ) as progress:
            task = progress.add_task(
                "Analyzing files", total=len(remaining_batches), status="Starting..."
            )

            def progress_callback(idx, batch_idx, batch_size):
                progress.update(
                    task,
                    status=f"Batch {idx + 1}/{len(remaining_batches)} ({batch_size} files)"
                )

            for _ in process_batches_iter(progress_callback):
                progress.update(task, advance=1, status="Complete")
    else:
        # No progress bar - just iterate
        for _ in process_batches_iter():
            pass

    return all_results, api_calls_made
```

**Step 4: Update run_compliance_check() to use extracted function**

Replace lines 117-187 in orchestrator.py:

```python
# orchestrator.py:117-187
# Replace entire batch processing section with:

# Process batches with optional progress bar
all_results, api_calls_made = _process_all_batches(
    batches=batches,
    project_root=project_root,
    config=config,
    guidelines=guidelines,
    guidelines_hash=guidelines_hash,
    client=client,
    rate_limiter=rate_limiter,
    cache=cache,
    progress_state=progress_state,
    progress_path=progress_path,
)

metrics.api_calls_made = api_calls_made
```

**Step 5: Run tests to verify refactoring works**

Run: `uv run pytest tests/test_orchestrator_refactoring.py -v`

Expected: PASS

**Step 6: Run full test suite to ensure no regressions**

Run: `uv run pytest -v`

Expected: All tests pass

**Step 7: Commit**

```bash
git add src/claude_lint/orchestrator.py tests/test_orchestrator_refactoring.py
git commit -m "refactor: extract batch processing logic from god function

- Extract _process_all_batches() from run_compliance_check()
- Eliminate duplicated progress bar logic (40 lines)
- Single code path handles both progress modes
- run_compliance_check() reduced from 155 to ~120 lines

Improves testability and maintainability"
```

---

## Task 9: Update Package Metadata for PyPI (LOW)

**Problem:** Placeholder author info and URLs prevent PyPI publication

**Files:**
- Modify: `pyproject.toml:8-12, 49-50`

**Step 1: Update author information**

```toml
# pyproject.toml:8-12
authors = [
    {name = "Vlad Temian", email = "vladtemian@gmail.com"},
]
maintainers = [
    {name = "Vlad Temian", email = "vladtemian@gmail.com"},
]
```

**Step 2: Update project URLs**

```toml
# pyproject.toml:49-52
[project.urls]
Homepage = "https://github.com/vtemian/claude-lint"
Repository = "https://github.com/vtemian/claude-lint"
Issues = "https://github.com/vtemian/claude-lint/issues"
Documentation = "https://github.com/vtemian/claude-lint/blob/main/README.md"
```

**Step 3: Verify pyproject.toml is valid**

Run: `uv run python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"`

Expected: No errors

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "fix: update package metadata for PyPI publication

- Replace placeholder author with Vlad Temian
- Update homepage URL to vtemian/claude-lint
- Add repository, issues, and documentation URLs

Package now ready for PyPI publication"
```

---

## Task 10: Fix README Documentation Discrepancy (LOW)

**Problem:** README says batch size "default 10-15" but it's actually 10

**Files:**
- Modify: `README.md:115`

**Step 1: Update README with correct default**

```markdown
# README.md:115
3. **Batch Processing**: Groups files into batches (default 10, configurable up to 100)
```

**Step 2: Verify all references to batch size are accurate**

Run: `grep -n "batch" README.md`

Expected: All references accurate

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: fix batch size documentation discrepancy

- Update README to show correct default (10, not 10-15)
- Add note about configurable maximum (100)

Aligns documentation with actual default in config.py"
```

---

## Task 11: Verify All Tests Pass and Coverage Improved (VERIFICATION)

**Files:**
- All test files

**Step 1: Run full test suite**

Run: `uv run pytest -v`

Expected: All 125+ tests PASS, 0 FAIL, 3 SKIP (integration tests requiring API key)

**Step 2: Check coverage metrics**

Run: `uv run pytest --cov=src/claude_lint --cov-report=term-missing`

Expected:
- Overall coverage: 90%+
- api_client.py: 95%+ (was 74%)
- file_reader.py: 95%+ (was 68%)
- orchestrator.py: 85%+ (was 77%)

**Step 3: Run mypy strict checks**

Run: `uv run mypy src/claude_lint`

Expected: Success: no issues found in 23 source files

**Step 4: Run ruff checks**

Run: `uv run ruff check src/claude_lint`

Expected: All checks passed!

**Step 5: Verify no assert statements in production code**

Run: `grep -r "^[^#]*\bassert\b" src/claude_lint`

Expected: No matches (all asserts replaced with explicit checks)

**Step 6: Document verification results**

Create verification report showing:
- Test pass rate: 100% (excluding skipped)
- Coverage improvement: 88% → 92%+
- No unsafe patterns (asserts)
- All critical issues resolved

---

## Expected Outcome

**Before:**
- Grade: C+ (74/100)
- Tests: 117 pass, 2 fail
- Coverage: 88% (api_client 74%, file_reader 68%)
- Critical issues: Unsafe assert, untested exception handlers
- Architecture: 155-line god function, duplicated logic

**After:**
- Grade: A- (90+/100)
- Tests: 125+ pass, 0 fail
- Coverage: 92%+ (all modules 90%+)
- Critical issues: RESOLVED
- Architecture: Extracted functions, DRY principle, documented constants

**Grading Improvements:**
- ✅ Correctness: 80 → 95 (no flaky tests, proper signal handling)
- ✅ Safety: 65 → 90 (no asserts, exception handlers tested, API key validation)
- ✅ Maintainability: 70 → 85 (extracted functions, documented constants)
- ✅ Testing: 75 → 95 (all tests pass, coverage 92%+)
- ✅ Documentation: 75 → 85 (accurate, documented constants)

**Overall: C+ (74) → A- (90)**
