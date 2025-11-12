# Production Readiness Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix critical and major production readiness issues to bring codebase from C+ (72/100) to A- (90/100)

**Architecture:** Systematic fixes to thread safety, input validation, error handling, type safety, and configuration. Focus on defensive programming and observability.

**Tech Stack:** Python 3.11+, threading, pydantic, mypy strict mode, pre-commit hooks

---

## Task 1: Add Thread Safety to RateLimiter

**Files:**
- Modify: `src/claude_lint/rate_limiter.py:1-67`
- Test: `tests/test_rate_limiter.py`

**Step 1: Write thread safety test**

```python
# Add to tests/test_rate_limiter.py
import threading
from claude_lint.rate_limiter import RateLimiter


def test_rate_limiter_thread_safety():
    """Test that rate limiter works correctly with concurrent access."""
    limiter = RateLimiter(max_requests=10, window_seconds=1.0)
    successful_acquires = []

    def worker():
        for _ in range(5):
            limiter.acquire()
            successful_acquires.append(1)

    # Start 5 threads, each making 5 requests = 25 total
    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All 25 requests should succeed
    assert len(successful_acquires) == 25


def test_try_acquire_non_blocking():
    """Test try_acquire doesn't block and returns False when at limit."""
    limiter = RateLimiter(max_requests=2, window_seconds=1.0)

    # First two should succeed
    assert limiter.try_acquire() is True
    assert limiter.try_acquire() is True

    # Third should fail without blocking
    assert limiter.try_acquire() is False

    # Wait for window to expire
    import time
    time.sleep(1.1)

    # Should succeed again
    assert limiter.try_acquire() is True
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_rate_limiter.py::test_rate_limiter_thread_safety -v`

Expected: FAIL (race conditions possible)

**Step 3: Add thread safety to RateLimiter**

```python
# src/claude_lint/rate_limiter.py
"""Rate limiting for API calls."""
import time
import threading
from collections import deque


class RateLimiter:
    """Thread-safe rate limiter with sliding window.

    Limits the number of requests within a time window using a sliding
    window algorithm for accurate rate limiting.

    Thread-safe: All operations are protected by a lock.
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
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Acquire a rate limit token, blocking if necessary.

        This method blocks until a token is available within the rate limit.
        Uses a sliding window to track requests.

        Thread-safe: Uses lock to prevent race conditions.
        """
        with self._lock:
            now = time.time()

            # Remove requests outside the current window
            while self.requests and self.requests[0] < now - self.window_seconds:
                self.requests.popleft()

            # If at limit, wait until oldest request expires
            if len(self.requests) >= self.max_requests:
                sleep_time = self.requests[0] + self.window_seconds - now
                if sleep_time > 0:
                    # Release lock while sleeping to allow other threads
                    self._lock.release()
                    try:
                        time.sleep(sleep_time)
                    finally:
                        self._lock.acquire()

                    # Re-check time after waking
                    now = time.time()
                    while self.requests and self.requests[0] < now - self.window_seconds:
                        self.requests.popleft()

                # Remove expired request if still at limit
                if len(self.requests) >= self.max_requests:
                    self.requests.popleft()

            # Record this request
            self.requests.append(time.time())

    def try_acquire(self) -> bool:
        """Try to acquire a token without blocking.

        Returns:
            True if token acquired, False if at rate limit

        Thread-safe: Uses lock to prevent race conditions.
        """
        with self._lock:
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

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_rate_limiter.py -v`

Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/claude_lint/rate_limiter.py tests/test_rate_limiter.py
git commit -m "fix: add thread safety to RateLimiter with lock protection"
```

---

## Task 2: Add Config Validation with Pydantic

**Files:**
- Modify: `src/claude_lint/config.py:1-90`
- Modify: `pyproject.toml:6-10`
- Create: `tests/test_config_validation.py`

**Step 1: Add pydantic dependency**

```toml
# pyproject.toml - update dependencies section
dependencies = [
    "anthropic>=0.18.0,<0.50",
    "click>=8.1.0",
    "rich>=13.0.0",
    "pydantic>=2.0.0,<3.0",
]
```

Run: `uv sync`

**Step 2: Write validation tests**

```python
# tests/test_config_validation.py
"""Tests for config validation."""
import pytest
from pydantic import ValidationError
from claude_lint.config import Config


def test_config_rejects_negative_batch_size():
    """Test that negative batch size raises error."""
    with pytest.raises(ValidationError, match="batch_size.*greater than 0"):
        Config(
            include=["**/*.py"],
            exclude=[],
            batch_size=-1
        )


def test_config_rejects_zero_batch_size():
    """Test that zero batch size raises error."""
    with pytest.raises(ValidationError, match="batch_size.*greater than 0"):
        Config(
            include=["**/*.py"],
            exclude=[],
            batch_size=0
        )


def test_config_rejects_negative_max_file_size():
    """Test that negative file size raises error."""
    with pytest.raises(ValidationError, match="max_file_size_mb.*greater than 0"):
        Config(
            include=["**/*.py"],
            exclude=[],
            batch_size=10,
            max_file_size_mb=-1.0
        )


def test_config_rejects_zero_api_rate_limit():
    """Test that zero rate limit raises error."""
    with pytest.raises(ValidationError, match="api_rate_limit.*greater than 0"):
        Config(
            include=["**/*.py"],
            exclude=[],
            batch_size=10,
            api_rate_limit=0
        )


def test_config_rejects_negative_api_timeout():
    """Test that negative timeout raises error."""
    with pytest.raises(ValidationError, match="api_timeout_seconds.*greater than 0"):
        Config(
            include=["**/*.py"],
            exclude=[],
            batch_size=10,
            api_timeout_seconds=-1.0
        )


def test_config_rejects_empty_include():
    """Test that empty include list raises error."""
    with pytest.raises(ValidationError, match="include.*at least 1 item"):
        Config(
            include=[],
            exclude=[],
            batch_size=10
        )


def test_config_accepts_valid_values():
    """Test that valid config is accepted."""
    config = Config(
        include=["**/*.py"],
        exclude=["tests/**"],
        batch_size=10,
        max_file_size_mb=2.0,
        api_rate_limit=5,
        api_timeout_seconds=120.0
    )
    assert config.batch_size == 10
    assert config.max_file_size_mb == 2.0
```

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_config_validation.py -v`

Expected: FAIL (Config doesn't validate yet)

**Step 4: Convert Config to Pydantic model**

```python
# src/claude_lint/config.py
"""Configuration management for lint-claude."""
import json
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class Config(BaseModel):
    """Configuration for lint-claude with validation."""

    include: list[str] = Field(min_length=1, description="File patterns to include")
    exclude: list[str] = Field(default_factory=list, description="File patterns to exclude")
    batch_size: int = Field(gt=0, le=100, description="Number of files per batch")
    model: str = Field(default="claude-sonnet-4-5-20250929", description="Claude model to use")
    max_file_size_mb: float = Field(default=1.0, gt=0, le=10, description="Maximum file size in MB")
    api_timeout_seconds: float = Field(default=60.0, gt=0, le=600, description="API timeout in seconds")
    api_rate_limit: int = Field(default=4, gt=0, le=50, description="Requests per second")
    api_rate_window_seconds: float = Field(default=1.0, gt=0, description="Rate limit window in seconds")
    show_progress: bool = Field(default=True, description="Show progress bars")
    api_key: Optional[str] = Field(default=None, description="Anthropic API key")

    @field_validator('include')
    @classmethod
    def validate_include_patterns(cls, v: list[str]) -> list[str]:
        """Ensure include patterns are non-empty strings."""
        if not v:
            raise ValueError("include must contain at least one pattern")
        for pattern in v:
            if not pattern.strip():
                raise ValueError("include patterns cannot be empty strings")
        return v

    class Config:
        frozen = False  # Allow modification for testing


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
        api_rate_limit=4,
        api_rate_window_seconds=1.0,
        show_progress=True,
        api_key=None
    )


def load_config(config_path: Path) -> Config:
    """Load configuration from file or return defaults.

    Supports both snake_case (preferred) and camelCase (backwards compat) keys.

    Args:
        config_path: Path to .lint-claude.json file

    Returns:
        Config object with loaded or default values

    Raises:
        ValueError: If configuration values are invalid
    """
    if not config_path.exists():
        return get_default_config()

    with open(config_path, encoding='utf-8') as f:
        data = json.load(f)

    defaults = get_default_config()

    # Build config dict with snake_case/camelCase support
    config_data = {
        "include": data.get("include", defaults.include),
        "exclude": data.get("exclude", defaults.exclude),
        "batch_size": data.get("batch_size", data.get("batchSize", defaults.batch_size)),
        "model": data.get("model", defaults.model),
        "max_file_size_mb": data.get(
            "max_file_size_mb",
            data.get("maxFileSizeMb", defaults.max_file_size_mb)
        ),
        "api_timeout_seconds": data.get(
            "api_timeout_seconds",
            data.get("apiTimeoutSeconds", defaults.api_timeout_seconds)
        ),
        "api_rate_limit": data.get(
            "api_rate_limit",
            data.get("apiRateLimit", defaults.api_rate_limit)
        ),
        "api_rate_window_seconds": data.get(
            "api_rate_window_seconds",
            data.get("apiRateWindowSeconds", defaults.api_rate_window_seconds)
        ),
        "show_progress": data.get(
            "show_progress",
            data.get("showProgress", defaults.show_progress)
        ),
        "api_key": data.get("api_key", data.get("apiKey"))
    }

    return Config(**config_data)
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_config_validation.py tests/test_config.py -v`

Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/claude_lint/config.py tests/test_config_validation.py pyproject.toml
git commit -m "feat: add pydantic validation to Config with bounds checking"
```

---

## Task 3: Fix Silent Exception Swallowing with Logging

**Files:**
- Modify: `src/claude_lint/batch_processor.py:72-87`
- Modify: `tests/test_batch_processor.py`

**Step 1: Write test for cache update failure logging**

```python
# Add to tests/test_batch_processor.py
import logging
from unittest.mock import patch, Mock
from claude_lint.batch_processor import process_batch


def test_process_batch_logs_cache_update_failures(tmp_path, caplog):
    """Test that cache update failures are logged."""
    from pathlib import Path
    from claude_lint.config import Config
    from claude_lint.cache import Cache
    from claude_lint.rate_limiter import RateLimiter

    # Create test file
    test_file = tmp_path / "test.py"
    test_file.write_text("print('test')")

    config = Config(
        include=["**/*.py"],
        exclude=[],
        batch_size=10
    )

    cache = Cache(claude_md_hash="test", entries={})
    rate_limiter = RateLimiter(max_requests=10, window_seconds=1.0)
    mock_client = Mock()

    with patch('claude_lint.batch_processor.analyze_files_with_client') as mock_api:
        mock_api.return_value = (
            '{"results": [{"file": "test.py", "violations": []}]}',
            Mock()
        )

        # Make compute_file_hash fail
        with patch('claude_lint.batch_processor.compute_file_hash') as mock_hash:
            mock_hash.side_effect = PermissionError("Access denied")

            with caplog.at_level(logging.WARNING):
                results = process_batch(
                    [test_file],
                    tmp_path,
                    config,
                    "# Guidelines",
                    "hash123",
                    mock_client,
                    rate_limiter,
                    cache
                )

    # Should still return results
    assert len(results) == 1

    # Should log the error
    assert any("Failed to update cache" in record.message for record in caplog.records)
    assert any("PermissionError" in record.message for record in caplog.records)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_batch_processor.py::test_process_batch_logs_cache_update_failures -v`

Expected: FAIL (no logging in exception handler)

**Step 3: Add logging to batch_processor**

```python
# src/claude_lint/batch_processor.py
"""Batch processing logic."""
from pathlib import Path
from typing import Any
from claude_lint.api_client import analyze_files_with_client
from claude_lint.cache import Cache, CacheEntry
from claude_lint.collector import compute_file_hash
from claude_lint.config import Config
from claude_lint.file_reader import read_batch_files
from claude_lint.logging_config import get_logger
from claude_lint.processor import build_xml_prompt, parse_response
from claude_lint.rate_limiter import RateLimiter
from claude_lint.retry import retry_with_backoff
from claude_lint.types import FileResult

logger = get_logger(__name__)


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
        except FileNotFoundError as e:
            # File was deleted between analysis and caching
            logger.debug(
                f"File {result['file']} not found during cache update, "
                f"likely deleted: {e}"
            )
        except (PermissionError, OSError) as e:
            # Permission or filesystem error
            logger.warning(
                f"Failed to update cache for {result['file']}: "
                f"{type(e).__name__}: {e}"
            )
        except Exception as e:
            # Unexpected error - log with full traceback
            logger.error(
                f"Unexpected error updating cache for {result['file']}: "
                f"{type(e).__name__}: {e}",
                exc_info=True
            )

    return batch_results_dict
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_batch_processor.py -v`

Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/claude_lint/batch_processor.py tests/test_batch_processor.py
git commit -m "fix: add specific exception handling and logging to cache updates"
```

---

## Task 4: Add UTF-8 Encoding to All File Operations

**Files:**
- Modify: `src/claude_lint/cache.py:39`
- Modify: `src/claude_lint/config.py` (already done in Task 2)

**Step 1: Write cross-platform encoding test**

```python
# Add to tests/test_cache.py
def test_cache_handles_non_ascii_content(tmp_path):
    """Test that cache handles files with non-ASCII characters."""
    from claude_lint.cache import Cache, CacheEntry, save_cache, load_cache

    cache = Cache(
        claude_md_hash="hash123",
        entries={
            "café.py": CacheEntry(
                file_hash="abc123",
                claude_md_hash="hash123",
                violations=[{
                    "type": "style",
                    "message": "Use café naming convention ☕",
                    "line": 1
                }],
                timestamp=1234567890
            )
        }
    )

    cache_path = tmp_path / ".cache.json"
    save_cache(cache, cache_path)

    # Reload and verify
    loaded = load_cache(cache_path)
    assert "café.py" in loaded.entries
    assert "café naming convention ☕" in loaded.entries["café.py"].violations[0]["message"]
```

**Step 2: Run test to verify current behavior**

Run: `uv run pytest tests/test_cache.py::test_cache_handles_non_ascii_content -v`

Expected: May fail on some platforms without explicit encoding

**Step 3: Add UTF-8 encoding to cache.py**

```python
# src/claude_lint/cache.py - update load_cache function
def load_cache(cache_path: Path) -> Cache:
    """Load cache from file or return empty cache.

    Args:
        cache_path: Path to cache file

    Returns:
        Cache object
    """
    if not cache_path.exists():
        return Cache(claude_md_hash="", entries={})

    try:
        with open(cache_path, encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return Cache(claude_md_hash="", entries={})

    entries = {}
    for file_path, entry_data in data.get("entries", {}).items():
        entries[file_path] = CacheEntry(**entry_data)

    return Cache(
        claude_md_hash=data.get("claudeMdHash", ""),
        entries=entries
    )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cache.py -v`

Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/claude_lint/cache.py tests/test_cache.py
git commit -m "fix: add explicit UTF-8 encoding to cache file operations"
```

---

## Task 5: Remove Useless Exception Re-raising

**Files:**
- Modify: `src/claude_lint/api_client.py:65-70`

**Step 1: Add logging for API errors**

```python
# src/claude_lint/api_client.py
"""Claude API client with prompt caching support."""
from typing import Any
from anthropic import Anthropic, APIError, APIConnectionError, RateLimitError, APITimeoutError
from claude_lint.logging_config import get_logger

logger = get_logger(__name__)


def create_client(api_key: str, timeout: float = 60.0) -> Anthropic:
    """Create Anthropic client with timeout.

    Args:
        api_key: Anthropic API key
        timeout: Request timeout in seconds (default: 60)

    Returns:
        Anthropic client instance configured with timeout
    """
    return Anthropic(api_key=api_key, timeout=timeout)


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
        APIError: If API call fails (includes connection, rate limit, timeout, etc.)
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
    except APITimeoutError as e:
        logger.error(f"API request timed out: {e}")
        raise
    except RateLimitError as e:
        logger.warning(f"Rate limit exceeded: {e}")
        raise
    except APIConnectionError as e:
        logger.error(f"API connection failed: {e}")
        raise
    except APIError as e:
        logger.error(f"API error: {e}")
        raise
    except (KeyboardInterrupt, SystemExit):
        # Never catch these
        raise

    # Validate response
    if not response.content:
        raise ValueError("API returned empty response content")

    # Extract text from first content block (must be TextBlock)
    first_block = response.content[0]
    if not hasattr(first_block, 'text'):
        raise ValueError(
            f"API returned non-text content (type: {type(first_block).__name__})"
        )
    return first_block.text, response


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


def get_usage_stats(response: Any) -> dict:
    """Get usage statistics from API response.

    Args:
        response: Response object from Claude API

    Returns:
        Dict with token usage stats
    """
    usage = response.usage
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_creation_tokens": getattr(usage, "cache_creation_input_tokens", 0),
        "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0)
    }
```

**Step 2: Run tests to verify nothing breaks**

Run: `uv run pytest tests/test_api_client.py -v`

Expected: ALL PASS

**Step 3: Commit**

```bash
git add src/claude_lint/api_client.py
git commit -m "refactor: add logging to API errors instead of useless re-raising"
```

---

## Task 6: Add Strict Mypy Configuration

**Files:**
- Create: `mypy.ini`
- Modify: `pyproject.toml:12-17`

**Step 1: Add mypy to dev dependencies**

```toml
# pyproject.toml
[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.1.0",
    "mypy>=1.8.0",
]
```

Run: `uv sync`

**Step 2: Create strict mypy configuration**

```ini
# mypy.ini
[mypy]
python_version = 3.11
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_any_generics = False
check_untyped_defs = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
strict_equality = True
strict_concatenate = True

# Be lenient with third-party libraries
[mypy-anthropic.*]
ignore_missing_imports = True

[mypy-click.*]
ignore_missing_imports = True

[mypy-rich.*]
ignore_missing_imports = True
```

**Step 3: Run mypy to find type issues**

Run: `uv run mypy src/claude_lint`

Expected: May show some errors to fix

**Step 4: Fix any mypy errors found**

(Address specific errors as they appear - common fixes below)

```python
# Example fixes for common mypy errors:

# 1. Add return type to functions
def some_function() -> None:  # was missing return type
    pass

# 2. Initialize attributes properly
class SomeClass:
    def __init__(self) -> None:
        self.value: int = 0  # explicit type

# 3. Use proper Protocol for callbacks
from typing import Protocol

class Callable(Protocol):
    def __call__(self) -> str: ...
```

**Step 5: Verify mypy passes**

Run: `uv run mypy src/claude_lint`

Expected: Success: no issues found

**Step 6: Commit**

```bash
git add mypy.ini pyproject.toml
git commit -m "feat: add strict mypy configuration for type safety"
```

---

## Task 7: Fix Weak Type Annotations

**Files:**
- Modify: `src/claude_lint/batch_processor.py:21`
- Modify: `src/claude_lint/api_client.py:24, 113`

**Step 1: Import proper types from anthropic**

```python
# src/claude_lint/api_client.py
"""Claude API client with prompt caching support."""
from anthropic import Anthropic, APIError, APIConnectionError, RateLimitError, APITimeoutError
from anthropic.types import Message
from claude_lint.logging_config import get_logger

logger = get_logger(__name__)


def create_client(api_key: str, timeout: float = 60.0) -> Anthropic:
    """Create Anthropic client with timeout.

    Args:
        api_key: Anthropic API key
        timeout: Request timeout in seconds (default: 60)

    Returns:
        Anthropic client instance configured with timeout
    """
    return Anthropic(api_key=api_key, timeout=timeout)


def analyze_files_with_client(
    client: Anthropic,
    guidelines: str,
    prompt: str,
    model: str = "claude-sonnet-4-5-20250929"
) -> tuple[str, Message]:
    """Analyze files using existing Claude API client.

    Args:
        client: Anthropic client instance
        guidelines: CLAUDE.md content (will be cached)
        prompt: Prompt with files to analyze
        model: Claude model to use

    Returns:
        Tuple of (response text, Message object)

    Raises:
        ValueError: If guidelines or prompt are empty or invalid
        APIError: If API call fails (includes connection, rate limit, timeout, etc.)
    """
    # ... rest of implementation unchanged ...
    return first_block.text, response


def analyze_files(
    api_key: str,
    guidelines: str,
    prompt: str,
    model: str = "claude-sonnet-4-5-20250929"
) -> tuple[str, Message]:
    """Analyze files using Claude API with cached guidelines.

    Convenience wrapper that creates client and makes call.
    For multiple calls, use create_client() and analyze_files_with_client().

    Args:
        api_key: Anthropic API key
        guidelines: CLAUDE.md content (will be cached)
        prompt: Prompt with files to analyze
        model: Claude model to use

    Returns:
        Tuple of (response text, Message object)

    Raises:
        ValueError: If guidelines or prompt are empty or invalid
        APIError: If API call fails (includes connection, rate limit, etc.)
    """
    client = create_client(api_key)
    return analyze_files_with_client(client, guidelines, prompt, model)


def get_usage_stats(response: Message) -> dict[str, int]:
    """Get usage statistics from API response.

    Args:
        response: Message object from Claude API

    Returns:
        Dict with token usage stats
    """
    usage = response.usage
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_creation_tokens": getattr(usage, "cache_creation_input_tokens", 0),
        "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0)
    }
```

**Step 2: Fix batch_processor types**

```python
# src/claude_lint/batch_processor.py
"""Batch processing logic."""
from pathlib import Path
from anthropic import Anthropic
from claude_lint.api_client import analyze_files_with_client
from claude_lint.cache import Cache, CacheEntry
from claude_lint.collector import compute_file_hash
from claude_lint.config import Config
from claude_lint.file_reader import read_batch_files
from claude_lint.logging_config import get_logger
from claude_lint.processor import build_xml_prompt, parse_response
from claude_lint.rate_limiter import RateLimiter
from claude_lint.retry import retry_with_backoff
from claude_lint.types import FileResult

logger = get_logger(__name__)


def process_batch(
    batch: list[Path],
    project_root: Path,
    config: Config,
    guidelines: str,
    guidelines_hash: str,
    client: Anthropic,  # Changed from Any
    rate_limiter: RateLimiter,
    cache: Cache
) -> list[dict[str, str | list]]:  # More specific than dict[str, Any]
    """Process a single batch of files.

    ... docstring unchanged ...
    """
    # ... implementation unchanged ...
```

**Step 3: Run mypy to verify**

Run: `uv run mypy src/claude_lint`

Expected: Success: no issues found

**Step 4: Run tests to verify nothing breaks**

Run: `uv run pytest tests/ -v --ignore=tests/integration`

Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/claude_lint/api_client.py src/claude_lint/batch_processor.py
git commit -m "fix: replace Any with proper Anthropic types for type safety"
```

---

## Task 8: Add Logging to Retry Mechanism

**Files:**
- Modify: `src/claude_lint/retry.py:1-50`
- Modify: `tests/test_retry.py`

**Step 1: Write test for retry logging**

```python
# Add to tests/test_retry.py
import logging
from claude_lint.retry import retry_with_backoff


def test_retry_logs_attempts(caplog):
    """Test that retry attempts are logged."""
    attempt_count = 0

    def flaky_function():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise RuntimeError(f"Attempt {attempt_count} failed")
        return "success"

    with caplog.at_level(logging.DEBUG):
        result = retry_with_backoff(flaky_function, max_retries=3)

    assert result == "success"
    assert attempt_count == 3

    # Check logging
    debug_messages = [r.message for r in caplog.records if r.levelname == "DEBUG"]
    assert len(debug_messages) >= 2  # At least 2 retry messages
    assert any("Attempt 1" in msg for msg in debug_messages)
    assert any("Attempt 2" in msg for msg in debug_messages)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_retry.py::test_retry_logs_attempts -v`

Expected: FAIL (no logging yet)

**Step 3: Add logging to retry.py**

```python
# src/claude_lint/retry.py
"""Retry logic with exponential backoff."""
import time
from typing import Callable, TypeVar
from claude_lint.logging_config import get_logger

T = TypeVar("T")
logger = get_logger(__name__)


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
                logger.debug(
                    f"Attempt {attempt + 1}/{max_retries} failed with "
                    f"{type(e).__name__}: {e}. Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
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

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_retry.py -v`

Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/claude_lint/retry.py tests/test_retry.py
git commit -m "feat: add debug logging to retry mechanism for visibility"
```

---

## Task 9: Add Jitter to Exponential Backoff

**Files:**
- Modify: `src/claude_lint/retry.py:1-50`
- Modify: `tests/test_retry.py`

**Step 1: Write test for jitter**

```python
# Add to tests/test_retry.py
import time
from claude_lint.retry import retry_with_backoff


def test_retry_uses_jitter():
    """Test that retry delays include jitter to prevent thundering herd."""
    attempt_count = 0
    sleep_times = []

    def flaky_function():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 4:
            raise RuntimeError("Fail")
        return "success"

    # Monkey-patch time.sleep to capture delays
    original_sleep = time.sleep
    def mock_sleep(duration):
        sleep_times.append(duration)
        # Don't actually sleep in test

    time.sleep = mock_sleep
    try:
        retry_with_backoff(flaky_function, max_retries=4, initial_delay=1.0)
    finally:
        time.sleep = original_sleep

    # Should have 3 sleeps (first attempt doesn't sleep)
    assert len(sleep_times) == 3

    # Delays should have jitter (not exact multiples)
    # Base: 1.0, 2.0, 4.0 but with ±50% jitter
    for i, delay in enumerate(sleep_times):
        base_delay = 1.0 * (2.0 ** i)
        min_delay = base_delay * 0.5
        max_delay = base_delay * 1.5
        assert min_delay <= delay <= max_delay, \
            f"Delay {delay} not in range [{min_delay}, {max_delay}]"

    # Verify delays are not all identical (would indicate no jitter)
    assert len(set(sleep_times)) > 1, "All delays are identical - no jitter!"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_retry.py::test_retry_uses_jitter -v`

Expected: FAIL (no jitter yet)

**Step 3: Add jitter to retry.py**

```python
# src/claude_lint/retry.py
"""Retry logic with exponential backoff."""
import random
import time
from typing import Callable, TypeVar
from claude_lint.logging_config import get_logger

T = TypeVar("T")
logger = get_logger(__name__)


def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
) -> T:
    """Retry a function with exponential backoff and jitter.

    Uses jitter to prevent thundering herd problem when multiple
    clients retry simultaneously.

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
                # Add jitter: random value between 0.5x and 1.5x the delay
                jittered_delay = delay * (0.5 + random.random())

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

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_retry.py -v`

Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/claude_lint/retry.py tests/test_retry.py
git commit -m "feat: add jitter to exponential backoff to prevent thundering herd"
```

---

## Task 10: Add Pre-commit Hooks Configuration

**Files:**
- Create: `.pre-commit-config.yaml`
- Modify: `pyproject.toml:12-17`

**Step 1: Add pre-commit to dev dependencies**

```toml
# pyproject.toml
[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.1.0",
    "mypy>=1.8.0",
    "pre-commit>=3.5.0",
]
```

Run: `uv sync`

**Step 2: Create pre-commit configuration**

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-json
      - id: check-toml
      - id: check-merge-conflict
      - id: detect-private-key

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.15
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies:
          - anthropic
          - click
          - rich
          - pydantic
        args: [--config-file=mypy.ini]

  - repo: local
    hooks:
      - id: pytest-check
        name: pytest-check
        entry: uv run pytest tests/ --ignore=tests/integration -x
        language: system
        pass_filenames: false
        always_run: true
```

**Step 3: Install pre-commit hooks**

Run: `uv run pre-commit install`

Expected: pre-commit installed at .git/hooks/pre-commit

**Step 4: Test pre-commit on all files**

Run: `uv run pre-commit run --all-files`

Expected: All hooks pass (or auto-fix issues)

**Step 5: Commit**

```bash
git add .pre-commit-config.yaml pyproject.toml
git commit -m "feat: add pre-commit hooks for code quality automation"
```

---

## Task 11: Pin Dependency Upper Bounds

**Files:**
- Modify: `pyproject.toml:6-10`

**Step 1: Update dependencies with upper bounds**

```toml
# pyproject.toml
[project]
name = "lint-claude"
version = "0.3.0"
description = "CLAUDE.md compliance checker using Claude API"
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.18.0,<0.50",
    "click>=8.1.0,<9.0",
    "rich>=13.0.0,<14.0",
    "pydantic>=2.0.0,<3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0,<8.0",
    "pytest-cov>=4.1.0,<5.0",
    "ruff>=0.1.0,<0.3",
    "mypy>=1.8.0,<2.0",
    "pre-commit>=3.5.0,<4.0",
]
```

**Step 2: Test that dependencies install correctly**

Run: `uv sync`

Expected: Dependencies install successfully

**Step 3: Run all tests**

Run: `uv run pytest tests/ --ignore=tests/integration -v`

Expected: ALL PASS

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "fix: pin dependency upper bounds to prevent breaking changes"
```

---

## Task 12: Add Enhanced Ruff Lint Rules

**Files:**
- Modify: `pyproject.toml:26-29`

**Step 1: Add comprehensive ruff configuration**

```toml
# pyproject.toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
    "SIM", # flake8-simplify
    "TCH", # flake8-type-checking
    "PTH", # flake8-use-pathlib
]

ignore = [
    "E501",  # line too long (handled by formatter)
    "B008",  # do not perform function calls in argument defaults
]

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = [
    "S101",  # allow assert in tests
]
```

**Step 2: Run ruff to check for issues**

Run: `uv run ruff check src/claude_lint`

Expected: May show some issues to fix

**Step 3: Auto-fix what ruff can fix**

Run: `uv run ruff check --fix src/claude_lint`

Expected: Some issues auto-fixed

**Step 4: Manually fix remaining issues**

Review and fix any remaining issues ruff identifies

**Step 5: Verify all tests still pass**

Run: `uv run pytest tests/ --ignore=tests/integration -v`

Expected: ALL PASS

**Step 6: Commit**

```bash
git add pyproject.toml src/
git commit -m "feat: add comprehensive ruff lint rules for code quality"
```

---

## Task 13: Complete Package Metadata

**Files:**
- Modify: `pyproject.toml:1-33`
- Verify: `README.md` exists and is referenced

**Step 1: Add complete project metadata**

```toml
# pyproject.toml
[project]
name = "lint-claude"
version = "0.3.0"
description = "CLAUDE.md compliance checker using Claude API"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"},
]
maintainers = [
    {name = "Your Name", email = "your.email@example.com"},
]
keywords = [
    "claude",
    "linter",
    "code-quality",
    "ai",
    "compliance",
    "static-analysis",
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Topic :: Software Development :: Quality Assurance",
    "Topic :: Software Development :: Testing",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Typing :: Typed",
]

[project.urls]
Homepage = "https://github.com/yourusername/lint-claude"
Documentation = "https://github.com/yourusername/lint-claude/blob/main/README.md"
Repository = "https://github.com/yourusername/lint-claude"
Issues = "https://github.com/yourusername/lint-claude/issues"
Changelog = "https://github.com/yourusername/lint-claude/blob/main/CHANGELOG.md"

dependencies = [
    "anthropic>=0.18.0,<0.50",
    "click>=8.1.0,<9.0",
    "rich>=13.0.0,<14.0",
    "pydantic>=2.0.0,<3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0,<8.0",
    "pytest-cov>=4.1.0,<5.0",
    "ruff>=0.1.0,<0.3",
    "mypy>=1.8.0,<2.0",
    "pre-commit>=3.5.0,<4.0",
]

[project.scripts]
lint-claude = "claude_lint.cli:main"

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
    "SIM", # flake8-simplify
    "TCH", # flake8-type-checking
    "PTH", # flake8-use-pathlib
]

ignore = [
    "E501",  # line too long (handled by formatter)
    "B008",  # do not perform function calls in argument defaults
]

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = [
    "S101",  # allow assert in tests
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

**Step 2: Verify README.md exists**

Run: `test -f README.md && echo "README exists" || echo "README missing"`

Expected: README exists

**Step 3: Test package build**

Run: `uv build`

Expected: Successfully built package

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "feat: complete package metadata for PyPI publication"
```

---

## Summary

**Plan Complete:** 13 tasks addressing all critical and major issues

**Estimated Total Time:** 4-6 hours

**Test Coverage Impact:** Expected to maintain >88% coverage, potentially increase to >92%

**Grade Improvement:** From C+ (72/100) to A- (90/100)

**Next Steps After Completion:**
1. Run full test suite: `uv run pytest tests/ -v`
2. Check coverage: `uv run pytest --cov=src/claude_lint --cov-report=term`
3. Verify mypy: `uv run mypy src/claude_lint`
4. Run pre-commit: `uv run pre-commit run --all-files`
5. Update CHANGELOG.md with v0.3.0 changes
6. Tag release: `git tag -a v0.3.0 -m "Production readiness fixes"`
