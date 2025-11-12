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
