from unittest.mock import Mock
import pytest
import logging
from claude_lint.retry import retry_with_backoff


def test_retry_success_on_first_attempt():
    """Test function succeeds on first attempt."""
    mock_func = Mock(return_value="success")

    result = retry_with_backoff(mock_func, max_retries=3)

    assert result == "success"
    assert mock_func.call_count == 1


def test_retry_success_after_failures():
    """Test function succeeds after some failures."""
    mock_func = Mock(side_effect=[Exception("fail1"), Exception("fail2"), "success"])

    result = retry_with_backoff(mock_func, max_retries=3, initial_delay=0.01)

    assert result == "success"
    assert mock_func.call_count == 3


def test_retry_exhausted():
    """Test all retries exhausted."""
    mock_func = Mock(side_effect=Exception("always fails"))

    with pytest.raises(Exception, match="always fails"):
        retry_with_backoff(mock_func, max_retries=3, initial_delay=0.01)

    assert mock_func.call_count == 3


def test_exponential_backoff_timing():
    """Test that backoff delays increase exponentially with jitter."""
    import time

    call_times = []

    def failing_func():
        call_times.append(time.time())
        if len(call_times) < 3:
            raise Exception("fail")
        return "success"

    retry_with_backoff(failing_func, max_retries=3, initial_delay=0.1)

    # Check delays between calls
    assert len(call_times) == 3
    delay1 = call_times[1] - call_times[0]
    delay2 = call_times[2] - call_times[1]

    # With jitter (0.5x to 1.5x), delays should be:
    # delay1: 0.1 * (0.5-1.5) = 0.05-0.15
    # delay2: 0.2 * (0.5-1.5) = 0.10-0.30
    # Check that delays are in reasonable ranges with jitter
    assert 0.05 <= delay1 <= 0.15, f"delay1 {delay1} outside expected range"
    assert 0.10 <= delay2 <= 0.30, f"delay2 {delay2} outside expected range"


def test_retry_logs_attempts(caplog):
    """Test that retry attempts are logged."""
    # Get the parent logger and ensure it has proper settings
    parent_logger = logging.getLogger("claude_lint")
    parent_logger.propagate = True  # Ensure propagation

    # Get retry logger and set level
    retry_logger = logging.getLogger("claude_lint.retry")
    original_level = retry_logger.level
    retry_logger.setLevel(logging.DEBUG)
    retry_logger.propagate = True  # Ensure propagation

    attempt_count = 0

    def flaky_function():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise RuntimeError(f"Attempt {attempt_count} failed")
        return "success"

    try:
        with caplog.at_level(logging.DEBUG, logger="claude_lint"):
            result = retry_with_backoff(flaky_function, max_retries=3, initial_delay=0.01)

        assert result == "success"
        assert attempt_count == 3

        # Check logging
        debug_messages = [r.message for r in caplog.records if r.levelname == "DEBUG"]
        assert len(debug_messages) >= 2  # At least 2 retry messages
        assert any("Attempt 1" in msg for msg in debug_messages)
        assert any("Attempt 2" in msg for msg in debug_messages)
    finally:
        # Restore original level
        retry_logger.setLevel(original_level)


def test_retry_uses_jitter():
    """Test that retry delays include jitter to prevent thundering herd."""
    import time

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

    # First check: no delay should be exactly the base delay (would indicate no jitter)
    base_delays = [1.0, 2.0, 4.0]
    for i, delay in enumerate(sleep_times):
        assert (
            delay != base_delays[i]
        ), f"Delay {i} is exactly {base_delays[i]} - no jitter applied!"

    # Second check: delays should be within jitter range
    # Base: 1.0, 2.0, 4.0 but with ±50% jitter
    for i, delay in enumerate(sleep_times):
        base_delay = 1.0 * (2.0**i)
        min_delay = base_delay * 0.5
        max_delay = base_delay * 1.5
        assert (
            min_delay <= delay <= max_delay
        ), f"Delay {delay} not in range [{min_delay}, {max_delay}]"
