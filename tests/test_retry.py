from unittest.mock import Mock
import pytest
from claude_lint.retry import retry_with_backoff


def test_retry_success_on_first_attempt():
    """Test function succeeds on first attempt."""
    mock_func = Mock(return_value="success")

    result = retry_with_backoff(mock_func, max_retries=3)

    assert result == "success"
    assert mock_func.call_count == 1


def test_retry_success_after_failures():
    """Test function succeeds after some failures."""
    mock_func = Mock(side_effect=[
        Exception("fail1"),
        Exception("fail2"),
        "success"
    ])

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
    """Test that backoff delays increase exponentially."""
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

    # Second delay should be roughly 2x first delay
    assert delay2 > delay1 * 1.8  # Account for timing variance
