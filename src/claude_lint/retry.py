"""Retry logic with exponential backoff."""
import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")


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
    """
    last_exception = None
    delay = initial_delay

    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e

            if attempt < max_retries - 1:  # Don't sleep on last attempt
                time.sleep(delay)
                delay *= backoff_factor

    # All retries exhausted
    raise last_exception
