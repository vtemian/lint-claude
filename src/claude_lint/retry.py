"""Retry logic with exponential backoff."""
import time
from typing import Callable, TypeVar

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
        KeyboardInterrupt: Immediately re-raised without retry
        SystemExit: Immediately re-raised without retry
    """
    last_exception: Exception | None = None
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
    if last_exception is not None:
        raise last_exception
    raise RuntimeError("retry_with_backoff: no retries executed")
