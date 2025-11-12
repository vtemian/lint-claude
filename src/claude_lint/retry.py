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
    backoff_factor: float = 2.0,
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
