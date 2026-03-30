"""Retry logic with exponential backoff."""

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
    max_retries: int = RETRY_MAX_ATTEMPTS,
    initial_delay: float = RETRY_INITIAL_DELAY,
    backoff_factor: float = RETRY_BACKOFF_FACTOR,
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
