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
