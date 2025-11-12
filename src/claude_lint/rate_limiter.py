"""Rate limiting for API calls."""
import threading
import time
from collections import deque


class RateLimiter:
    """Thread-safe rate limiter with sliding window.

    Limits the number of requests within a time window using a sliding
    window algorithm for accurate rate limiting.

    Thread-safe: All operations are protected by a lock and condition variable.
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
        self._condition = threading.Condition(self._lock)

    def acquire(self) -> None:
        """Acquire a rate limit token, blocking if necessary.

        This method blocks until a token is available within the rate limit.
        Uses a sliding window to track requests.

        Thread-safe: Uses condition variable to wait without releasing lock unsafely.
        """
        with self._condition:
            now = time.time()

            # Remove requests outside the current window
            while self.requests and self.requests[0] < now - self.window_seconds:
                self.requests.popleft()

            # Wait while at limit
            while len(self.requests) >= self.max_requests:
                # Calculate how long to wait for oldest request to expire
                sleep_time = self.requests[0] + self.window_seconds - time.time()

                if sleep_time > 0:
                    # Wait with condition - atomically releases and reacquires lock
                    self._condition.wait(timeout=sleep_time)

                    # After waking, re-check time and clean up expired requests
                    now = time.time()
                    while self.requests and self.requests[0] < now - self.window_seconds:
                        self.requests.popleft()
                else:
                    # Oldest request already expired, remove it
                    self.requests.popleft()

            # Record this request
            self.requests.append(time.time())
            # Notify waiting threads that a request completed (slot may be available)
            self._condition.notify()

    def try_acquire(self) -> bool:
        """Try to acquire a token without blocking.

        Returns:
            True if token acquired, False if at rate limit

        Thread-safe: Uses condition variable for consistent locking.
        """
        with self._condition:
            now = time.time()

            # Remove requests outside the current window
            while self.requests and self.requests[0] < now - self.window_seconds:
                self.requests.popleft()

            # Check if we're at limit
            if len(self.requests) >= self.max_requests:
                return False

            # Record this request
            self.requests.append(now)
            # Notify waiting threads that a request completed (slot may be available)
            self._condition.notify()
            return True
