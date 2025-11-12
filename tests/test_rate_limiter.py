"""Tests for rate limiting functionality."""
import time
import threading
from claude_lint.rate_limiter import RateLimiter


def test_rate_limiter_allows_requests_under_limit():
    """Test that requests under limit go through immediately."""
    limiter = RateLimiter(max_requests=5, window_seconds=1.0)

    start = time.time()
    for _ in range(3):
        limiter.acquire()
    elapsed = time.time() - start

    # Should be nearly instant (< 100ms)
    assert elapsed < 0.1


def test_rate_limiter_blocks_when_limit_exceeded():
    """Test that rate limiter blocks when limit is exceeded."""
    limiter = RateLimiter(max_requests=2, window_seconds=1.0)

    # First 2 requests should be instant
    start = time.time()
    limiter.acquire()
    limiter.acquire()
    elapsed_first_two = time.time() - start
    assert elapsed_first_two < 0.1

    # Third request should block
    start = time.time()
    limiter.acquire()
    elapsed_third = time.time() - start

    # Should have waited ~1 second for window to roll
    assert elapsed_third >= 0.9
    assert elapsed_third < 1.2


def test_rate_limiter_sliding_window():
    """Test that rate limiter uses sliding window."""
    limiter = RateLimiter(max_requests=2, window_seconds=1.0)

    # Make 2 requests
    limiter.acquire()
    time.sleep(0.6)  # Wait 600ms
    limiter.acquire()

    # Wait another 500ms (total 1.1s from first request)
    time.sleep(0.5)

    # Third request should be allowed (first request is now outside window)
    start = time.time()
    limiter.acquire()
    elapsed = time.time() - start

    # Should be instant since first request expired
    assert elapsed < 0.1


def test_config_has_rate_limit_fields():
    """Test that config includes rate limiting options."""
    from claude_lint.config import get_default_config

    config = get_default_config()
    assert hasattr(config, "api_rate_limit")
    assert hasattr(config, "api_rate_window_seconds")
    assert config.api_rate_limit == 4  # Conservative default
    assert config.api_rate_window_seconds == 1.0


def test_rate_limiter_thread_safety():
    """Test that rate limiter works correctly with concurrent access."""
    limiter = RateLimiter(max_requests=10, window_seconds=1.0)
    successful_acquires = []
    lock = threading.Lock()

    def worker():
        for _ in range(5):
            limiter.acquire()
            with lock:
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
