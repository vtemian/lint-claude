"""Tests for configuration constants."""

from claude_lint.constants import (
    API_MAX_TOKENS,
    MIN_API_KEY_LENGTH,
    RETRY_BACKOFF_FACTOR,
    RETRY_INITIAL_DELAY,
    RETRY_JITTER_MAX,
    RETRY_JITTER_MIN,
    RETRY_MAX_ATTEMPTS,
)


def test_api_constants_are_positive():
    """Test that API constants have valid positive values."""
    assert API_MAX_TOKENS > 0
    assert isinstance(API_MAX_TOKENS, int)


def test_retry_constants_are_valid():
    """Test that retry constants have valid ranges."""
    assert RETRY_MAX_ATTEMPTS >= 1
    assert RETRY_INITIAL_DELAY > 0
    assert RETRY_BACKOFF_FACTOR >= 1.0
    assert 0 < RETRY_JITTER_MIN < 1.0
    assert RETRY_JITTER_MAX > 1.0
    assert RETRY_JITTER_MIN < RETRY_JITTER_MAX


def test_jitter_range_is_reasonable():
    """Test that jitter creates reasonable variance."""
    # With 0.5 and 1.5, jitter creates ±50% variance
    variance = RETRY_JITTER_MAX - RETRY_JITTER_MIN
    assert 0.5 <= variance <= 2.0  # Reasonable range


def test_validation_constants_are_positive():
    """Test that validation constants have valid positive values."""
    assert MIN_API_KEY_LENGTH > 0
    assert isinstance(MIN_API_KEY_LENGTH, int)
