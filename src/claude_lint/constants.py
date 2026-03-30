"""Configuration constants for claude-lint.

This module centralizes magic numbers used throughout the codebase
with clear documentation for why each value was chosen.
"""

# API Client Settings
# ------------------

API_MAX_TOKENS = 4096
"""Maximum tokens for Claude API responses.

Value: 4096
Rationale: Max output tokens for lint responses. Claude supports higher limits
but 4096 is sufficient for linting output.
"""

# Retry Settings
# -------------

RETRY_MAX_ATTEMPTS = 3
"""Maximum number of retry attempts for failed API calls.

Value: 3
Rationale: Balance between reliability and latency. Most transient failures
resolve within 1-2 retries. 3 attempts = original + 2 retries.
Total time: ~1s + ~2s = ~3s for exponential backoff.
"""

RETRY_INITIAL_DELAY = 1.0
"""Initial delay in seconds before first retry.

Value: 1.0
Rationale: Start conservative. Anthropic rate limits are per-minute, so 1s
gives time for rate limit windows to expire. Shorter delays waste retries
on rate limit errors.
"""

RETRY_BACKOFF_FACTOR = 2.0
"""Exponential backoff multiplier for retry delays.

Value: 2.0
Rationale: Standard exponential backoff (doubles each retry).
Delays: 1s → 2s → 4s. Aggressive enough to resolve quickly, conservative
enough to avoid overwhelming rate limits.
"""

RETRY_JITTER_MIN = 0.5
"""Minimum jitter multiplier for retry delays.

Value: 0.5 (50% of base delay)
Rationale: Prevents thundering herd when multiple clients retry simultaneously.
Range: 0.5x to 1.5x creates ±50% variance around base delay.
"""

RETRY_JITTER_MAX = 1.5
"""Maximum jitter multiplier for retry delays.

Value: 1.5 (150% of base delay)
Rationale: Upper bound of jitter range. Creates ±50% variance with JITTER_MIN.
"""

# Validation Settings
# ------------------

MIN_API_KEY_LENGTH = 40
"""Minimum expected length for Anthropic API keys.

Value: 40
Rationale: Anthropic API keys are typically 40+ characters. This catches
common errors like incomplete keys, placeholder values, or test strings.
"""
