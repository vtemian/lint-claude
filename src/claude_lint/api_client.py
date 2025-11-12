"""Claude API client with prompt caching support."""
from anthropic import Anthropic, APIError, APIConnectionError, RateLimitError, APITimeoutError
from anthropic.types import Message
from claude_lint.logging_config import get_logger

logger = get_logger(__name__)


def create_client(api_key: str, timeout: float = 60.0) -> Anthropic:
    """Create Anthropic client with timeout.

    Args:
        api_key: Anthropic API key
        timeout: Request timeout in seconds (default: 60)

    Returns:
        Anthropic client instance configured with timeout
    """
    return Anthropic(api_key=api_key, timeout=timeout)


def analyze_files_with_client(
    client: Anthropic, guidelines: str, prompt: str, model: str = "claude-sonnet-4-5-20250929"
) -> tuple[str, Message]:
    """Analyze files using existing Claude API client.

    Args:
        client: Anthropic client instance
        guidelines: CLAUDE.md content (will be cached)
        prompt: Prompt with files to analyze
        model: Claude model to use

    Returns:
        Tuple of (response text, Message object)

    Raises:
        ValueError: If guidelines or prompt are empty or invalid
        APIError: If API call fails (includes connection, rate limit, timeout, etc.)
    """
    # Input validation
    if not guidelines or not isinstance(guidelines, str) or not guidelines.strip():
        raise ValueError("guidelines must be a non-empty string")
    if not prompt or not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    # Use prompt caching for guidelines
    try:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=[{"type": "text", "text": guidelines, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
        )
    except APITimeoutError as e:
        logger.error(f"API request timed out: {e}")
        raise
    except RateLimitError as e:
        logger.warning(f"Rate limit exceeded: {e}")
        raise
    except APIConnectionError as e:
        logger.error(f"API connection failed: {e}")
        raise
    except APIError as e:
        logger.error(f"API error: {e}")
        raise
    except (KeyboardInterrupt, SystemExit):
        # Never catch these
        raise

    # Validate response
    if not response.content:
        raise ValueError("API returned empty response content")

    # Extract text from first content block (must be TextBlock)
    first_block = response.content[0]
    if not hasattr(first_block, "text"):
        raise ValueError(f"API returned non-text content (type: {type(first_block).__name__})")
    return first_block.text, response


def analyze_files(
    api_key: str, guidelines: str, prompt: str, model: str = "claude-sonnet-4-5-20250929"
) -> tuple[str, Message]:
    """Analyze files using Claude API with cached guidelines.

    Convenience wrapper that creates client and makes call.
    For multiple calls, use create_client() and analyze_files_with_client().

    Args:
        api_key: Anthropic API key
        guidelines: CLAUDE.md content (will be cached)
        prompt: Prompt with files to analyze
        model: Claude model to use

    Returns:
        Tuple of (response text, Message object)

    Raises:
        ValueError: If guidelines or prompt are empty or invalid
        APIError: If API call fails (includes connection, rate limit, etc.)
    """
    client = create_client(api_key)
    return analyze_files_with_client(client, guidelines, prompt, model)


def get_usage_stats(response: Message) -> dict[str, int]:
    """Get usage statistics from API response.

    Args:
        response: Message object from Claude API

    Returns:
        Dict with token usage stats
    """
    usage = response.usage
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_creation_tokens": getattr(usage, "cache_creation_input_tokens", 0),
        "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0),
    }
