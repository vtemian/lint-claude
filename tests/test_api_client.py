from unittest.mock import Mock, patch
import pytest
from claude_lint.api_client import ClaudeClient


@patch("claude_lint.api_client.Anthropic")
def test_claude_client_initialization(mock_anthropic):
    """Test initializing Claude API client."""
    client = ClaudeClient(api_key="test-key")

    mock_anthropic.assert_called_once_with(api_key="test-key")


@patch("claude_lint.api_client.Anthropic")
def test_analyze_with_caching(mock_anthropic):
    """Test making API call with prompt caching."""
    # Setup mock
    mock_response = Mock()
    mock_response.content = [Mock(text='{"results": []}')]
    mock_anthropic.return_value.messages.create.return_value = mock_response

    client = ClaudeClient(api_key="test-key")

    # Make request
    response = client.analyze_files(
        guidelines="# Guidelines",
        prompt="Check these files"
    )

    # Verify caching was used
    call_args = mock_anthropic.return_value.messages.create.call_args
    assert call_args[1]["model"] == "claude-sonnet-4-5-20250929"
    assert call_args[1]["max_tokens"] == 4096

    # Check system message uses cache_control
    system_messages = call_args[1]["system"]
    assert len(system_messages) == 1
    assert system_messages[0]["type"] == "text"
    assert system_messages[0]["text"] == "# Guidelines"
    assert system_messages[0]["cache_control"] == {"type": "ephemeral"}

    # Check user message
    messages = call_args[1]["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Check these files"


@patch("claude_lint.api_client.Anthropic")
def test_get_usage_stats(mock_anthropic):
    """Test extracting usage statistics from response."""
    mock_response = Mock()
    mock_response.content = [Mock(text='{"results": []}')]
    mock_response.usage = Mock(
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=200,
        cache_read_input_tokens=0
    )
    mock_anthropic.return_value.messages.create.return_value = mock_response

    client = ClaudeClient(api_key="test-key")
    response = client.analyze_files(
        guidelines="# Guidelines",
        prompt="Check files"
    )

    stats = client.get_last_usage_stats()

    assert stats["input_tokens"] == 100
    assert stats["output_tokens"] == 50
    assert stats["cache_creation_tokens"] == 200
    assert stats["cache_read_tokens"] == 0


@patch("claude_lint.api_client.Anthropic")
def test_get_usage_stats_no_request(mock_anthropic):
    """Test get_last_usage_stats returns None when no request has been made."""
    client = ClaudeClient(api_key="test-key")

    stats = client.get_last_usage_stats()

    assert stats is None
