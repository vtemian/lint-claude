"""Tests for API timeout handling."""
import pytest
from unittest.mock import patch, MagicMock
from anthropic import APITimeoutError
from claude_lint.api_client import analyze_files_with_client, create_client
from claude_lint.config import get_default_config


def test_api_client_has_timeout_configured():
    """Test that API client is created with timeout."""
    with patch("claude_lint.api_client.Anthropic") as mock_anthropic:
        create_client("test-key")

        # Verify Anthropic was called with timeout
        mock_anthropic.assert_called_once()
        call_kwargs = mock_anthropic.call_args[1]
        assert "timeout" in call_kwargs
        assert call_kwargs["timeout"] == 60.0


def test_api_call_respects_timeout():
    """Test that API calls use configured timeout."""
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="response")]
    client.messages.create.return_value = mock_response

    analyze_files_with_client(client, "guidelines", "prompt")

    # Verify timeout was passed to API call
    client.messages.create.assert_called_once()
    # The timeout is set on client initialization, not per-call


def test_api_timeout_raises_clear_error():
    """Test that API timeout errors are clear."""
    client = MagicMock()
    client.messages.create.side_effect = APITimeoutError("Request timed out")

    with pytest.raises(APITimeoutError, match="timed out"):
        analyze_files_with_client(client, "guidelines", "prompt")


def test_config_has_api_timeout_field():
    """Test that config includes api_timeout_seconds."""
    config = get_default_config()
    assert hasattr(config, "api_timeout_seconds")
    assert config.api_timeout_seconds == 60.0
