"""Tests for config validation."""
import pytest
from pydantic import ValidationError
from claude_lint.config import Config


def test_config_rejects_negative_batch_size():
    """Test that negative batch size raises error."""
    with pytest.raises(ValidationError, match="batch_size"):
        Config(
            include=["**/*.py"],
            exclude=[],
            batch_size=-1
        )


def test_config_rejects_zero_batch_size():
    """Test that zero batch size raises error."""
    with pytest.raises(ValidationError, match="batch_size"):
        Config(
            include=["**/*.py"],
            exclude=[],
            batch_size=0
        )


def test_config_rejects_negative_max_file_size():
    """Test that negative file size raises error."""
    with pytest.raises(ValidationError, match="max_file_size_mb"):
        Config(
            include=["**/*.py"],
            exclude=[],
            batch_size=10,
            max_file_size_mb=-1.0
        )


def test_config_rejects_zero_api_rate_limit():
    """Test that zero rate limit raises error."""
    with pytest.raises(ValidationError, match="api_rate_limit"):
        Config(
            include=["**/*.py"],
            exclude=[],
            batch_size=10,
            api_rate_limit=0
        )


def test_config_rejects_negative_api_timeout():
    """Test that negative timeout raises error."""
    with pytest.raises(ValidationError, match="api_timeout_seconds"):
        Config(
            include=["**/*.py"],
            exclude=[],
            batch_size=10,
            api_timeout_seconds=-1.0
        )


def test_config_rejects_empty_include():
    """Test that empty include list raises error."""
    with pytest.raises(ValidationError, match="include"):
        Config(
            include=[],
            exclude=[],
            batch_size=10
        )


def test_config_accepts_valid_values():
    """Test that valid config is accepted."""
    config = Config(
        include=["**/*.py"],
        exclude=["tests/**"],
        batch_size=10,
        max_file_size_mb=2.0,
        api_rate_limit=5,
        api_timeout_seconds=120.0
    )
    assert config.batch_size == 10
    assert config.max_file_size_mb == 2.0
