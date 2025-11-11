import tempfile
from pathlib import Path
import pytest
from claude_lint.validation import (
    validate_project_root,
    validate_mode,
    validate_batch_size,
    validate_api_key
)


def test_validate_project_root_exists():
    """Test validation of existing directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        validate_project_root(Path(tmpdir))  # Should not raise


def test_validate_project_root_not_exists():
    """Test validation of non-existent directory."""
    with pytest.raises(ValueError, match="does not exist"):
        validate_project_root(Path("/nonexistent/path"))


def test_validate_project_root_is_file():
    """Test validation fails for file instead of directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "file.txt"
        file_path.write_text("content")

        with pytest.raises(ValueError, match="not a directory"):
            validate_project_root(file_path)


def test_validate_mode_valid():
    """Test validation of valid modes."""
    for mode in ["full", "diff", "working", "staged"]:
        validate_mode(mode)  # Should not raise


def test_validate_mode_invalid():
    """Test validation of invalid mode."""
    with pytest.raises(ValueError, match="Invalid mode"):
        validate_mode("invalid")


def test_validate_batch_size_valid():
    """Test validation of valid batch sizes."""
    validate_batch_size(1)
    validate_batch_size(10)
    validate_batch_size(100)


def test_validate_batch_size_zero():
    """Test validation fails for zero."""
    with pytest.raises(ValueError, match="must be positive"):
        validate_batch_size(0)


def test_validate_batch_size_negative():
    """Test validation fails for negative."""
    with pytest.raises(ValueError, match="must be positive"):
        validate_batch_size(-5)


def test_validate_api_key_valid():
    """Test validation of valid API key."""
    validate_api_key("sk-ant-1234567890")  # Should not raise


def test_validate_api_key_empty():
    """Test validation fails for empty key."""
    with pytest.raises(ValueError, match="API key is required"):
        validate_api_key("")


def test_validate_api_key_none():
    """Test validation fails for None."""
    with pytest.raises(ValueError, match="API key is required"):
        validate_api_key(None)
