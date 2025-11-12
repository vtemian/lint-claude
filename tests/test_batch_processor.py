"""Tests for batch processing."""
import logging
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest
from claude_lint.batch_processor import process_batch
from claude_lint.cache import Cache
from claude_lint.config import Config
from claude_lint.rate_limiter import RateLimiter


def test_process_batch_success():
    """Test successful batch processing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        (tmpdir / "file1.py").write_text("code")

        batch = [tmpdir / "file1.py"]
        config = Config(
            include=["**/*.py"],
            exclude=[],
            batch_size=10,
            api_key="test"
        )
        cache = Cache(claude_md_hash="hash", entries={})
        rate_limiter = RateLimiter(max_requests=10, window_seconds=1.0)

        client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "response"

        with patch('claude_lint.batch_processor.analyze_files_with_client') as mock_api:
            mock_api.return_value = (
                '{"results": [{"file": "file1.py", "violations": []}]}',
                mock_response
            )

            results = process_batch(
                batch, tmpdir, config, "guidelines", "hash",
                client, rate_limiter, cache
            )

            assert len(results) == 1
            assert results[0]["file"] == "file1.py"
            assert "file1.py" in cache.entries


def test_process_batch_empty_after_filtering():
    """Test batch with all files filtered out."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # File that's too large
        large_file = tmpdir / "large.py"
        large_file.write_text("x" * 1024 * 1024 * 2)  # 2MB

        batch = [large_file]
        config = Config(
            include=["**/*.py"],
            exclude=[],
            batch_size=10,
            max_file_size_mb=0.5,  # 500KB limit
            api_key="test"
        )
        cache = Cache(claude_md_hash="hash", entries={})
        rate_limiter = RateLimiter(max_requests=10, window_seconds=1.0)

        results = process_batch(
            batch, tmpdir, config, "guidelines", "hash",
            MagicMock(), rate_limiter, cache
        )

        # Should return empty - no API call made
        assert len(results) == 0


def test_process_batch_logs_cache_update_failures(tmp_path, caplog):
    """Test that cache update failures are logged."""
    # Create test file
    test_file = tmp_path / "test.py"
    test_file.write_text("print('test')")

    config = Config(
        include=["**/*.py"],
        exclude=[],
        batch_size=10
    )

    cache = Cache(claude_md_hash="test", entries={})
    rate_limiter = RateLimiter(max_requests=10, window_seconds=1.0)
    mock_client = Mock()

    with patch('claude_lint.batch_processor.analyze_files_with_client') as mock_api:
        mock_api.return_value = (
            '{"results": [{"file": "test.py", "violations": []}]}',
            Mock()
        )

        # Make compute_file_hash fail
        with patch('claude_lint.batch_processor.compute_file_hash') as mock_hash:
            mock_hash.side_effect = PermissionError("Access denied")

            with caplog.at_level(logging.WARNING):
                results = process_batch(
                    [test_file],
                    tmp_path,
                    config,
                    "# Guidelines",
                    "hash123",
                    mock_client,
                    rate_limiter,
                    cache
                )

    # Should still return results
    assert len(results) == 1

    # Should log the error
    assert any("Failed to update cache" in record.message for record in caplog.records)
    assert any("PermissionError" in record.message for record in caplog.records)
