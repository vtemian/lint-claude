"""Tests for safe file reading."""

import logging
import tempfile
from pathlib import Path

import pytest
from claude_lint.file_reader import read_batch_files, read_file_safely


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset claude_lint logger to allow cap log to capture logs."""
    logger = logging.getLogger("claude_lint")
    logger.propagate = True
    logger.handlers.clear()
    logger.setLevel(logging.NOTSET)
    yield
    # Reset after test
    logger.propagate = True
    logger.handlers.clear()
    logger.setLevel(logging.NOTSET)


def test_read_file_safely_utf8():
    """Test reading valid UTF-8 file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_file = tmpdir / "test.py"
        test_file.write_text("print('hello')")

        content = read_file_safely(test_file, tmpdir, max_size_bytes=1024 * 1024)

        assert content == "print('hello')"


def test_read_file_with_valid_utf8():
    """Test reading file with valid UTF-8 including non-ASCII characters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_file = tmpdir / "test.py"
        test_file.write_text("# UTF-8: café ☕", encoding="utf-8")

        content = read_file_safely(test_file, tmpdir, max_size_bytes=1024)

        assert content is not None
        assert "café" in content
        assert "☕" in content


def test_read_file_safely_exceeds_size():
    """Test file exceeding size limit is skipped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_file = tmpdir / "large.py"
        test_file.write_text("x" * 2000)

        # Set limit to 1000 bytes
        content = read_file_safely(test_file, tmpdir, max_size_bytes=1000)

        assert content is None


def test_read_file_safely_invalid_utf8():
    """Test fallback to latin-1 for invalid UTF-8."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_file = tmpdir / "latin.py"
        # Write latin-1 encoded content
        test_file.write_bytes("# Café".encode("latin-1"))

        content = read_file_safely(test_file, tmpdir, max_size_bytes=1024 * 1024)

        assert content is not None
        assert "Caf" in content


def test_read_file_with_latin1_fallback():
    """Test fallback to latin-1 when UTF-8 fails."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_file = tmpdir / "test.py"

        # Write latin-1 encoded content that's invalid UTF-8
        content_latin1 = "# Latin-1: café"
        test_file.write_bytes(content_latin1.encode("latin-1"))

        content = read_file_safely(test_file, tmpdir, max_size_bytes=1024)

        # Should read successfully with latin-1 fallback
        assert content is not None
        assert "café" in content


def test_read_file_logs_encoding_fallback(caplog):
    """Test that encoding fallback is logged."""
    import logging

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_file = tmpdir / "test.py"

        # Write content with byte that's invalid UTF-8
        test_file.write_bytes(b"print('hello')\n\x80\x81\x82")

        # Capture all logs at WARNING level
        caplog.set_level(logging.WARNING)
        content = read_file_safely(test_file, tmpdir, max_size_bytes=1024)

        # Should log warning about UTF-8 failure
        assert any("not valid UTF-8" in record.message for record in caplog.records)

        # But should still return content via latin-1
        assert content is not None


def test_read_file_with_unreadable_file(caplog):
    """Test handling of file with permission denied."""
    import sys

    # Skip on Windows where chmod doesn't work the same way
    if sys.platform == "win32":
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_file = tmpdir / "test.py"
        test_file.write_text("content")

        # Make file unreadable
        test_file.chmod(0o000)

        try:
            caplog.set_level(logging.WARNING)
            content = read_file_safely(test_file, tmpdir, max_size_bytes=1024)

            # Should return None for unreadable file
            assert content is None

            # Should log warning about error
            assert any("Error reading file" in record.message for record in caplog.records)
        finally:
            # Restore permissions for cleanup
            test_file.chmod(0o644)


def test_read_file_with_totally_unreadable_file(caplog):
    """Test handling of completely unreadable file."""
    import sys

    # Skip on Windows where chmod doesn't work the same way
    if sys.platform == "win32":
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_file = tmpdir / "test.py"
        test_file.write_text("content")

        # Make file unreadable
        test_file.chmod(0o000)

        try:
            caplog.set_level(logging.WARNING)
            content = read_file_safely(test_file, tmpdir, max_size_bytes=1024)

            # Should return None for unreadable file
            assert content is None
        finally:
            # Restore permissions for cleanup
            test_file.chmod(0o644)


def test_read_batch_files():
    """Test reading multiple files in batch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        (tmpdir / "file1.py").write_text("code1")
        (tmpdir / "file2.py").write_text("code2")
        (tmpdir / "file3.py").write_text("x" * 2000)  # Too large

        files = [tmpdir / "file1.py", tmpdir / "file2.py", tmpdir / "file3.py"]

        contents = read_batch_files(files, tmpdir, max_size_mb=0.001)  # 1KB limit

        assert "file1.py" in contents
        assert "file2.py" in contents
        assert "file3.py" not in contents


def test_read_batch_files_skips_unreadable():
    """Test that batch reading skips unreadable files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create readable file
        good_file = tmpdir / "good.py"
        good_file.write_text("# Good file")

        # Create file with invalid encoding
        bad_file = tmpdir / "bad.py"
        bad_file.write_bytes(b"\x80\x81\x82")

        # Read batch
        result = read_batch_files([good_file, bad_file], tmpdir, max_size_mb=1.0)

        # Should include good file
        assert "good.py" in result
        assert "# Good file" in result["good.py"]

        # Should include bad file (read via latin-1 fallback)
        assert "bad.py" in result


def test_read_file_nonexistent(caplog):
    """Test handling of file that doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        nonexistent = tmpdir / "does_not_exist.py"

        caplog.set_level(logging.WARNING)
        content = read_file_safely(nonexistent, tmpdir, max_size_bytes=1024)

        # Should return None
        assert content is None

        # Should log warning about stat failure (file doesn't exist)
        assert any("Cannot stat file" in record.message for record in caplog.records)


def test_read_file_stat_error(caplog):
    """Test handling of OSError during stat."""
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_file = tmpdir / "test.py"
        test_file.write_text("content")

        # Mock stat to raise OSError
        caplog.set_level(logging.WARNING)
        with patch.object(Path, "stat", side_effect=OSError("Stat failed")):
            content = read_file_safely(test_file, tmpdir, max_size_bytes=1024)

        # Should return None
        assert content is None

        # Should log warning about stat failure
        assert any("Cannot stat file" in record.message for record in caplog.records)


def test_read_file_disappears_before_read(caplog):
    """Test handling of file that disappears between stat and read."""
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_file = tmpdir / "test.py"
        test_file.write_text("content")

        # Mock read_text to raise FileNotFoundError
        caplog.set_level(logging.WARNING)
        with patch.object(Path, "read_text", side_effect=FileNotFoundError("File vanished")):
            content = read_file_safely(test_file, tmpdir, max_size_bytes=1024)

        # Should return None
        assert content is None

        # Should log warning about file not found
        assert any("File not found" in record.message for record in caplog.records)


def test_read_file_latin1_fallback_fails(caplog):
    """Test handling of exception during latin-1 fallback."""
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_file = tmpdir / "test.py"
        test_file.write_bytes(b"\x80\x81\x82")  # Invalid UTF-8

        # Mock read_text to raise UnicodeDecodeError on first call (UTF-8)
        # and generic Exception on second call (latin-1)
        call_count = [0]

        def mock_read_text(encoding=None):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call (UTF-8) fails with UnicodeDecodeError
                raise UnicodeDecodeError("utf-8", b"\x80", 0, 1, "invalid start byte")
            else:
                # Second call (latin-1) fails with generic exception
                raise OSError("Disk read error")

        caplog.set_level(logging.WARNING)
        with patch.object(Path, "read_text", side_effect=mock_read_text):
            content = read_file_safely(test_file, tmpdir, max_size_bytes=1024)

        # Should return None
        assert content is None

        # Should log both warnings: UTF-8 failure and decode failure
        messages = [record.message for record in caplog.records]
        assert any("not valid UTF-8" in msg for msg in messages)
        assert any("Unable to decode file" in msg for msg in messages)
