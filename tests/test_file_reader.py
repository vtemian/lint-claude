"""Tests for safe file reading."""
import tempfile
from pathlib import Path
from claude_lint.file_reader import read_file_safely, read_batch_files


def test_read_file_safely_utf8():
    """Test reading valid UTF-8 file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_file = tmpdir / "test.py"
        test_file.write_text("print('hello')")

        content = read_file_safely(test_file, tmpdir, max_size_bytes=1024 * 1024)

        assert content == "print('hello')"


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
