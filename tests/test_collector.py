import tempfile
from pathlib import Path
import pytest
from claude_lint.collector import FileCollector
from claude_lint.config import Config


def test_collect_files_with_patterns():
    """Test collecting files with include/exclude patterns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create file structure
        (tmpdir / "src").mkdir()
        (tmpdir / "src" / "main.py").write_text("code")
        (tmpdir / "src" / "utils.py").write_text("code")
        (tmpdir / "tests").mkdir()
        (tmpdir / "tests" / "test_main.py").write_text("test")
        (tmpdir / "node_modules").mkdir()
        (tmpdir / "node_modules" / "lib.js").write_text("js")

        config = Config(
            include=["**/*.py"],
            exclude=["node_modules/**"],
            batch_size=10
        )

        collector = FileCollector(tmpdir, config)
        files = collector.collect_all()

        # Should include Python files but exclude node_modules
        file_names = [f.name for f in files]
        assert "main.py" in file_names
        assert "utils.py" in file_names
        assert "test_main.py" in file_names
        assert "lib.js" not in file_names


def test_filter_by_file_list():
    """Test filtering collected files by a specific list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create files
        (tmpdir / "file1.py").write_text("code")
        (tmpdir / "file2.py").write_text("code")
        (tmpdir / "file3.py").write_text("code")

        config = Config(
            include=["**/*.py"],
            exclude=[],
            batch_size=10
        )

        collector = FileCollector(tmpdir, config)

        # Filter to only specific files
        filtered = collector.filter_by_list(["file1.py", "file3.py"])

        file_names = [f.name for f in filtered]
        assert "file1.py" in file_names
        assert "file3.py" in file_names
        assert "file2.py" not in file_names


def test_compute_file_hash():
    """Test computing hash of file content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_file = tmpdir / "test.py"
        content = "print('hello')"
        test_file.write_text(content)

        config = Config(include=["**/*.py"], exclude=[], batch_size=10)
        collector = FileCollector(tmpdir, config)

        file_hash = collector.compute_hash(test_file)

        # Hash should be consistent
        import hashlib
        expected = hashlib.sha256(content.encode()).hexdigest()
        assert file_hash == expected
