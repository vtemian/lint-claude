import tempfile
from pathlib import Path
from claude_lint.collector import collect_all_files, is_excluded
from claude_lint.config import Config


def test_pattern_matching_nested_double_star():
    """Test pattern with multiple ** segments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create nested structure
        (tmpdir / "src" / "tests" / "unit").mkdir(parents=True)
        (tmpdir / "src" / "tests" / "unit" / "test.py").write_text("code")
        (tmpdir / "lib" / "vendor" / "package").mkdir(parents=True)
        (tmpdir / "lib" / "vendor" / "package" / "file.py").write_text("code")

        config = Config(
            include=["**/*.py"],
            exclude=["**/tests/**"],
            batch_size=10
        )

        files = collect_all_files(tmpdir, config)
        file_names = [f.name for f in files]

        # Should exclude anything under tests directory
        assert "test.py" not in file_names
        assert "file.py" in file_names


def test_is_excluded_consistent_with_include():
    """Test that exclusion uses same matching as inclusion."""
    # Both should use pathlib.PurePath.match() for consistency
    test_cases = [
        ("src/tests/test.py", ["**/tests/**"], True),
        ("tests/test.py", ["**/tests/**"], True),
        ("src/test.py", ["**/tests/**"], False),
        ("node_modules/lib/file.js", ["node_modules/**"], True),
        ("src/node_modules/file.js", ["node_modules/**"], True),
    ]

    for path, patterns, expected in test_cases:
        result = is_excluded(Path(path), patterns)
        assert result == expected, f"Path {path} with {patterns} should be {expected}"


def test_pattern_matching_consistency():
    """Test that include and exclude use same pattern matching."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        (tmpdir / "src").mkdir()
        (tmpdir / "src" / "main.py").write_text("code")
        (tmpdir / "tests").mkdir()
        (tmpdir / "tests" / "test.py").write_text("code")

        # Include all .py, exclude tests - should use same matching
        config = Config(
            include=["**/*.py"],
            exclude=["tests/**"],
            batch_size=10
        )

        files = collect_all_files(tmpdir, config)
        file_names = [f.name for f in files]

        assert "main.py" in file_names
        assert "test.py" not in file_names
