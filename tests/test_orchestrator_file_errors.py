import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
from claude_lint.orchestrator import run_compliance_check
from claude_lint.config import Config


def test_file_with_invalid_utf8():
    """Test handling of files with invalid UTF-8."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create file with invalid UTF-8
        binary_file = tmpdir / "binary.py"
        binary_file.write_bytes(b"print('hello')\n\x80\x81\x82")

        (tmpdir / "CLAUDE.md").write_text("# Guidelines")

        config = Config(include=["**/*.py"], exclude=[], batch_size=10, api_key="test-key")

        # Should skip binary file with warning (check logs)
        with patch("claude_lint.batch_processor.analyze_files_with_client") as mock_api:
            with patch("claude_lint.orchestrator.create_client") as mock_create:
                mock_create.return_value = Mock()
                mock_api.return_value = ('{"results": []}', Mock())

                # File should be skipped, not crash
                results, metrics = run_compliance_check(tmpdir, config, mode="full")

                # Should return empty or handle gracefully
                assert isinstance(results, list)


def test_file_reading_fallback_encoding():
    """Test that files are attempted with fallback encoding."""
    # This tests the new behavior where we try UTF-8, then latin-1
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create file with latin-1 encoding
        latin_file = tmpdir / "latin.py"
        content = "# Café"
        latin_file.write_bytes(content.encode("latin-1"))

        (tmpdir / "CLAUDE.md").write_text("# Guidelines")

        config = Config(include=["**/*.py"], exclude=[], batch_size=10, api_key="test-key")

        with patch("claude_lint.batch_processor.analyze_files_with_client") as mock_api:
            with patch("claude_lint.orchestrator.create_client") as mock_create:
                mock_create.return_value = Mock()
                mock_api.return_value = (
                    '{"results": [{"file": "latin.py", "violations": []}]}',
                    Mock(),
                )

                results, metrics = run_compliance_check(tmpdir, config, mode="full")

                # Should successfully read with fallback
                assert mock_api.called
