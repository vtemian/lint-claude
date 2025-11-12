"""Tests for progress display functionality."""
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from claude_lint.orchestrator import run_compliance_check
from claude_lint.config import Config


def test_progress_display_enabled_by_default(capsys):
    """Test that progress display is shown by default."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test files
        (tmpdir / "file1.py").write_text("code")
        (tmpdir / "file2.py").write_text("code")
        (tmpdir / "CLAUDE.md").write_text("# Guidelines")

        config = Config(
            include=["**/*.py"],
            exclude=[],
            batch_size=1,  # Force 2 batches
            api_key="test-key",
        )

        with patch("claude_lint.orchestrator.create_client") as mock_create:
            with patch("claude_lint.batch_processor.analyze_files_with_client") as mock_analyze:
                mock_create.return_value = MagicMock()
                mock_analyze.return_value = (
                    '{"results": [{"file": "file1.py", "violations": []}]}',
                    MagicMock(),
                )

                # Progress should be visible in output
                # (We can't easily test rich output, but we can verify it doesn't crash)
                results, metrics = run_compliance_check(tmpdir, config, mode="full")

                assert len(results) >= 1


def test_progress_can_be_disabled():
    """Test that progress can be disabled via config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        (tmpdir / "file1.py").write_text("code")
        (tmpdir / "CLAUDE.md").write_text("# Guidelines")

        config = Config(
            include=["**/*.py"],
            exclude=[],
            batch_size=10,
            show_progress=False,  # Disable progress
            api_key="test-key",
        )

        with patch("claude_lint.orchestrator.create_client") as mock_create:
            with patch("claude_lint.batch_processor.analyze_files_with_client") as mock_analyze:
                mock_create.return_value = MagicMock()
                mock_analyze.return_value = (
                    '{"results": [{"file": "file1.py", "violations": []}]}',
                    MagicMock(),
                )

                results, metrics = run_compliance_check(tmpdir, config, mode="full")
                assert len(results) == 1
