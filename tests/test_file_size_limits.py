import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
from claude_lint.orchestrator import run_compliance_check
from claude_lint.config import Config


def test_skip_large_files():
    """Test that files over size limit are skipped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create small file
        small_file = tmpdir / "small.py"
        small_file.write_text("print('hello')")

        # Create large file (over 1MB)
        large_file = tmpdir / "large.py"
        large_file.write_text("x" * (2 * 1024 * 1024))  # 2MB

        (tmpdir / "CLAUDE.md").write_text("# Guidelines")

        config = Config(
            include=["**/*.py"],
            exclude=[],
            batch_size=10,
            max_file_size_mb=1.0,  # 1MB limit
            api_key="test-key",
        )

        with patch("claude_lint.batch_processor.analyze_files_with_client") as mock_api:
            with patch("claude_lint.orchestrator.create_client") as mock_create:
                mock_create.return_value = Mock()
                mock_api.return_value = (
                    '{"results": [{"file": "small.py", "violations": []}]}',
                    Mock(),
                )

                results, metrics = run_compliance_check(tmpdir, config, mode="full")

                # Only small file should be analyzed
                call_args = mock_api.call_args
                prompt = call_args[0][2]  # Third argument is prompt

                assert "small.py" in prompt
                assert "large.py" not in prompt


def test_default_file_size_limit():
    """Test default file size limit."""
    from claude_lint.config import get_default_config

    config = get_default_config()
    assert config.max_file_size_mb == 1.0
