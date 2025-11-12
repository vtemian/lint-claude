"""Tests for refactored orchestrator functions."""
from pathlib import Path
from unittest.mock import Mock, patch

from claude_lint.config import Config


def test_process_all_batches_with_progress():
    """Test batch processing with progress bar enabled."""
    from claude_lint.orchestrator import _process_all_batches

    # Mock dependencies
    mock_client = Mock()
    mock_rate_limiter = Mock()
    mock_cache = Mock()
    mock_progress_state = Mock(results=[], completed_batch_indices=[], total_batches=2)

    batches = [
        [Path("file1.py")],
        [Path("file2.py")],
    ]

    with patch("claude_lint.orchestrator.process_batch") as mock_process:
        with patch("claude_lint.orchestrator.update_progress") as mock_update:
            with patch("claude_lint.orchestrator.save_progress"):
                with patch("claude_lint.orchestrator.save_cache"):
                    # Mock batch results
                    mock_process.return_value = [{"file": "file1.py", "violations": []}]
                    mock_update.return_value = mock_progress_state

                    config = Config(
                        include=["**/*.py"],
                        batch_size=10,
                        show_progress=True,
                    )

                    results, api_calls = _process_all_batches(
                        batches=batches,
                        project_root=Path("/tmp"),
                        config=config,
                        guidelines="# Test",
                        guidelines_hash="abc123",
                        client=mock_client,
                        rate_limiter=mock_rate_limiter,
                        cache=mock_cache,
                        progress_state=mock_progress_state,
                        progress_path=Path("/tmp/progress.json"),
                    )

                    # Should process both batches
                    assert mock_process.call_count == 2
                    assert api_calls == 2


def test_process_all_batches_without_progress():
    """Test batch processing with progress bar disabled."""
    from claude_lint.orchestrator import _process_all_batches

    mock_client = Mock()
    mock_rate_limiter = Mock()
    mock_cache = Mock()
    mock_progress_state = Mock(results=[], completed_batch_indices=[], total_batches=1)

    batches = [[Path("file1.py")]]

    with patch("claude_lint.orchestrator.process_batch") as mock_process:
        with patch("claude_lint.orchestrator.update_progress") as mock_update:
            with patch("claude_lint.orchestrator.save_progress"):
                with patch("claude_lint.orchestrator.save_cache"):
                    mock_process.return_value = [{"file": "file1.py", "violations": []}]
                    mock_update.return_value = mock_progress_state

                    config = Config(
                        include=["**/*.py"],
                        batch_size=10,
                        show_progress=False,  # Disable progress
                    )

                    results, api_calls = _process_all_batches(
                        batches=batches,
                        project_root=Path("/tmp"),
                        config=config,
                        guidelines="# Test",
                        guidelines_hash="abc123",
                        client=mock_client,
                        rate_limiter=mock_rate_limiter,
                        cache=mock_cache,
                        progress_state=mock_progress_state,
                        progress_path=Path("/tmp/progress.json"),
                    )

                    assert mock_process.call_count == 1
                    assert api_calls == 1
