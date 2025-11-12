import tempfile
from pathlib import Path
import pytest
from claude_lint.progress import (
    create_progress_state,
    update_progress,
    save_progress,
    load_progress,
    get_remaining_batch_indices,
    is_progress_complete,
    get_progress_percentage,
    cleanup_progress,
)


def test_progress_state_initialization():
    """Test initializing progress state."""
    state = create_progress_state(total_batches=5)

    assert state.total_batches == 5
    assert len(state.completed_batch_indices) == 0
    assert is_progress_complete(state) is False


def test_update_progress():
    """Test updating progress."""
    state = create_progress_state(total_batches=3)

    state = update_progress(state, batch_index=0, results=[{"file": "a.py", "violations": []}])
    state = update_progress(state, batch_index=1, results=[{"file": "b.py", "violations": []}])

    assert len(state.completed_batch_indices) == 2
    assert get_progress_percentage(state) == pytest.approx(66.7, rel=0.1)


def test_save_and_load_progress():
    """Test saving and loading progress."""
    with tempfile.TemporaryDirectory() as tmpdir:
        progress_file = Path(tmpdir) / ".agent-lint-progress.json"

        # Create and save progress
        state = create_progress_state(total_batches=5)
        state = update_progress(state, batch_index=0, results=[{"file": "a.py", "violations": []}])
        state = update_progress(state, batch_index=1, results=[{"file": "b.py", "violations": []}])
        save_progress(state, progress_file)

        # Load progress
        loaded = load_progress(progress_file)

        assert loaded.total_batches == 5
        assert len(loaded.completed_batch_indices) == 2
        assert 0 in loaded.completed_batch_indices
        assert 1 in loaded.completed_batch_indices


def test_resume_from_progress():
    """Test resuming from saved progress."""
    with tempfile.TemporaryDirectory() as tmpdir:
        progress_file = Path(tmpdir) / ".agent-lint-progress.json"

        # Create initial progress
        state = create_progress_state(total_batches=5)
        state = update_progress(state, batch_index=0, results=[{"file": "a.py", "violations": []}])
        state = update_progress(state, batch_index=2, results=[{"file": "c.py", "violations": []}])
        save_progress(state, progress_file)

        # Resume
        resumed = load_progress(progress_file)
        remaining = get_remaining_batch_indices(resumed)

        assert remaining == [1, 3, 4]


def test_cleanup_on_complete():
    """Test that progress file is cleaned up when complete."""
    with tempfile.TemporaryDirectory() as tmpdir:
        progress_file = Path(tmpdir) / ".agent-lint-progress.json"

        state = create_progress_state(total_batches=2)
        state = update_progress(state, batch_index=0, results=[])
        state = update_progress(state, batch_index=1, results=[])
        save_progress(state, progress_file)

        assert is_progress_complete(state) is True
        cleanup_progress(progress_file)

        assert not progress_file.exists()
