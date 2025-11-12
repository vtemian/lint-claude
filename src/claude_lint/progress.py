"""Progress tracking and resume capability."""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from claude_lint.file_utils import atomic_write_json


@dataclass
class ProgressState:
    """State for progress tracking."""

    total_batches: int
    completed_batch_indices: list[int] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)


def create_progress_state(total_batches: int) -> ProgressState:
    """Create a new progress state.

    Args:
        total_batches: Total number of batches to process

    Returns:
        New ProgressState
    """
    return ProgressState(total_batches=total_batches)


def update_progress(
    state: ProgressState, batch_index: int, results: list[dict[str, Any]]
) -> ProgressState:
    """Update progress with batch results.

    Args:
        state: Current progress state
        batch_index: Index of completed batch
        results: Results from this batch

    Returns:
        Updated progress state
    """
    if batch_index not in state.completed_batch_indices:
        state.completed_batch_indices.append(batch_index)

    state.results.extend(results)
    return state


def save_progress(state: ProgressState, progress_file: Path) -> None:
    """Save progress to file atomically.

    Args:
        state: Progress state to save
        progress_file: Path to progress file
    """
    data = asdict(state)
    atomic_write_json(data, progress_file)


def load_progress(progress_file: Path) -> ProgressState:
    """Load progress from file.

    Args:
        progress_file: Path to progress file

    Returns:
        ProgressState with loaded data
    """
    with progress_file.open() as f:
        data = json.load(f)

    return ProgressState(**data)


def get_remaining_batch_indices(state: ProgressState) -> list[int]:
    """Get list of batch indices that still need processing.

    Args:
        state: Current progress state

    Returns:
        List of remaining batch indices
    """
    all_indices = set(range(state.total_batches))
    completed = set(state.completed_batch_indices)
    return sorted(all_indices - completed)


def is_progress_complete(state: ProgressState) -> bool:
    """Check if all batches are complete.

    Args:
        state: Current progress state

    Returns:
        True if all batches processed
    """
    return len(state.completed_batch_indices) == state.total_batches


def get_progress_percentage(state: ProgressState) -> float:
    """Get progress as percentage.

    Args:
        state: Current progress state

    Returns:
        Progress percentage (0-100)
    """
    completed_batches = len(state.completed_batch_indices)
    return (completed_batches / state.total_batches) * 100


def cleanup_progress(progress_file: Path) -> None:
    """Remove progress file.

    Args:
        progress_file: Path to progress file
    """
    if progress_file.exists():
        progress_file.unlink()
