"""Input validation functions."""
from pathlib import Path

VALID_MODES = {"full", "diff", "working", "staged"}


def validate_project_root(project_root: Path) -> None:
    """Validate project root directory exists.

    Args:
        project_root: Path to validate

    Raises:
        ValueError: If path does not exist or is not a directory
    """
    if not project_root.exists():
        raise ValueError(f"Project root does not exist: {project_root}")

    if not project_root.is_dir():
        raise ValueError(f"Project root is not a directory: {project_root}")


def validate_mode(mode: str) -> None:
    """Validate check mode.

    Args:
        mode: Mode to validate

    Raises:
        ValueError: If mode is not valid
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode: {mode}. Must be one of: {', '.join(sorted(VALID_MODES))}")


def validate_batch_size(batch_size: int) -> None:
    """Validate batch size.

    Args:
        batch_size: Batch size to validate

    Raises:
        ValueError: If batch size is not positive
    """
    if batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got: {batch_size}")


def validate_api_key(api_key: str | None) -> None:
    """Validate API key.

    Args:
        api_key: API key to validate

    Raises:
        ValueError: If API key is missing or empty
    """
    if not api_key:
        raise ValueError(
            "API key is required. Set ANTHROPIC_API_KEY environment variable "
            "or provide in config file."
        )
