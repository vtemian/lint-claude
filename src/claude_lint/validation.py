"""Input validation functions."""

from pathlib import Path

from claude_lint.constants import MIN_API_KEY_LENGTH

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
    """Validate API key is present and well-formed.

    Args:
        api_key: API key to validate

    Raises:
        ValueError: If API key is missing or malformed
    """
    if not api_key or not api_key.strip():
        raise ValueError(
            "API key is required. Set ANTHROPIC_API_KEY environment variable "
            "or add 'api_key' to .agent-lint.json"
        )

    key = api_key.strip()

    # Check prefix
    if not key.startswith("sk-ant-"):
        raise ValueError(
            "Invalid API key format. Anthropic API keys should start with 'sk-ant-'. "
            "Check your key at https://console.anthropic.com/"
        )

    # Check minimum length (Anthropic keys are typically 40+ chars)
    if len(key) < MIN_API_KEY_LENGTH:
        raise ValueError(
            f"API key appears too short ({len(key)} chars). "
            f"Anthropic API keys are typically {MIN_API_KEY_LENGTH}+ characters. "
            "Check your key at https://console.anthropic.com/"
        )
