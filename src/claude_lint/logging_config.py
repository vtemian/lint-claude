"""Logging configuration for claude-lint."""
import logging
import sys


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging for claude-lint.

    Args:
        verbose: Enable verbose (INFO level) logging
        quiet: Enable quiet (ERROR only) logging
    """
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    # Configure root logger for claude_lint namespace
    logger = logging.getLogger("claude_lint")
    logger.setLevel(level)
    logger.propagate = False  # Prevent propagation to root logger

    # Remove existing handlers
    logger.handlers.clear()

    # Create console handler
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get logger for module.

    Args:
        name: Module name (will be prefixed with 'claude_lint.')

    Returns:
        Logger instance
    """
    if not name.startswith("claude_lint."):
        name = f"claude_lint.{name}"
    return logging.getLogger(name)
