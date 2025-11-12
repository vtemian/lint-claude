"""File reading with encoding fallback and size limits."""
from pathlib import Path

from claude_lint.logging_config import get_logger

logger = get_logger(__name__)


def read_file_safely(file_path: Path, project_root: Path, max_size_bytes: int) -> str | None:
    """Read file with encoding fallback and size checking.

    Tries UTF-8 first, falls back to latin-1. Checks file size before reading.

    Args:
        file_path: Absolute path to file
        project_root: Project root for relative path logging
        max_size_bytes: Maximum allowed file size in bytes

    Returns:
        File content as string, or None if file should be skipped
    """
    rel_path = file_path.relative_to(project_root)

    # Check file size
    try:
        file_size = file_path.stat().st_size
        if file_size > max_size_bytes:
            logger.warning(
                f"File {rel_path} exceeds size limit "
                f"({file_size / 1024 / 1024:.2f}MB > "
                f"{max_size_bytes / 1024 / 1024:.2f}MB), skipping"
            )
            return None
    except OSError as e:
        logger.warning(f"Cannot stat file {rel_path}, skipping: {e}")
        return None

    # Try reading with encoding fallback
    try:
        # Try UTF-8 first
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fall back to latin-1 which accepts all byte sequences
        try:
            logger.warning(f"File {rel_path} is not valid UTF-8, trying latin-1")
            return file_path.read_text(encoding="latin-1")
        except Exception as e:
            logger.warning(f"Unable to decode file {rel_path}, skipping: {e}")
            return None
    except FileNotFoundError:
        logger.warning(f"File not found, skipping: {rel_path}")
        return None
    except Exception as e:
        logger.warning(f"Error reading file {rel_path}, skipping: {e}")
        return None


def read_batch_files(batch: list[Path], project_root: Path, max_size_mb: float) -> dict[str, str]:
    """Read multiple files for a batch.

    Args:
        batch: List of file paths to read
        project_root: Project root directory
        max_size_mb: Maximum file size in megabytes

    Returns:
        Dict mapping relative paths to file contents
    """
    file_contents = {}
    max_size_bytes = int(max_size_mb * 1024 * 1024)

    for file_path in batch:
        rel_path = str(file_path.relative_to(project_root))
        content = read_file_safely(file_path, project_root, max_size_bytes)

        if content is not None:
            file_contents[rel_path] = content

    return file_contents
