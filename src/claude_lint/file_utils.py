"""File operation utilities."""
import json
from pathlib import Path
from typing import Any


def atomic_write_json(data: Any, target_path: Path) -> None:
    """Write JSON data atomically to prevent corruption.

    Writes to a temporary file first, then atomically replaces the target.
    This ensures the target file is never in a partially-written state.

    Args:
        data: Data to serialize as JSON
        target_path: Target file path

    Raises:
        IOError: If write fails
        ValueError: If data cannot be serialized
    """
    # Create temp file in same directory as target for atomic replace
    tmp_path = target_path.with_suffix(".tmp")

    try:
        # Write to temporary file
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # Atomic replace (POSIX guarantees atomicity)
        tmp_path.replace(target_path)
    except Exception:
        # Clean up temp file on failure
        if tmp_path.exists():
            tmp_path.unlink()
        raise
