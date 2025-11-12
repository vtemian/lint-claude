"""File collection with pattern matching."""
import hashlib
from pathlib import Path, PurePath

from claude_lint.config import Config


def collect_all_files(root_path: Path, config: Config) -> list[Path]:
    """Collect all files matching patterns.

    Args:
        root_path: Root directory to search from
        config: Configuration with include/exclude patterns

    Returns:
        List of matching file paths
    """
    all_files = []

    for pattern in config.include:
        # Use rglob for ** patterns, glob otherwise
        if "**" in pattern:
            # For ** patterns, extract the file pattern
            # e.g., "**/*.py" -> "*.py", "src/**/*.py" -> walk and match
            parts = pattern.split("/")
            if pattern.startswith("**/"):
                # Simple case: **/*.ext
                glob_pattern = "/".join(parts[1:])
                matching = root_path.rglob(glob_pattern)
            else:
                # Complex case: use rglob with ** and filter
                glob_pattern = "/".join(parts)
                matching = root_path.glob(glob_pattern)
        else:
            matching = root_path.glob(pattern)

        for file_path in matching:
            if file_path.is_file():
                # Check if excluded
                relative = file_path.relative_to(root_path)
                if not is_excluded(relative, config.exclude):
                    all_files.append(file_path)

    return list(set(all_files))  # Remove duplicates


def filter_files_by_list(root_path: Path, file_list: list[str], config: Config) -> list[Path]:
    """Filter collected files by specific list.

    Args:
        root_path: Root directory
        file_list: List of file paths (relative to root)
        config: Configuration with include/exclude patterns

    Returns:
        List of matching file paths that pass include/exclude filters
    """
    filtered = []

    for file_str in file_list:
        file_path = root_path / file_str
        if file_path.is_file():
            relative = Path(file_str)

            # Check if matches include patterns
            # PurePath.match() handles ** patterns correctly for nested paths
            # but requires at least one directory level for ** patterns
            matches_include = any(
                PurePath(str(relative)).match(pattern)
                or (pattern.startswith("**/") and PurePath(str(relative)).match(pattern[3:]))
                for pattern in config.include
            )

            if matches_include and not is_excluded(relative, config.exclude):
                filtered.append(file_path)

    return filtered


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of file content.

    Args:
        file_path: Path to file

    Returns:
        Hex digest of file content hash
    """
    # Use streaming approach for large files
    hash_obj = hashlib.sha256()
    try:
        # Try reading as text first
        with file_path.open("rb") as f:
            # Read in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(65536), b""):
                hash_obj.update(chunk)
    except OSError as e:
        # Handle file read errors
        raise OSError(f"Failed to read file {file_path}: {e}") from e

    return hash_obj.hexdigest()


def is_excluded(relative_path: Path, exclude_patterns: list[str]) -> bool:
    """Check if file is excluded by patterns.

    Uses pathlib.PurePath.match() for consistent glob-style matching
    with include patterns. Supports ** for recursive matching.

    Args:
        relative_path: File path relative to root
        exclude_patterns: List of exclude patterns

    Returns:
        True if file should be excluded
    """
    path_obj = PurePath(relative_path)

    for pattern in exclude_patterns:
        # PurePath.match() handles ** for nested paths, but we need to handle
        # different pattern forms specially

        # First try direct match
        if path_obj.match(pattern):
            return True

        # Handle patterns like "**/tests/**" - check if any part of path matches
        if "**" in pattern:
            # For patterns like "**/dirname/**", check if dirname is in path parts
            if pattern.startswith("**/") and pattern.endswith("/**"):
                dir_name = pattern[3:-3]  # Extract "dirname" from "**/dirname/**"
                if dir_name in path_obj.parts:
                    return True

            # For patterns like "dirname/**", check if path starts with dirname
            elif pattern.endswith("/**") and not pattern.startswith("**/"):
                prefix = pattern[:-3]  # Remove "/**"
                # Check if any parent of path matches the prefix
                for parent in [path_obj] + list(path_obj.parents):
                    if parent.match(prefix) or str(parent) == prefix:
                        return True

            # For patterns like "**/filename.ext", check against path
            elif pattern.startswith("**/"):
                suffix_pattern = pattern[3:]
                if path_obj.match(suffix_pattern):
                    return True

    return False
