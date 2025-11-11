"""File collection with pattern matching."""
import hashlib
from pathlib import Path
from typing import Optional
import fnmatch

from claude_lint.config import Config


class FileCollector:
    """Collects files based on include/exclude patterns."""

    def __init__(self, root_path: Path, config: Config):
        """Initialize file collector.

        Args:
            root_path: Root directory to search from
            config: Configuration with include/exclude patterns
        """
        self.root_path = root_path
        self.config = config

    def collect_all(self) -> list[Path]:
        """Collect all files matching patterns.

        Returns:
            List of matching file paths
        """
        all_files = []

        for pattern in self.config.include:
            # Use rglob for recursive patterns
            if "**" in pattern:
                # Extract the extension pattern (e.g., "*.py" from "**/*.py")
                glob_pattern = pattern.split("**/")[-1]
                matching = self.root_path.rglob(glob_pattern)
            else:
                matching = self.root_path.glob(pattern)

            for file_path in matching:
                if file_path.is_file():
                    # Check if excluded
                    relative = file_path.relative_to(self.root_path)
                    if not self._is_excluded(relative):
                        all_files.append(file_path)

        return list(set(all_files))  # Remove duplicates

    def filter_by_list(self, file_list: list[str]) -> list[Path]:
        """Filter collected files by specific list.

        Args:
            file_list: List of file paths (relative to root)

        Returns:
            List of matching file paths that pass include/exclude filters
        """
        filtered = []

        for file_str in file_list:
            file_path = self.root_path / file_str
            if file_path.is_file():
                relative = Path(file_str)

                # Check if matches include patterns
                # Handle both "**/*.py" and "*.py" patterns
                matches_include = any(
                    fnmatch.fnmatch(str(relative), pattern) or
                    fnmatch.fnmatch(str(relative), pattern.replace("**/", ""))
                    for pattern in self.config.include
                )

                if matches_include and not self._is_excluded(relative):
                    filtered.append(file_path)

        return filtered

    def compute_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file content.

        Args:
            file_path: Path to file

        Returns:
            Hex digest of file content hash
        """
        content = file_path.read_text()
        return hashlib.sha256(content.encode()).hexdigest()

    def _is_excluded(self, relative_path: Path) -> bool:
        """Check if file is excluded by patterns.

        Args:
            relative_path: File path relative to root

        Returns:
            True if file should be excluded
        """
        return any(
            fnmatch.fnmatch(str(relative_path), pattern)
            for pattern in self.config.exclude
        )
