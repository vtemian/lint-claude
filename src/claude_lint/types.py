"""Type definitions for claude-lint."""
from typing import TypedDict, Optional


class Violation(TypedDict):
    """Single violation structure."""
    type: str
    message: str
    line: Optional[int]


class FileResult(TypedDict):
    """Result for a single file."""
    file: str
    violations: list[Violation]
