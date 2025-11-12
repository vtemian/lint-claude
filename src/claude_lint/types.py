"""Type definitions for claude-lint."""
from typing import TypedDict


class Violation(TypedDict):
    """Single violation structure."""

    type: str
    message: str
    line: int | None


class FileResult(TypedDict):
    """Result for a single file."""

    file: str
    violations: list[Violation]
