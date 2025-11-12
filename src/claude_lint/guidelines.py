"""CLAUDE.md guidelines reader and hash tracker."""
import hashlib
from pathlib import Path


def read_claude_md(project_root: Path, fallback_home: Path | None = None) -> str:
    """Read CLAUDE.md from project root or ~/.claude/ fallback.

    Args:
        project_root: Project root directory
        fallback_home: Optional fallback directory (defaults to ~/.claude)

    Returns:
        Content of CLAUDE.md

    Raises:
        FileNotFoundError: If CLAUDE.md not found in either location
    """
    # Try project root first
    project_claude_md = project_root / "CLAUDE.md"
    if project_claude_md.exists():
        return project_claude_md.read_text()

    # Try fallback (default to ~/.claude)
    if fallback_home is None:
        fallback_home = Path.home() / ".claude"

    home_claude_md = fallback_home / "CLAUDE.md"
    if home_claude_md.exists():
        return home_claude_md.read_text()

    raise FileNotFoundError(f"CLAUDE.md not found in {project_root} or {fallback_home}")


def get_claude_md_hash(content: str) -> str:
    """Compute SHA256 hash of CLAUDE.md content.

    Args:
        content: CLAUDE.md content

    Returns:
        Hex digest of SHA256 hash
    """
    return hashlib.sha256(content.encode()).hexdigest()
