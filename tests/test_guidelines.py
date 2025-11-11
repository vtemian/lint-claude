import hashlib
import tempfile
from pathlib import Path
import pytest
from claude_lint.guidelines import read_claude_md, get_claude_md_hash


def test_read_claude_md_from_project():
    """Test reading CLAUDE.md from project root."""
    with tempfile.TemporaryDirectory() as tmpdir:
        claude_md = Path(tmpdir) / "CLAUDE.md"
        content = "# Guidelines\n\nFollow TDD."
        claude_md.write_text(content)

        result = read_claude_md(Path(tmpdir))

        assert result == content


def test_read_claude_md_from_home():
    """Test reading CLAUDE.md from ~/.claude/ if not in project."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Project has no CLAUDE.md
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()

        # But ~/.claude/CLAUDE.md exists
        home_claude_dir = Path(tmpdir) / ".claude"
        home_claude_dir.mkdir()
        home_claude_md = home_claude_dir / "CLAUDE.md"
        content = "# Home Guidelines\n\nUse TDD."
        home_claude_md.write_text(content)

        result = read_claude_md(project_dir, fallback_home=home_claude_dir)

        assert result == content


def test_read_claude_md_not_found():
    """Test FileNotFoundError when CLAUDE.md not found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        project_dir = tmpdir / "project"
        project_dir.mkdir()

        home_dir = tmpdir / "home"
        home_dir.mkdir()

        with pytest.raises(FileNotFoundError) as exc_info:
            read_claude_md(project_dir, fallback_home=home_dir)

        assert "CLAUDE.md not found" in str(exc_info.value)


def test_get_claude_md_hash():
    """Test computing hash of CLAUDE.md content."""
    content = "# Guidelines\n\nFollow TDD."
    expected_hash = hashlib.sha256(content.encode()).hexdigest()

    result = get_claude_md_hash(content)

    assert result == expected_hash
