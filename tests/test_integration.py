"""End-to-end integration tests."""
import os
import tempfile
from pathlib import Path
import subprocess
import pytest


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
def test_full_integration():
    """Test full end-to-end workflow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create project structure
        (tmpdir / "src").mkdir()
        (tmpdir / "src" / "good.py").write_text(
            '''
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b
    """
    return a + b
'''
        )

        (tmpdir / "src" / "bad.py").write_text(
            """
def process(data):
    if data:
        if data.get("value"):
            if data["value"] > 0:
                return data["value"] * 2
    return None
"""
        )

        # Create CLAUDE.md
        (tmpdir / "CLAUDE.md").write_text(
            """
# Guidelines

- Follow TDD
- Avoid nested complexity
- Use type hints
- Write docstrings
"""
        )

        # Create config
        (tmpdir / ".agent-lint.json").write_text(
            """
{
  "include": ["**/*.py"],
  "exclude": ["tests/**"],
  "batchSize": 10
}
"""
        )

        # Run claude-lint
        result = subprocess.run(
            ["python", "-m", "claude_lint.cli", "--full"],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            env={**os.environ},
        )

        # Verify output
        assert "good.py" in result.stdout
        assert "bad.py" in result.stdout

        # bad.py should have violations
        assert result.returncode == 1
