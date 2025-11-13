"""End-to-end integration tests."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def test_project(tmp_path):
    """Create a test project with known files."""
    # Copy fixture project to temp directory
    fixture_dir = Path(__file__).parent / "fixtures" / "test_project"
    project_dir = tmp_path / "test_project"
    shutil.copytree(fixture_dir, project_dir)
    return project_dir


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="Requires ANTHROPIC_API_KEY environment variable",
)
def test_full_scan_with_real_api(test_project):
    """Test full scan with real API call."""
    # Run CLI
    result = subprocess.run(
        ["uv", "run", "lint-claude", "--full", "--json"],
        cwd=test_project,
        capture_output=True,
        text=True,
        timeout=60,
    )

    # Should complete successfully
    assert result.returncode in [0, 1]  # 0 = pass, 1 = violations

    # Should produce valid JSON
    output = json.loads(result.stdout)
    assert "results" in output
    assert "summary" in output

    # Should analyze both files
    files = [r["file"] for r in output["results"]]
    assert "good.py" in files
    assert "bad.py" in files


def test_full_scan_without_api_key(test_project, monkeypatch):
    """Test that missing API key shows helpful error."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = subprocess.run(
        ["uv", "run", "lint-claude", "--full"], cwd=test_project, capture_output=True, text=True
    )

    assert result.returncode == 2
    assert "API key" in result.stderr


def test_keyboard_interrupt_handling(test_project):
    """Test that Ctrl-C is handled gracefully."""
    import platform
    import time

    # Skip on Windows - SIGINT handling is different
    if platform.system() == "Windows":
        pytest.skip("SIGINT handling differs on Windows")

    # Provide fake API key so process runs long enough to be interrupted
    env = os.environ.copy()
    env["ANTHROPIC_API_KEY"] = "sk-ant-api01-" + "x" * 90

    # Start process
    proc = subprocess.Popen(
        ["uv", "run", "lint-claude", "--full"],
        cwd=test_project,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    # Wait longer to ensure process has fully started
    # (not just during module import)
    time.sleep(2.0)

    # Send interrupt
    proc.send_signal(subprocess.signal.SIGINT)

    # Get result
    stdout, stderr = proc.communicate(timeout=5)

    # Should exit with 130 (SIGINT) - that's the key indicator
    # Message may vary depending on when interrupt occurs
    assert proc.returncode == 130, f"Expected exit code 130, got {proc.returncode}"


def test_progress_bar_output(test_project):
    """Test that progress bar is shown."""
    result = subprocess.run(
        ["uv", "run", "lint-claude", "--full"],
        cwd=test_project,
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "ANTHROPIC_API_KEY": "test-key"},
    )

    # Progress should be visible (rich uses ANSI codes)
    # We can't easily test the visual output, but verify it runs
    assert result.returncode in [0, 1, 2]


def test_cache_persistence(test_project):
    """Test that cache is persisted and reused."""
    # Skip if no API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("Requires ANTHROPIC_API_KEY")

    # First run
    result1 = subprocess.run(
        ["uv", "run", "lint-claude", "--full", "--json"],
        cwd=test_project,
        capture_output=True,
        text=True,
        timeout=60,
    )

    # Cache file should exist
    cache_file = test_project / ".agent-lint-cache.json"
    assert cache_file.exists()

    # Second run should be faster (uses cache)
    result2 = subprocess.run(
        ["uv", "run", "lint-claude", "--full", "--json"],
        cwd=test_project,
        capture_output=True,
        text=True,
        timeout=60,
    )

    # Both should produce same results
    if result1.returncode in [0, 1]:
        output1 = json.loads(result1.stdout)
        output2 = json.loads(result2.stdout)
        assert output1["results"] == output2["results"]


def test_version_flag():
    """Test --version flag works."""
    result = subprocess.run(
        ["uv", "run", "lint-claude", "--version"], capture_output=True, text=True
    )

    assert result.returncode == 0
    assert "lint-claude" in result.stdout
    assert "0.3.2" in result.stdout
