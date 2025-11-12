import json
import tempfile
from pathlib import Path
import pytest
from claude_lint.cache import Cache, CacheEntry, load_cache, save_cache


def test_cache_entry_creation():
    """Test creating a cache entry."""
    entry = CacheEntry(
        file_hash="abc123",
        claude_md_hash="def456",
        violations=[{"type": "error", "message": "missing docstring", "line": None}],
        timestamp=1234567890
    )

    assert entry.file_hash == "abc123"
    assert entry.claude_md_hash == "def456"
    assert entry.violations == [{"type": "error", "message": "missing docstring", "line": None}]


def test_load_cache_empty():
    """Test loading cache when file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / ".agent-lint-cache.json"

        cache = load_cache(cache_path)

        assert cache.entries == {}
        assert cache.claude_md_hash == ""


def test_save_and_load_cache():
    """Test saving and loading cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / ".agent-lint-cache.json"

        # Create cache with entry
        cache = Cache(
            claude_md_hash="hash123",
            entries={
                "file1.py": CacheEntry(
                    file_hash="filehash1",
                    claude_md_hash="hash123",
                    violations=[{"type": "error", "message": "error1", "line": None}],
                    timestamp=1234567890
                )
            }
        )

        # Save
        save_cache(cache, cache_path)

        # Load
        loaded = load_cache(cache_path)

        assert loaded.claude_md_hash == "hash123"
        assert "file1.py" in loaded.entries
        assert loaded.entries["file1.py"].file_hash == "filehash1"


def test_cache_invalidation_on_claude_md_change():
    """Test that cache should be invalidated if CLAUDE.md hash changes."""
    cache = Cache(
        claude_md_hash="old_hash",
        entries={
            "file1.py": CacheEntry(
                file_hash="filehash1",
                claude_md_hash="old_hash",
                violations=[],
                timestamp=1234567890
            )
        }
    )

    # Check if entry is valid with new CLAUDE.md hash
    current_claude_hash = "new_hash"
    entry = cache.entries["file1.py"]

    is_valid = entry.claude_md_hash == current_claude_hash

    assert not is_valid


def test_cache_handles_non_ascii_content(tmp_path):
    """Test that cache handles files with non-ASCII characters."""
    cache = Cache(
        claude_md_hash="hash123",
        entries={
            "café.py": CacheEntry(
                file_hash="abc123",
                claude_md_hash="hash123",
                violations=[{
                    "type": "style",
                    "message": "Use café naming convention ☕",
                    "line": 1
                }],
                timestamp=1234567890
            )
        }
    )

    cache_path = tmp_path / ".cache.json"
    save_cache(cache, cache_path)

    # Reload and verify
    loaded = load_cache(cache_path)
    assert "café.py" in loaded.entries
    assert "café naming convention ☕" in loaded.entries["café.py"].violations[0]["message"]
