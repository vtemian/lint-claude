import tempfile
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from claude_lint.file_utils import atomic_write_json


def test_atomic_write_json_success():
    """Test successful atomic write."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        target = tmpdir / "test.json"
        data = {"key": "value", "list": [1, 2, 3]}

        atomic_write_json(data, target)

        # Verify file exists and contains correct data
        assert target.exists()
        with open(target) as f:
            loaded = json.load(f)
        assert loaded == data


def test_atomic_write_json_no_corruption_on_failure():
    """Test that existing file is not corrupted on write failure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        target = tmpdir / "test.json"

        # Write initial data
        original_data = {"original": "data"}
        target.write_text(json.dumps(original_data))

        # Mock json.dump to fail
        with patch("json.dump", side_effect=RuntimeError("Write failed")):
            with pytest.raises(RuntimeError, match="Write failed"):
                atomic_write_json({"new": "data"}, target)

        # Verify original file is unchanged
        with open(target) as f:
            loaded = json.load(f)
        assert loaded == original_data


def test_atomic_write_json_tmp_cleaned_up():
    """Test that temporary file is cleaned up on failure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        target = tmpdir / "test.json"

        # Mock json.dump to fail
        with patch("json.dump", side_effect=RuntimeError("Write failed")):
            with pytest.raises(RuntimeError):
                atomic_write_json({"data": "value"}, target)

        # Verify no .tmp files left behind
        tmp_files = list(tmpdir.glob("*.tmp"))
        assert len(tmp_files) == 0
