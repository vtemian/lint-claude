"""Tests for snake_case config support."""
import tempfile
import json
from pathlib import Path
from claude_lint.config import load_config


def test_load_config_snake_case():
    """Test loading config with snake_case keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / ".agent-lint.json"
        config_file.write_text(
            json.dumps({"batch_size": 20, "max_file_size_mb": 2.0, "api_key": "test-key"})
        )

        config = load_config(config_file)

        assert config.batch_size == 20
        assert config.max_file_size_mb == 2.0
        assert config.api_key == "test-key"


def test_load_config_backwards_compat_camel_case():
    """Test that camelCase is still supported for backwards compatibility."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / ".agent-lint.json"
        config_file.write_text(
            json.dumps({"batchSize": 20, "maxFileSizeMb": 2.0, "apiKey": "test-key"})
        )

        config = load_config(config_file)

        assert config.batch_size == 20
        assert config.max_file_size_mb == 2.0
        assert config.api_key == "test-key"
