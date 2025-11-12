import json
import tempfile
from pathlib import Path
from claude_lint.config import load_config


def test_load_config_with_defaults():
    """Test loading config with default values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / ".agent-lint.json"
        config_path.write_text(json.dumps({}))

        config = load_config(config_path)

        assert config.include == ["**/*.py", "**/*.js", "**/*.ts"]
        assert config.exclude == ["node_modules/**", "dist/**", ".git/**"]
        assert config.batch_size == 10


def test_load_config_with_custom_values():
    """Test loading config with custom values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / ".agent-lint.json"
        config_data = {"include": ["src/**/*.py"], "exclude": ["tests/**"], "batchSize": 5}
        config_path.write_text(json.dumps(config_data))

        config = load_config(config_path)

        assert config.include == ["src/**/*.py"]
        assert config.exclude == ["tests/**"]
        assert config.batch_size == 5


def test_load_config_missing_file():
    """Test loading config when file doesn't exist uses defaults."""
    config = load_config(Path("/nonexistent/.agent-lint.json"))

    assert config.include == ["**/*.py", "**/*.js", "**/*.ts"]
    assert config.batch_size == 10


def test_config_model_default():
    """Test default model configuration."""
    from claude_lint.config import get_default_config

    config = get_default_config()
    assert config.model == "claude-sonnet-4-5-20250929"


def test_load_config_with_model():
    """Test loading config with custom model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / ".agent-lint.json"
        config_file.write_text(json.dumps({"model": "claude-opus-4-5-20250929"}))

        config = load_config(config_file)
        assert config.model == "claude-opus-4-5-20250929"
