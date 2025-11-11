"""Configuration management for claude-lint."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Configuration for claude-lint."""
    include: list[str]
    exclude: list[str]
    batch_size: int
    api_key: Optional[str] = None

    @classmethod
    def defaults(cls) -> "Config":
        """Return default configuration."""
        return cls(
            include=["**/*.py", "**/*.js", "**/*.ts"],
            exclude=["node_modules/**", "dist/**", ".git/**"],
            batch_size=10,
            api_key=None
        )


def load_config(config_path: Path) -> Config:
    """Load configuration from file or return defaults.

    Args:
        config_path: Path to .agent-lint.json file

    Returns:
        Config object with loaded or default values
    """
    if not config_path.exists():
        return Config.defaults()

    with open(config_path) as f:
        data = json.load(f)

    defaults = Config.defaults()

    return Config(
        include=data.get("include", defaults.include),
        exclude=data.get("exclude", defaults.exclude),
        batch_size=data.get("batchSize", defaults.batch_size),
        api_key=data.get("apiKey")
    )
