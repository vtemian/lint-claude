"""Configuration management for claude-lint."""
import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class Config(BaseModel):
    """Configuration for claude-lint with validation."""

    include: list[str] = Field(min_length=1, description="File patterns to include")
    exclude: list[str] = Field(default_factory=list, description="File patterns to exclude")
    batch_size: int = Field(gt=0, le=100, description="Number of files per batch")
    model: str = Field(default="claude-sonnet-4-5-20250929", description="Claude model to use")
    max_file_size_mb: float = Field(default=1.0, gt=0, le=10, description="Maximum file size in MB")
    api_timeout_seconds: float = Field(
        default=60.0, gt=0, le=600, description="API timeout in seconds"
    )
    api_rate_limit: int = Field(default=4, gt=0, le=50, description="Requests per second")
    api_rate_window_seconds: float = Field(
        default=1.0, gt=0, description="Rate limit window in seconds"
    )
    show_progress: bool = Field(default=True, description="Show progress bars")
    api_key: str | None = Field(default=None, description="Anthropic API key")

    @field_validator("include")
    @classmethod
    def validate_include_patterns(cls, v: list[str]) -> list[str]:
        """Ensure include patterns are non-empty strings."""
        if not v:
            raise ValueError("include must contain at least one pattern")
        for pattern in v:
            if not pattern.strip():
                raise ValueError("include patterns cannot be empty strings")
        return v

    model_config = {"frozen": False}  # Allow modification for testing


def get_default_config() -> Config:
    """Return default configuration.

    Returns:
        Config with default values
    """
    return Config(
        include=["**/*.py", "**/*.js", "**/*.ts"],
        exclude=["node_modules/**", "dist/**", ".git/**"],
        batch_size=10,
        model="claude-sonnet-4-5-20250929",
        max_file_size_mb=1.0,
        api_timeout_seconds=60.0,
        api_rate_limit=4,  # Conservative: 4 requests/second
        api_rate_window_seconds=1.0,
        show_progress=True,
        api_key=None,
    )


def load_config(config_path: Path) -> Config:
    """Load configuration from file or return defaults.

    Supports both snake_case (preferred) and camelCase (backwards compat) keys.

    Args:
        config_path: Path to .agent-lint.json file

    Returns:
        Config object with loaded or default values

    Raises:
        ValueError: If configuration values are invalid
    """
    if not config_path.exists():
        return get_default_config()

    with config_path.open(encoding="utf-8") as f:
        data = json.load(f)

    defaults = get_default_config()

    # Build config dict with snake_case/camelCase support
    config_data = {
        "include": data.get("include", defaults.include),
        "exclude": data.get("exclude", defaults.exclude),
        "batch_size": data.get("batch_size", data.get("batchSize", defaults.batch_size)),
        "model": data.get("model", defaults.model),
        "max_file_size_mb": data.get(
            "max_file_size_mb", data.get("maxFileSizeMb", defaults.max_file_size_mb)
        ),
        "api_timeout_seconds": data.get(
            "api_timeout_seconds", data.get("apiTimeoutSeconds", defaults.api_timeout_seconds)
        ),
        "api_rate_limit": data.get(
            "api_rate_limit", data.get("apiRateLimit", defaults.api_rate_limit)
        ),
        "api_rate_window_seconds": data.get(
            "api_rate_window_seconds",
            data.get("apiRateWindowSeconds", defaults.api_rate_window_seconds),
        ),
        "show_progress": data.get(
            "show_progress", data.get("showProgress", defaults.show_progress)
        ),
        "api_key": data.get("api_key", data.get("apiKey")),
    }

    return Config(**config_data)
