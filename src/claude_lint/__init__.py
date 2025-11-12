"""Claude-lint: CLAUDE.md compliance checker."""

from claude_lint.__version__ import __version__
from claude_lint.config import Config, get_default_config, load_config
from claude_lint.orchestrator import run_compliance_check
from claude_lint.types import FileResult, Violation

__all__ = [
    "__version__",
    "Config",
    "load_config",
    "get_default_config",
    "run_compliance_check",
    "Violation",
    "FileResult",
]
