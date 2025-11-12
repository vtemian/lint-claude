# Claude-Lint Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Python CLI tool that checks project files for compliance with CLAUDE.md guidelines using Claude API with prompt caching.

**Architecture:** Smart change detection with git integration, batched file processing (default 10 files per API call, configurable up to 100), persistent caching for results and CLAUDE.md hash, retry logic with exponential backoff, and resumable progress tracking.

**Tech Stack:** Python 3.11+, anthropic SDK, GitPython, click (CLI), pytest

---

## Task 1: Project Initialization

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/claude_lint/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "lint-claude"
version = "0.1.0"
description = "CLAUDE.md compliance checker using Claude API"
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.18.0",
    "gitpython>=3.1.0",
    "click>=8.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.1.0",
]

[project.scripts]
lint-claude = "claude_lint.cli:main"

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

**Step 2: Create .gitignore**

```
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
.pytest_cache/
.coverage
htmlcov/
.lint-claude-cache.json
.venv/
venv/
```

**Step 3: Create basic README**

```markdown
# lint-claude

CLAUDE.md compliance checker using Claude API with prompt caching.

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Check changed files from main branch
lint-claude --diff main

# Check working directory changes
lint-claude --working

# Full project scan
lint-claude --full
```

## Configuration

Create `.lint-claude.json`:

```json
{
  "include": ["**/*.py", "**/*.js", "**/*.ts"],
  "exclude": ["node_modules/**", "dist/**", "*.test.js"],
  "batchSize": 10
}
```
```

**Step 4: Create package structure**

```bash
mkdir -p src/claude_lint tests
touch src/claude_lint/__init__.py
touch tests/__init__.py
```

**Step 5: Verify structure**

Run: `ls -la src/claude_lint/ tests/`
Expected: Directory structure created with __init__.py files

**Step 6: Commit**

```bash
git init
git add pyproject.toml .gitignore README.md src/ tests/
git commit -m "chore: initialize lint-claude project structure"
```

---

## Task 2: Configuration Schema

**Files:**
- Create: `src/claude_lint/config.py`
- Create: `tests/test_config.py`
- Create: `.lint-claude.json` (example)

**Step 1: Write failing test for config loading**

Create `tests/test_config.py`:

```python
import json
import tempfile
from pathlib import Path
import pytest
from claude_lint.config import Config, load_config


def test_load_config_with_defaults():
    """Test loading config with default values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / ".lint-claude.json"
        config_path.write_text(json.dumps({}))

        config = load_config(config_path)

        assert config.include == ["**/*.py", "**/*.js", "**/*.ts"]
        assert config.exclude == ["node_modules/**", "dist/**", ".git/**"]
        assert config.batch_size == 10


def test_load_config_with_custom_values():
    """Test loading config with custom values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / ".lint-claude.json"
        config_data = {
            "include": ["src/**/*.py"],
            "exclude": ["tests/**"],
            "batchSize": 5
        }
        config_path.write_text(json.dumps(config_data))

        config = load_config(config_path)

        assert config.include == ["src/**/*.py"]
        assert config.exclude == ["tests/**"]
        assert config.batch_size == 5


def test_load_config_missing_file():
    """Test loading config when file doesn't exist uses defaults."""
    config = load_config(Path("/nonexistent/.lint-claude.json"))

    assert config.include == ["**/*.py", "**/*.js", "**/*.ts"]
    assert config.batch_size == 10
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with "No module named 'claude_lint.config'"

**Step 3: Implement config module**

Create `src/claude_lint/config.py`:

```python
"""Configuration management for lint-claude."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Configuration for lint-claude."""
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
        config_path: Path to .lint-claude.json file

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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: All 3 tests PASS

**Step 5: Create example config file**

Create `.lint-claude.json`:

```json
{
  "include": ["**/*.py", "**/*.js", "**/*.ts"],
  "exclude": ["node_modules/**", "dist/**", "*.test.js", "__pycache__/**"],
  "batchSize": 10
}
```

**Step 6: Commit**

```bash
git add src/claude_lint/config.py tests/test_config.py .lint-claude.json
git commit -m "feat: add configuration schema and loader"
```

---

## Task 3: CLAUDE.md Reader with Hash Tracking

**Files:**
- Create: `src/claude_lint/guidelines.py`
- Create: `tests/test_guidelines.py`

**Step 1: Write failing test for CLAUDE.md reading**

Create `tests/test_guidelines.py`:

```python
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


def test_get_claude_md_hash():
    """Test computing hash of CLAUDE.md content."""
    content = "# Guidelines\n\nFollow TDD."
    expected_hash = hashlib.sha256(content.encode()).hexdigest()

    result = get_claude_md_hash(content)

    assert result == expected_hash
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_guidelines.py -v`
Expected: FAIL with "No module named 'claude_lint.guidelines'"

**Step 3: Implement guidelines module**

Create `src/claude_lint/guidelines.py`:

```python
"""CLAUDE.md guidelines reader and hash tracker."""
import hashlib
from pathlib import Path
from typing import Optional


def read_claude_md(project_root: Path, fallback_home: Optional[Path] = None) -> str:
    """Read CLAUDE.md from project root or ~/.claude/ fallback.

    Args:
        project_root: Project root directory
        fallback_home: Optional fallback directory (defaults to ~/.claude)

    Returns:
        Content of CLAUDE.md

    Raises:
        FileNotFoundError: If CLAUDE.md not found in either location
    """
    # Try project root first
    project_claude_md = project_root / "CLAUDE.md"
    if project_claude_md.exists():
        return project_claude_md.read_text()

    # Try fallback (default to ~/.claude)
    if fallback_home is None:
        fallback_home = Path.home() / ".claude"

    home_claude_md = fallback_home / "CLAUDE.md"
    if home_claude_md.exists():
        return home_claude_md.read_text()

    raise FileNotFoundError(
        f"CLAUDE.md not found in {project_root} or {fallback_home}"
    )


def get_claude_md_hash(content: str) -> str:
    """Compute SHA256 hash of CLAUDE.md content.

    Args:
        content: CLAUDE.md content

    Returns:
        Hex digest of SHA256 hash
    """
    return hashlib.sha256(content.encode()).hexdigest()
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_guidelines.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/claude_lint/guidelines.py tests/test_guidelines.py
git commit -m "feat: add CLAUDE.md reader with hash tracking"
```

---

## Task 4: Cache Manager

**Files:**
- Create: `src/claude_lint/cache.py`
- Create: `tests/test_cache.py`

**Step 1: Write failing test for cache operations**

Create `tests/test_cache.py`:

```python
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
        violations=["missing docstring"],
        timestamp=1234567890
    )

    assert entry.file_hash == "abc123"
    assert entry.claude_md_hash == "def456"
    assert entry.violations == ["missing docstring"]


def test_load_cache_empty():
    """Test loading cache when file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / ".lint-claude-cache.json"

        cache = load_cache(cache_path)

        assert cache.entries == {}
        assert cache.claude_md_hash == ""


def test_save_and_load_cache():
    """Test saving and loading cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / ".lint-claude-cache.json"

        # Create cache with entry
        cache = Cache(
            claude_md_hash="hash123",
            entries={
                "file1.py": CacheEntry(
                    file_hash="filehash1",
                    claude_md_hash="hash123",
                    violations=["error1"],
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cache.py -v`
Expected: FAIL with "No module named 'claude_lint.cache'"

**Step 3: Implement cache module**

Create `src/claude_lint/cache.py`:

```python
"""Cache management for file analysis results."""
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class CacheEntry:
    """Cache entry for a single file."""
    file_hash: str
    claude_md_hash: str
    violations: list[str]
    timestamp: int


@dataclass
class Cache:
    """Cache for analysis results."""
    claude_md_hash: str
    entries: dict[str, CacheEntry]


def load_cache(cache_path: Path) -> Cache:
    """Load cache from file or return empty cache.

    Args:
        cache_path: Path to cache file

    Returns:
        Cache object
    """
    if not cache_path.exists():
        return Cache(claude_md_hash="", entries={})

    with open(cache_path) as f:
        data = json.load(f)

    entries = {}
    for file_path, entry_data in data.get("entries", {}).items():
        entries[file_path] = CacheEntry(**entry_data)

    return Cache(
        claude_md_hash=data.get("claudeMdHash", ""),
        entries=entries
    )


def save_cache(cache: Cache, cache_path: Path) -> None:
    """Save cache to file.

    Args:
        cache: Cache object to save
        cache_path: Path to cache file
    """
    entries_dict = {}
    for file_path, entry in cache.entries.items():
        entries_dict[file_path] = asdict(entry)

    data = {
        "claudeMdHash": cache.claude_md_hash,
        "entries": entries_dict
    }

    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cache.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/claude_lint/cache.py tests/test_cache.py
git commit -m "feat: add cache manager for analysis results"
```

---

## Task 5: Git Integration

**Files:**
- Create: `src/claude_lint/git_utils.py`
- Create: `tests/test_git_utils.py`

**Step 1: Write failing test for git operations**

Create `tests/test_git_utils.py`:

```python
import tempfile
from pathlib import Path
import subprocess
import pytest
from claude_lint.git_utils import (
    get_changed_files_from_branch,
    get_working_directory_files,
    get_staged_files,
    is_git_repo
)


def setup_git_repo(tmpdir: Path) -> Path:
    """Helper to set up a git repo for testing."""
    repo_dir = tmpdir / "repo"
    repo_dir.mkdir()

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_dir,
        check=True,
        capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_dir,
        check=True,
        capture_output=True
    )

    # Create initial commit
    (repo_dir / "file1.py").write_text("print('hello')")
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=repo_dir,
        check=True,
        capture_output=True
    )

    return repo_dir


def test_is_git_repo():
    """Test checking if directory is a git repo."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        repo_dir = setup_git_repo(tmpdir)

        assert is_git_repo(repo_dir) is True
        assert is_git_repo(tmpdir / "nonexistent") is False


def test_get_changed_files_from_branch():
    """Test getting files changed from a branch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        repo_dir = setup_git_repo(tmpdir)

        # Create a new file
        (repo_dir / "file2.py").write_text("print('world')")
        subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "add file2"],
            cwd=repo_dir,
            check=True,
            capture_output=True
        )

        # Get files changed from HEAD~1
        files = get_changed_files_from_branch(repo_dir, "HEAD~1")

        assert "file2.py" in files
        assert "file1.py" not in files


def test_get_working_directory_files():
    """Test getting modified and untracked files in working directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        repo_dir = setup_git_repo(tmpdir)

        # Modify existing file
        (repo_dir / "file1.py").write_text("print('modified')")

        # Add untracked file
        (repo_dir / "file3.py").write_text("print('new')")

        files = get_working_directory_files(repo_dir)

        assert "file1.py" in files  # modified
        assert "file3.py" in files  # untracked


def test_get_staged_files():
    """Test getting staged files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        repo_dir = setup_git_repo(tmpdir)

        # Modify and stage file
        (repo_dir / "file1.py").write_text("print('staged')")
        subprocess.run(["git", "add", "file1.py"], cwd=repo_dir, check=True, capture_output=True)

        # Create unstaged file
        (repo_dir / "file3.py").write_text("print('unstaged')")

        files = get_staged_files(repo_dir)

        assert "file1.py" in files
        assert "file3.py" not in files
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_git_utils.py -v`
Expected: FAIL with "No module named 'claude_lint.git_utils'"

**Step 3: Implement git utils module**

Create `src/claude_lint/git_utils.py`:

```python
"""Git integration utilities."""
from pathlib import Path
import subprocess
from typing import Optional


def is_git_repo(path: Path) -> bool:
    """Check if path is inside a git repository.

    Args:
        path: Directory path to check

    Returns:
        True if path is in a git repo, False otherwise
    """
    try:
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_changed_files_from_branch(repo_path: Path, base_branch: str) -> list[str]:
    """Get files changed from a base branch.

    Args:
        repo_path: Path to git repository
        base_branch: Base branch to compare against (e.g., 'main', 'HEAD~1')

    Returns:
        List of changed file paths relative to repo root
    """
    result = subprocess.run(
        ["git", "diff", "--name-only", base_branch],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True
    )

    files = [f.strip() for f in result.stdout.split("\n") if f.strip()]
    return files


def get_working_directory_files(repo_path: Path) -> list[str]:
    """Get modified and untracked files in working directory.

    Args:
        repo_path: Path to git repository

    Returns:
        List of modified/untracked file paths relative to repo root
    """
    # Get modified files
    result = subprocess.run(
        ["git", "diff", "--name-only"],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True
    )
    modified = [f.strip() for f in result.stdout.split("\n") if f.strip()]

    # Get untracked files
    result = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True
    )
    untracked = [f.strip() for f in result.stdout.split("\n") if f.strip()]

    return list(set(modified + untracked))


def get_staged_files(repo_path: Path) -> list[str]:
    """Get staged files.

    Args:
        repo_path: Path to git repository

    Returns:
        List of staged file paths relative to repo root
    """
    result = subprocess.run(
        ["git", "diff", "--name-only", "--cached"],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True
    )

    files = [f.strip() for f in result.stdout.split("\n") if f.strip()]
    return files
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_git_utils.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/claude_lint/git_utils.py tests/test_git_utils.py
git commit -m "feat: add git integration utilities"
```

---

## Task 6: File Collector

**Files:**
- Create: `src/claude_lint/collector.py`
- Create: `tests/test_collector.py`

**Step 1: Write failing test for file collection**

Create `tests/test_collector.py`:

```python
import tempfile
from pathlib import Path
import pytest
from claude_lint.collector import FileCollector
from claude_lint.config import Config


def test_collect_files_with_patterns():
    """Test collecting files with include/exclude patterns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create file structure
        (tmpdir / "src").mkdir()
        (tmpdir / "src" / "main.py").write_text("code")
        (tmpdir / "src" / "utils.py").write_text("code")
        (tmpdir / "tests").mkdir()
        (tmpdir / "tests" / "test_main.py").write_text("test")
        (tmpdir / "node_modules").mkdir()
        (tmpdir / "node_modules" / "lib.js").write_text("js")

        config = Config(
            include=["**/*.py"],
            exclude=["node_modules/**"],
            batch_size=10
        )

        collector = FileCollector(tmpdir, config)
        files = collector.collect_all()

        # Should include Python files but exclude node_modules
        file_names = [f.name for f in files]
        assert "main.py" in file_names
        assert "utils.py" in file_names
        assert "test_main.py" in file_names
        assert "lib.js" not in file_names


def test_filter_by_file_list():
    """Test filtering collected files by a specific list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create files
        (tmpdir / "file1.py").write_text("code")
        (tmpdir / "file2.py").write_text("code")
        (tmpdir / "file3.py").write_text("code")

        config = Config(
            include=["**/*.py"],
            exclude=[],
            batch_size=10
        )

        collector = FileCollector(tmpdir, config)

        # Filter to only specific files
        filtered = collector.filter_by_list(["file1.py", "file3.py"])

        file_names = [f.name for f in filtered]
        assert "file1.py" in file_names
        assert "file3.py" in file_names
        assert "file2.py" not in file_names


def test_compute_file_hash():
    """Test computing hash of file content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_file = tmpdir / "test.py"
        content = "print('hello')"
        test_file.write_text(content)

        config = Config(include=["**/*.py"], exclude=[], batch_size=10)
        collector = FileCollector(tmpdir, config)

        file_hash = collector.compute_hash(test_file)

        # Hash should be consistent
        import hashlib
        expected = hashlib.sha256(content.encode()).hexdigest()
        assert file_hash == expected
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_collector.py -v`
Expected: FAIL with "No module named 'claude_lint.collector'"

**Step 3: Implement collector module**

Create `src/claude_lint/collector.py`:

```python
"""File collection with pattern matching."""
import hashlib
from pathlib import Path
from typing import Optional
import fnmatch

from claude_lint.config import Config


class FileCollector:
    """Collects files based on include/exclude patterns."""

    def __init__(self, root_path: Path, config: Config):
        """Initialize file collector.

        Args:
            root_path: Root directory to search from
            config: Configuration with include/exclude patterns
        """
        self.root_path = root_path
        self.config = config

    def collect_all(self) -> list[Path]:
        """Collect all files matching patterns.

        Returns:
            List of matching file paths
        """
        all_files = []

        for pattern in self.config.include:
            # Use rglob for recursive patterns
            if "**" in pattern:
                glob_pattern = pattern.replace("**", "**/*")
                matching = self.root_path.rglob(glob_pattern.lstrip("**/"))
            else:
                matching = self.root_path.glob(pattern)

            for file_path in matching:
                if file_path.is_file():
                    # Check if excluded
                    relative = file_path.relative_to(self.root_path)
                    if not self._is_excluded(relative):
                        all_files.append(file_path)

        return list(set(all_files))  # Remove duplicates

    def filter_by_list(self, file_list: list[str]) -> list[Path]:
        """Filter collected files by specific list.

        Args:
            file_list: List of file paths (relative to root)

        Returns:
            List of matching file paths that pass include/exclude filters
        """
        filtered = []

        for file_str in file_list:
            file_path = self.root_path / file_str
            if file_path.is_file():
                relative = Path(file_str)

                # Check if matches include patterns
                matches_include = any(
                    fnmatch.fnmatch(str(relative), pattern)
                    for pattern in self.config.include
                )

                if matches_include and not self._is_excluded(relative):
                    filtered.append(file_path)

        return filtered

    def compute_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file content.

        Args:
            file_path: Path to file

        Returns:
            Hex digest of file content hash
        """
        content = file_path.read_text()
        return hashlib.sha256(content.encode()).hexdigest()

    def _is_excluded(self, relative_path: Path) -> bool:
        """Check if file is excluded by patterns.

        Args:
            relative_path: File path relative to root

        Returns:
            True if file should be excluded
        """
        return any(
            fnmatch.fnmatch(str(relative_path), pattern)
            for pattern in self.config.exclude
        )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_collector.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/claude_lint/collector.py tests/test_collector.py
git commit -m "feat: add file collector with pattern matching"
```

---

## Task 7: Batch Processor with XML Prompts

**Files:**
- Create: `src/claude_lint/processor.py`
- Create: `tests/test_processor.py`

**Step 1: Write failing test for batch processor**

Create `tests/test_processor.py`:

```python
from pathlib import Path
import pytest
from claude_lint.processor import BatchProcessor, build_xml_prompt


def test_build_xml_prompt():
    """Test building XML prompt for Claude API."""
    claude_md_content = "# Guidelines\n\nFollow TDD."
    files = {
        "src/main.py": "def main():\n    pass",
        "src/utils.py": "def helper():\n    return 42"
    }

    prompt = build_xml_prompt(claude_md_content, files)

    # Check structure
    assert "<guidelines>" in prompt
    assert "Follow TDD" in prompt
    assert "</guidelines>" in prompt
    assert "<files>" in prompt
    assert '<file path="src/main.py">' in prompt
    assert '<file path="src/utils.py">' in prompt
    assert "def main()" in prompt
    assert "def helper()" in prompt
    assert "</files>" in prompt


def test_batch_files():
    """Test batching files into groups."""
    files = [f"file{i}.py" for i in range(25)]
    batch_size = 10

    processor = BatchProcessor(batch_size)
    batches = processor.create_batches(files)

    assert len(batches) == 3  # 10, 10, 5
    assert len(batches[0]) == 10
    assert len(batches[1]) == 10
    assert len(batches[2]) == 5


def test_parse_response():
    """Test parsing Claude API response."""
    response = """
    Here are the compliance issues:

    ```json
    {
      "results": [
        {
          "file": "src/main.py",
          "violations": [
            {
              "type": "missing-pattern",
              "message": "No tests found for this module",
              "line": null
            }
          ]
        },
        {
          "file": "src/utils.py",
          "violations": []
        }
      ]
    }
    ```
    """

    processor = BatchProcessor(batch_size=10)
    results = processor.parse_response(response)

    assert len(results) == 2
    assert results[0]["file"] == "src/main.py"
    assert len(results[0]["violations"]) == 1
    assert results[1]["file"] == "src/utils.py"
    assert len(results[1]["violations"]) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_processor.py -v`
Expected: FAIL with "No module named 'claude_lint.processor'"

**Step 3: Implement processor module**

Create `src/claude_lint/processor.py`:

```python
"""Batch processing and XML prompt generation."""
import json
import re
from typing import Any


def build_xml_prompt(claude_md_content: str, files: dict[str, str]) -> str:
    """Build XML prompt for Claude API.

    Args:
        claude_md_content: Content of CLAUDE.md
        files: Dict mapping file paths to their content

    Returns:
        XML formatted prompt
    """
    # Build files XML
    files_xml = ""
    for file_path, content in files.items():
        files_xml += f'  <file path="{file_path}">\n{content}\n  </file>\n'

    prompt = f"""<guidelines>
{claude_md_content}
</guidelines>

Check the following files for compliance with the guidelines above.
For each file, evaluate:
1. Pattern compliance - Does the code follow specific patterns mentioned?
2. Principle adherence - Does the code embody the philosophy described?
3. Anti-pattern detection - Does the code contain things warned against?

<files>
{files_xml}</files>

Return results in this JSON format:
{{
  "results": [
    {{
      "file": "path/to/file",
      "violations": [
        {{
          "type": "missing-pattern|principle-violation|anti-pattern",
          "message": "Description of the issue",
          "line": null or line number
        }}
      ]
    }}
  ]
}}

If a file has no violations, include it with an empty violations array.
"""

    return prompt


class BatchProcessor:
    """Processes files in batches."""

    def __init__(self, batch_size: int):
        """Initialize batch processor.

        Args:
            batch_size: Number of files per batch
        """
        self.batch_size = batch_size

    def create_batches(self, items: list[Any]) -> list[list[Any]]:
        """Split items into batches.

        Args:
            items: List of items to batch

        Returns:
            List of batches
        """
        batches = []
        for i in range(0, len(items), self.batch_size):
            batches.append(items[i:i + self.batch_size])
        return batches

    def parse_response(self, response: str) -> list[dict[str, Any]]:
        """Parse Claude API response to extract results.

        Args:
            response: Raw response text from Claude

        Returns:
            List of file results with violations
        """
        # Extract JSON from markdown code blocks if present
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return []

        try:
            data = json.loads(json_str)
            return data.get("results", [])
        except json.JSONDecodeError:
            return []
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_processor.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/claude_lint/processor.py tests/test_processor.py
git commit -m "feat: add batch processor with XML prompt generation"
```

---

## Task 8: Claude API Integration with Prompt Caching

**Files:**
- Create: `src/claude_lint/api_client.py`
- Create: `tests/test_api_client.py`

**Step 1: Write failing test for API client**

Create `tests/test_api_client.py`:

```python
from unittest.mock import Mock, patch
import pytest
from claude_lint.api_client import ClaudeClient


@patch("claude_lint.api_client.Anthropic")
def test_claude_client_initialization(mock_anthropic):
    """Test initializing Claude API client."""
    client = ClaudeClient(api_key="test-key")

    mock_anthropic.assert_called_once_with(api_key="test-key")


@patch("claude_lint.api_client.Anthropic")
def test_analyze_with_caching(mock_anthropic):
    """Test making API call with prompt caching."""
    # Setup mock
    mock_response = Mock()
    mock_response.content = [Mock(text='{"results": []}')]
    mock_anthropic.return_value.messages.create.return_value = mock_response

    client = ClaudeClient(api_key="test-key")

    # Make request
    response = client.analyze_files(
        guidelines="# Guidelines",
        prompt="Check these files"
    )

    # Verify caching was used
    call_args = mock_anthropic.return_value.messages.create.call_args
    assert call_args[1]["model"] == "claude-sonnet-4-5-20250929"

    # Check system message uses cache_control
    system_messages = call_args[1]["system"]
    assert len(system_messages) == 1
    assert system_messages[0]["type"] == "text"
    assert system_messages[0]["text"] == "# Guidelines"
    assert system_messages[0]["cache_control"] == {"type": "ephemeral"}

    # Check user message
    messages = call_args[1]["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Check these files"


@patch("claude_lint.api_client.Anthropic")
def test_get_usage_stats(mock_anthropic):
    """Test extracting usage statistics from response."""
    mock_response = Mock()
    mock_response.content = [Mock(text='{"results": []}')]
    mock_response.usage = Mock(
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=200,
        cache_read_input_tokens=0
    )
    mock_anthropic.return_value.messages.create.return_value = mock_response

    client = ClaudeClient(api_key="test-key")
    response = client.analyze_files(
        guidelines="# Guidelines",
        prompt="Check files"
    )

    stats = client.get_last_usage_stats()

    assert stats["input_tokens"] == 100
    assert stats["output_tokens"] == 50
    assert stats["cache_creation_tokens"] == 200
    assert stats["cache_read_tokens"] == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_client.py -v`
Expected: FAIL with "No module named 'claude_lint.api_client'"

**Step 3: Implement API client module**

Create `src/claude_lint/api_client.py`:

```python
"""Claude API client with prompt caching support."""
from typing import Optional
from anthropic import Anthropic


class ClaudeClient:
    """Client for Claude API with prompt caching."""

    def __init__(self, api_key: str):
        """Initialize Claude API client.

        Args:
            api_key: Anthropic API key
        """
        self.client = Anthropic(api_key=api_key)
        self.last_response = None

    def analyze_files(self, guidelines: str, prompt: str) -> str:
        """Analyze files using Claude API with cached guidelines.

        Args:
            guidelines: CLAUDE.md content (will be cached)
            prompt: Prompt with files to analyze

        Returns:
            Response text from Claude
        """
        # Use prompt caching for guidelines
        response = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": guidelines,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        self.last_response = response
        return response.content[0].text

    def get_last_usage_stats(self) -> Optional[dict]:
        """Get usage statistics from last API call.

        Returns:
            Dict with token usage stats or None if no request made
        """
        if self.last_response is None:
            return None

        usage = self.last_response.usage
        return {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_creation_tokens": getattr(usage, "cache_creation_input_tokens", 0),
            "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0)
        }
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_api_client.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/claude_lint/api_client.py tests/test_api_client.py
git commit -m "feat: add Claude API client with prompt caching"
```

---

## Task 9: Retry Logic with Exponential Backoff

**Files:**
- Create: `src/claude_lint/retry.py`
- Create: `tests/test_retry.py`

**Step 1: Write failing test for retry logic**

Create `tests/test_retry.py`:

```python
from unittest.mock import Mock
import pytest
from claude_lint.retry import retry_with_backoff


def test_retry_success_on_first_attempt():
    """Test function succeeds on first attempt."""
    mock_func = Mock(return_value="success")

    result = retry_with_backoff(mock_func, max_retries=3)

    assert result == "success"
    assert mock_func.call_count == 1


def test_retry_success_after_failures():
    """Test function succeeds after some failures."""
    mock_func = Mock(side_effect=[
        Exception("fail1"),
        Exception("fail2"),
        "success"
    ])

    result = retry_with_backoff(mock_func, max_retries=3, initial_delay=0.01)

    assert result == "success"
    assert mock_func.call_count == 3


def test_retry_exhausted():
    """Test all retries exhausted."""
    mock_func = Mock(side_effect=Exception("always fails"))

    with pytest.raises(Exception, match="always fails"):
        retry_with_backoff(mock_func, max_retries=3, initial_delay=0.01)

    assert mock_func.call_count == 3


def test_exponential_backoff_timing():
    """Test that backoff delays increase exponentially."""
    import time

    call_times = []

    def failing_func():
        call_times.append(time.time())
        if len(call_times) < 3:
            raise Exception("fail")
        return "success"

    retry_with_backoff(failing_func, max_retries=3, initial_delay=0.1)

    # Check delays between calls
    assert len(call_times) == 3
    delay1 = call_times[1] - call_times[0]
    delay2 = call_times[2] - call_times[1]

    # Second delay should be roughly 2x first delay
    assert delay2 > delay1 * 1.8  # Account for timing variance
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_retry.py -v`
Expected: FAIL with "No module named 'claude_lint.retry'"

**Step 3: Implement retry module**

Create `src/claude_lint/retry.py`:

```python
"""Retry logic with exponential backoff."""
import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
) -> T:
    """Retry a function with exponential backoff.

    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry

    Returns:
        Result from successful function call

    Raises:
        Exception: The last exception if all retries exhausted
    """
    last_exception = None
    delay = initial_delay

    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e

            if attempt < max_retries - 1:  # Don't sleep on last attempt
                time.sleep(delay)
                delay *= backoff_factor

    # All retries exhausted
    raise last_exception
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_retry.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/claude_lint/retry.py tests/test_retry.py
git commit -m "feat: add retry logic with exponential backoff"
```

---

## Task 10: Progress Tracking and Resume Capability

**Files:**
- Create: `src/claude_lint/progress.py`
- Create: `tests/test_progress.py`

**Step 1: Write failing test for progress tracking**

Create `tests/test_progress.py`:

```python
import json
import tempfile
from pathlib import Path
import pytest
from claude_lint.progress import ProgressTracker, ProgressState


def test_progress_tracker_initialization():
    """Test initializing progress tracker."""
    with tempfile.TemporaryDirectory() as tmpdir:
        progress_file = Path(tmpdir) / ".lint-claude-progress.json"

        tracker = ProgressTracker(progress_file, total_batches=5)

        assert tracker.total_batches == 5
        assert tracker.completed_batches == 0
        assert tracker.is_complete() is False


def test_update_progress():
    """Test updating progress."""
    with tempfile.TemporaryDirectory() as tmpdir:
        progress_file = Path(tmpdir) / ".lint-claude-progress.json"

        tracker = ProgressTracker(progress_file, total_batches=3)

        tracker.update(batch_index=0, results=[{"file": "a.py", "violations": []}])
        tracker.update(batch_index=1, results=[{"file": "b.py", "violations": []}])

        assert tracker.completed_batches == 2
        assert tracker.get_progress_percentage() == pytest.approx(66.7, rel=0.1)


def test_save_and_load_progress():
    """Test saving and loading progress."""
    with tempfile.TemporaryDirectory() as tmpdir:
        progress_file = Path(tmpdir) / ".lint-claude-progress.json"

        # Create and save progress
        tracker = ProgressTracker(progress_file, total_batches=5)
        tracker.update(batch_index=0, results=[{"file": "a.py", "violations": []}])
        tracker.update(batch_index=1, results=[{"file": "b.py", "violations": []}])
        tracker.save()

        # Load progress
        loaded = ProgressTracker.load(progress_file)

        assert loaded.total_batches == 5
        assert loaded.completed_batches == 2
        assert 0 in loaded.state.completed_batch_indices
        assert 1 in loaded.state.completed_batch_indices


def test_resume_from_progress():
    """Test resuming from saved progress."""
    with tempfile.TemporaryDirectory() as tmpdir:
        progress_file = Path(tmpdir) / ".lint-claude-progress.json"

        # Create initial progress
        tracker = ProgressTracker(progress_file, total_batches=5)
        tracker.update(batch_index=0, results=[{"file": "a.py", "violations": []}])
        tracker.update(batch_index=2, results=[{"file": "c.py", "violations": []}])
        tracker.save()

        # Resume
        resumed = ProgressTracker.load(progress_file)
        remaining = resumed.get_remaining_batch_indices()

        assert remaining == [1, 3, 4]


def test_cleanup_on_complete():
    """Test that progress file is cleaned up when complete."""
    with tempfile.TemporaryDirectory() as tmpdir:
        progress_file = Path(tmpdir) / ".lint-claude-progress.json"

        tracker = ProgressTracker(progress_file, total_batches=2)
        tracker.update(batch_index=0, results=[])
        tracker.update(batch_index=1, results=[])
        tracker.save()

        assert tracker.is_complete() is True
        tracker.cleanup()

        assert not progress_file.exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_progress.py -v`
Expected: FAIL with "No module named 'claude_lint.progress'"

**Step 3: Implement progress tracking module**

Create `src/claude_lint/progress.py`:

```python
"""Progress tracking and resume capability."""
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


@dataclass
class ProgressState:
    """State for progress tracking."""
    total_batches: int
    completed_batch_indices: list[int] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)


class ProgressTracker:
    """Tracks progress of file analysis for resume capability."""

    def __init__(self, progress_file: Path, total_batches: int):
        """Initialize progress tracker.

        Args:
            progress_file: Path to progress file
            total_batches: Total number of batches to process
        """
        self.progress_file = progress_file
        self.total_batches = total_batches
        self.state = ProgressState(total_batches=total_batches)

    @property
    def completed_batches(self) -> int:
        """Get number of completed batches."""
        return len(self.state.completed_batch_indices)

    def update(self, batch_index: int, results: list[dict[str, Any]]) -> None:
        """Update progress with batch results.

        Args:
            batch_index: Index of completed batch
            results: Results from this batch
        """
        if batch_index not in self.state.completed_batch_indices:
            self.state.completed_batch_indices.append(batch_index)

        self.state.results.extend(results)

    def save(self) -> None:
        """Save progress to file."""
        data = asdict(self.state)
        with open(self.progress_file, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, progress_file: Path) -> "ProgressTracker":
        """Load progress from file.

        Args:
            progress_file: Path to progress file

        Returns:
            ProgressTracker with loaded state
        """
        with open(progress_file) as f:
            data = json.load(f)

        tracker = cls(progress_file, total_batches=data["total_batches"])
        tracker.state = ProgressState(**data)
        return tracker

    def get_remaining_batch_indices(self) -> list[int]:
        """Get list of batch indices that still need processing.

        Returns:
            List of remaining batch indices
        """
        all_indices = set(range(self.total_batches))
        completed = set(self.state.completed_batch_indices)
        return sorted(all_indices - completed)

    def is_complete(self) -> bool:
        """Check if all batches are complete.

        Returns:
            True if all batches processed
        """
        return len(self.state.completed_batch_indices) == self.total_batches

    def get_progress_percentage(self) -> float:
        """Get progress as percentage.

        Returns:
            Progress percentage (0-100)
        """
        return (self.completed_batches / self.total_batches) * 100

    def cleanup(self) -> None:
        """Remove progress file."""
        if self.progress_file.exists():
            self.progress_file.unlink()
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_progress.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/claude_lint/progress.py tests/test_progress.py
git commit -m "feat: add progress tracking with resume capability"
```

---

## Task 11: Report Formatter

**Files:**
- Create: `src/claude_lint/reporter.py`
- Create: `tests/test_reporter.py`

**Step 1: Write failing test for reporter**

Create `tests/test_reporter.py`:

```python
import json
import pytest
from claude_lint.reporter import Reporter, format_detailed_report, format_json_report


def test_format_detailed_report():
    """Test formatting detailed human-readable report."""
    results = [
        {
            "file": "src/main.py",
            "violations": [
                {
                    "type": "missing-pattern",
                    "message": "No tests found",
                    "line": None
                },
                {
                    "type": "anti-pattern",
                    "message": "Nested complexity detected",
                    "line": 42
                }
            ]
        },
        {
            "file": "src/utils.py",
            "violations": []
        }
    ]

    report = format_detailed_report(results)

    assert "src/main.py" in report
    assert "2 violation(s)" in report
    assert "No tests found" in report
    assert "line 42" in report
    assert "src/utils.py" in report
    assert "✓ No violations" in report


def test_format_json_report():
    """Test formatting JSON report."""
    results = [
        {
            "file": "src/main.py",
            "violations": [
                {
                    "type": "missing-pattern",
                    "message": "No tests found",
                    "line": None
                }
            ]
        }
    ]

    report = format_json_report(results)
    data = json.loads(report)

    assert "results" in data
    assert len(data["results"]) == 1
    assert data["results"][0]["file"] == "src/main.py"
    assert data["summary"]["total_files"] == 1
    assert data["summary"]["files_with_violations"] == 1


def test_reporter_get_exit_code():
    """Test getting exit code based on results."""
    reporter = Reporter()

    # No violations
    clean_results = [{"file": "a.py", "violations": []}]
    assert reporter.get_exit_code(clean_results) == 0

    # Has violations
    dirty_results = [
        {"file": "a.py", "violations": []},
        {"file": "b.py", "violations": [{"type": "error", "message": "bad"}]}
    ]
    assert reporter.get_exit_code(dirty_results) == 1


def test_reporter_print_summary():
    """Test printing summary statistics."""
    results = [
        {"file": "a.py", "violations": []},
        {"file": "b.py", "violations": [{"type": "error", "message": "e1"}]},
        {"file": "c.py", "violations": [
            {"type": "error", "message": "e2"},
            {"type": "warn", "message": "e3"}
        ]}
    ]

    reporter = Reporter()
    summary = reporter.get_summary(results)

    assert summary["total_files"] == 3
    assert summary["files_with_violations"] == 2
    assert summary["total_violations"] == 3
    assert summary["clean_files"] == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_reporter.py -v`
Expected: FAIL with "No module named 'claude_lint.reporter'"

**Step 3: Implement reporter module**

Create `src/claude_lint/reporter.py`:

```python
"""Report formatting and output."""
import json
from typing import Any


def format_detailed_report(results: list[dict[str, Any]]) -> str:
    """Format results as detailed human-readable report.

    Args:
        results: List of file results with violations

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 70)
    lines.append("CLAUDE.MD COMPLIANCE REPORT")
    lines.append("=" * 70)
    lines.append("")

    for result in results:
        file_path = result["file"]
        violations = result["violations"]

        if violations:
            lines.append(f"📄 {file_path}")
            lines.append(f"   {len(violations)} violation(s) found:")
            lines.append("")

            for violation in violations:
                vtype = violation["type"]
                message = violation["message"]
                line = violation.get("line")

                line_info = f" (line {line})" if line else ""
                lines.append(f"   ⚠️  [{vtype}]{line_info}")
                lines.append(f"      {message}")
                lines.append("")
        else:
            lines.append(f"✓ {file_path}")
            lines.append("   No violations")
            lines.append("")

    return "\n".join(lines)


def format_json_report(results: list[dict[str, Any]]) -> str:
    """Format results as JSON.

    Args:
        results: List of file results with violations

    Returns:
        JSON string
    """
    total_files = len(results)
    files_with_violations = sum(1 for r in results if r["violations"])
    total_violations = sum(len(r["violations"]) for r in results)

    report = {
        "results": results,
        "summary": {
            "total_files": total_files,
            "files_with_violations": files_with_violations,
            "clean_files": total_files - files_with_violations,
            "total_violations": total_violations
        }
    }

    return json.dumps(report, indent=2)


class Reporter:
    """Handles result reporting and output."""

    def get_exit_code(self, results: list[dict[str, Any]]) -> int:
        """Get exit code based on results.

        Args:
            results: List of file results

        Returns:
            0 if no violations, 1 if violations found
        """
        has_violations = any(r["violations"] for r in results)
        return 1 if has_violations else 0

    def get_summary(self, results: list[dict[str, Any]]) -> dict[str, int]:
        """Get summary statistics.

        Args:
            results: List of file results

        Returns:
            Dict with summary counts
        """
        total_files = len(results)
        files_with_violations = sum(1 for r in results if r["violations"])
        total_violations = sum(len(r["violations"]) for r in results)

        return {
            "total_files": total_files,
            "files_with_violations": files_with_violations,
            "clean_files": total_files - files_with_violations,
            "total_violations": total_violations
        }
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_reporter.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/claude_lint/reporter.py tests/test_reporter.py
git commit -m "feat: add report formatter with detailed and JSON output"
```

---

## Task 12: Main Orchestrator

**Files:**
- Create: `src/claude_lint/orchestrator.py`
- Create: `tests/test_orchestrator.py`

**Step 1: Write failing test for orchestrator**

Create `tests/test_orchestrator.py`:

```python
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
from claude_lint.orchestrator import Orchestrator
from claude_lint.config import Config


@patch("claude_lint.orchestrator.ClaudeClient")
@patch("claude_lint.orchestrator.is_git_repo")
def test_orchestrator_full_scan(mock_is_git, mock_claude_client):
    """Test full project scan mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test files
        (tmpdir / "file1.py").write_text("print('hello')")
        (tmpdir / "file2.py").write_text("print('world')")
        (tmpdir / "CLAUDE.md").write_text("# Guidelines\n\nUse TDD.")

        # Mock git check
        mock_is_git.return_value = False

        # Mock Claude API
        mock_client = Mock()
        mock_client.analyze_files.return_value = """
        ```json
        {
          "results": [
            {"file": "file1.py", "violations": []},
            {"file": "file2.py", "violations": []}
          ]
        }
        ```
        """
        mock_claude_client.return_value = mock_client

        # Run orchestrator
        config = Config(
            include=["**/*.py"],
            exclude=["tests/**"],
            batch_size=10,
            api_key="test-key"
        )

        orchestrator = Orchestrator(tmpdir, config)
        results = orchestrator.run(mode="full")

        # Verify
        assert len(results) == 2
        assert mock_client.analyze_files.called


@patch("claude_lint.orchestrator.ClaudeClient")
@patch("claude_lint.orchestrator.get_changed_files_from_branch")
@patch("claude_lint.orchestrator.is_git_repo")
def test_orchestrator_diff_mode(mock_is_git, mock_git_diff, mock_claude_client):
    """Test diff mode with git."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test files
        (tmpdir / "file1.py").write_text("code")
        (tmpdir / "file2.py").write_text("code")
        (tmpdir / "CLAUDE.md").write_text("# Guidelines")

        # Mock git
        mock_is_git.return_value = True
        mock_git_diff.return_value = ["file1.py"]  # Only file1 changed

        # Mock Claude API
        mock_client = Mock()
        mock_client.analyze_files.return_value = """
        ```json
        {"results": [{"file": "file1.py", "violations": []}]}
        ```
        """
        mock_claude_client.return_value = mock_client

        # Run orchestrator
        config = Config(
            include=["**/*.py"],
            exclude=[],
            batch_size=10,
            api_key="test-key"
        )

        orchestrator = Orchestrator(tmpdir, config)
        results = orchestrator.run(mode="diff", base_branch="main")

        # Verify only file1 was checked
        assert len(results) == 1
        assert results[0]["file"] == "file1.py"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_orchestrator.py -v`
Expected: FAIL with "No module named 'claude_lint.orchestrator'"

**Step 3: Implement orchestrator module**

Create `src/claude_lint/orchestrator.py`:

```python
"""Main orchestrator coordinating all components."""
import os
from pathlib import Path
from typing import Any, Optional

from claude_lint.api_client import ClaudeClient
from claude_lint.cache import Cache, CacheEntry, load_cache, save_cache
from claude_lint.collector import FileCollector
from claude_lint.config import Config
from claude_lint.git_utils import (
    is_git_repo,
    get_changed_files_from_branch,
    get_working_directory_files,
    get_staged_files
)
from claude_lint.guidelines import read_claude_md, get_claude_md_hash
from claude_lint.processor import BatchProcessor, build_xml_prompt
from claude_lint.progress import ProgressTracker
from claude_lint.retry import retry_with_backoff


class Orchestrator:
    """Main orchestrator for lint-claude."""

    def __init__(self, project_root: Path, config: Config):
        """Initialize orchestrator.

        Args:
            project_root: Project root directory
            config: Configuration
        """
        self.project_root = project_root
        self.config = config

        # Get API key
        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("API key required via config or ANTHROPIC_API_KEY env var")

        self.client = ClaudeClient(api_key)
        self.collector = FileCollector(project_root, config)
        self.processor = BatchProcessor(config.batch_size)

        # Load CLAUDE.md
        self.guidelines = read_claude_md(project_root)
        self.guidelines_hash = get_claude_md_hash(self.guidelines)

        # Load cache
        cache_path = project_root / ".lint-claude-cache.json"
        self.cache = load_cache(cache_path)
        self.cache_path = cache_path

        # Progress tracking
        progress_path = project_root / ".lint-claude-progress.json"
        self.progress_path = progress_path

    def run(
        self,
        mode: str = "full",
        base_branch: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Run compliance check.

        Args:
            mode: Check mode - 'full', 'diff', 'working', 'staged'
            base_branch: Base branch for diff mode

        Returns:
            List of results for all checked files
        """
        # Collect files to check
        files_to_check = self._collect_files(mode, base_branch)

        if not files_to_check:
            return []

        # Filter using cache
        files_needing_check = self._filter_cached(files_to_check)

        if not files_needing_check:
            # All files cached, return cached results
            return self._get_cached_results(files_to_check)

        # Create batches
        batches = self.processor.create_batches(files_needing_check)

        # Check for resumable progress
        tracker = self._init_progress_tracker(len(batches))

        # Process batches
        all_results = list(tracker.state.results)  # Start with resumed results

        for batch_idx in tracker.get_remaining_batch_indices():
            batch = batches[batch_idx]

            # Read file contents
            file_contents = {}
            for file_path in batch:
                rel_path = file_path.relative_to(self.project_root)
                content = file_path.read_text()
                file_contents[str(rel_path)] = content

            # Build prompt
            prompt = build_xml_prompt(self.guidelines, file_contents)

            # Make API call with retry
            def api_call():
                return self.client.analyze_files(self.guidelines, prompt)

            response = retry_with_backoff(api_call)

            # Parse results
            batch_results = self.processor.parse_response(response)
            all_results.extend(batch_results)

            # Update cache
            for result in batch_results:
                file_path = self.project_root / result["file"]
                file_hash = self.collector.compute_hash(file_path)

                self.cache.entries[result["file"]] = CacheEntry(
                    file_hash=file_hash,
                    claude_md_hash=self.guidelines_hash,
                    violations=result["violations"],
                    timestamp=int(Path(file_path).stat().st_mtime)
                )

            # Save progress
            tracker.update(batch_idx, batch_results)
            tracker.save()
            save_cache(self.cache, self.cache_path)

        # Cleanup progress on completion
        if tracker.is_complete():
            tracker.cleanup()

        # Update cache hash
        self.cache.claude_md_hash = self.guidelines_hash
        save_cache(self.cache, self.cache_path)

        return all_results

    def _collect_files(
        self,
        mode: str,
        base_branch: Optional[str]
    ) -> list[Path]:
        """Collect files based on mode."""
        if mode == "full":
            return self.collector.collect_all()

        if not is_git_repo(self.project_root):
            raise ValueError(f"Mode '{mode}' requires git repository")

        if mode == "diff":
            if not base_branch:
                raise ValueError("diff mode requires base_branch")
            changed = get_changed_files_from_branch(self.project_root, base_branch)
        elif mode == "working":
            changed = get_working_directory_files(self.project_root)
        elif mode == "staged":
            changed = get_staged_files(self.project_root)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return self.collector.filter_by_list(changed)

    def _filter_cached(self, files: list[Path]) -> list[Path]:
        """Filter out files that are cached and valid."""
        needs_check = []

        for file_path in files:
            rel_path = str(file_path.relative_to(self.project_root))

            # Check if cached
            if rel_path not in self.cache.entries:
                needs_check.append(file_path)
                continue

            entry = self.cache.entries[rel_path]

            # Check if CLAUDE.md changed
            if entry.claude_md_hash != self.guidelines_hash:
                needs_check.append(file_path)
                continue

            # Check if file changed
            current_hash = self.collector.compute_hash(file_path)
            if entry.file_hash != current_hash:
                needs_check.append(file_path)
                continue

        return needs_check

    def _get_cached_results(self, files: list[Path]) -> list[dict[str, Any]]:
        """Get results from cache for files."""
        results = []

        for file_path in files:
            rel_path = str(file_path.relative_to(self.project_root))

            if rel_path in self.cache.entries:
                entry = self.cache.entries[rel_path]
                results.append({
                    "file": rel_path,
                    "violations": entry.violations
                })

        return results

    def _init_progress_tracker(self, total_batches: int) -> ProgressTracker:
        """Initialize or load progress tracker."""
        if self.progress_path.exists():
            return ProgressTracker.load(self.progress_path)
        return ProgressTracker(self.progress_path, total_batches)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_orchestrator.py -v`
Expected: All 2 tests PASS

**Step 5: Commit**

```bash
git add src/claude_lint/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: add main orchestrator coordinating all components"
```

---

## Task 13: CLI Interface

**Files:**
- Create: `src/claude_lint/cli.py`
- Create: `tests/test_cli.py`

**Step 1: Write failing test for CLI**

Create `tests/test_cli.py`:

```python
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from click.testing import CliRunner
import pytest
from claude_lint.cli import main


@patch("claude_lint.cli.Orchestrator")
def test_cli_full_scan(mock_orchestrator):
    """Test CLI with full scan."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        # Create config
        Path(".lint-claude.json").write_text('{"include": ["**/*.py"]}')
        Path("CLAUDE.md").write_text("# Guidelines")

        # Mock orchestrator
        mock_orch = Mock()
        mock_orch.run.return_value = [
            {"file": "test.py", "violations": []}
        ]
        mock_orchestrator.return_value = mock_orch

        # Run CLI
        result = runner.invoke(main, ["--full"])

        assert result.exit_code == 0
        assert "test.py" in result.output


@patch("claude_lint.cli.Orchestrator")
def test_cli_diff_mode(mock_orchestrator):
    """Test CLI with diff mode."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path(".lint-claude.json").write_text('{}')
        Path("CLAUDE.md").write_text("# Guidelines")

        mock_orch = Mock()
        mock_orch.run.return_value = []
        mock_orchestrator.return_value = mock_orch

        result = runner.invoke(main, ["--diff", "main"])

        assert mock_orch.run.called
        call_args = mock_orch.run.call_args
        assert call_args[1]["mode"] == "diff"
        assert call_args[1]["base_branch"] == "main"


@patch("claude_lint.cli.Orchestrator")
def test_cli_json_output(mock_orchestrator):
    """Test CLI with JSON output format."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path(".lint-claude.json").write_text('{}')
        Path("CLAUDE.md").write_text("# Guidelines")

        mock_orch = Mock()
        mock_orch.run.return_value = [
            {"file": "test.py", "violations": []}
        ]
        mock_orchestrator.return_value = mock_orch

        result = runner.invoke(main, ["--full", "--json"])

        assert result.exit_code == 0
        assert '"results"' in result.output
        assert '"summary"' in result.output


@patch("claude_lint.cli.Orchestrator")
def test_cli_exit_code_on_violations(mock_orchestrator):
    """Test CLI returns exit code 1 when violations found."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path(".lint-claude.json").write_text('{}')
        Path("CLAUDE.md").write_text("# Guidelines")

        mock_orch = Mock()
        mock_orch.run.return_value = [
            {
                "file": "test.py",
                "violations": [{"type": "error", "message": "bad", "line": None}]
            }
        ]
        mock_orchestrator.return_value = mock_orch

        result = runner.invoke(main, ["--full"])

        assert result.exit_code == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL with "No module named 'claude_lint.cli'"

**Step 3: Implement CLI module**

Create `src/claude_lint/cli.py`:

```python
"""Command-line interface for lint-claude."""
import sys
from pathlib import Path
import click

from claude_lint.config import load_config
from claude_lint.orchestrator import Orchestrator
from claude_lint.reporter import Reporter, format_detailed_report, format_json_report


@click.command()
@click.option("--full", is_flag=True, help="Full project scan")
@click.option("--diff", type=str, help="Check files changed from branch")
@click.option("--working", is_flag=True, help="Check working directory changes")
@click.option("--staged", is_flag=True, help="Check staged files")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option("--config", type=click.Path(exists=True), help="Path to config file")
def main(full, diff, working, staged, output_json, config):
    """Claude-lint: CLAUDE.md compliance checker."""
    # Determine mode
    mode_count = sum([full, bool(diff), working, staged])
    if mode_count == 0:
        click.echo("Error: Must specify one mode: --full, --diff, --working, or --staged")
        sys.exit(2)
    elif mode_count > 1:
        click.echo("Error: Only one mode can be specified")
        sys.exit(2)

    if full:
        mode = "full"
        base_branch = None
    elif diff:
        mode = "diff"
        base_branch = diff
    elif working:
        mode = "working"
        base_branch = None
    elif staged:
        mode = "staged"
        base_branch = None

    # Load config
    project_root = Path.cwd()
    config_path = Path(config) if config else project_root / ".lint-claude.json"
    cfg = load_config(config_path)

    try:
        # Run orchestrator
        orchestrator = Orchestrator(project_root, cfg)
        results = orchestrator.run(mode=mode, base_branch=base_branch)

        # Format output
        if output_json:
            output = format_json_report(results)
        else:
            output = format_detailed_report(results)

        click.echo(output)

        # Exit with appropriate code
        reporter = Reporter()
        exit_code = reporter.get_exit_code(results)
        sys.exit(exit_code)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/claude_lint/cli.py tests/test_cli.py
git commit -m "feat: add CLI interface with click"
```

---

## Task 14: Integration Testing and Documentation

**Files:**
- Create: `tests/test_integration.py`
- Update: `README.md`

**Step 1: Write integration test**

Create `tests/test_integration.py`:

```python
"""End-to-end integration tests."""
import os
import tempfile
from pathlib import Path
import subprocess
import pytest


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)
def test_full_integration():
    """Test full end-to-end workflow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create project structure
        (tmpdir / "src").mkdir()
        (tmpdir / "src" / "good.py").write_text('''
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b
    """
    return a + b
''')

        (tmpdir / "src" / "bad.py").write_text('''
def process(data):
    if data:
        if data.get("value"):
            if data["value"] > 0:
                return data["value"] * 2
    return None
''')

        # Create CLAUDE.md
        (tmpdir / "CLAUDE.md").write_text('''
# Guidelines

- Follow TDD
- Avoid nested complexity
- Use type hints
- Write docstrings
''')

        # Create config
        (tmpdir / ".lint-claude.json").write_text('''
{
  "include": ["**/*.py"],
  "exclude": ["tests/**"],
  "batchSize": 10
}
''')

        # Run lint-claude
        result = subprocess.run(
            ["python", "-m", "claude_lint.cli", "--full"],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            env={**os.environ}
        )

        # Verify output
        assert "good.py" in result.stdout
        assert "bad.py" in result.stdout

        # bad.py should have violations
        assert result.returncode == 1
```

**Step 2: Update README with comprehensive documentation**

Update `README.md`:

```markdown
# lint-claude

CLAUDE.md compliance checker using Claude API with prompt caching.

## Features

- 🔍 Smart change detection with git integration
- 💾 Persistent caching for fast re-runs
- 🔄 Resume capability for interrupted scans
- 📦 Batch processing with configurable size
- 🚀 Prompt caching for efficient API usage
- 🔁 Automatic retry with exponential backoff
- 📊 Detailed and JSON output formats
- ✅ CI/CD friendly with exit codes

## Installation

```bash
# From source
git clone https://github.com/yourusername/lint-claude.git
cd lint-claude
pip install -e .

# Or with pip (once published)
pip install lint-claude
```

## Configuration

Create `.lint-claude.json` in your project root:

```json
{
  "include": ["**/*.py", "**/*.js", "**/*.ts"],
  "exclude": ["node_modules/**", "dist/**", "*.test.js"],
  "batchSize": 10
}
```

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

Or specify in config:

```json
{
  "apiKey": "your-api-key",
  ...
}
```

## Usage

### Full Project Scan

```bash
lint-claude --full
```

### Check Changes from Branch

```bash
lint-claude --diff main
lint-claude --diff origin/develop
```

### Check Working Directory

```bash
# Check modified and untracked files
lint-claude --working
```

### Check Staged Files

```bash
# Check only staged files
lint-claude --staged
```

### JSON Output

```bash
lint-claude --full --json > results.json
```

## CI/CD Integration

Claude-lint returns exit code 0 for clean scans and 1 when violations are found:

```yaml
# GitHub Actions example
- name: Check CLAUDE.md compliance
  run: |
    pip install lint-claude
    lint-claude --diff origin/main
```

## How It Works

1. **File Collection**: Gathers files based on mode (full/diff/working/staged) and include/exclude patterns
2. **Cache Check**: Skips files that haven't changed since last scan
3. **Batch Processing**: Groups files into batches (default 10, configurable up to 100)
4. **API Analysis**: Sends batches to Claude API with cached CLAUDE.md in system prompt
5. **Result Parsing**: Extracts violations from Claude's analysis
6. **Caching**: Stores results and file hashes for future runs
7. **Reporting**: Outputs detailed or JSON format with exit code

## Caching Strategy

- **CLAUDE.md Hash**: Triggers full re-scan when guidelines change
- **File Hashes**: Only re-checks modified files
- **API Prompt Caching**: Claude's prompt caching keeps CLAUDE.md cached across requests
- **Result Cache**: Stores previous analysis results in `.lint-claude-cache.json`

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=claude_lint --cov-report=html

# Lint code
ruff check src/

# Format code
ruff format src/
```

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR.
```

**Step 3: Run integration test (if API key available)**

Run: `pytest tests/test_integration.py -v`
Expected: PASS (if ANTHROPIC_API_KEY set) or SKIPPED

**Step 4: Run all tests**

Run: `pytest -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add tests/test_integration.py README.md
git commit -m "docs: add integration tests and comprehensive README"
```

---

## Task 15: Final Verification and Cleanup

**Step 1: Install in development mode**

Run: `pip install -e .`
Expected: Package installed successfully

**Step 2: Verify CLI is available**

Run: `lint-claude --help`
Expected: Help text displayed

**Step 3: Run full test suite**

Run: `pytest -v --cov=claude_lint`
Expected: All tests pass with good coverage

**Step 4: Test on actual project (if API key available)**

```bash
# In a test project with CLAUDE.md
lint-claude --full
```

Expected: Analysis completes successfully

**Step 5: Clean up cache files from testing**

Run: `find . -name ".lint-claude-cache.json" -delete && find . -name ".lint-claude-progress.json" -delete`
Expected: Cache files removed

**Step 6: Final commit**

```bash
git add -A
git commit -m "chore: final verification and cleanup"
git tag v0.1.0
```

---

## Plan Complete

All components implemented:
- ✅ Configuration management
- ✅ CLAUDE.md reader with hashing
- ✅ Cache system
- ✅ Git integration (4 modes)
- ✅ File collection with patterns
- ✅ Batch processing with XML prompts
- ✅ Claude API with prompt caching
- ✅ Retry logic with backoff
- ✅ Progress tracking and resume
- ✅ Report formatting (detailed + JSON)
- ✅ CLI interface
- ✅ Comprehensive tests
- ✅ Documentation

**Skills Referenced:**
- @test-driven-development - Every component built with failing test first
- @verification-before-completion - Tests run before marking complete
- @systematic-debugging - If issues arise, use structured approach

**Next Steps:**
1. Test with real CLAUDE.md on actual projects
2. Gather feedback on violation detection quality
3. Consider adding fix suggestions
4. Consider adding `.claudeignore` file support
5. Publish to PyPI
