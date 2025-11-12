# Contributing to lint-claude

Thank you for considering contributing to `lint-claude`! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful, constructive, and professional. We're all here to build better software.

## How to Contribute

### Reporting Bugs

1. **Check existing issues** to avoid duplicates
2. **Use the bug report template** (if available)
3. **Include**:
   - Clear description of the issue
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, lint-claude version)
   - Relevant logs or error messages

**Example:**
```markdown
## Bug: Rate limiter not respecting RPM limit

### Steps to Reproduce
1. Set `rate_limit_rpm: 10` in config
2. Run `uvx lint-claude --full` on 50 files
3. Observe more than 10 requests per minute

### Expected
No more than 10 API requests per minute

### Actual
15-20 requests per minute

### Environment
- OS: macOS 14.0
- Python: 3.11.5
- lint-claude: 0.3.0
```

### Suggesting Features

1. **Check existing feature requests** in issues
2. **Open a discussion** before implementing large features
3. **Explain**:
   - The problem your feature solves
   - Proposed solution
   - Alternative approaches considered
   - Any breaking changes

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch** from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes** following the guidelines below

4. **Write tests** for new functionality

5. **Run the test suite**
   ```bash
   uv run pytest
   uv run mypy src/
   uv run ruff check src/
   ```

6. **Commit with clear messages**
   ```bash
   git commit -m "feat: add support for custom output formats"
   git commit -m "fix: race condition in batch processor"
   git commit -m "docs: update configuration examples"
   ```

7. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Fill out the PR template** with:
   - Description of changes
   - Related issues (if any)
   - Testing performed
   - Breaking changes (if any)

## Development Setup

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- Git

### Initial Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/lint-claude.git
cd lint-claude

# Add upstream remote
git remote add upstream https://github.com/vtemian/lint-claude.git

# Install dependencies
uv sync --all-extras --dev

# Set up pre-commit hooks (optional)
uv run pre-commit install
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=claude_lint --cov-report=html

# Run specific test file
uv run pytest tests/test_orchestrator.py

# Run specific test
uv run pytest tests/test_orchestrator.py::test_orchestrator_full_scan

# Run with verbose output
uv run pytest -v

# Run with debugging output
uv run pytest -vv -s
```

### Code Quality

```bash
# Run linter (check for issues)
uv run ruff check src/ tests/

# Run linter (auto-fix issues)
uv run ruff check --fix src/ tests/

# Run formatter (check only)
uv run ruff format --check src/ tests/

# Run formatter (apply)
uv run ruff format src/ tests/

# Run type checker
uv run mypy src/
```

### Manual Testing

```bash
# Install in development mode
uv pip install -e .

# Test CLI
lint-claude --help

# Test with sample project
cd /path/to/test-project
export ANTHROPIC_API_KEY="your-key"
lint-claude --full
```

## Coding Guidelines

### Python Style

- **Follow PEP 8** (enforced by Ruff)
- **Line length**: 100 characters max
- **Type hints**: Required for all functions
- **Docstrings**: Required for public functions (Google style)

### Code Organization

```python
# 1. Standard library imports
import logging
from pathlib import Path
from typing import Any

# 2. Third-party imports
import click
from anthropic import Anthropic

# 3. Local imports
from claude_lint.config import Config
from claude_lint.types import FileResult
```

### Naming Conventions

- **Variables/functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: `_leading_underscore`

### Type Hints

```python
# Good
def process_files(
    files: list[Path],
    config: Config,
    batch_size: int = 10
) -> list[FileResult]:
    """Process files in batches."""
    ...

# Bad (missing types)
def process_files(files, config, batch_size=10):
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def analyze_code(file_path: Path, guidelines: str) -> FileResult:
    """Analyze a code file against guidelines.

    Args:
        file_path: Path to the file to analyze.
        guidelines: CLAUDE.md content with guidelines.

    Returns:
        FileResult with violations found.

    Raises:
        FileNotFoundError: If file_path doesn't exist.
        ValueError: If guidelines are empty.
    """
    ...
```

### Error Handling

```python
# Good: Specific exceptions with context
try:
    content = file_path.read_text()
except FileNotFoundError:
    logger.error(f"File not found: {file_path}")
    raise
except UnicodeDecodeError as e:
    logger.warning(f"Cannot decode {file_path}: {e}")
    return None

# Bad: Bare except
try:
    content = file_path.read_text()
except:
    return None
```

### Testing

- **Test naming**: `test_<function>_<scenario>`
- **One assertion per test** (when possible)
- **Use fixtures** for common setup
- **Mock external dependencies** (API calls, file I/O)
- **Test edge cases**: empty inputs, invalid data, errors

```python
def test_collect_files_excludes_patterns():
    """Test that exclude patterns are respected."""
    config = Config(
        include=["**/*.py"],
        exclude=["tests/**", "*.test.py"]
    )

    files = collect_files(Path("."), config)

    assert not any("test" in str(f) for f in files)
```

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test changes
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `chore`: Build/tooling changes

**Examples:**
```bash
feat(cli): add --json output format
fix(cache): prevent race condition in cache writes
docs(readme): update installation instructions
test(orchestrator): add error path tests
refactor(processor): extract XML parsing logic
perf(collector): optimize glob pattern matching
chore(deps): bump anthropic to 0.25.0
```

## Release Process

Only maintainers can publish releases. See [docs/PUBLISHING.md](docs/PUBLISHING.md) for details.

## Questions?

- **Issues**: [GitHub Issues](https://github.com/vtemian/lint-claude/issues)
- **Discussions**: [GitHub Discussions](https://github.com/vtemian/lint-claude/discussions)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
