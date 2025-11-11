# claude-lint

CLAUDE.md compliance checker using Claude API with prompt caching.

## Features

- Smart caching based on file and CLAUDE.md hashes
- Multiple scan modes: full project, git diff, working directory, staged files
- Batch processing for large projects
- Progress tracking with resume capability
- Prompt caching for cost efficiency
- Configurable Claude model (Sonnet, Opus, etc.)
- File size limits to prevent API overload
- Atomic file writes prevent cache corruption
- Comprehensive logging with --verbose and --quiet flags
- Input validation for robust operation
- Automatic retry with exponential backoff
- Detailed and JSON output formats
- CI/CD friendly with exit codes

## Installation

```bash
# Using uvx (recommended)
uvx claude-lint --help

# From source
git clone https://github.com/vtemian/claude-lint.git
cd claude-lint
uv sync
uv pip install -e .
```

## Configuration

Create `.agent-lint.json` in your project root:

```json
{
  "include": ["**/*.py", "**/*.js", "**/*.ts"],
  "exclude": ["node_modules/**", "dist/**", "*.test.js"],
  "batch_size": 10,
  "max_file_size_mb": 1.0,
  "model": "claude-sonnet-4-5-20250929"
}
```

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

Or specify in config (snake_case or camelCase):

```json
{
  "api_key": "your-api-key",
  ...
}
```

All configuration keys are optional. Snake_case is preferred, but camelCase is also supported for backwards compatibility.

## Usage

### Full Project Scan

```bash
claude-lint --full
```

### Check Changes from Branch

```bash
claude-lint --diff main
claude-lint --diff origin/develop
```

### Check Working Directory

```bash
# Check modified and untracked files
claude-lint --working
```

### Check Staged Files

```bash
# Check only staged files
claude-lint --staged
```

### JSON Output

```bash
claude-lint --full --json > results.json
```

## CI/CD Integration

Claude-lint returns exit code 0 for clean scans and 1 when violations are found:

```yaml
# GitHub Actions example
- name: Check CLAUDE.md compliance
  run: |
    uvx claude-lint --diff origin/main
```

## How It Works

1. **File Collection**: Gathers files based on mode (full/diff/working/staged) and include/exclude patterns
2. **Cache Check**: Skips files that haven't changed since last scan
3. **Batch Processing**: Groups files into batches (default 10-15)
4. **API Analysis**: Sends batches to Claude API with cached CLAUDE.md in system prompt
5. **Result Parsing**: Extracts violations from Claude's analysis
6. **Caching**: Stores results and file hashes for future runs
7. **Reporting**: Outputs detailed or JSON format with exit code

## Caching Strategy

- **CLAUDE.md Hash**: Triggers full re-scan when guidelines change
- **File Hashes**: Only re-checks modified files
- **API Prompt Caching**: Claude's prompt caching keeps CLAUDE.md cached across requests
- **Result Cache**: Stores previous analysis results in `.agent-lint-cache.json`

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - Design decisions and module structure
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions

## Development

```bash
# Install dev dependencies
uv sync

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=claude_lint --cov-report=html

# Lint code
uv run ruff check src/

# Format code
uv run ruff format src/
```

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR.
