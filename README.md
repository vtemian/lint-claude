# lint-claude

**Automatically enforce your CLAUDE.md guidelines using Claude AI**

[![PyPI version](https://badge.fury.io/py/lint-claude.svg)](https://badge.fury.io/py/lint-claude)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## The Problem

You've written a [CLAUDE.md](https://github.com/anthropics/claude-code/blob/main/codebase_instructions.md) file with coding guidelines, architecture rules, and best practices for your project. But how do you ensure your team actually follows them?

**Manual code reviews miss things.** Reviewers can't remember every guideline, and catching violations is tedious and inconsistent.

**Traditional linters are rigid.** They can't understand context-specific rules like "use functional patterns for data transformations" or "avoid nested conditionals in business logic."

**You need intelligent, automated guideline enforcement** that understands your specific rules and checks every file change before it's merged.

## The Solution

`lint-claude` uses Claude AI to analyze your code against your CLAUDE.md guidelines automatically. It catches violations during development and in CI/CD, ensuring consistent adherence to your team's standards.

```bash
# Check changed files before committing
uvx lint-claude --working

# In CI/CD: fail the build if guidelines are violated
uvx lint-claude --diff main
```

**Smart & Fast:**
- **Prompt caching** reduces API costs by ~90%
- **File hashing** skips unchanged files
- **Batch processing** handles large projects efficiently
- **Git-aware** checks only what changed

**Team-Friendly:**
- **Multiple scan modes** (full, diff, working, staged)
- **CI/CD integration** with exit codes
- **JSON output** for custom tooling
- **Resume support** for interrupted scans

## Quick Start

### 1. Install with uvx (no installation needed)

```bash
# Run directly without installing
uvx lint-claude --help
```

Or install globally:

```bash
# Install with uv
uv tool install lint-claude

# Install with pip
pip install lint-claude
```

### 2. Set your API key

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

Get your API key from [Anthropic Console](https://console.anthropic.com/).

### 3. Create CLAUDE.md

Create a `CLAUDE.md` file in your project root with your guidelines:

```markdown
# Project Guidelines

## Code Style
- Use descriptive variable names (no single letters except loop counters)
- Functions should be under 50 lines
- Prefer composition over inheritance

## Architecture
- Keep business logic in src/domain/
- All API calls must use the retry wrapper
- Database queries must be in repository classes

## Testing
- Every public function needs a test
- Test files must be colocated with source files
```

### 4. Run your first scan

```bash
# Check all files
uvx lint-claude --full

# Check only changed files (faster)
uvx lint-claude --working
```

That's it! Claude will analyze your code and report any guideline violations.

## Usage

### Scan Modes

```bash
# Full project scan
uvx lint-claude --full

# Check files changed from a branch (great for PRs)
uvx lint-claude --diff main
uvx lint-claude --diff origin/develop

# Check uncommitted changes (working directory)
uvx lint-claude --working

# Check only staged files (pre-commit hook)
uvx lint-claude --staged
```

### Output Formats

```bash
# Human-readable output (default)
uvx lint-claude --full

# JSON output for tooling integration
uvx lint-claude --full --json

# Quiet mode (only errors)
uvx lint-claude --full --quiet

# Verbose mode (detailed logging)
uvx lint-claude --full --verbose
```

### Configuration

Create `.lint-claude.json` in your project root:

```json
{
  "include": ["**/*.py", "**/*.js", "**/*.ts"],
  "exclude": ["node_modules/**", "dist/**", "*.test.js"],
  "batch_size": 10,
  "max_file_size_mb": 1.0,
  "model": "claude-sonnet-4-5-20250929"
}
```

**All settings are optional.** Defaults work for most projects.

**Available options:**

| Option | Default | Description |
|--------|---------|-------------|
| `include` | `["**/*"]` | Glob patterns for files to check |
| `exclude` | `[]` | Glob patterns to skip |
| `batch_size` | `10` | Files per API request (1-100) |
| `max_file_size_mb` | `10.0` | Skip files larger than this |
| `model` | `claude-sonnet-4-5-20250929` | Claude model to use |
| `api_key` | (env var) | Override ANTHROPIC_API_KEY |
| `api_timeout_seconds` | `300` | API request timeout |
| `rate_limit_rpm` | `50` | API requests per minute |

Use `snake_case` (preferred) or `camelCase` for keys.

## CI/CD Integration

`lint-claude` returns exit code `0` for clean scans and `1` when violations are found, making it perfect for CI/CD.

### GitHub Actions

```yaml
name: CLAUDE.md Compliance

on: [pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Need full history for --diff

      - name: Check guidelines
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          uvx lint-claude --diff origin/main
```

### GitLab CI

```yaml
lint-claude:
  image: python:3.11
  script:
    - pip install uv
    - uvx lint-claude --diff origin/main
  only:
    - merge_requests
```

### Pre-commit Hook

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: lint-claude
        name: CLAUDE.md compliance
        entry: uvx lint-claude --staged
        language: system
        pass_filenames: false
```

## How It Works

1. **File Collection** - Gathers files based on scan mode and include/exclude patterns
2. **Cache Check** - Skips files unchanged since last scan (based on file hash + CLAUDE.md hash)
3. **Batch Processing** - Groups files into batches (default 10, configurable)
4. **API Analysis** - Sends batches to Claude with CLAUDE.md in cached system prompt
5. **Violation Detection** - Parses Claude's structured XML response
6. **Result Caching** - Stores results for future runs
7. **Reporting** - Outputs violations with file paths and line numbers

### Caching Strategy

`lint-claude` is designed to be fast and cost-effective:

- **Prompt Caching** - CLAUDE.md is cached by Claude's API, reducing costs by ~90%
- **File Hash Cache** - Only re-checks modified files (`.lint-claude-cache.json`)
- **CLAUDE.md Hash** - Triggers full re-scan when guidelines change
- **Resume Support** - Interrupted scans resume from last batch (`.lint-claude-progress.json`)

**Example:** Checking 100 files costs ~$0.50 on first run, then ~$0.05 for incremental checks.

## Examples

### Example Output

```bash
$ uvx lint-claude --working

🔍 Checking 3 files against CLAUDE.md...

❌ src/auth/login.py
  Line 45: Function `authenticate_user` is 73 lines long.
           Guideline requires functions under 50 lines.

  Line 12: Variable name `u` is not descriptive.
           Guideline requires descriptive names.

❌ src/api/users.py
  Line 23: Database query not in repository class.
           Guideline requires all database queries in repositories.

✅ src/utils/format.py - No violations

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Summary: 2 files with violations, 1 file clean

Exit code: 1
```

### Example JSON Output

```bash
$ uvx lint-claude --staged --json
```

```json
{
  "results": [
    {
      "file": "src/auth/login.py",
      "violations": [
        {
          "type": "error",
          "line": 45,
          "message": "Function `authenticate_user` is 73 lines long. Guideline requires functions under 50 lines."
        }
      ]
    }
  ],
  "summary": {
    "files_checked": 3,
    "files_with_violations": 2,
    "files_clean": 1,
    "total_violations": 3
  },
  "metrics": {
    "duration_seconds": 4.2,
    "cache_hit_rate": 0.67,
    "api_calls_made": 1
  }
}
```

## Advanced Usage

### Custom Model

Use different Claude models for different needs:

```bash
# Fast and cheap (Haiku)
uvx lint-claude --full --model claude-haiku-20250301

# More thorough (Opus)
uvx lint-claude --full --model claude-opus-20250229

# Default (Sonnet - best balance)
uvx lint-claude --full --model claude-sonnet-4-5-20250929
```

### Batch Size Tuning

```bash
# Smaller batches (more accurate, more API calls)
uvx lint-claude --full --batch-size 5

# Larger batches (faster, cheaper, may miss context)
uvx lint-claude --full --batch-size 50
```

### Skip Cache

```bash
# Force re-check all files
rm .lint-claude-cache.json
uvx lint-claude --full
```

## Troubleshooting

### API Rate Limits

If you hit rate limits, reduce RPM in config:

```json
{
  "rate_limit_rpm": 30
}
```

### Large Files Skipped

Files larger than 10MB are skipped by default. To check larger files:

```json
{
  "max_file_size_mb": 20.0
}
```

### Cost Concerns

- Use `--diff` or `--working` instead of `--full` to check fewer files
- Reduce `batch_size` for more granular caching
- Use Haiku model for cheaper analysis

**Cost estimate:** ~$0.005 per file on first check, ~$0.0005 on cached re-checks.

### Interrupted Scans

If a scan is interrupted, it will resume from the last completed batch:

```bash
# Resume automatically
uvx lint-claude --full
```

To start fresh:

```bash
rm .lint-claude-progress.json
uvx lint-claude --full
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - Design decisions and module structure
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and detailed solutions
- [Contributing](CONTRIBUTING.md) - Development setup and guidelines

## Development

```bash
# Clone repository
git clone https://github.com/vtemian/lint-claude.git
cd lint-claude

# Install dependencies
uv sync

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=claude_lint --cov-report=html

# Lint code
uv run ruff check src/

# Format code
uv run ruff format src/

# Type check
uv run mypy src/
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`uv run pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## FAQ

### How is this different from traditional linters?

Traditional linters (ESLint, Pylint, Ruff) are rule-based and can't understand context. `lint-claude` uses AI to understand your specific guidelines in natural language, like "avoid god classes" or "use repository pattern for database access."

### Does it replace traditional linters?

No, use both! Traditional linters catch syntax errors and common issues. `lint-claude` catches architectural and stylistic violations specific to your team's guidelines.

### How much does it cost?

~$0.005 per file on first scan, ~$0.0005 on incremental scans. A typical PR with 10 changed files costs less than $0.01.

### Can I use it offline?

No, it requires Claude API access. However, it caches aggressively to minimize API calls.

### What models are supported?

All Claude 3+ models: Haiku, Sonnet, Opus. Sonnet (default) offers the best balance of speed, cost, and accuracy.

### Can I customize the prompt?

The system prompt is fixed to ensure reliable output parsing. Customize behavior through your CLAUDE.md guidelines instead.

## Acknowledgments

- Built with [Anthropic Claude API](https://www.anthropic.com/)
- Inspired by [CLAUDE.md specification](https://github.com/anthropics/claude-code/blob/main/codebase_instructions.md)

## Support

- **Issues:** [GitHub Issues](https://github.com/vtemian/lint-claude/issues)
- **Discussions:** [GitHub Discussions](https://github.com/vtemian/lint-claude/discussions)
- **Changelog:** [CHANGELOG.md](CHANGELOG.md)

---

**Made with ❤️ by developers who believe in enforceable standards**
