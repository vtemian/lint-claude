# Development Guidelines

## Package Management

- Use `uv` for all dependency management
- Use `uvx` for running CLI tools
- Never use `pip` or `pip install`

## Installation

```bash
# Install dependencies
uv sync

# Run the tool
uvx claude-lint --help
```

## Development Setup

```bash
# Install in development mode
uv pip install -e .

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=claude_lint --cov-report=html

# Run linter
uv run ruff check src/

# Format code
uv run ruff format src/
```

## Communication Style

- Never use emojis in code, documentation, or output
- Keep messages clear and professional
- Use plain text markers instead of emojis (e.g., "[OK]", "[ERROR]", "[INFO]")

## Code Style

- Follow PEP 8 conventions
- Use type hints throughout
- Keep functions focused and under 50 lines
- Write comprehensive docstrings with Args/Returns sections
- Maintain 90%+ test coverage

## Testing

- Follow TDD approach: write failing tests first
- Use pytest for all tests
- Mock external dependencies (API calls, file system when appropriate)
- Test both happy paths and error cases
- Name tests descriptively: `test_<function>_<scenario>`

## Git Workflow

- Write clear, descriptive commit messages
- Use conventional commits format: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`
- Commit frequently with atomic changes
- Tag releases with semantic versioning (v0.1.0, v0.2.0, etc.)

## Error Handling

- Always handle potential errors explicitly
- Provide clear error messages
- Use appropriate exceptions (ValueError, FileNotFoundError, etc.)
- Log errors for debugging
- Never silently fail

## Documentation

- Update README when adding features
- Keep examples up-to-date
- Document configuration options
- Include usage examples for all major features
- Maintain accurate type hints as documentation
