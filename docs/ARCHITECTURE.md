# Architecture

## Overview

Claude-lint is designed as a functional Python CLI that checks code compliance with CLAUDE.md guidelines using the Claude API.

## Design Principles

1. **Functional Programming**: No classes (except pure dataclasses), only functions
2. **Separation of Concerns**: Each module has single responsibility
3. **Testability**: Pure functions, dependency injection
4. **Robustness**: Comprehensive validation, atomic operations
5. **Performance**: Caching, batching, client reuse

## Module Structure

```
src/claude_lint/
├── __init__.py          # Public API exports
├── __version__.py       # Version tracking
├── types.py             # TypedDict definitions
├── cli.py               # Click CLI interface
├── config.py            # Configuration loading
├── validation.py        # Input validation
├── orchestrator.py      # Main coordination logic
├── collector.py         # File collection & filtering
├── processor.py         # Batch creation & XML prompts
├── api_client.py        # Claude API integration
├── cache.py             # Result caching
├── progress.py          # Progress tracking
├── git_utils.py         # Git integration
├── guidelines.py        # CLAUDE.md reading
├── retry.py             # Exponential backoff
├── reporter.py          # Output formatting
├── logging_config.py    # Logging setup
└── file_utils.py        # Atomic file operations
```

## Data Flow

1. **CLI Entry** (`cli.py`)
   - Parse command line arguments
   - Setup logging
   - Call orchestrator

2. **Validation** (`validation.py`)
   - Validate project root, mode, batch size, API key
   - Early failure on invalid inputs

3. **File Collection** (`collector.py`)
   - Collect files based on mode (full/diff/working/staged)
   - Apply include/exclude patterns
   - Filter by file size limits

4. **Caching** (`cache.py`)
   - Load existing cache
   - Filter out unchanged files (file hash + CLAUDE.md hash)
   - Return cached results for unchanged files

5. **Batch Processing** (`processor.py`)
   - Create batches of files
   - Build XML prompts for each batch
   - Escape file paths and contents

6. **API Calls** (`api_client.py`)
   - Create client once, reuse across batches
   - Use prompt caching for CLAUDE.md
   - Retry on transient failures

7. **Progress Tracking** (`progress.py`)
   - Save progress after each batch
   - Enable resume on interruption
   - Clean up on completion

8. **Reporting** (`reporter.py`)
   - Format results (detailed or JSON)
   - Calculate exit code
   - Generate summary statistics

## Key Design Decisions

### Why No Classes?

User requirement: "NEVER USE CLASSES". This forces:
- Functional decomposition
- Pure functions
- Explicit dependencies
- Testability without mocking complex state

### Why Atomic Writes?

Cache and progress files must never be partially written. Atomic writes ensure:
- Write to `.tmp` file
- Atomically replace original
- No corruption on crashes

### Why Client Reuse?

Creating Anthropic client has overhead. Reusing across batches:
- Reduces connection overhead
- Maintains connection pooling
- Improves performance

### Why Pattern Matching Unification?

Originally used fnmatch for excludes, PurePath.match() for includes. This caused:
- Different matching semantics
- Confusing behavior
- Bugs with `**` patterns

Unified to `PurePath.match()` for consistency.

### Why Fallback Encoding?

Files may not be UTF-8. Strategy:
1. Try UTF-8 (most common)
2. Fall back to latin-1 (accepts all bytes)
3. Skip if both fail

This handles most files while warning on issues.

## Performance Characteristics

- **File Collection**: O(n) where n = files in project
- **Cache Lookup**: O(n) where n = files to check
- **Batching**: O(n/b) API calls where b = batch size
- **API Calls**: ~1-3 seconds per batch (depends on model)

For 100 files with batch size 10:
- ~10 API calls
- ~10-30 seconds total (with caching)
- Subsequent runs: <1 second (cache hits)

## Extension Points

### Adding New Modes

1. Add mode to `VALID_MODES` in `validation.py`
2. Add git function to `git_utils.py` if needed
3. Add case to `collect_files_for_mode()` in `orchestrator.py`
4. Add CLI option in `cli.py`

### Adding New Config Options

1. Add field to `Config` dataclass in `config.py`
2. Update `get_default_config()` and `load_config()`
3. Use config value in relevant module
4. Add tests

### Custom Output Formats

1. Add format function to `reporter.py`
2. Add CLI option in `cli.py`
3. Call format function based on option

## Testing Strategy

- **Unit Tests**: Each module tested in isolation
- **Integration Tests**: Orchestrator with mocked API
- **Property Tests**: Pattern matching, encoding handling
- **Error Tests**: Timeouts, exceptions, edge cases

## Security Considerations

- Input validation prevents path traversal
- Subprocess calls use list form (no shell injection)
- Timeouts prevent hanging
- File size limits prevent DoS
- API key not logged or exposed
