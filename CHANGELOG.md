# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2025-11-11

### Critical Fixes
- Added 30-second timeout to all git subprocess calls to prevent hanging
- Replaced print() statements with proper logging framework
- Implemented atomic file writes for cache and progress files
- Fixed exception handling to not catch KeyboardInterrupt/SystemExit
- Removed unused gitpython dependency

### Major Improvements
- Unified pattern matching to use PurePath.match() consistently
- Made Claude model configurable via config file
- Improved file encoding handling with UTF-8 and latin-1 fallback
- Added comprehensive input validation for all parameters
- Implemented Anthropic client reuse for better performance
- Added --verbose, --quiet, and --version CLI flags
- Added file size limits (configurable, default 1MB)

### Minor Improvements
- Added TypedDict for violation structures
- Added __all__ exports for public API
- Normalized config keys to snake_case (with camelCase backwards compat)
- Comprehensive documentation (architecture, troubleshooting)
- Improved error messages to stderr
- Better type safety throughout
- Fixed type checking issues (mypy) and linting (ruff)

### Documentation
- Added ARCHITECTURE.md with design decisions
- Added TROUBLESHOOTING.md for common issues
- Updated README with all new features
- Added inline documentation improvements

## [0.1.0] - 2025-11-11

Initial release with core functionality:
- Smart caching based on file and CLAUDE.md hashes
- Multiple scan modes: full project, git diff, working directory, staged files
- Batch processing for large projects
- Progress tracking with resume capability
- Prompt caching for cost efficiency
