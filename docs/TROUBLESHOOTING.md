# Troubleshooting Guide

## Common Issues

### API Key Not Found

**Error:** `ValueError: API key is required`

**Solution:** Set the `ANTHROPIC_API_KEY` environment variable or add `api_key` to `.agent-lint.json`:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Git Command Timeout

**Error:** `RuntimeError: Git command timed out after 30s`

**Solution:**
- Check if git is hanging (network issues with remote)
- Large repositories may need optimization
- Check git config for hanging processes

### Files Skipped Due to Encoding

**Warning:** `File X is not valid UTF-8, trying latin-1`

**Info:** Claude-lint tries UTF-8 first, falls back to latin-1. If both fail, the file is skipped.

**Solution:**
- Check file encoding: `file -I <filename>`
- Convert to UTF-8 if possible
- Binary files will be skipped (expected)

### File Size Limit

**Warning:** `File X exceeds size limit (2.5MB > 1.0MB), skipping`

**Solution:** Increase `max_file_size_mb` in config:

```json
{
  "max_file_size_mb": 5.0
}
```

### Cache Corruption

**Error:** Issues loading cache file

**Solution:** Delete cache and progress files:

```bash
rm .agent-lint-cache.json .agent-lint-progress.json
```

Claude-lint will rebuild the cache on next run.

### Pattern Matching Not Working

**Issue:** Files not being included/excluded as expected

**Solution:**
- Use `**` for recursive patterns: `**/*.py`
- Patterns are matched against relative paths from project root
- Test with `--verbose` to see which files are collected
- Both include and exclude use same glob pattern matching

### Git Not Found

**Error:** `FileNotFoundError: git`

**Solution:** Install git and ensure it's in your PATH.

## Debug Mode

Run with `--verbose` for detailed logging:

```bash
claude-lint --full --verbose
```

This shows:
- Which files are being collected
- Which files are skipped and why
- Cache hit/miss information
- API call details

## Getting Help

1. Check this troubleshooting guide
2. Review [Architecture docs](ARCHITECTURE.md)
3. Run with `--verbose` for detailed output
4. File an issue at https://github.com/vtemian/claude-lint/issues
