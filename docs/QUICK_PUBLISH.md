# Quick Publish Guide

**TL;DR**: Create a GitHub release, and the package publishes automatically to PyPI.

## One-Time Setup (First Release)

### 1. Configure PyPI Trusted Publisher

Go to [https://pypi.org/manage/account/publishing/](https://pypi.org/manage/account/publishing/):

- **PyPI Project Name**: `lint-claude`
- **Owner**: `vtemian`
- **Repository name**: `lint-claude`
- **Workflow name**: `publish.yml`
- **Environment name**: `pypi`

### 2. Create GitHub Environment

Go to Repository Settings → Environments → New environment:

- **Name**: `pypi`
- **Protection rules** (optional): Add reviewers, restrict to `main` branch

## Publishing a New Version

### Standard Release

```bash
# 1. Update version in TWO places
# - pyproject.toml (line 3)
# - src/claude_lint/__version__.py (line 3)

# 2. Update CHANGELOG.md
# Add release notes under new version heading

# 3. Commit and push
git add pyproject.toml src/claude_lint/__version__.py CHANGELOG.md
git commit -m "chore: bump version to 0.4.0"
git push origin main

# 4. Create GitHub release
gh release create v0.4.0 \
  --title "v0.4.0" \
  --notes "$(sed -n '/## \[0.4.0\]/,/## \[/p' CHANGELOG.md | head -n -1)"

# Or via GitHub UI: Releases → Draft a new release
```

That's it! The workflow handles the rest:
- ✅ Runs tests on Python 3.11 and 3.12
- ✅ Builds the package
- ✅ Publishes to PyPI

### Pre-release (Alpha/Beta/RC)

```bash
# 1. Use pre-release version format
# 0.4.0a1  (alpha)
# 0.4.0b1  (beta)
# 0.4.0rc1 (release candidate)

# 2. Create release and check "This is a pre-release"
gh release create v0.4.0a1 \
  --title "v0.4.0 Alpha 1" \
  --notes "Testing new features..." \
  --prerelease
```

## Verification

```bash
# Check PyPI
open https://pypi.org/project/lint-claude/

# Test installation
uvx lint-claude@0.4.0 --version
```

## Common Issues

**"Trusted publisher not found"**
- Wait a few minutes after configuring
- Ensure workflow name is exactly `publish.yml`
- Ensure environment name is exactly `pypi`

**"File already exists"**
- Cannot re-upload same version
- Bump version and try again

**Tests failed**
- Fix failing tests before releasing
- Push fixes and create release again

## Rollback

If you publish a broken version:

```bash
# 1. Yank on PyPI
# Go to https://pypi.org/manage/project/lint-claude/releases/
# Click version → Options → Yank release

# 2. Publish fixed version
# Bump patch version (e.g., 0.4.0 → 0.4.1)
# Follow standard release process
```

## Manual Publish (Emergency)

```bash
# Build
rm -rf dist/
uv build

# Publish
uv publish
# Enter PyPI API token when prompted
```

---

**See [PUBLISHING.md](PUBLISHING.md) for detailed documentation.**
