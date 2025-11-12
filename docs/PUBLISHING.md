# Publishing lint-claude to PyPI

This guide explains how to publish new versions of `lint-claude` to PyPI.

## Prerequisites

1. **PyPI Account**: Create an account at [pypi.org](https://pypi.org/account/register/)
2. **Test PyPI Account** (optional): Create an account at [test.pypi.org](https://test.pypi.org/account/register/)
3. **Repository Access**: You must have maintainer access to the GitHub repository

## Setup: Trusted Publishing (Recommended)

We use PyPI's [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) which eliminates the need for API tokens.

### 1. Configure PyPI Trusted Publisher

1. Go to [https://pypi.org/manage/account/publishing/](https://pypi.org/manage/account/publishing/)
2. Scroll to "Add a new pending publisher"
3. Fill in:
   - **PyPI Project Name**: `lint-claude`
   - **Owner**: `vtemian` (or your GitHub username)
   - **Repository name**: `lint-claude`
   - **Workflow name**: `publish.yml`
   - **Environment name**: `pypi`
4. Click "Add"

### 2. Create GitHub Environment

1. Go to your repository's Settings → Environments
2. Create a new environment named `pypi`
3. Add protection rules (optional but recommended):
   - Required reviewers: Add yourself
   - Deployment branches: Selected branches → `main`

## Publishing Process

### Automated Publishing (Recommended)

1. **Update Version**
   ```bash
   # Edit both files:
   # - pyproject.toml (line 3)
   # - src/claude_lint/__version__.py (line 3)

   # Update to new version (e.g., 0.3.1)
   ```

2. **Update CHANGELOG.md**
   ```bash
   # Add release notes for new version
   # Follow Keep a Changelog format
   ```

3. **Commit Changes**
   ```bash
   git add pyproject.toml src/claude_lint/__version__.py CHANGELOG.md
   git commit -m "chore: bump version to 0.3.1"
   git push origin main
   ```

4. **Create GitHub Release**
   ```bash
   # Via GitHub UI:
   # 1. Go to Releases → Draft a new release
   # 2. Click "Choose a tag" → Create new tag: v0.3.1
   # 3. Release title: "v0.3.1"
   # 4. Description: Copy from CHANGELOG.md
   # 5. Click "Publish release"

   # Or via gh CLI:
   gh release create v0.3.1 \
     --title "v0.3.1" \
     --notes "$(sed -n '/## \[0.3.1\]/,/## \[/p' CHANGELOG.md | head -n -1)"
   ```

5. **Workflow Triggers Automatically**
   - GitHub Actions workflow `.github/workflows/publish.yml` runs automatically
   - Tests run on Python 3.11 and 3.12
   - Package is built
   - Published to PyPI using trusted publishing

6. **Verify Publication**
   ```bash
   # Check PyPI
   open https://pypi.org/project/lint-claude/

   # Test installation
   uvx lint-claude@0.3.1 --version
   ```

### Manual Publishing (Fallback)

If you need to publish manually:

1. **Build Package**
   ```bash
   # Clean previous builds
   rm -rf dist/

   # Build package
   uv build

   # Verify contents
   tar -tzf dist/lint-claude-0.3.1.tar.gz
   unzip -l dist/claude_lint-0.3.1-py3-none-any.whl
   ```

2. **Test with Test PyPI** (optional)
   ```bash
   # Upload to Test PyPI
   uv publish --publish-url https://test.pypi.org/legacy/

   # Test installation
   uvx --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ \
     lint-claude --version
   ```

3. **Publish to PyPI**
   ```bash
   # Will prompt for API token or use __token__ from environment
   uv publish

   # Or with explicit token
   uv publish --token $PYPI_TOKEN
   ```

## Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **Major (X.0.0)**: Breaking changes
- **Minor (0.X.0)**: New features, backward compatible
- **Patch (0.0.X)**: Bug fixes, backward compatible

Examples:
- `0.3.0` → `0.3.1`: Bug fix
- `0.3.1` → `0.4.0`: New feature
- `0.4.0` → `1.0.0`: Stable release with breaking changes

## Pre-release Versions

For testing:

```bash
# Alpha release
0.4.0a1, 0.4.0a2

# Beta release
0.4.0b1, 0.4.0b2

# Release candidate
0.4.0rc1, 0.4.0rc2
```

Update version in `pyproject.toml` and `__version__.py`, then create a GitHub release with "pre-release" checked.

## Checklist Before Publishing

- [ ] All tests pass (`uv run pytest`)
- [ ] Type checking passes (`uv run mypy src/`)
- [ ] Linting passes (`uv run ruff check src/`)
- [ ] Version bumped in both files
- [ ] CHANGELOG.md updated with release notes
- [ ] README.md is up to date
- [ ] Git tag matches version
- [ ] GitHub release created
- [ ] Trusted publishing configured on PyPI

## Troubleshooting

### "Project name already exists"

Your first upload must be done manually or the project must be registered on PyPI first. After the first upload, trusted publishing can be configured.

### "Invalid or expired token"

If using API tokens (not trusted publishing):
1. Generate new token at [https://pypi.org/manage/account/token/](https://pypi.org/manage/account/token/)
2. Set scopes to "Entire account" or specific to `lint-claude`
3. Save token securely (you won't see it again)
4. Use in publish command: `uv publish --token pypi-...`

### "Trusted publisher configuration not found"

1. Verify the workflow name is exactly `publish.yml`
2. Verify the environment name is exactly `pypi`
3. Wait a few minutes after configuring (can take time to propagate)
4. Ensure the release is published (not draft)

### "File already exists"

You cannot re-upload the same version. Either:
- Delete the release on PyPI (if it's broken)
- Bump to a new version (recommended)

### Build Fails

```bash
# Clean everything and rebuild
rm -rf dist/ build/ *.egg-info
uv build

# Check for issues
uv run twine check dist/*
```

## Rollback

If you publish a broken version:

1. **Yank the Release on PyPI**
   ```bash
   # Via web interface:
   # https://pypi.org/manage/project/lint-claude/releases/
   # Click version → Options → Yank release

   # Provide reason: "Broken: [describe issue]"
   ```

2. **Publish Fixed Version**
   ```bash
   # Bump patch version
   # e.g., 0.3.1 (broken) → 0.3.2 (fixed)

   # Follow normal publishing process
   ```

3. **Update Documentation**
   ```bash
   # Add note to CHANGELOG.md
   ## [0.3.2] - 2025-11-12
   ### Fixed
   - Fixed [issue from 0.3.1]

   ## [0.3.1] - 2025-11-12 [YANKED]
   - Reason: [describe issue]
   ```

**Note**: Yanked versions can still be installed explicitly (`uvx lint-claude@0.3.1`) but won't be installed by default.

## Security

- **Never commit API tokens** to git
- Use GitHub Secrets for manual workflows
- Prefer trusted publishing over API tokens
- Rotate tokens regularly if using them
- Use environment-specific tokens (not account-wide)

## References

- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
- [uv Build & Publish](https://docs.astral.sh/uv/guides/publish/)
- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
