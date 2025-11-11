"""Main orchestrator coordinating all components."""
import os
from pathlib import Path
from typing import Any, Optional

from claude_lint.api_client import create_client, analyze_files_with_client
from claude_lint.cache import Cache, CacheEntry, load_cache, save_cache
from claude_lint.collector import collect_all_files, filter_files_by_list, compute_file_hash
from claude_lint.config import Config
from claude_lint.git_utils import (
    is_git_repo,
    get_changed_files_from_branch,
    get_working_directory_files,
    get_staged_files
)
from claude_lint.guidelines import read_claude_md, get_claude_md_hash
from claude_lint.logging_config import get_logger
from claude_lint.processor import create_batches, parse_response, build_xml_prompt
from claude_lint.progress import (
    create_progress_state,
    update_progress,
    save_progress,
    load_progress,
    get_remaining_batch_indices,
    is_progress_complete,
    cleanup_progress,
    ProgressState
)
from claude_lint.retry import retry_with_backoff
from claude_lint.types import FileResult
from claude_lint.validation import (
    validate_project_root,
    validate_mode,
    validate_batch_size,
    validate_api_key
)

logger = get_logger(__name__)


def run_compliance_check(
    project_root: Path,
    config: Config,
    mode: str = "full",
    base_branch: Optional[str] = None
) -> list[dict[str, Any]]:
    """Run compliance check.

    Args:
        project_root: Project root directory
        config: Configuration
        mode: Check mode - 'full', 'diff', 'working', 'staged'
        base_branch: Base branch for diff mode

    Returns:
        List of results for all checked files

    Raises:
        ValueError: If inputs are invalid
    """
    # Input validation
    validate_project_root(project_root)
    validate_mode(mode)
    validate_batch_size(config.batch_size)

    # Get API key
    api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
    validate_api_key(api_key)

    # Load CLAUDE.md
    guidelines = read_claude_md(project_root)
    guidelines_hash = get_claude_md_hash(guidelines)

    # Load cache
    cache_path = project_root / ".agent-lint-cache.json"
    cache = load_cache(cache_path)

    # Progress tracking
    progress_path = project_root / ".agent-lint-progress.json"

    # Collect files to check
    files_to_check = collect_files_for_mode(
        project_root, config, mode, base_branch
    )

    if not files_to_check:
        return []

    # Filter using cache
    files_needing_check = filter_cached_files(
        files_to_check, cache, project_root, guidelines_hash
    )

    if not files_needing_check:
        # All files cached, return cached results
        return get_cached_results(files_to_check, cache, project_root)

    # Create batches
    batches = create_batches(files_needing_check, config.batch_size)

    # Check for resumable progress
    progress_state = init_or_load_progress(progress_path, len(batches))

    # Create API client once for all batches
    assert api_key is not None  # Validated above
    client = create_client(api_key)

    # Process batches
    all_results = list(progress_state.results)  # Start with resumed results

    for batch_idx in get_remaining_batch_indices(progress_state):
        batch = batches[batch_idx]

        # Read file contents
        file_contents = {}
        max_size_bytes = int(config.max_file_size_mb * 1024 * 1024)

        for file_path in batch:
            rel_path = file_path.relative_to(project_root)

            # Check file size
            try:
                file_size = file_path.stat().st_size
                if file_size > max_size_bytes:
                    logger.warning(
                        f"File {rel_path} exceeds size limit "
                        f"({file_size / 1024 / 1024:.2f}MB > "
                        f"{config.max_file_size_mb}MB), skipping"
                    )
                    continue
            except OSError as e:
                logger.warning(f"Cannot stat file {rel_path}, skipping: {e}")
                continue

            try:
                # Try UTF-8 first
                content = file_path.read_text(encoding='utf-8')
                file_contents[str(rel_path)] = content
            except UnicodeDecodeError:
                # Fall back to latin-1 which accepts all byte sequences
                try:
                    logger.warning(
                        f"File {rel_path} is not valid UTF-8, trying latin-1"
                    )
                    content = file_path.read_text(encoding='latin-1')
                    file_contents[str(rel_path)] = content
                except Exception as e:
                    logger.warning(
                        f"Unable to decode file {rel_path}, skipping: {e}"
                    )
                    continue
            except FileNotFoundError:
                logger.warning(f"File not found, skipping: {rel_path}")
                continue
            except Exception as e:
                logger.warning(f"Error reading file {rel_path}, skipping: {e}")
                continue

        # Build prompt
        prompt = build_xml_prompt(guidelines, file_contents)

        # Make API call with retry
        def api_call():
            response_text, response_obj = analyze_files_with_client(
                client, guidelines, prompt, model=config.model
            )
            return response_text

        response = retry_with_backoff(api_call)

        # Parse results
        batch_results: list[FileResult] = parse_response(response)
        # Convert FileResult to dict for compatibility with progress state
        batch_results_dict: list[dict[str, Any]] = [dict(r) for r in batch_results]
        all_results.extend(batch_results_dict)

        # Update cache
        for result in batch_results:
            file_path = project_root / result["file"]
            file_hash = compute_file_hash(file_path)

            cache.entries[result["file"]] = CacheEntry(
                file_hash=file_hash,
                claude_md_hash=guidelines_hash,
                violations=result["violations"],
                timestamp=int(Path(file_path).stat().st_mtime)
            )

        # Save progress
        progress_state = update_progress(progress_state, batch_idx, batch_results_dict)
        save_progress(progress_state, progress_path)
        save_cache(cache, cache_path)

    # Cleanup progress on completion
    if is_progress_complete(progress_state):
        cleanup_progress(progress_path)

    # Update cache hash
    cache.claude_md_hash = guidelines_hash
    save_cache(cache, cache_path)

    return all_results


def collect_files_for_mode(
    project_root: Path,
    config: Config,
    mode: str,
    base_branch: Optional[str]
) -> list[Path]:
    """Collect files based on mode.

    Args:
        project_root: Project root directory
        config: Configuration
        mode: Check mode
        base_branch: Base branch for diff mode

    Returns:
        List of files to check
    """
    if mode == "full":
        return collect_all_files(project_root, config)

    if not is_git_repo(project_root):
        raise ValueError(f"Mode '{mode}' requires git repository")

    if mode == "diff":
        if not base_branch:
            raise ValueError("diff mode requires base_branch")
        changed = get_changed_files_from_branch(project_root, base_branch)
    elif mode == "working":
        changed = get_working_directory_files(project_root)
    elif mode == "staged":
        changed = get_staged_files(project_root)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return filter_files_by_list(project_root, changed, config)


def filter_cached_files(
    files: list[Path],
    cache: Cache,
    project_root: Path,
    guidelines_hash: str
) -> list[Path]:
    """Filter out files that are cached and valid.

    Args:
        files: Files to check
        cache: Cache object
        project_root: Project root directory
        guidelines_hash: Hash of CLAUDE.md

    Returns:
        List of files that need checking
    """
    needs_check = []

    for file_path in files:
        rel_path = str(file_path.relative_to(project_root))

        # Check if cached
        if rel_path not in cache.entries:
            needs_check.append(file_path)
            continue

        entry = cache.entries[rel_path]

        # Check if CLAUDE.md changed
        if entry.claude_md_hash != guidelines_hash:
            needs_check.append(file_path)
            continue

        # Check if file changed
        current_hash = compute_file_hash(file_path)
        if entry.file_hash != current_hash:
            needs_check.append(file_path)
            continue

    return needs_check


def get_cached_results(
    files: list[Path],
    cache: Cache,
    project_root: Path
) -> list[dict[str, Any]]:
    """Get results from cache for files.

    Args:
        files: Files to get results for
        cache: Cache object
        project_root: Project root directory

    Returns:
        List of cached results
    """
    results = []

    for file_path in files:
        rel_path = str(file_path.relative_to(project_root))

        if rel_path in cache.entries:
            entry = cache.entries[rel_path]
            results.append({
                "file": rel_path,
                "violations": entry.violations
            })

    return results


def init_or_load_progress(
    progress_path: Path,
    total_batches: int
) -> ProgressState:
    """Initialize or load progress state.

    Args:
        progress_path: Path to progress file
        total_batches: Total number of batches

    Returns:
        ProgressState object
    """
    if progress_path.exists():
        return load_progress(progress_path)
    return create_progress_state(total_batches)
