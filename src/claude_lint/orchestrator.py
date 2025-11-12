"""Main orchestrator coordinating all components."""
import os
from pathlib import Path
from typing import Any

from claude_lint.api_client import create_client
from claude_lint.batch_processor import process_batch
from claude_lint.cache import Cache, load_cache, save_cache
from claude_lint.collector import collect_all_files, compute_file_hash, filter_files_by_list
from claude_lint.config import Config
from claude_lint.git_utils import (
    get_changed_files_from_branch,
    get_staged_files,
    get_working_directory_files,
    is_git_repo,
)
from claude_lint.guidelines import get_claude_md_hash, read_claude_md
from claude_lint.logging_config import get_logger
from claude_lint.metrics import AnalysisMetrics
from claude_lint.processor import create_batches
from claude_lint.progress import (
    ProgressState,
    cleanup_progress,
    create_progress_state,
    get_remaining_batch_indices,
    is_progress_complete,
    load_progress,
    save_progress,
    update_progress,
)
from claude_lint.rate_limiter import RateLimiter
from claude_lint.validation import (
    validate_api_key,
    validate_batch_size,
    validate_mode,
    validate_project_root,
)

logger = get_logger(__name__)


def _process_all_batches(
    batches: list[list[Path]],
    project_root: Path,
    config: Config,
    guidelines: str,
    guidelines_hash: str,
    client: Any,
    rate_limiter: RateLimiter,
    cache: Cache,
    progress_state: ProgressState,
    progress_path: Path,
) -> tuple[list[dict[str, Any]], int]:
    """Process all batches with optional progress bar.

    Args:
        batches: List of file batches to process
        project_root: Project root directory
        config: Configuration
        guidelines: CLAUDE.md content
        guidelines_hash: Hash of CLAUDE.md
        client: Anthropic API client
        rate_limiter: Rate limiter for API calls
        cache: Cache object
        progress_state: Progress tracking state
        progress_path: Path to progress file

    Returns:
        Tuple of (all results, API calls made)
    """
    all_results = list(progress_state.results)
    api_calls_made = 0

    # Determine if we should show progress
    show_progress = config.show_progress and not os.environ.get("CLAUDE_LINT_NO_PROGRESS")

    remaining_batches = list(get_remaining_batch_indices(progress_state))
    cache_path = project_root / ".agent-lint-cache.json"

    # Common batch processing logic
    def process_batches_iter(
        progress_callback: Any = None,
    ) -> Any:
        nonlocal api_calls_made
        nonlocal progress_state

        for idx, batch_idx in enumerate(remaining_batches):
            batch = batches[batch_idx]

            if progress_callback:
                progress_callback(idx, batch_idx, len(batch))

            batch_results = process_batch(
                batch,
                project_root,
                config,
                guidelines,
                guidelines_hash,
                client,
                rate_limiter,
                cache,
            )

            all_results.extend(batch_results)
            api_calls_made += 1

            progress_state = update_progress(progress_state, batch_idx, batch_results)
            save_progress(progress_state, progress_path)
            save_cache(cache, cache_path)

            yield batch_results

    if show_progress:
        from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[bold cyan]{task.fields[status]}"),
        ) as progress:
            task = progress.add_task(
                "Analyzing files", total=len(remaining_batches), status="Starting..."
            )

            def progress_callback(idx: int, batch_idx: int, batch_size: int) -> None:
                progress.update(
                    task, status=f"Batch {idx + 1}/{len(remaining_batches)} ({batch_size} files)"
                )

            for _ in process_batches_iter(progress_callback):
                progress.update(task, advance=1, status="Complete")
    else:
        # No progress bar - just iterate
        for _ in process_batches_iter():
            pass

    return all_results, api_calls_made


def run_compliance_check(
    project_root: Path, config: Config, mode: str = "full", base_branch: str | None = None
) -> tuple[list[dict[str, Any]], AnalysisMetrics]:
    """Run compliance check.

    Args:
        project_root: Project root directory
        config: Configuration
        mode: Check mode - 'full', 'diff', 'working', 'staged'
        base_branch: Base branch for diff mode

    Returns:
        Tuple of (results list, metrics object)

    Raises:
        ValueError: If inputs are invalid
    """
    metrics = AnalysisMetrics()
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
    files_to_check = collect_files_for_mode(project_root, config, mode, base_branch)

    metrics.total_files_collected = len(files_to_check)

    if not files_to_check:
        metrics.finish()
        return [], metrics

    # Filter using cache
    files_needing_check = filter_cached_files(files_to_check, cache, project_root, guidelines_hash)

    metrics.files_from_cache = len(files_to_check) - len(files_needing_check)
    metrics.cache_hits = metrics.files_from_cache
    metrics.cache_misses = len(files_needing_check)
    metrics.files_analyzed = len(files_needing_check)

    if not files_needing_check:
        # All files cached, return cached results
        metrics.finish()
        return get_cached_results(files_to_check, cache, project_root), metrics

    # Create batches
    batches = create_batches(files_needing_check, config.batch_size)

    # Check for resumable progress
    progress_state = init_or_load_progress(progress_path, len(batches))

    # Create API client once for all batches
    # Type narrowing: validate_api_key() ensures api_key is str (not None)
    assert api_key is not None  # For mypy type narrowing only
    client = create_client(api_key, timeout=config.api_timeout_seconds)

    # Create rate limiter for API calls
    rate_limiter = RateLimiter(
        max_requests=config.api_rate_limit, window_seconds=config.api_rate_window_seconds
    )

    # Process batches with optional progress bar
    all_results, api_calls_made = _process_all_batches(
        batches=batches,
        project_root=project_root,
        config=config,
        guidelines=guidelines,
        guidelines_hash=guidelines_hash,
        client=client,
        rate_limiter=rate_limiter,
        cache=cache,
        progress_state=progress_state,
        progress_path=progress_path,
    )

    metrics.api_calls_made = api_calls_made

    # Cleanup progress on completion
    if is_progress_complete(progress_state):
        cleanup_progress(progress_path)

    # Cache is already saved after each batch - no need to save again here
    # The cache.claude_md_hash is already set correctly

    metrics.finish()
    return all_results, metrics


def collect_files_for_mode(
    project_root: Path, config: Config, mode: str, base_branch: str | None
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
    files: list[Path], cache: Cache, project_root: Path, guidelines_hash: str
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


def get_cached_results(files: list[Path], cache: Cache, project_root: Path) -> list[dict[str, Any]]:
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
            results.append({"file": rel_path, "violations": entry.violations})

    return results


def init_or_load_progress(progress_path: Path, total_batches: int) -> ProgressState:
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
