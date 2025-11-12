"""Batch processing logic."""
from pathlib import Path
from typing import TYPE_CHECKING, Any

from anthropic import Anthropic

from claude_lint.api_client import analyze_files_with_client
from claude_lint.cache import Cache, CacheEntry
from claude_lint.collector import compute_file_hash
from claude_lint.config import Config
from claude_lint.file_reader import read_batch_files
from claude_lint.logging_config import get_logger
from claude_lint.processor import build_xml_prompt, parse_response
from claude_lint.rate_limiter import RateLimiter
from claude_lint.retry import retry_with_backoff

if TYPE_CHECKING:
    from claude_lint.types import FileResult

logger = get_logger(__name__)


def process_batch(
    batch: list[Path],
    project_root: Path,
    config: Config,
    guidelines: str,
    guidelines_hash: str,
    client: Anthropic,
    rate_limiter: RateLimiter,
    cache: Cache,
) -> list[dict[str, Any]]:
    """Process a single batch of files.

    This function:
    1. Reads file contents with size/encoding checks
    2. Builds XML prompt
    3. Makes rate-limited API call with retry
    4. Parses results
    5. Updates cache

    Args:
        batch: List of file paths to process
        project_root: Project root directory
        config: Configuration
        guidelines: CLAUDE.md content
        guidelines_hash: Hash of CLAUDE.md
        client: Anthropic client
        rate_limiter: Rate limiter for API calls
        cache: Cache object to update

    Returns:
        List of file results as dicts
    """
    # Read files
    file_contents = read_batch_files(batch, project_root, config.max_file_size_mb)

    # Skip if no files to process
    if not file_contents:
        return []

    # Build prompt
    prompt = build_xml_prompt(guidelines, file_contents)

    # Make rate-limited API call with retry
    def api_call() -> str:
        rate_limiter.acquire()
        response_text, _ = analyze_files_with_client(client, guidelines, prompt, model=config.model)
        return response_text

    response = retry_with_backoff(api_call)

    # Parse results
    batch_results: list[FileResult] = parse_response(response)
    batch_results_dict: list[dict[str, Any]] = [dict(r) for r in batch_results]

    # Update cache
    for result in batch_results:
        try:
            file_path = project_root / result["file"]
            file_hash = compute_file_hash(file_path)

            cache.entries[result["file"]] = CacheEntry(
                file_hash=file_hash,
                claude_md_hash=guidelines_hash,
                violations=result["violations"],
                timestamp=int(file_path.stat().st_mtime),
            )
        except FileNotFoundError as e:
            # File was deleted between analysis and caching
            logger.debug(
                f"File {result['file']} not found during cache update, " f"likely deleted: {e}"
            )
        except (PermissionError, OSError) as e:
            # Permission or filesystem error
            logger.warning(
                f"Failed to update cache for {result['file']}: " f"{type(e).__name__}: {e}"
            )
        except Exception as e:
            # Unexpected error - log with full traceback
            logger.error(
                f"Unexpected error updating cache for {result['file']}: "
                f"{type(e).__name__}: {e}",
                exc_info=True,
            )

    return batch_results_dict
