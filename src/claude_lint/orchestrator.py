"""Main orchestrator coordinating all components."""
import os
from pathlib import Path
from typing import Any, Optional

from claude_lint.api_client import ClaudeClient
from claude_lint.cache import Cache, CacheEntry, load_cache, save_cache
from claude_lint.collector import FileCollector
from claude_lint.config import Config
from claude_lint.git_utils import (
    is_git_repo,
    get_changed_files_from_branch,
    get_working_directory_files,
    get_staged_files
)
from claude_lint.guidelines import read_claude_md, get_claude_md_hash
from claude_lint.processor import BatchProcessor, build_xml_prompt
from claude_lint.progress import ProgressTracker
from claude_lint.retry import retry_with_backoff


class Orchestrator:
    """Main orchestrator for claude-lint."""

    def __init__(self, project_root: Path, config: Config):
        """Initialize orchestrator.

        Args:
            project_root: Project root directory
            config: Configuration
        """
        self.project_root = project_root
        self.config = config

        # Get API key
        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("API key required via config or ANTHROPIC_API_KEY env var")

        self.client = ClaudeClient(api_key)
        self.collector = FileCollector(project_root, config)
        self.processor = BatchProcessor(config.batch_size)

        # Load CLAUDE.md
        self.guidelines = read_claude_md(project_root)
        self.guidelines_hash = get_claude_md_hash(self.guidelines)

        # Load cache
        cache_path = project_root / ".agent-lint-cache.json"
        self.cache = load_cache(cache_path)
        self.cache_path = cache_path

        # Progress tracking
        progress_path = project_root / ".agent-lint-progress.json"
        self.progress_path = progress_path

    def run(
        self,
        mode: str = "full",
        base_branch: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Run compliance check.

        Args:
            mode: Check mode - 'full', 'diff', 'working', 'staged'
            base_branch: Base branch for diff mode

        Returns:
            List of results for all checked files
        """
        # Collect files to check
        files_to_check = self._collect_files(mode, base_branch)

        if not files_to_check:
            return []

        # Filter using cache
        files_needing_check = self._filter_cached(files_to_check)

        if not files_needing_check:
            # All files cached, return cached results
            return self._get_cached_results(files_to_check)

        # Create batches
        batches = self.processor.create_batches(files_needing_check)

        # Check for resumable progress
        tracker = self._init_progress_tracker(len(batches))

        # Process batches
        all_results = list(tracker.state.results)  # Start with resumed results

        for batch_idx in tracker.get_remaining_batch_indices():
            batch = batches[batch_idx]

            # Read file contents
            file_contents = {}
            for file_path in batch:
                rel_path = file_path.relative_to(self.project_root)
                content = file_path.read_text()
                file_contents[str(rel_path)] = content

            # Build prompt
            prompt = build_xml_prompt(self.guidelines, file_contents)

            # Make API call with retry
            def api_call():
                return self.client.analyze_files(self.guidelines, prompt)

            response = retry_with_backoff(api_call)

            # Parse results
            batch_results = self.processor.parse_response(response)
            all_results.extend(batch_results)

            # Update cache
            for result in batch_results:
                file_path = self.project_root / result["file"]
                file_hash = self.collector.compute_hash(file_path)

                self.cache.entries[result["file"]] = CacheEntry(
                    file_hash=file_hash,
                    claude_md_hash=self.guidelines_hash,
                    violations=result["violations"],
                    timestamp=int(Path(file_path).stat().st_mtime)
                )

            # Save progress
            tracker.update(batch_idx, batch_results)
            tracker.save()
            save_cache(self.cache, self.cache_path)

        # Cleanup progress on completion
        if tracker.is_complete():
            tracker.cleanup()

        # Update cache hash
        self.cache.claude_md_hash = self.guidelines_hash
        save_cache(self.cache, self.cache_path)

        return all_results

    def _collect_files(
        self,
        mode: str,
        base_branch: Optional[str]
    ) -> list[Path]:
        """Collect files based on mode."""
        if mode == "full":
            return self.collector.collect_all()

        if not is_git_repo(self.project_root):
            raise ValueError(f"Mode '{mode}' requires git repository")

        if mode == "diff":
            if not base_branch:
                raise ValueError("diff mode requires base_branch")
            changed = get_changed_files_from_branch(self.project_root, base_branch)
        elif mode == "working":
            changed = get_working_directory_files(self.project_root)
        elif mode == "staged":
            changed = get_staged_files(self.project_root)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return self.collector.filter_by_list(changed)

    def _filter_cached(self, files: list[Path]) -> list[Path]:
        """Filter out files that are cached and valid."""
        needs_check = []

        for file_path in files:
            rel_path = str(file_path.relative_to(self.project_root))

            # Check if cached
            if rel_path not in self.cache.entries:
                needs_check.append(file_path)
                continue

            entry = self.cache.entries[rel_path]

            # Check if CLAUDE.md changed
            if entry.claude_md_hash != self.guidelines_hash:
                needs_check.append(file_path)
                continue

            # Check if file changed
            current_hash = self.collector.compute_hash(file_path)
            if entry.file_hash != current_hash:
                needs_check.append(file_path)
                continue

        return needs_check

    def _get_cached_results(self, files: list[Path]) -> list[dict[str, Any]]:
        """Get results from cache for files."""
        results = []

        for file_path in files:
            rel_path = str(file_path.relative_to(self.project_root))

            if rel_path in self.cache.entries:
                entry = self.cache.entries[rel_path]
                results.append({
                    "file": rel_path,
                    "violations": entry.violations
                })

        return results

    def _init_progress_tracker(self, total_batches: int) -> ProgressTracker:
        """Initialize or load progress tracker."""
        if self.progress_path.exists():
            return ProgressTracker.load(self.progress_path)
        return ProgressTracker(self.progress_path, total_batches)
