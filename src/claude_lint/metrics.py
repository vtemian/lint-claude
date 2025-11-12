"""Metrics and telemetry tracking."""
import time
from dataclasses import dataclass, field


@dataclass
class AnalysisMetrics:
    """Metrics collected during analysis."""

    # Timing
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None

    # Files
    total_files_collected: int = 0
    files_from_cache: int = 0
    files_analyzed: int = 0
    files_skipped: int = 0

    # API calls
    api_calls_made: int = 0
    api_retries: int = 0

    # Cache
    cache_hits: int = 0
    cache_misses: int = 0

    # Rate limiting
    rate_limit_waits: int = 0
    total_wait_time: float = 0.0

    def finish(self) -> None:
        """Mark analysis as finished."""
        self.end_time = time.time()

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def cache_hit_rate(self) -> float:
        """Get cache hit rate as percentage."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return (self.cache_hits / total) * 100

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "total_files_collected": self.total_files_collected,
            "files_from_cache": self.files_from_cache,
            "files_analyzed": self.files_analyzed,
            "files_skipped": self.files_skipped,
            "api_calls_made": self.api_calls_made,
            "api_retries": self.api_retries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(self.cache_hit_rate, 1),
            "rate_limit_waits": self.rate_limit_waits,
            "total_wait_time": round(self.total_wait_time, 2),
        }
