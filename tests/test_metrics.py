"""Tests for metrics tracking."""
from claude_lint.metrics import AnalysisMetrics
import time


def test_metrics_initialization():
    """Test metrics starts with zeros."""
    metrics = AnalysisMetrics()

    assert metrics.total_files_collected == 0
    assert metrics.api_calls_made == 0
    assert metrics.cache_hits == 0


def test_metrics_elapsed_time():
    """Test elapsed time calculation."""
    metrics = AnalysisMetrics()
    time.sleep(0.1)
    metrics.finish()

    assert metrics.elapsed_seconds >= 0.1
    assert metrics.elapsed_seconds < 1.0


def test_metrics_cache_hit_rate():
    """Test cache hit rate calculation."""
    metrics = AnalysisMetrics()

    metrics.cache_hits = 80
    metrics.cache_misses = 20

    assert metrics.cache_hit_rate == 80.0


def test_metrics_to_dict():
    """Test conversion to dictionary."""
    metrics = AnalysisMetrics()
    metrics.total_files_collected = 100
    metrics.api_calls_made = 10
    metrics.finish()

    d = metrics.to_dict()

    assert d["total_files_collected"] == 100
    assert d["api_calls_made"] == 10
    assert "elapsed_seconds" in d
