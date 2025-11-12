import json
from claude_lint.reporter import (
    get_exit_code,
    get_summary,
    format_detailed_report,
    format_json_report,
)
from claude_lint.metrics import AnalysisMetrics


def test_format_detailed_report():
    """Test formatting detailed human-readable report."""
    results = [
        {
            "file": "src/main.py",
            "violations": [
                {"type": "missing-pattern", "message": "No tests found", "line": None},
                {"type": "anti-pattern", "message": "Nested complexity detected", "line": 42},
            ],
        },
        {"file": "src/utils.py", "violations": []},
    ]

    metrics = AnalysisMetrics()
    metrics.total_files_collected = 2
    metrics.files_analyzed = 2
    metrics.api_calls_made = 1
    metrics.finish()

    report = format_detailed_report(results, metrics)

    assert "src/main.py" in report
    assert "2 violation(s)" in report
    assert "No tests found" in report
    assert "line 42" in report
    assert "src/utils.py" in report
    assert "No violations" in report
    assert "ANALYSIS METRICS" in report
    assert "Elapsed time:" in report


def test_format_json_report():
    """Test formatting JSON report."""
    results = [
        {
            "file": "src/main.py",
            "violations": [{"type": "missing-pattern", "message": "No tests found", "line": None}],
        }
    ]

    metrics = AnalysisMetrics()
    metrics.total_files_collected = 1
    metrics.files_analyzed = 1
    metrics.api_calls_made = 1
    metrics.finish()

    report = format_json_report(results, metrics)
    data = json.loads(report)

    assert "results" in data
    assert len(data["results"]) == 1
    assert data["results"][0]["file"] == "src/main.py"
    assert data["summary"]["total_files"] == 1
    assert data["summary"]["files_with_violations"] == 1
    assert "metrics" in data
    assert data["metrics"]["api_calls_made"] == 1


def test_reporter_get_exit_code():
    """Test getting exit code based on results."""
    # No violations
    clean_results = [{"file": "a.py", "violations": []}]
    assert get_exit_code(clean_results) == 0

    # Has violations
    dirty_results = [
        {"file": "a.py", "violations": []},
        {"file": "b.py", "violations": [{"type": "error", "message": "bad"}]},
    ]
    assert get_exit_code(dirty_results) == 1


def test_reporter_print_summary():
    """Test printing summary statistics."""
    results = [
        {"file": "a.py", "violations": []},
        {"file": "b.py", "violations": [{"type": "error", "message": "e1"}]},
        {
            "file": "c.py",
            "violations": [{"type": "error", "message": "e2"}, {"type": "warn", "message": "e3"}],
        },
    ]

    summary = get_summary(results)

    assert summary["total_files"] == 3
    assert summary["files_with_violations"] == 2
    assert summary["total_violations"] == 3
    assert summary["clean_files"] == 1
