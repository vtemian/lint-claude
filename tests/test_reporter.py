import json
import pytest
from claude_lint.reporter import Reporter, format_detailed_report, format_json_report


def test_format_detailed_report():
    """Test formatting detailed human-readable report."""
    results = [
        {
            "file": "src/main.py",
            "violations": [
                {
                    "type": "missing-pattern",
                    "message": "No tests found",
                    "line": None
                },
                {
                    "type": "anti-pattern",
                    "message": "Nested complexity detected",
                    "line": 42
                }
            ]
        },
        {
            "file": "src/utils.py",
            "violations": []
        }
    ]

    report = format_detailed_report(results)

    assert "src/main.py" in report
    assert "2 violation(s)" in report
    assert "No tests found" in report
    assert "line 42" in report
    assert "src/utils.py" in report
    assert "No violations" in report


def test_format_json_report():
    """Test formatting JSON report."""
    results = [
        {
            "file": "src/main.py",
            "violations": [
                {
                    "type": "missing-pattern",
                    "message": "No tests found",
                    "line": None
                }
            ]
        }
    ]

    report = format_json_report(results)
    data = json.loads(report)

    assert "results" in data
    assert len(data["results"]) == 1
    assert data["results"][0]["file"] == "src/main.py"
    assert data["summary"]["total_files"] == 1
    assert data["summary"]["files_with_violations"] == 1


def test_reporter_get_exit_code():
    """Test getting exit code based on results."""
    reporter = Reporter()

    # No violations
    clean_results = [{"file": "a.py", "violations": []}]
    assert reporter.get_exit_code(clean_results) == 0

    # Has violations
    dirty_results = [
        {"file": "a.py", "violations": []},
        {"file": "b.py", "violations": [{"type": "error", "message": "bad"}]}
    ]
    assert reporter.get_exit_code(dirty_results) == 1


def test_reporter_print_summary():
    """Test printing summary statistics."""
    results = [
        {"file": "a.py", "violations": []},
        {"file": "b.py", "violations": [{"type": "error", "message": "e1"}]},
        {"file": "c.py", "violations": [
            {"type": "error", "message": "e2"},
            {"type": "warn", "message": "e3"}
        ]}
    ]

    reporter = Reporter()
    summary = reporter.get_summary(results)

    assert summary["total_files"] == 3
    assert summary["files_with_violations"] == 2
    assert summary["total_violations"] == 3
    assert summary["clean_files"] == 1
