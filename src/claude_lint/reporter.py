"""Report formatting and output."""
import json
from typing import Any

from claude_lint.metrics import AnalysisMetrics
from claude_lint.types import FileResult


def format_detailed_report(results: list[dict[str, Any]], metrics: AnalysisMetrics) -> str:
    """Format results as detailed human-readable report.

    Args:
        results: List of file results with violations
        metrics: Analysis metrics

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 70)
    lines.append("CLAUDE.MD COMPLIANCE REPORT")
    lines.append("=" * 70)
    lines.append("")

    for result in results:
        file_path = result["file"]
        violations = result["violations"]

        if violations:
            lines.append(f"[FILE] {file_path}")
            lines.append(f"   {len(violations)} violation(s) found:")
            lines.append("")

            for violation in violations:
                vtype = violation["type"]
                message = violation["message"]
                line = violation.get("line")

                line_info = f" (line {line})" if line else ""
                lines.append(f"   [WARNING] [{vtype}]{line_info}")
                lines.append(f"      {message}")
                lines.append("")
        else:
            lines.append(f"[OK] {file_path}")
            lines.append("   No violations")
            lines.append("")

    # Add metrics section
    lines.append("=" * 70)
    lines.append("ANALYSIS METRICS")
    lines.append("=" * 70)
    lines.append(f"Elapsed time: {metrics.elapsed_seconds:.2f}s")
    lines.append(f"Files collected: {metrics.total_files_collected}")
    lines.append(f"Files from cache: {metrics.files_from_cache}")
    lines.append(f"Files analyzed: {metrics.files_analyzed}")
    lines.append(f"API calls made: {metrics.api_calls_made}")
    lines.append(f"Cache hit rate: {metrics.cache_hit_rate:.1f}%")
    lines.append("")

    return "\n".join(lines)


def format_json_report(results: list[dict[str, Any]], metrics: AnalysisMetrics) -> str:
    """Format results as JSON.

    Args:
        results: List of file results with violations
        metrics: Analysis metrics

    Returns:
        JSON string
    """
    total_files = len(results)
    files_with_violations = sum(1 for r in results if r["violations"])
    total_violations = sum(len(r["violations"]) for r in results)

    report = {
        "results": results,
        "summary": {
            "total_files": total_files,
            "files_with_violations": files_with_violations,
            "clean_files": total_files - files_with_violations,
            "total_violations": total_violations,
        },
        "metrics": metrics.to_dict(),
    }

    return json.dumps(report, indent=2)


def get_exit_code(results: list[dict[str, Any]]) -> int:
    """Get exit code based on results.

    Args:
        results: List of file results

    Returns:
        0 if no violations, 1 if violations found
    """
    has_violations = any(r["violations"] for r in results)
    return 1 if has_violations else 0


def get_summary(results: list[FileResult]) -> dict[str, int]:
    """Get summary statistics.

    Args:
        results: List of file results

    Returns:
        Dict with summary counts
    """
    total_files = len(results)
    files_with_violations = sum(1 for r in results if r["violations"])
    total_violations = sum(len(r["violations"]) for r in results)

    return {
        "total_files": total_files,
        "files_with_violations": files_with_violations,
        "clean_files": total_files - files_with_violations,
        "total_violations": total_violations,
    }
