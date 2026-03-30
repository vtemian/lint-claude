"""Command-line interface for claude-lint."""

import json
import sys
from pathlib import Path
from typing import Any, TextIO

import click

from claude_lint.__version__ import __version__
from claude_lint.config import load_config
from claude_lint.logging_config import get_logger, setup_logging
from claude_lint.orchestrator import run_compliance_check
from claude_lint.reporter import format_detailed_report, format_json_report, get_exit_code


def _write_batch_results(
    file_handle: TextIO, batch_results: list[dict[str, Any]], is_json: bool
) -> None:
    """Write batch results to file in streaming fashion.

    Args:
        file_handle: Open file handle to write to
        batch_results: List of file results from this batch
        is_json: Whether to format as JSON (JSON Lines format)
    """
    if is_json:
        # JSON Lines format - one JSON object per line
        for result in batch_results:
            file_handle.write(json.dumps(result) + "\n")
    else:
        # Text format - format each file result
        for result in batch_results:
            file_path = result.get("file", "unknown")
            violations = result.get("violations", [])

            if not violations:
                file_handle.write(f"[OK] {file_path}\n")
                file_handle.write("   No violations\n\n")
            else:
                file_handle.write(f"[FILE] {file_path}\n")
                file_handle.write(f"   {len(violations)} violation(s) found:\n\n")

                for violation in violations:
                    vtype = violation.get("type", "unknown")
                    message = violation.get("message", "No message")
                    line = violation.get("line")

                    line_str = f" (line {line})" if line else ""
                    file_handle.write(f"   [WARNING] [{vtype}]{line_str}\n")
                    file_handle.write(f"      {message}\n\n")

    file_handle.flush()


@click.command()
@click.version_option(version=__version__, prog_name="lint-claude")
@click.option("--full", is_flag=True, help="Full project scan")
@click.option("--diff", type=str, help="Check files changed from branch")
@click.option("--working", is_flag=True, help="Check working directory changes")
@click.option("--staged", is_flag=True, help="Check staged files")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option("--output", "-o", type=click.Path(), help="Write output to file")
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
@click.option("--quiet", is_flag=True, help="Suppress warnings (errors only)")
@click.option("--config", type=click.Path(exists=True), help="Path to config file")
def main(
    full: bool,
    diff: str | None,
    working: bool,
    staged: bool,
    output_json: bool,
    output: str | None,
    verbose: bool,
    quiet: bool,
    config: str | None,
) -> None:
    """Claude-lint: CLAUDE.md compliance checker."""
    try:
        _run_main(full, diff, working, staged, output_json, output, verbose, quiet, config)
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user", err=True)
        sys.exit(130)  # Standard SIGINT exit code


def _run_main(
    full: bool,
    diff: str | None,
    working: bool,
    staged: bool,
    output_json: bool,
    output: str | None,
    verbose: bool,
    quiet: bool,
    config: str | None,
) -> None:
    """Internal main logic - extracted for KeyboardInterrupt handling."""
    # Setup logging
    setup_logging(verbose=verbose, quiet=quiet)
    # Determine mode
    mode_count = sum([full, bool(diff), working, staged])
    if mode_count == 0:
        click.echo("Error: Must specify one mode: --full, --diff, --working, or --staged", err=True)
        sys.exit(2)
    elif mode_count > 1:
        click.echo("Error: Only one mode can be specified", err=True)
        sys.exit(2)

    if full:
        mode = "full"
        base_branch = None
    elif diff:
        mode = "diff"
        base_branch = diff
    elif working:
        mode = "working"
        base_branch = None
    elif staged:
        mode = "staged"
        base_branch = None

    # Load config
    project_root = Path.cwd()
    config_path = Path(config) if config else project_root / ".agent-lint.json"
    cfg = load_config(config_path)

    try:
        # Setup streaming output if file specified
        output_file_handle = None
        if output:
            output_path = Path(output)
            try:
                output_file_handle = output_path.open("w")
                # Write header for text format
                if not output_json:
                    output_file_handle.write("=" * 70 + "\n")
                    output_file_handle.write("CLAUDE.MD COMPLIANCE REPORT (STREAMING)\n")
                    output_file_handle.write("=" * 70 + "\n\n")
                    output_file_handle.flush()
            except (OSError, PermissionError) as e:
                click.echo(f"Error opening {output_path}: {e}", err=True)
                sys.exit(2)

        try:
            # Run compliance check with streaming callback
            def stream_results(batch_results: list) -> None:
                if output_file_handle:
                    _write_batch_results(output_file_handle, batch_results, output_json)

            results, metrics = run_compliance_check(
                project_root,
                cfg,
                mode=mode,
                base_branch=base_branch,
                stream_callback=stream_results if output else None,
            )

            # Format full output for stdout
            if output_json:
                report = format_json_report(results, metrics)
            else:
                report = format_detailed_report(results, metrics)

            # Write summary to file if streaming
            if output_file_handle:
                if not output_json:
                    output_file_handle.write("\n" + "=" * 70 + "\n")
                    output_file_handle.write("SUMMARY\n")
                    output_file_handle.write("=" * 70 + "\n")
                    # Extract summary from full report
                    summary_lines = report.split("\n")
                    in_summary = False
                    for line in summary_lines:
                        if "SUMMARY" in line or in_summary:
                            in_summary = True
                            output_file_handle.write(line + "\n")
                    output_file_handle.flush()
                else:
                    # For JSON, write final summary object
                    import json

                    summary_obj = {
                        "summary": {
                            "total_files": metrics.total_files_collected,
                            "files_analyzed": metrics.files_analyzed,
                            "api_calls": metrics.api_calls_made,
                        }
                    }
                    output_file_handle.write(json.dumps(summary_obj) + "\n")
                    output_file_handle.flush()

                output_file_handle.close()
                click.echo(f"Results written to {output_path}", err=True)

            # Always print to stdout as well
            click.echo(report)

            # Exit with appropriate code
            exit_code_val = get_exit_code(results)
            sys.exit(exit_code_val)

        finally:
            if output_file_handle and not output_file_handle.closed:
                output_file_handle.close()

    except KeyboardInterrupt:
        # Re-raise to be caught by main()
        raise
    except (ValueError, FileNotFoundError) as e:
        # Expected errors with helpful messages
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)
    except Exception as e:
        # Unexpected errors - log details in verbose mode
        logger = get_logger(__name__)
        logger.exception("Unexpected error during execution")
        click.echo(f"An unexpected error occurred: {e}\nRun with --verbose for details.", err=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
