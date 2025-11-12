"""Command-line interface for claude-lint."""
import sys
from pathlib import Path
import click

from claude_lint.__version__ import __version__
from claude_lint.config import load_config
from claude_lint.logging_config import setup_logging, get_logger
from claude_lint.orchestrator import run_compliance_check
from claude_lint.reporter import get_exit_code, format_detailed_report, format_json_report


@click.command()
@click.version_option(version=__version__, prog_name="claude-lint")
@click.option("--full", is_flag=True, help="Full project scan")
@click.option("--diff", type=str, help="Check files changed from branch")
@click.option("--working", is_flag=True, help="Check working directory changes")
@click.option("--staged", is_flag=True, help="Check staged files")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
@click.option("--quiet", is_flag=True, help="Suppress warnings (errors only)")
@click.option("--config", type=click.Path(exists=True), help="Path to config file")
def main(
    full: bool,
    diff: str | None,
    working: bool,
    staged: bool,
    output_json: bool,
    verbose: bool,
    quiet: bool,
    config: str | None,
) -> None:
    """Claude-lint: CLAUDE.md compliance checker."""
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
        # Run compliance check
        results, metrics = run_compliance_check(
            project_root, cfg, mode=mode, base_branch=base_branch
        )

        # Format output
        if output_json:
            output = format_json_report(results, metrics)
        else:
            output = format_detailed_report(results, metrics)

        click.echo(output)

        # Exit with appropriate code
        exit_code_val = get_exit_code(results)
        sys.exit(exit_code_val)

    except KeyboardInterrupt:
        # User cancelled with Ctrl-C
        click.echo("\nOperation cancelled by user", err=True)
        sys.exit(130)  # Standard SIGINT exit code
    except (ValueError, FileNotFoundError) as e:
        # Expected errors with helpful messages
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)
    except Exception as e:
        # Unexpected errors - log details in verbose mode
        logger = get_logger(__name__)
        logger.exception("Unexpected error during execution")
        click.echo(
            f"An unexpected error occurred: {e}\n" "Run with --verbose for details.", err=True
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
