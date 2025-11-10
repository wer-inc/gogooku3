"""
Argument parser for gogooku5-dataset CLI.

Organizes arguments into 7 logical groups:
A. Period (date range specification)
B. Chunking (chunk control and resume logic)
C. Data Sources (JQuants API, offline mode)
D. Compute Resources (GPU/CPU, workers)
E. Features (feature groups and toggles)
F. Output (paths, tags, artifacts)
G. Debug/Validation (logging, checks)
"""

import argparse
from datetime import datetime, timedelta


def build_parser() -> argparse.ArgumentParser:
    """
    Build the unified CLI argument parser.

    Returns:
        ArgumentParser with all subcommands and options configured
    """
    parser = argparse.ArgumentParser(
        prog="gogooku5-dataset",
        description="Unified CLI for gogooku5 dataset builder with automatic chunking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic build (auto-detects chunking)
  %(prog)s build --start 2020-01-01 --end 2025-01-01

  # Short period (full mode)
  %(prog)s build --start 2025-01-01 --end 2025-03-31

  # Chunked build with auto-merge
  %(prog)s build --start 2020-01-01 --end 2025-01-01 --chunk-months 3 --merge

  # Resume failed build
  %(prog)s build --start 2020-01-01 --end 2025-01-01 --resume

  # Update latest chunk only
  %(prog)s build --start 2020-01-01 --end 2025-01-01 --latest

  # Dry-run (show execution plan)
  %(prog)s build --start 2020-01-01 --end 2025-01-01 --dry-run

  # Environment check
  %(prog)s check --strict

  # Merge existing chunks
  %(prog)s merge --chunks-dir output/chunks --allow-partial
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )

    # Subcommands
    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available commands",
        required=True,
    )

    # Build command
    build_parser = subparsers.add_parser(
        "build",
        help="Build dataset (with automatic chunking)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_build_arguments(build_parser)

    # Merge command
    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge completed chunks into final dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_merge_arguments(merge_parser)

    # Check command
    check_parser = subparsers.add_parser(
        "check",
        help="Validate environment and dependencies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_check_arguments(check_parser)

    return parser


def _add_build_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for 'build' command."""

    # ===== Group A: Period Specification =====
    period_group = parser.add_argument_group(
        "Period",
        "Date range specification (mutually exclusive with --lookback-years)",
    )

    period_group.add_argument(
        "--start",
        type=str,
        metavar="DATE",
        help="Start date (YYYY-MM-DD). Required unless --lookback-years specified.",
    )

    period_group.add_argument(
        "--end",
        type=str,
        metavar="DATE",
        help="End date (YYYY-MM-DD). Defaults to today if not specified.",
    )

    period_group.add_argument(
        "--lookback-years",
        type=int,
        metavar="N",
        default=None,
        help="Build dataset for past N years (alternative to --start/--end). Default: 5 if neither --start nor --lookback-years specified.",
    )

    # ===== Group B: Chunking Control =====
    chunk_group = parser.add_argument_group(
        "Chunking",
        "Chunk splitting and execution control",
    )

    chunk_group.add_argument(
        "--chunk-months",
        type=int,
        metavar="N",
        default=3,
        help="Chunk size in months (default: 3). Used when auto-chunking is triggered.",
    )

    chunk_group.add_argument(
        "--chunk-mode",
        type=str,
        choices=["auto", "full", "chunks", "latest"],
        default="auto",
        help="""
Execution mode (default: auto):
  - auto: Auto-detect based on date range (full if ≤180 days, chunks if >180 days)
  - full: Single build without chunking
  - chunks: Force chunked execution
  - latest: Build only the latest chunk (incremental update)
        """.strip(),
    )

    chunk_group.add_argument(
        "--resume",
        action="store_true",
        help="Skip already completed chunks (checks status.json). Use with --chunk-mode=chunks.",
    )

    chunk_group.add_argument(
        "--force",
        action="store_true",
        help="Force re-build all chunks (ignore status.json). Overrides --resume.",
    )

    chunk_group.add_argument(
        "--latest",
        action="store_true",
        help="Build only the latest chunk (shortcut for --chunk-mode=latest).",
    )

    # ===== Group C: Data Sources =====
    source_group = parser.add_argument_group(
        "Data Sources",
        "API and data fetching configuration",
    )

    source_group.add_argument(
        "--jquants",
        action="store_true",
        default=True,
        help="Use JQuants API for data fetching (default: enabled).",
    )

    source_group.add_argument(
        "--offline",
        action="store_true",
        help="Offline mode: use local Parquet files instead of API. Automatically searches for cached data.",
    )

    source_group.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force API fetch, ignore all caches (USE_CACHE will be temporarily disabled).",
    )

    source_group.add_argument(
        "--refresh-listed",
        action="store_true",
        help="Refresh listed info data before building.",
    )

    # ===== Group D: Compute Resources =====
    compute_group = parser.add_argument_group(
        "Compute Resources",
        "GPU/CPU selection and parallel execution",
    )

    compute_group.add_argument(
        "--gpu",
        action="store_true",
        default=None,
        help="Enable GPU-ETL (RAPIDS/cuDF). Auto-detected by default.",
    )

    compute_group.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU-only mode (disable GPU-ETL).",
    )

    compute_group.add_argument(
        "--rmm-pool-size",
        type=str,
        metavar="SIZE",
        default=None,
        help="RMM GPU memory pool size (e.g., '40GB'). Default: read from .env (RMM_POOL_SIZE) or '40GB'.",
    )

    compute_group.add_argument(
        "--workers",
        type=int,
        metavar="N",
        default=1,
        help="Number of parallel chunk workers (default: 1). Note: Parallel execution not yet fully implemented.",
    )

    # ===== Group E: Features =====
    feature_group = parser.add_argument_group(
        "Features",
        "Feature engineering configuration",
    )

    feature_group.add_argument(
        "--features",
        type=str,
        choices=["basic", "advanced", "all"],
        default="all",
        help="""
Feature preset (default: all):
  - basic: Technical indicators, quality financial features
  - advanced: Graph, flow, volatility, sector aggregation
  - all: All available features (basic + advanced + margin + short selling)
        """.strip(),
    )

    feature_group.add_argument(
        "--enable-graph",
        action="store_true",
        default=True,
        help="Enable correlation graph features (default: enabled).",
    )

    feature_group.add_argument(
        "--disable-graph",
        action="store_true",
        help="Disable correlation graph features.",
    )

    feature_group.add_argument(
        "--enable-sector",
        action="store_true",
        default=True,
        help="Enable sector aggregation features (default: enabled).",
    )

    feature_group.add_argument(
        "--enable-margin",
        action="store_true",
        default=True,
        help="Enable daily margin interest features (default: enabled).",
    )

    feature_group.add_argument(
        "--enable-short-selling",
        action="store_true",
        default=True,
        help="Enable short selling data features (default: enabled).",
    )

    feature_group.add_argument(
        "--futures-continuous",
        action="store_true",
        help="Enable continuous futures price features (requires Premium plan).",
    )

    # ===== Group F: Output =====
    output_group = parser.add_argument_group(
        "Output",
        "Output paths and artifact configuration",
    )

    output_group.add_argument(
        "--output-dir",
        type=str,
        metavar="DIR",
        default="output",
        help="Output directory for datasets and chunks (default: output/).",
    )

    output_group.add_argument(
        "--tag",
        type=str,
        metavar="TAG",
        default="full",
        help="Dataset tag/suffix for naming (default: full).",
    )

    output_group.add_argument(
        "--merge",
        action="store_true",
        help="Automatically merge chunks after successful build.",
    )

    output_group.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow partial chunk merge (useful for testing). Use with --merge.",
    )

    output_group.add_argument(
        "--background",
        action="store_true",
        help="Run in background (daemonize). PID saved to {output-dir}/dataset.pid.",
    )

    # ===== Group G: Debug/Validation =====
    debug_group = parser.add_argument_group(
        "Debug/Validation",
        "Logging, validation, and dry-run",
    )

    debug_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Show execution plan without actually building. Useful for validating arguments.",
    )

    debug_group.add_argument(
        "--check",
        action="store_true",
        help="Run environment checks only (GPU, dependencies, credentials). Exit after checks.",
    )

    debug_group.add_argument(
        "--check-strict",
        action="store_true",
        help="Strict environment check (fail if GPU not available). Use with --check.",
    )

    debug_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging (DEBUG level).",
    )

    debug_group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Minimal logging (WARNING level and above).",
    )

    debug_group.add_argument(
        "--log-file",
        type=str,
        metavar="PATH",
        default=None,
        help="Log file path. Auto-generated if not specified.",
    )


def _add_merge_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for 'merge' command."""

    parser.add_argument(
        "--chunks-dir",
        type=str,
        metavar="DIR",
        default="output/chunks",
        help="Directory containing completed chunks (default: output/chunks).",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        metavar="DIR",
        default="output",
        help="Output directory for merged dataset (default: output/).",
    )

    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow merging even if some chunks are incomplete.",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately on any error (default: collect errors and show summary).",
    )

    parser.add_argument(
        "--tag",
        type=str,
        metavar="TAG",
        default="full",
        help="Dataset tag/suffix for naming (default: full).",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging (DEBUG level).",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Minimal logging (WARNING level and above).",
    )


def _add_check_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for 'check' command."""

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode: fail if GPU not available or any dependency missing.",
    )

    parser.add_argument(
        "--gpu-required",
        action="store_true",
        help="Require GPU availability (fail if not detected).",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging (DEBUG level).",
    )


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate argument combinations and apply defaults.

    Raises:
        ValueError: If invalid argument combination detected
    """
    # Validate date specification
    if args.command == "build":
        if not args.start and not args.lookback_years:
            # Default to 5 years
            args.lookback_years = 5

        if args.start and args.lookback_years:
            raise ValueError("Cannot specify both --start and --lookback-years. Choose one.")

        # Convert lookback_years to start/end
        if args.lookback_years:
            today = datetime.now().date()
            args.end = args.end or today.isoformat()
            end_date = datetime.fromisoformat(args.end).date()
            start_date = end_date - timedelta(days=args.lookback_years * 365)
            args.start = start_date.isoformat()

        # Default end to today
        if not args.end:
            args.end = datetime.now().date().isoformat()

        # Validate date format
        try:
            start_date = datetime.fromisoformat(args.start).date()
            end_date = datetime.fromisoformat(args.end).date()
        except ValueError as e:
            raise ValueError(f"Invalid date format: {e}. Use YYYY-MM-DD.")

        if start_date >= end_date:
            raise ValueError(f"Start date ({args.start}) must be before end date ({args.end}).")

        # Handle --latest shortcut
        if args.latest:
            args.chunk_mode = "latest"

        # Validate GPU flags
        if args.gpu and args.no_gpu:
            raise ValueError("Cannot specify both --gpu and --no-gpu.")

        # Validate logging flags
        if args.verbose and args.quiet:
            raise ValueError("Cannot specify both --verbose and --quiet.")

        # Validate chunk mode
        if args.resume and args.chunk_mode == "full":
            raise ValueError("--resume requires chunked execution. Use --chunk-mode=chunks or --chunk-mode=auto.")

        if args.force and args.resume:
            # Force overrides resume
            args.resume = False
