"""
Main entry point for gogooku5-dataset CLI.

Handles:
- Argument parsing
- Configuration loading
- Validation
- Command dispatching
- Logging setup
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from .args import build_parser, validate_args
from .config import load_config
from .validators import (
    ValidationError,
    check_environment,
    print_check_report,
    validate_config,
)


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """
    Configure logging for CLI.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers,
    )


def main(argv: Optional[list] = None) -> int:
    """
    Main entry point for CLI.

    Args:
        argv: Command-line arguments (default: sys.argv[1:])

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Parse arguments
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        # Validate arguments
        validate_args(args)
    except ValueError as e:
        print(f"❌ Argument validation error: {e}", file=sys.stderr)
        return 1

    # Load configuration
    try:
        config = load_config(args)
    except Exception as e:
        print(f"❌ Configuration loading error: {e}", file=sys.stderr)
        return 1

    # Setup logging
    setup_logging(
        level=config.log_level,
        log_file=config.log_file,
    )

    logger = logging.getLogger(__name__)
    logger.info("🚀 gogooku5-dataset CLI v1.0.0")
    logger.info(f"📋 Command: {config.command}")

    # Validate configuration
    try:
        # Check command runs validation separately
        if config.command != "check":
            strict = getattr(config, "check_strict", False)
            validate_config(config, strict=strict)
    except ValidationError as e:
        logger.error(str(e))
        return 1

    # Dispatch to command handler
    try:
        if config.command == "build":
            from .commands.build import BuildCommand

            cmd = BuildCommand(config)
            return cmd.execute()

        elif config.command == "merge":
            from .commands.merge import MergeCommand

            cmd = MergeCommand(config)
            return cmd.execute()

        elif config.command == "check":
            strict = config.check_strict or config.gpu_required
            success, report = check_environment(config, strict=strict)
            print_check_report(report)

            if success:
                logger.info("✅ All checks passed")
                return 0
            else:
                logger.error("❌ Some checks failed")
                return 1

        else:
            logger.error(f"Unknown command: {config.command}")
            return 1

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user")
        return 130

    except Exception as e:
        logger.exception(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
