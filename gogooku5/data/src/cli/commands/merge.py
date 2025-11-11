"""
Merge command implementation.

Wraps gogooku5/data/tools/merge_chunks.py so CLI users can merge
completed chunks into a final dataset artifact.
"""

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ..config import Config

# Add tools directory to path for merge_chunks import
tools_dir = Path(__file__).parents[3] / "tools"
if str(tools_dir) not in sys.path:
    sys.path.insert(0, str(tools_dir))

import merge_chunks  # noqa: E402 - dynamic path addition required

logger = logging.getLogger(__name__)


class MergeCommand:
    """Merge completed chunks into final dataset artifacts."""

    def __init__(self, config: "Config"):
        self.config = config

    def execute(self) -> int:
        """
        Execute merge command by delegating to merge_chunks.main().

        Returns:
            Exit code from merge_chunks CLI.
        """
        logger.info("\n" + "=" * 70)
        logger.info("🔗 Starting MERGE")
        logger.info("=" * 70)
        logger.info(f"Chunks Dir:   {self.config.chunks_dir}")
        logger.info(f"Output Dir:   {self.config.output_dir}")
        logger.info(f"Allow Partial: {self.config.allow_partial}")
        logger.info(f"Strict Mode:   {self.config.strict}")

        argv = self._build_argv()

        try:
            exit_code = merge_chunks.main(argv)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception(f"Merge command failed: {exc}")
            return 1

        if exit_code == 0:
            logger.info("✅ Merge completed successfully")
        else:
            logger.error("❌ Merge command returned exit code %s", exit_code)
        return exit_code

    def _build_argv(self) -> List[str]:
        """Construct argument list for merge_chunks.main"""
        argv: List[str] = []

        chunks_dir = Path(self.config.chunks_dir)
        output_dir = Path(self.config.output_dir)

        argv.extend(["--chunks-dir", str(chunks_dir)])
        argv.extend(["--output-dir", str(output_dir)])

        if self.config.allow_partial:
            argv.append("--allow-partial")
        if self.config.strict:
            argv.append("--strict")

        return argv
