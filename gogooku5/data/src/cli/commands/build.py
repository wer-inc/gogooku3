"""
Build command implementation.

Implements:
- Automatic mode detection (full/chunks/latest)
- Execution plan generation
- Dry-run display
- DatasetBuilder integration for full mode
- Chunked execution with optional auto-merge
- Check-only mode
"""

import logging
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config

from ..plan import ExecutionPlan, determine_mode, display_plan

logger = logging.getLogger(__name__)


class BuildCommand:
    """
    Build dataset with automatic chunking.

    Implements Phase 2: Core build logic with mode detection,
    plan generation, and DatasetBuilder integration.
    """

    def __init__(self, config: "Config"):
        """
        Initialize build command.

        Args:
            config: Configuration object
        """
        self.config = config
        logger.debug("BuildCommand initialized")

    def execute(self) -> int:
        """
        Execute build command.

        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            # Step 1: Check-only mode
            if self.config.check_only:
                logger.info("✅ Configuration validated. Use --dry-run to see execution plan.")
                return 0

            # Step 2: Determine execution mode
            mode = self._determine_mode()
            logger.info(f"📊 Execution mode: {mode.upper()}")

            # Step 3: Create execution plan
            plan = self._create_plan(mode)

            # Step 4: Dry-run mode
            if self.config.dry_run:
                self._display_dry_run(plan)
                return 0

            # Step 5: Execute based on mode
            if mode == "full":
                return self._execute_full_build(plan)
            elif mode == "chunks":
                return self._execute_chunked_build(plan)
            elif mode == "latest":
                return self._execute_latest_chunk(plan)
            else:
                logger.error(f"Unknown execution mode: {mode}")
                return 1

        except Exception as e:
            logger.exception(f"Build command failed: {e}")
            return 1

    def _determine_mode(self) -> str:
        """
        Determine execution mode.

        Returns:
            Execution mode: "full", "chunks", or "latest"
        """
        start_date = datetime.fromisoformat(self.config.start_date).date()
        end_date = datetime.fromisoformat(self.config.end_date).date()

        mode = determine_mode(
            start_date=start_date,
            end_date=end_date,
            chunk_mode=self.config.chunk_mode,
            latest=self.config.latest,
        )

        logger.debug(
            f"Mode determination: range={start_date} to {end_date}, "
            f"chunk_mode={self.config.chunk_mode}, latest={self.config.latest} "
            f"→ {mode}"
        )

        return mode

    def _create_plan(self, mode: str) -> ExecutionPlan:
        """
        Create execution plan.

        Args:
            mode: Execution mode

        Returns:
            ExecutionPlan object
        """
        start_date = datetime.fromisoformat(self.config.start_date).date()
        end_date = datetime.fromisoformat(self.config.end_date).date()
        total_days = (end_date - start_date).days

        plan = ExecutionPlan(
            mode=mode,
            start_date=start_date,
            end_date=end_date,
            total_days=total_days,
            output_dir=self.config.output_dir,
            tag=self.config.tag,
            gpu_enabled=self.config.gpu_enabled,
            features=self.config.feature_preset,
        )

        if mode in ("chunks", "latest"):
            # Calculate number of chunks
            months_total = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
            num_chunks = (months_total + self.config.chunk_months - 1) // self.config.chunk_months

            plan.chunk_months = self.config.chunk_months
            plan.num_chunks = num_chunks if mode == "chunks" else 1
            plan.resume = self.config.resume
            plan.force = self.config.force

        return plan

    def _display_dry_run(self, plan: ExecutionPlan) -> None:
        """
        Display dry-run information.

        Args:
            plan: ExecutionPlan object
        """
        logger.info("\n" + "=" * 70)
        logger.info("🔍 DRY-RUN MODE (No actual execution)")
        logger.info("=" * 70)

        # Display plan
        print(plan)

        # Display chunk details for chunked mode
        if plan.mode in ("chunks", "latest"):
            # Import ChunkPlanner
            sys.path.insert(0, str(Path(__file__).parents[3]))
            from builder.chunks import ChunkPlanner
            from builder.config.settings import DatasetBuilderSettings

            # Get warmup days
            warmup_days = self._get_warmup_days()

            # Initialize settings and planner
            settings = DatasetBuilderSettings()
            planner = ChunkPlanner(
                settings=settings,
                warmup_days=warmup_days,
                output_root=plan.output_dir / "chunks",
                months_per_chunk=plan.chunk_months,
            )

            # Calculate chunks
            chunks = planner.plan(
                start=plan.start_date.isoformat(),
                end=plan.end_date.isoformat(),
            )

            if plan.mode == "latest":
                chunks = [chunks[-1]]  # Only latest chunk

            display_plan(plan, chunks)

        # Display next steps
        logger.info("\n📝 Next Steps:")
        logger.info("  1. Remove --dry-run to execute")
        logger.info("  2. Monitor progress in logs")
        logger.info("  3. Check output in: %s", plan.output_dir)

    def _get_warmup_days(self) -> int:
        """
        Get warmup days from config or calculate automatically.

        Returns:
            Number of warmup days
        """
        warmup_days_str = getattr(self.config, "warmup_days", "auto")

        if warmup_days_str == "auto":
            # Auto-calculate: lookback + horizon + embargo + safety
            lookback = 252  # 1 year
            horizon = 20  # Max prediction horizon
            embargo = 5  # Walk-forward embargo
            safety = 60  # Safety margin
            return lookback + horizon + embargo + safety  # 337 days
        else:
            return int(warmup_days_str)

    def _execute_full_build(self, plan: ExecutionPlan) -> int:
        """
        Execute full build (single run without chunking).

        Args:
            plan: ExecutionPlan object

        Returns:
            Exit code
        """
        logger.info("\n" + "=" * 70)
        logger.info("🚀 Starting FULL BUILD")
        logger.info("=" * 70)

        # Import DatasetBuilder and settings
        try:
            sys.path.insert(0, str(Path(__file__).parents[3]))
            from builder.config.settings import DatasetBuilderSettings
            from builder.pipelines.dataset_builder import DatasetBuilder

            logger.debug("DatasetBuilder imported from gogooku5/data")
        except ImportError as e:
            logger.error(
                "DatasetBuilder not found. "
                "Please ensure gogooku5/data/src/builder/pipelines/dataset_builder.py exists."
            )
            logger.error(f"Import error: {e}")
            return 1

        try:
            # Initialize DatasetBuilderSettings (auto-loads from .env)
            logger.info("Loading DatasetBuilder settings from .env...")
            settings = DatasetBuilderSettings()

            # Override with CLI-specified values
            if plan.output_dir:
                settings.data_output_dir = plan.output_dir
            if plan.tag:
                settings.dataset_tag = plan.tag
            if plan.gpu_enabled is not None:
                settings.use_gpu_etl = plan.gpu_enabled

            # Initialize DatasetBuilder
            logger.info("Initializing DatasetBuilder...")
            builder = DatasetBuilder(settings=settings)

            # Execute build
            start_str = plan.start_date.isoformat()
            end_str = plan.end_date.isoformat()
            logger.info(f"Building dataset: {start_str} → {end_str}")

            # Call DatasetBuilder.build()
            output_path = builder.build(
                start=start_str,
                end=end_str,
                refresh_listed=False,
            )

            logger.info("✅ Full build completed")
            logger.info(f"   Output: {output_path}")

            return 0

        except Exception as e:
            logger.exception(f"Full build failed: {e}")
            return 1

    def _execute_chunked_build(self, plan: ExecutionPlan) -> int:
        """
        Execute chunked build.

        Args:
            plan: ExecutionPlan object

        Returns:
            Exit code
        """
        # Import ChunkPlanner from builder
        sys.path.insert(0, str(Path(__file__).parents[3]))
        from builder.chunks import ChunkPlanner
        from builder.config.settings import DatasetBuilderSettings

        # Initialize settings
        settings = DatasetBuilderSettings()

        # Calculate chunks using ChunkPlanner
        warmup_days = self._get_warmup_days()
        planner = ChunkPlanner(
            settings=settings,
            warmup_days=warmup_days,
            output_root=plan.output_dir / "chunks",
            months_per_chunk=plan.chunk_months,
        )

        chunks = planner.plan(
            start=plan.start_date.isoformat(),
            end=plan.end_date.isoformat(),
        )

        # Execute with ChunkExecutor
        from ..executors.chunk_executor import ChunkExecutor

        executor = ChunkExecutor(self.config, plan)
        exit_code = executor.execute(chunks)

        if exit_code != 0:
            return exit_code

        if self.config.auto_merge:
            return self._run_auto_merge(plan)

        return 0

    def _execute_latest_chunk(self, plan: ExecutionPlan) -> int:
        """
        Execute latest chunk only.

        Args:
            plan: ExecutionPlan object

        Returns:
            Exit code
        """
        # Import ChunkPlanner from builder
        sys.path.insert(0, str(Path(__file__).parents[3]))
        from builder.chunks import ChunkPlanner
        from builder.config.settings import DatasetBuilderSettings

        # Initialize settings
        settings = DatasetBuilderSettings()

        # Calculate all chunks using ChunkPlanner
        warmup_days = self._get_warmup_days()
        planner = ChunkPlanner(
            settings=settings,
            warmup_days=warmup_days,
            output_root=plan.output_dir / "chunks",
            months_per_chunk=plan.chunk_months,
        )

        all_chunks = planner.plan(
            start=plan.start_date.isoformat(),
            end=plan.end_date.isoformat(),
        )

        # Take only the latest chunk
        latest_chunk = [all_chunks[-1]] if all_chunks else []

        if not latest_chunk:
            logger.error("No chunks to execute")
            return 1

        logger.info(f"Latest chunk: {latest_chunk[0].chunk_id}")

        # Execute with ChunkExecutor
        from ..executors.chunk_executor import ChunkExecutor

        executor = ChunkExecutor(self.config, plan)
        exit_code = executor.execute(latest_chunk)

        if exit_code != 0:
            return exit_code

        if self.config.auto_merge:
            return self._run_auto_merge(plan)

        return 0

    def _run_auto_merge(self, plan: ExecutionPlan) -> int:
        """
        Run merge command automatically after successful chunk build.
        """
        from ..commands.merge import MergeCommand
        from ..config import Config as CLIConfig

        logger.info("\n" + "=" * 70)
        logger.info("🔗 Auto-merge enabled: merging completed chunks")
        logger.info("=" * 70)

        merge_args = Namespace(
            command="merge",
            chunks_dir=str(plan.output_dir / "chunks"),
            output_dir=str(plan.output_dir),
            allow_partial=self.config.allow_partial,
            strict=False,
            tag=self.config.tag,
            verbose=False,
            quiet=False,
            log_file=None,
        )

        merge_config = CLIConfig(merge_args)
        merge_cmd = MergeCommand(merge_config)
        return merge_cmd.execute()
