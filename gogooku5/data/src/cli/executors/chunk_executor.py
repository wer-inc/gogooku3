"""
Chunk executor for dataset building.

Executes chunk builds with:
- Status tracking (StatusManager)
- Progress monitoring (ProgressTracker)
- Resume/force logic
- Error handling and retry
"""

import logging
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ..plan import ChunkSpec, ExecutionPlan
    from ..config import Config

from builder.config.settings import DatasetBuilderSettings
from builder.pipelines.dataset_builder import DatasetBuilder

from .progress import ProgressTracker
from .status_manager import StatusManager

logger = logging.getLogger(__name__)


class ChunkExecutor:
    """
    Executor for chunked dataset builds.

    Manages chunk-by-chunk execution with status tracking,
    progress monitoring, and resume capability.
    """

    def __init__(self, config: "Config", plan: "ExecutionPlan"):
        """
        Initialize chunk executor.

        Args:
            config: Configuration object
            plan: Execution plan
        """
        self.config = config
        self.plan = plan
        self.status_mgr = StatusManager()
        self.builder = self._create_builder()

    def execute(self, chunks: List["ChunkSpec"]) -> int:
        """
        Execute chunk builds.

        Args:
            chunks: List of ChunkSpec objects

        Returns:
            Exit code (0 for success, non-zero for error)
        """
        logger.info("\n" + "=" * 70)
        logger.info("🚀 Starting CHUNKED BUILD")
        logger.info("=" * 70)
        logger.info(f"Total chunks: {len(chunks)}")
        logger.info(f"Resume mode: {self.plan.resume}")
        logger.info(f"Force mode: {self.plan.force}")

        # Filter chunks based on resume/force logic
        chunks_to_execute = self._filter_chunks(chunks)

        if not chunks_to_execute:
            logger.info("✅ All chunks already completed (use --force to rebuild)")
            return 0

        logger.info(f"Chunks to execute: {len(chunks_to_execute)}/{len(chunks)}")

        # Execute chunks with progress tracking
        failed_chunks = []
        refresh_flag = getattr(self.config, "refresh_listed", False)
        with ProgressTracker(len(chunks_to_execute), desc="Building chunks") as progress:
            for chunk in chunks_to_execute:
                try:
                    self._execute_chunk(chunk, refresh_listed=refresh_flag)
                    refresh_flag = False
                    progress.update(1)
                except Exception as e:
                    logger.error(f"❌ Chunk {chunk.chunk_id} failed: {e}")
                    failed_chunks.append((chunk, str(e)))

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("📊 Execution Summary")
        logger.info("=" * 70)

        summary = self.status_mgr.get_chunk_summary(self.plan.output_dir / "chunks")
        logger.info(f"Completed: {summary['completed']}")
        logger.info(f"Failed:    {summary['failed']}")
        logger.info(f"Running:   {summary['running']}")
        logger.info(f"Pending:   {summary['pending']}")
        logger.info(f"Total:     {summary['total']}")

        if failed_chunks:
            logger.error("\n❌ Failed chunks:")
            for chunk, error in failed_chunks:
                logger.error(f"  - {chunk.chunk_id}: {error}")
            return 1
        else:
            logger.info("\n✅ All chunks completed successfully")
            return 0

    def _filter_chunks(self, chunks: List["ChunkSpec"]) -> List["ChunkSpec"]:
        """
        Filter chunks based on resume/force logic.

        Args:
            chunks: All chunks

        Returns:
            Filtered list of chunks to execute
        """
        filtered = []
        for chunk in chunks:
            # Use status_path from builder's ChunkSpec
            status_path = chunk.status_path if hasattr(chunk, "status_path") else chunk.output_dir / "status.json"

            should_skip = self.status_mgr.should_skip_chunk(
                status_path,
                resume=self.plan.resume,
                force=self.plan.force,
            )
            if not should_skip:
                filtered.append(chunk)

        return filtered

    def _execute_chunk(self, chunk: "ChunkSpec", *, refresh_listed: bool) -> None:
        """
        Execute single chunk build.

        Args:
            chunk: ChunkSpec object (from builder.chunks)
        """
        logger.info(f"\n{'=' * 70}")
        logger.info(f"📦 Building chunk: {chunk.chunk_id}")
        logger.info(f"{'=' * 70}")
        logger.info(f"Output range: {chunk.output_start} → {chunk.output_end}")
        logger.info(f"Input range:  {chunk.input_start} → {chunk.input_end}")

        try:
            self.builder.build_chunk(chunk, refresh_listed=refresh_listed)
            logger.info(f"✅ Chunk {chunk.chunk_id} completed")

        except Exception as e:
            logger.exception(f"Chunk build failed: {e}")
            raise

    def _create_builder(self) -> DatasetBuilder:
        """
        Initialize a DatasetBuilder for chunk execution.
        """
        settings = DatasetBuilderSettings()

        if self.plan.output_dir:
            self.plan.output_dir.mkdir(parents=True, exist_ok=True)
            settings.data_output_dir = self.plan.output_dir
        if self.plan.tag:
            settings.dataset_tag = self.plan.tag
        if self.plan.gpu_enabled is not None:
            settings.use_gpu_etl = self.plan.gpu_enabled

        return DatasetBuilder(settings=settings)
