#!/usr/bin/env python3
"""
Memory-safe dataset generator with chunked processing and checkpointing.

This script generates ML datasets in a memory-efficient way by:
1. Processing data in yearly chunks
2. Saving intermediate checkpoints
3. Monitoring memory usage
4. Resuming from interruptions

Usage:
    python scripts/data/dataset_generator_safe.py \\
        --start-date 2020-09-01 --end-date 2025-09-01 \\
        [--chunk-years 1] [--resume] [--force]

Features:
    - Chunked processing (default: 1 year per chunk)
    - Automatic checkpointing
    - Resume from interruption
    - Memory monitoring
    - Progress tracking
"""

import argparse
import asyncio
import json
import logging
import os
import psutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

# Load environment variables
env_path = ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)

# Setup logging
LOG_DIR = Path("_logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"dataset_safe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# Checkpoint directory
CHECKPOINT_DIR = Path("output/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


class MemorySafeDatasetGenerator:
    """Generate ML datasets with memory safety and checkpointing."""

    def __init__(
        self,
        start_date: str,
        end_date: str,
        chunk_years: int = 1,
        memory_limit_pct: float = 70.0,
        resume: bool = False,
        force: bool = False,
    ):
        """Initialize generator.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            chunk_years: Years per chunk (default: 1)
            memory_limit_pct: Memory limit as percentage (default: 70%)
            resume: Resume from checkpoint
            force: Force overwrite existing checkpoint
        """
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.chunk_years = chunk_years
        self.memory_limit_pct = memory_limit_pct
        self.resume = resume
        self.force = force

        # State
        self.checkpoint_file = CHECKPOINT_DIR / f"dataset_{start_date}_{end_date}.json"
        self.checkpoint_data: Dict = {}
        self.chunks: List[Tuple[datetime, datetime]] = []
        self.completed_chunks: List[int] = []

        # Memory tracking
        self.total_memory = psutil.virtual_memory().total / (1024**3)  # GB
        self.memory_limit_gb = self.total_memory * self.memory_limit_pct / 100

        logger.info(f"Total memory: {self.total_memory:.1f} GB")
        logger.info(f"Memory limit: {self.memory_limit_gb:.1f} GB ({self.memory_limit_pct}%)")

    def create_chunks(self) -> List[Tuple[datetime, datetime]]:
        """Create date range chunks."""
        chunks = []
        current = self.start_date

        while current < self.end_date:
            chunk_end = min(
                current + timedelta(days=365 * self.chunk_years),
                self.end_date
            )
            chunks.append((current, chunk_end))
            current = chunk_end

        logger.info(f"Created {len(chunks)} chunks ({self.chunk_years} year(s) each)")
        return chunks

    def load_checkpoint(self) -> bool:
        """Load checkpoint from file.

        Returns:
            True if checkpoint was loaded, False otherwise
        """
        if not self.checkpoint_file.exists():
            logger.info("No checkpoint file found")
            return False

        try:
            with open(self.checkpoint_file, "r") as f:
                self.checkpoint_data = json.load(f)

            self.completed_chunks = self.checkpoint_data.get("completed_chunks", [])
            logger.info(f"Loaded checkpoint: {len(self.completed_chunks)} chunks completed")
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def save_checkpoint(self, chunk_idx: int, chunk_path: Path):
        """Save checkpoint after completing a chunk."""
        self.checkpoint_data.update({
            "start_date": self.start_date.strftime("%Y-%m-%d"),
            "end_date": self.end_date.strftime("%Y-%m-%d"),
            "chunk_years": self.chunk_years,
            "completed_chunks": self.completed_chunks + [chunk_idx],
            "chunks": [
                {
                    "index": i,
                    "start": start.strftime("%Y-%m-%d"),
                    "end": end.strftime("%Y-%m-%d"),
                    "completed": i in self.completed_chunks or i == chunk_idx,
                    "path": str(chunk_path) if i == chunk_idx else self.checkpoint_data.get("chunks", [{}])[i].get("path", "")
                }
                for i, (start, end) in enumerate(self.chunks)
            ],
            "last_updated": datetime.now().isoformat(),
        })

        try:
            with open(self.checkpoint_file, "w") as f:
                json.dump(self.checkpoint_data, f, indent=2)
            logger.info(f"Checkpoint saved: {self.checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def check_memory(self) -> Tuple[float, float]:
        """Check current memory usage.

        Returns:
            (used_gb, used_pct)
        """
        vm = psutil.virtual_memory()
        used_gb = vm.used / (1024**3)
        used_pct = vm.percent
        return used_gb, used_pct

    async def generate_chunk(self, chunk_idx: int, start: datetime, end: datetime) -> Optional[Path]:
        """Generate dataset for a single chunk.

        Args:
            chunk_idx: Chunk index
            start: Start date
            end: End date

        Returns:
            Path to generated chunk file, or None if failed
        """
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        logger.info("=" * 70)
        logger.info(f"Processing chunk {chunk_idx + 1}/{len(self.chunks)}")
        logger.info(f"Date range: {start_str} to {end_str}")
        logger.info("=" * 70)

        # Check memory before starting
        used_gb, used_pct = self.check_memory()
        logger.info(f"Memory before: {used_gb:.1f} GB ({used_pct:.1f}%)")

        if used_gb > self.memory_limit_gb:
            logger.error(f"Memory usage ({used_gb:.1f} GB) exceeds limit ({self.memory_limit_gb:.1f} GB)")
            logger.error("Aborting to prevent OOM. Try reducing chunk size with --chunk-years")
            return None

        # Import run_full_dataset
        try:
            from scripts.pipelines.run_full_dataset import main as run_full_dataset
        except ImportError as e:
            logger.error(f"Failed to import run_full_dataset: {e}")
            return None

        # Set environment variables for this chunk
        chunk_env = os.environ.copy()
        chunk_env.update({
            # Reduce parallel processing
            "MAX_CONCURRENT_FETCH": str(int(os.getenv("MAX_CONCURRENT_FETCH", "40"))),
            "MAX_PARALLEL_WORKERS": str(int(os.getenv("MAX_PARALLEL_WORKERS", "8"))),
            # GPU memory limit
            "RMM_POOL_SIZE": os.getenv("RMM_POOL_SIZE", "30GB"),
            # Enable memory monitoring
            "ENABLE_MEMORY_MONITOR": "1",
        })

        # Temporarily override sys.argv for run_full_dataset
        original_argv = sys.argv.copy()
        sys.argv = [
            "run_full_dataset.py",
            "--jquants",
            "--start-date", start_str,
            "--end-date", end_str,
        ]

        try:
            # Run dataset generation for this chunk
            logger.info("Starting dataset generation...")
            result = await run_full_dataset()

            if result != 0:
                logger.error(f"Chunk generation failed with code {result}")
                return None

            # Find generated chunk file
            chunk_pattern = f"ml_dataset_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_full.parquet"
            chunk_path = Path("output/datasets") / chunk_pattern

            if not chunk_path.exists():
                # Try alternative location
                chunk_path = Path("output") / chunk_pattern

            if not chunk_path.exists():
                logger.error(f"Chunk file not found: {chunk_path}")
                return None

            logger.info(f"Chunk generated: {chunk_path}")

            # Check memory after generation
            used_gb, used_pct = self.check_memory()
            logger.info(f"Memory after: {used_gb:.1f} GB ({used_pct:.1f}%)")

            return chunk_path

        except Exception as e:
            logger.error(f"Chunk generation failed: {e}", exc_info=True)
            return None

        finally:
            # Restore sys.argv
            sys.argv = original_argv

    async def merge_chunks(self, chunk_paths: List[Path]) -> Optional[Path]:
        """Merge chunk files into final dataset.

        Args:
            chunk_paths: List of chunk file paths

        Returns:
            Path to final merged dataset, or None if failed
        """
        logger.info("=" * 70)
        logger.info("Merging chunks into final dataset")
        logger.info("=" * 70)

        try:
            import polars as pl

            # Read and concatenate all chunks
            logger.info(f"Reading {len(chunk_paths)} chunk files...")
            dfs = []
            for i, chunk_path in enumerate(chunk_paths, 1):
                logger.info(f"  [{i}/{len(chunk_paths)}] Reading {chunk_path.name}...")
                df = pl.read_parquet(chunk_path)
                dfs.append(df)
                logger.info(f"    Shape: {df.shape}, Memory: {df.estimated_size() / 1024**3:.2f} GB")

            logger.info("Concatenating chunks...")
            final_df = pl.concat(dfs)
            logger.info(f"Final dataset shape: {final_df.shape}")

            # Sort by Date and Code
            logger.info("Sorting by Date and Code...")
            final_df = final_df.sort(["Date", "Code"])

            # Save final dataset
            output_path = Path("output") / f"ml_dataset_{self.start_date.strftime('%Y%m%d')}_{self.end_date.strftime('%Y%m%d')}_full.parquet"
            logger.info(f"Saving final dataset to {output_path}...")
            final_df.write_parquet(output_path)

            # Create symlink
            symlink_path = Path("output") / "ml_dataset_latest_full.parquet"
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()
            symlink_path.symlink_to(output_path.name)
            logger.info(f"Created symlink: {symlink_path} -> {output_path.name}")

            logger.info(f"Final dataset saved: {output_path}")
            logger.info(f"Final size: {output_path.stat().st_size / 1024**3:.2f} GB")

            return output_path

        except Exception as e:
            logger.error(f"Merge failed: {e}", exc_info=True)
            return None

    async def run(self) -> int:
        """Main execution flow.

        Returns:
            Exit code (0 = success, 1 = failure)
        """
        logger.info("Memory-safe dataset generator started")
        logger.info(f"Date range: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"Chunk size: {self.chunk_years} year(s)")
        logger.info(f"Log file: {LOG_FILE}")

        # Create chunks
        self.chunks = self.create_chunks()

        # Load checkpoint if resuming
        if self.resume:
            if self.load_checkpoint():
                logger.info(f"Resuming from checkpoint: {len(self.completed_chunks)} chunks already completed")
            else:
                logger.warning("Resume requested but no checkpoint found, starting from beginning")
        elif self.checkpoint_file.exists() and not self.force:
            logger.warning(f"Checkpoint file exists: {self.checkpoint_file}")
            logger.warning("Use --resume to continue or --force to overwrite")
            return 1

        # Process chunks
        chunk_paths: List[Path] = []

        for idx, (start, end) in enumerate(self.chunks):
            # Skip completed chunks
            if idx in self.completed_chunks:
                logger.info(f"Skipping completed chunk {idx + 1}/{len(self.chunks)}")
                # Try to find existing chunk file
                chunk_info = next((c for c in self.checkpoint_data.get("chunks", []) if c["index"] == idx), None)
                if chunk_info and chunk_info.get("path"):
                    existing_path = Path(chunk_info["path"])
                    if existing_path.exists():
                        chunk_paths.append(existing_path)
                        continue

            # Generate chunk
            chunk_path = await self.generate_chunk(idx, start, end)

            if chunk_path is None:
                logger.error(f"Failed to generate chunk {idx + 1}/{len(self.chunks)}")
                logger.info(f"Progress saved in checkpoint: {self.checkpoint_file}")
                logger.info("You can resume with: python scripts/data/dataset_generator_safe.py --resume")
                return 1

            # Save checkpoint
            self.completed_chunks.append(idx)
            self.save_checkpoint(idx, chunk_path)
            chunk_paths.append(chunk_path)

        # Merge chunks
        if len(chunk_paths) > 1:
            final_path = await self.merge_chunks(chunk_paths)
            if final_path is None:
                logger.error("Failed to merge chunks")
                return 1
        elif len(chunk_paths) == 1:
            # Single chunk, just copy to final location
            final_path = chunk_paths[0]
            logger.info(f"Single chunk generated: {final_path}")
        else:
            logger.error("No chunks were generated")
            return 1

        # Clean up checkpoint
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info("Checkpoint cleaned up")

        logger.info("=" * 70)
        logger.info("Dataset generation completed successfully!")
        logger.info(f"Final dataset: {final_path}")
        logger.info("=" * 70)

        return 0


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Generate ML dataset with memory safety and checkpointing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--chunk-years",
        type=int,
        default=int(os.getenv("DATASET_CHUNK_YEARS", "1")),
        help="Years per chunk (default: 1)",
    )
    parser.add_argument(
        "--memory-limit",
        type=float,
        default=float(os.getenv("DATASET_MEMORY_LIMIT", "70")),
        help="Memory limit as percentage (default: 70%%)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing checkpoint",
    )

    args = parser.parse_args()

    # Create generator
    generator = MemorySafeDatasetGenerator(
        start_date=args.start_date,
        end_date=args.end_date,
        chunk_years=args.chunk_years,
        memory_limit_pct=args.memory_limit,
        resume=args.resume,
        force=args.force,
    )

    # Run
    try:
        exit_code = asyncio.run(generator.run())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        logger.info(f"Progress saved. Resume with: python {__file__} --resume")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
