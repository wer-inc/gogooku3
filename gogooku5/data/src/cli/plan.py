"""
Execution plan data structures for gogooku5-dataset CLI.

Defines ExecutionPlan and related classes for representing
dataset build plans with automatic chunking.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional


@dataclass
class ExecutionPlan:
    """
    High-level execution plan for dataset build.

    Attributes:
        mode: Execution mode (full, chunks, latest)
        start_date: Start date
        end_date: End date
        total_days: Total days in range
        output_dir: Output directory
        tag: Dataset tag
        gpu_enabled: Whether GPU-ETL is enabled
        features: Feature preset name
    """

    mode: str  # "full", "chunks", "latest"
    start_date: date
    end_date: date
    total_days: int
    output_dir: Path
    tag: str
    gpu_enabled: bool
    features: str

    # Chunking-specific (None for full mode)
    chunk_months: Optional[int] = None
    num_chunks: Optional[int] = None
    resume: bool = False
    force: bool = False

    def __str__(self) -> str:
        """Human-readable plan summary."""
        lines = [
            "=" * 70,
            "📋 Execution Plan",
            "=" * 70,
            f"Mode:        {self.mode.upper()}",
            f"Period:      {self.start_date} → {self.end_date} ({self.total_days} days)",
            f"Output:      {self.output_dir}",
            f"Tag:         {self.tag}",
            f"GPU:         {'Enabled' if self.gpu_enabled else 'Disabled'}",
            f"Features:    {self.features}",
        ]

        if self.mode == "chunks":
            lines.extend(
                [
                    "",
                    "Chunking:",
                    f"  Chunk Size:  {self.chunk_months} months",
                    f"  Num Chunks:  {self.num_chunks}",
                    f"  Resume:      {'Yes' if self.resume else 'No'}",
                    f"  Force:       {'Yes' if self.force else 'No'}",
                ]
            )

        lines.append("=" * 70)
        return "\n".join(lines)


@dataclass
class ChunkSpec:
    """
    Specification for a single chunk.

    Attributes:
        chunk_id: Unique identifier (e.g., "20200101-20200331")
        start_date: Chunk start date (output range)
        end_date: Chunk end date (output range)
        input_start_date: Input start date (includes warmup)
        input_end_date: Input end date
        output_dir: Chunk output directory
        status_file: Path to status.json
    """

    chunk_id: str
    start_date: date
    end_date: date
    input_start_date: date
    input_end_date: date
    output_dir: Path
    status_file: Path

    @property
    def output_days(self) -> int:
        """Number of output days."""
        return (self.end_date - self.start_date).days

    @property
    def warmup_days(self) -> int:
        """Number of warmup days."""
        return (self.start_date - self.input_start_date).days

    def __str__(self) -> str:
        """Human-readable chunk summary."""
        return (
            f"{self.chunk_id}: "
            f"{self.start_date} → {self.end_date} "
            f"({self.output_days} days, warmup: {self.warmup_days} days)"
        )


def determine_mode(
    start_date: date,
    end_date: date,
    chunk_mode: str,
    latest: bool,
) -> str:
    """
    Determine execution mode based on arguments and date range.

    Args:
        start_date: Start date
        end_date: End date
        chunk_mode: User-specified mode (auto, full, chunks, latest)
        latest: Whether --latest flag is set

    Returns:
        Execution mode: "full", "chunks", or "latest"
    """
    # Explicit mode takes precedence
    if latest or chunk_mode == "latest":
        return "latest"

    if chunk_mode in ("full", "chunks"):
        return chunk_mode

    # Auto-detect based on date range
    assert chunk_mode == "auto", f"Invalid chunk_mode: {chunk_mode}"

    total_days = (end_date - start_date).days

    if total_days <= 180:  # 6 months or less
        return "full"
    else:
        return "chunks"


def calculate_chunk_specs(
    start_date: date,
    end_date: date,
    chunk_months: int,
    warmup_days: int,
    output_dir: Path,
) -> List[ChunkSpec]:
    """
    Calculate chunk specifications for chunked execution.

    Args:
        start_date: Overall start date
        end_date: Overall end date
        chunk_months: Chunk size in months
        warmup_days: Number of warmup days for each chunk
        output_dir: Base output directory

    Returns:
        List of ChunkSpec objects
    """
    from dateutil.relativedelta import relativedelta  # type: ignore[import-untyped]

    chunks = []
    current_start = start_date

    while current_start < end_date:
        # Calculate chunk end (exclusive)
        chunk_end = min(current_start + relativedelta(months=chunk_months), end_date)

        # Calculate input range (includes warmup)
        input_start = current_start - relativedelta(days=warmup_days)

        # Chunk ID format: YYYYMMDD-YYYYMMDD
        chunk_id = f"{current_start.strftime('%Y%m%d')}-{chunk_end.strftime('%Y%m%d')}"

        # Chunk output directory
        chunk_dir = output_dir / "chunks" / chunk_id
        status_file = chunk_dir / "status.json"

        chunk = ChunkSpec(
            chunk_id=chunk_id,
            start_date=current_start,
            end_date=chunk_end,
            input_start_date=input_start,
            input_end_date=chunk_end,
            output_dir=chunk_dir,
            status_file=status_file,
        )
        chunks.append(chunk)

        # Move to next chunk
        current_start = chunk_end

    return chunks


def display_plan(plan: ExecutionPlan, chunks: Optional[List[ChunkSpec]] = None) -> None:
    """
    Display execution plan in human-readable format.

    Args:
        plan: ExecutionPlan object
        chunks: Optional list of ChunkSpec objects (for chunked mode)
    """
    print(plan)

    if chunks:
        print("\n📦 Chunk Details:")
        print("=" * 70)
        for i, chunk in enumerate(chunks, 1):
            # Use status_path from builder's ChunkSpec
            status_path = chunk.status_path if hasattr(chunk, "status_path") else chunk.output_dir / "status.json"
            status = "⏭️  Will skip" if status_path.exists() else "✅ Will execute"

            # Format chunk info
            chunk_info = f"{chunk.chunk_id}: {chunk.output_start} → {chunk.output_end}"
            print(f"  {i:2d}. {chunk_info} - {status}")
        print("=" * 70)

    print("\n💡 Tip: Use --resume to skip completed chunks")
    print("💡 Tip: Use --force to rebuild all chunks")
