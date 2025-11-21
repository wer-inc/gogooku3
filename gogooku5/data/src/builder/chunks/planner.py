"""Chunk planning utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List

from ..config import DatasetBuilderSettings, get_settings
from ..utils.datetime import shift_trading_days
from ..utils.logger import get_logger

# Default warmup period matches legacy single-run behaviour.
DEFAULT_WARMUP_DAYS: int = 85


@dataclass(frozen=True)
class ChunkSpec:
    """Describe a dataset chunk build."""

    chunk_id: str
    input_start: str
    input_end: str
    output_start: str
    output_end: str
    output_dir: Path

    @property
    def dataset_path(self) -> Path:
        return self.output_dir / "ml_dataset.parquet"

    @property
    def metadata_path(self) -> Path:
        return self.output_dir / "metadata.json"

    @property
    def status_path(self) -> Path:
        return self.output_dir / "status.json"


class ChunkPlanner:
    """Plan dataset builds over fixed-month chunks with warmup handling."""

    def __init__(
        self,
        *,
        settings: DatasetBuilderSettings | None = None,
        warmup_days: int = DEFAULT_WARMUP_DAYS,
        output_root: Path | None = None,
        months_per_chunk: int = 3,
    ) -> None:
        if months_per_chunk <= 0:
            raise ValueError("months_per_chunk must be >= 1")
        self.settings = settings or get_settings()
        self.warmup_days = warmup_days
        self.output_root = output_root or (self.settings.data_output_dir / "chunks")
        self.months_per_chunk = months_per_chunk
        self._logger = get_logger("builder.chunks")

    def plan(self, *, start: str, end: str) -> List[ChunkSpec]:
        """Return chunk specs covering start..end inclusive."""

        start_date = _parse_date(start)
        end_date = _parse_date(end)
        if end_date < start_date:
            raise ValueError("end date must be on or after start date")

        specs: List[ChunkSpec] = []
        chunk_start = _align_chunk_start(start_date, self.months_per_chunk)
        while chunk_start <= end_date:
            chunk_end = _chunk_end(chunk_start, self.months_per_chunk)
            output_start = max(start_date, chunk_start)
            output_end = min(end_date, chunk_end)
            if output_start <= output_end:
                chunk_id = _chunk_id_for_span(chunk_start, chunk_end, self.months_per_chunk)
                output_dir = self.output_root / chunk_id
                input_start = self._compute_input_start(output_start)
                spec = ChunkSpec(
                    chunk_id=chunk_id,
                    input_start=input_start,
                    input_end=output_end.strftime("%Y-%m-%d"),
                    output_start=output_start.strftime("%Y-%m-%d"),
                    output_end=output_end.strftime("%Y-%m-%d"),
                    output_dir=output_dir,
                )
                specs.append(spec)
            chunk_start = _add_months(chunk_start, self.months_per_chunk)

        return specs

    def _compute_input_start(self, output_start: date) -> str:
        """Shift output_start backwards by warmup trading days."""

        iso_output = output_start.strftime("%Y-%m-%d")
        if self.warmup_days <= 0:
            return iso_output

        try:
            return shift_trading_days(iso_output, -self.warmup_days)
        except RuntimeError as exc:
            # Align with legacy DatasetBuilder fallback so environments without
            # holiday libraries can still produce a dataset (warmup equals output).
            self._logger.warning(
                "Failed to shift trading days for %s (warmup=%d): %s. Falling back to chunk output start.",
                iso_output,
                self.warmup_days,
                exc,
            )
            return iso_output


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _align_chunk_start(dt: date, months_per_chunk: int) -> date:
    """Return the first day of the chunk containing ``dt``."""

    month = ((dt.month - 1) // months_per_chunk) * months_per_chunk + 1
    return date(dt.year, month, 1)


def _add_months(dt: date, months: int) -> date:
    """Return the first day after advancing ``months`` months."""

    total = (dt.year * 12) + (dt.month - 1) + months
    year = total // 12
    month = total % 12 + 1
    return date(year, month, 1)


def _chunk_end(start: date, months_per_chunk: int) -> date:
    """Return the final day included in a chunk starting at ``start``."""

    next_start = _add_months(start, months_per_chunk)
    return next_start - timedelta(days=1)


def _chunk_id_for_span(start: date, end: date, months_per_chunk: int) -> str:
    """Return a readable chunk identifier for the span."""

    if months_per_chunk == 3:
        quarter = ((start.month - 1) // 3) + 1
        return f"{start.year}Q{quarter}"
    if months_per_chunk == 1:
        return f"{start.year}M{start.month:02d}"
    if start.year == end.year:
        return f"{start.year}M{start.month:02d}-{end.month:02d}"
    return f"{start.year}M{start.month:02d}-{end.year}M{end.month:02d}"
