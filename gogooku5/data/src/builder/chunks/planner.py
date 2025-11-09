"""Quarterly chunk planning utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from calendar import monthrange
from pathlib import Path
from typing import Iterable, List

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
    """Plan dataset builds over quarterly chunks with warmup handling."""

    def __init__(
        self,
        *,
        settings: DatasetBuilderSettings | None = None,
        warmup_days: int = DEFAULT_WARMUP_DAYS,
        output_root: Path | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.warmup_days = warmup_days
        self.output_root = output_root or (self.settings.data_output_dir / "chunks")
        self._logger = get_logger("builder.chunks")

    def plan(self, *, start: str, end: str) -> List[ChunkSpec]:
        """Return chunk specs covering start..end inclusive."""

        start_date = _parse_date(start)
        end_date = _parse_date(end)
        if end_date < start_date:
            raise ValueError("end date must be on or after start date")

        specs: List[ChunkSpec] = []
        for quarter_start in _iterate_quarter_starts(start_date, end_date):
            quarter_end = _quarter_end(quarter_start)
            output_start = max(start_date, quarter_start)
            output_end = min(end_date, quarter_end)
            if output_start > output_end:
                continue

            chunk_id = _chunk_id_for_date(quarter_start)
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
                "Failed to shift trading days for %s (warmup=%d): %s. "
                "Falling back to chunk output start.",
                iso_output,
                self.warmup_days,
                exc,
            )
            return iso_output


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _chunk_id_for_date(dt: date) -> str:
    quarter = ((dt.month - 1) // 3) + 1
    return f"{dt.year}Q{quarter}"


def _iterate_quarter_starts(start: date, end: date) -> Iterable[date]:
    """Yield the first day of each quarter overlapping the range."""

    current = _quarter_start(start)
    while current <= end:
        yield current
        # Advance to next quarter
        month = current.month + 3
        year = current.year
        if month > 12:
            month -= 12
            year += 1
        current = date(year, month, 1)


def _quarter_start(dt: date) -> date:
    month = ((dt.month - 1) // 3) * 3 + 1
    return date(dt.year, month, 1)


def _quarter_end(dt: date) -> date:
    """Return the final day of the quarter for the quarter containing dt."""

    start = _quarter_start(dt)
    month = start.month + 2
    year = start.year
    if month > 12:
        month -= 12
        year += 1
    last_day = monthrange(year, month)[1]
    return date(year, month, last_day)

