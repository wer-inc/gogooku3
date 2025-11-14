"""Raw data source catalog for chunking + manifest generation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence


DateKind = Literal["date", "datetime", "string"]


@dataclass(frozen=True, slots=True)
class RawSourceSpec:
    """Declarative description of a cached raw data source."""

    name: str
    input_glob: str
    date_column: str
    output_dir: Path
    date_kind: DateKind = "date"
    date_format: str | None = None
    prefer_latest: bool = True
    sort_by: Sequence[str] = field(default_factory=tuple)
    chunk_name_override: str | None = None
    description: str | None = None

    @property
    def chunk_dir(self) -> Path:
        return self.output_dir


RAW_SOURCE_SPECS: dict[str, RawSourceSpec] = {
    "prices": RawSourceSpec(
        name="prices",
        input_glob="output/raw/prices/daily_quotes_*.parquet",
        date_column="Date",
        output_dir=Path("output/raw/prices/chunks"),
        description="Daily price quotes (all sessions).",
    ),
    "prices_am": RawSourceSpec(
        name="prices_am",
        input_glob="output/raw/prices_am/prices_am_*.parquet",
        date_column="Date",
        output_dir=Path("output/raw/prices_am/chunks"),
        description="Morning session quotes.",
    ),
    "margin_daily": RawSourceSpec(
        name="margin_daily",
        input_glob="output/raw/margin/daily_margin_interest_*.parquet",
        date_column="ApplicationDate",
        output_dir=Path("output/raw/margin/chunks"),
        description="Daily margin interest (TSE).",
        sort_by=("ApplicationDate", "Code"),
    ),
    "margin_weekly": RawSourceSpec(
        name="margin_weekly",
        input_glob="output/raw/margin/weekly_margin_interest_*.parquet",
        date_column="Date",
        output_dir=Path("output/raw/margin/chunks"),
        description="Weekly margin interest (TSE).",
        sort_by=("Date", "Code"),
    ),
    "short_selling": RawSourceSpec(
        name="short_selling",
        input_glob="output/raw/short_selling/short_selling_*.parquet",
        date_column="Date",
        output_dir=Path("output/raw/short_selling/chunks"),
        description="Daily short selling by issue.",
        sort_by=("Date", "Code"),
    ),
    "short_selling_sector": RawSourceSpec(
        name="short_selling_sector",
        input_glob="output/raw/short_selling/sector_short_selling_*.parquet",
        date_column="Date",
        output_dir=Path("output/raw/short_selling/chunks"),
        description="Daily short selling aggregated by sector33.",
        sort_by=("Date", "Sector33Code"),
    ),
    "flow_trades_spec": RawSourceSpec(
        name="flow_trades_spec",
        input_glob="output/raw/flow/trades_spec_history_*.parquet",
        date_column="EndDate",
        output_dir=Path("output/raw/flow/chunks"),
        description="Weekly trading flow by participant category.",
        sort_by=("EndDate", "Section"),
    ),
    "indices_topix": RawSourceSpec(
        name="indices_topix",
        input_glob="output/raw/indices/topix_history_*.parquet",
        date_column="Date",
        output_dir=Path("output/raw/indices/chunks"),
        description="TOPIX historical quotes.",
        sort_by=("Date",),
    ),
    "statements": RawSourceSpec(
        name="statements",
        input_glob="output/raw/statements/event_raw_statements_*.parquet",
        date_column="DisclosedDate",
        date_kind="string",
        date_format="%Y-%m-%d",
        output_dir=Path("output/raw/statements/chunks"),
        description="Timely disclosure statements.",
        sort_by=("DisclosedDate", "DisclosedTime", "LocalCode"),
    ),
    "jquants_listed": RawSourceSpec(
        name="jquants_listed",
        input_glob="output/raw/jquants/listed_info_history_*.parquet",
        date_column="Date",
        output_dir=Path("output/raw/jquants/chunks"),
        description="J-Quants listed company master snapshots.",
        sort_by=("Date", "Code"),
    ),
}
