"""Dataset parity comparison utilities."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import polars as pl

from ..features.utils.lazy_io import lazy_load


def _is_numeric(dtype: pl.DataType) -> bool:
    if dtype.is_numeric():
        return True
    return "decimal" in repr(dtype).lower()


@dataclass
class ColumnDiff:
    column: str
    dtype_reference: str | None
    dtype_candidate: str | None
    max_abs_diff: float | None = None
    mean_abs_diff: float | None = None
    reference_only: bool = False
    candidate_only: bool = False


@dataclass
class ParityResult:
    reference_path: str
    candidate_path: str
    key_columns: Sequence[str]
    rows_reference: int
    rows_candidate: int
    rows_only_reference: int
    rows_only_candidate: int
    column_diffs: List[ColumnDiff]
    schema_mismatch: bool

    def to_json(self) -> str:
        return json.dumps(
            {
                "reference_path": self.reference_path,
                "candidate_path": self.candidate_path,
                "key_columns": list(self.key_columns),
                "rows_reference": self.rows_reference,
                "rows_candidate": self.rows_candidate,
                "rows_only_reference": self.rows_only_reference,
                "rows_only_candidate": self.rows_only_candidate,
                "schema_mismatch": self.schema_mismatch,
                "column_diffs": [asdict(diff) for diff in self.column_diffs],
            },
            indent=2,
        )


def compare_datasets(
    reference: Path,
    candidate: Path,
    *,
    key_columns: Sequence[str] = ("Code", "Date"),
    sample_rows: int | None = None,
) -> ParityResult:
    """Compare two dataset parquet files and return parity report."""

    ref_df = lazy_load(reference, prefer_ipc=True)
    cand_df = lazy_load(candidate, prefer_ipc=True)

    if sample_rows is not None:
        ref_df = ref_df.head(sample_rows)
        cand_df = cand_df.head(sample_rows)

    ref_cols = set(ref_df.columns)
    cand_cols = set(cand_df.columns)

    column_diffs: list[ColumnDiff] = []
    schema_mismatch = False

    # Columns only in one side
    for col in sorted(ref_cols - cand_cols):
        column_diffs.append(
            ColumnDiff(
                column=col,
                dtype_reference=str(ref_df.schema[col]),
                dtype_candidate=None,
                reference_only=True,
            )
        )
        schema_mismatch = True
    for col in sorted(cand_cols - ref_cols):
        column_diffs.append(
            ColumnDiff(
                column=col,
                dtype_reference=None,
                dtype_candidate=str(cand_df.schema[col]),
                candidate_only=True,
            )
        )
        schema_mismatch = True

    shared_columns = [c for c in ref_cols & cand_cols]

    # Type mismatches and numeric diffs
    for col in sorted(shared_columns):
        ref_type = ref_df.schema[col]
        cand_type = cand_df.schema[col]
        if ref_type != cand_type:
            schema_mismatch = True
            column_diffs.append(
                ColumnDiff(
                    column=col,
                    dtype_reference=str(ref_type),
                    dtype_candidate=str(cand_type),
                )
            )
            continue

        if not _is_numeric(ref_type):
            continue

        # Align by keys to compute diffs
        numeric_diff = _compute_numeric_diff(ref_df, cand_df, col, key_columns)
        if numeric_diff is not None:
            column_diffs.append(
                ColumnDiff(
                    column=col,
                    dtype_reference=str(ref_type),
                    dtype_candidate=str(cand_type),
                    max_abs_diff=numeric_diff["max_abs_diff"],
                    mean_abs_diff=numeric_diff["mean_abs_diff"],
                )
            )

    rows_only_reference = _count_missing_rows(ref_df, cand_df, key_columns)
    rows_only_candidate = _count_missing_rows(cand_df, ref_df, key_columns)

    return ParityResult(
        reference_path=str(reference),
        candidate_path=str(candidate),
        key_columns=list(key_columns),
        rows_reference=ref_df.height,
        rows_candidate=cand_df.height,
        rows_only_reference=rows_only_reference,
        rows_only_candidate=rows_only_candidate,
        column_diffs=column_diffs,
        schema_mismatch=schema_mismatch,
    )


def _compute_numeric_diff(
    ref_df: pl.DataFrame,
    cand_df: pl.DataFrame,
    column: str,
    key_columns: Sequence[str],
) -> Dict[str, float] | None:
    """Compute numeric difference metrics for a shared column."""

    keys = [k for k in key_columns if k in ref_df.columns and k in cand_df.columns]
    if not keys:
        return None

    ref_subset = ref_df.select(keys + [column])
    cand_subset = cand_df.select(keys + [column])
    joined = ref_subset.join(
        cand_subset,
        on=keys,
        how="inner",
        suffix="_candidate",
    )
    if joined.is_empty():
        return None

    ref_col = column
    cand_col = f"{column}_candidate"
    diff_expr = (pl.col(ref_col) - pl.col(cand_col)).abs()
    stats = joined.select(
        diff_expr.max().alias("max_abs_diff"),
        diff_expr.mean().alias("mean_abs_diff"),
    )
    result = stats.row(0)
    return {
        "max_abs_diff": float(result[0]) if result[0] is not None else 0.0,
        "mean_abs_diff": float(result[1]) if result[1] is not None else 0.0,
    }


def _count_missing_rows(left: pl.DataFrame, right: pl.DataFrame, key_columns: Sequence[str]) -> int:
    keys = [k for k in key_columns if k in left.columns and k in right.columns]
    if not keys:
        return 0
    missing = left.select(keys).join(right.select(keys), on=keys, how="anti").height
    return missing
