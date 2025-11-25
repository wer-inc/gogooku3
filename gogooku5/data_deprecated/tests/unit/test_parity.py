from __future__ import annotations

from pathlib import Path

import polars as pl
from builder.validation.parity import compare_datasets


def test_compare_datasets_reports_diffs(tmp_path: Path) -> None:
    ref_path = tmp_path / "ref.parquet"
    cand_path = tmp_path / "cand.parquet"

    ref_df = pl.DataFrame(
        {
            "Code": ["1301", "1305"],
            "Date": ["2024-01-01", "2024-01-01"],
            "Close": [100.0, 200.0],
            "Volume": [10_000, 5_000],
        }
    )
    cand_df = pl.DataFrame(
        {
            "Code": ["1301", "1305", "1306"],
            "Date": ["2024-01-01", "2024-01-01", "2024-01-01"],
            "Close": [101.0, 199.5, 300.0],
            "Volume": [9_500, 5_500, 4_000],
            "macro_vix_close": [20.0, 20.0, 20.0],
        }
    )

    ref_df.write_parquet(ref_path)
    cand_df.write_parquet(cand_path)

    result = compare_datasets(ref_path, cand_path, key_columns=("Code", "Date"))

    assert result.rows_reference == 2
    assert result.rows_candidate == 3
    assert result.rows_only_candidate == 1
    assert any(diff.column == "macro_vix_close" and diff.candidate_only for diff in result.column_diffs)
    assert any(diff.column == "Close" and diff.max_abs_diff is not None for diff in result.column_diffs)
