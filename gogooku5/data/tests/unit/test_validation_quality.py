from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from builder.validation.quality import run_quality_checks


def test_run_quality_checks_includes_feature_missingness(tmp_path: Path) -> None:
    """Light feature quality check should report per-feature missing rates."""

    path = tmp_path / "dataset.parquet"
    df = pl.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            "code": ["A", "B", "C"],
            "feature_ok": [1.0, 2.0, 3.0],
            "feature_missing": [None, 1.0, None],
            "target_1d": [0.1, 0.2, 0.3],
        }
    )
    df.write_parquet(path)

    summary, errors, warnings = run_quality_checks(
        path,
        date_col="date",
        code_col="code",
        targets=["target_1d"],
        asof_pairs=[],
        allow_future_days=0,
        sample_rows=5,
    )

    # Base checks should pass without errors for this small dataset.
    assert "primary_key" in summary
    assert "future_dates" in summary
    assert "targets" in summary
    assert "asof" in summary

    # New feature-level check is present.
    assert "features" in summary
    feature_res = summary["features"]
    assert feature_res["status"] in {"ok", "warning"}

    details = feature_res.get("details", {})
    assert details.get("feature_count") == 2  # feature_ok, feature_missing
    assert details.get("total_rows") == 3

    top_missing = details.get("top_missing_features") or []
    # feature_missing should appear as an offender with non-zero missing_rate.
    offender_names = {entry["feature"] for entry in top_missing}
    assert "feature_missing" in offender_names

