import polars as pl
from pathlib import Path
from src.pipeline.incremental_updater import IncrementalDatasetUpdater


def test_incremental_merge_basic(tmp_path: Path):
    old = pl.DataFrame({
        "Date": ["2025-01-01", "2025-01-02"],
        "Code": ["A", "A"],
        "x": [1, 2],
    })
    inc = pl.DataFrame({
        "Date": ["2025-01-02", "2025-01-03"],
        "Code": ["A", "A"],
        "x": [20, 3],
    })
    p_old = tmp_path / "old.parquet"; old.write_parquet(p_old)
    p_inc = tmp_path / "inc.parquet"; inc.write_parquet(p_inc)
    upd = IncrementalDatasetUpdater(tmp_path)
    merged = upd.merge_enriched(p_old, p_inc)
    assert merged.height == 3
    # latest for 2025-01-02 should be from incremental (20)
    assert merged.filter((pl.col("Date") == "2025-01-02") & (pl.col("Code") == "A"))["x"].item() == 20

