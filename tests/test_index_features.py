import os
import sys
from pathlib import Path

import polars as pl

# Ensure repository root on sys.path for `src.*` imports when running in CI sandboxes
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gogooku3.features.index_features import (
    build_all_index_features,
    attach_index_features_to_equity,
    attach_sector_index_features,
)


def _toy_indices() -> pl.DataFrame:
    rows = []
    # 6 business days
    dates = [f"2024-01-0{i}" for i in range(1, 7)]
    # Codes: TOPIX(0000), Value(8100), Growth(8200), Core30(0028), Small(002D), Prime(0500), Standard(0501)
    codes = ["0000", "8100", "8200", "0028", "002D", "0500", "0501", "0040"]
    base = {
        "0000": 2000.0,
        "8100": 1500.0,
        "8200": 1500.0,
        "0028": 3000.0,
        "002D": 1200.0,
        "0500": 1800.0,
        "0501": 900.0,
        "0040": 800.0,
    }
    for code in codes:
        px = base[code]
        for d in dates:
            # Small drift by code to make spreads non-zero
            bump = 1.001 if code in ("8100", "0028", "0500") else 0.999
            px = px * bump
            rows.append({
                "Code": code,
                "Date": d,
                "Open": px * 0.99,
                "High": px * 1.01,
                "Low": px * 0.98,
                "Close": px,
            })
    return pl.DataFrame(rows).with_columns(pl.col("Date").str.strptime(pl.Date, strict=False))


def _toy_quotes() -> pl.DataFrame:
    # Minimal equity panel with Code, Date
    rows = []
    for code in ["1001", "1002", "1003"]:
        for d in [f"2024-01-0{i}" for i in range(1, 7)]:
            rows.append({"Code": code, "Date": d})
    return pl.DataFrame(rows).with_columns(pl.col("Date").str.strptime(pl.Date, strict=False))


def test_build_all_index_features_basic():
    idx = _toy_indices()
    per_idx, daily = build_all_index_features(idx)
    assert not per_idx.is_empty()
    assert not daily.is_empty()
    # Check essential per-index columns
    for c in ["idx_r_1d", "idx_r_5d", "idx_vol_20d", "idx_z_close_20"]:
        assert c in per_idx.columns
    # Check spreads present
    assert any(c.startswith("spread_") for c in daily.columns), "expected at least one spread series"
    # Check breadth present
    assert "breadth_sector_gt_ma50" in daily.columns


def test_attach_index_features_to_equity():
    idx = _toy_indices()
    per_idx, daily = build_all_index_features(idx)
    quotes = _toy_quotes()
    out = attach_index_features_to_equity(quotes, per_idx, daily)
    assert out.height == quotes.height
    # spread columns are added
    spread_cols = [c for c in out.columns if c.startswith("spread_")]
    assert spread_cols, "no spread columns attached"
    # Confirm values are same across all codes per date (join by Date)
    sample_day = out.filter(pl.col("Date") == pl.date(2024, 1, 3))
    for c in spread_cols:
        vals = sample_day[c].drop_nulls().unique().to_list()
        assert len(vals) <= 1


def test_attach_sector_index_features_basic():
    idx = _toy_indices()
    quotes = _toy_quotes()
    # Build a minimal listed_info mapping with Sector33Code hints
    listed_rows = []
    for code in quotes.select("Code").unique()["Code"].to_list():
        listed_rows.append({"Code": code, "Sector33Code": "6050"})  # 小売業 → 005A
    listed = pl.DataFrame(listed_rows)
    out = attach_sector_index_features(quotes, idx, listed, prefix="sect_")
    # Should add some sect_* columns
    sect_cols = [c for c in out.columns if c.startswith("sect_")]
    assert sect_cols, "no sector columns joined"
