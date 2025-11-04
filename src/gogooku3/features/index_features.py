from __future__ import annotations

"""
Index OHLC feature engineering and cross-index aggregates.

This module computes:
 - Per-index features: log returns, intraday/overnight returns, ATR/NATR,
   realized vol, SMA/STD bands, Z-scores, and basic trend metrics.
 - Cross-index features: relative-to-benchmark (per family) and daily spreads
   such as Value vs Growth, Large vs Small, Prime vs Standard/Growth.
 - Breadth: share of sector indices above 50-day SMA.

All computations are done per Code via Polars over("Code") to avoid leakage
between different indices. Inputs are expected to have Date, Code, Open, High,
Low, Close with Date parsed as pl.Date.
"""

import json
from pathlib import Path

import polars as pl


def _ensure_types(df: pl.DataFrame) -> pl.DataFrame:
    df2 = df
    if "Date" in df2.columns and df2["Date"].dtype == pl.Utf8:
        df2 = df2.with_columns(pl.col("Date").str.strptime(pl.Date, strict=False))
    # Normalize numeric types if present
    for c in ("Open", "High", "Low", "Close"):
        if c in df2.columns:
            df2 = df2.with_columns(pl.col(c).cast(pl.Float64, strict=False))
    if "Code" in df2.columns:
        df2 = df2.with_columns(pl.col("Code").cast(pl.Utf8))
    return df2.sort(["Code", "Date"]) if {"Code", "Date"}.issubset(df2.columns) else df2


def assign_family_and_benchmark(df: pl.DataFrame) -> pl.DataFrame:
    """Assign index family and benchmark code for relative features.

    Families:
      - MARKET: market-wide indices (TOPIX=0000, market segments 0500/0501/0502)
      - SECTOR: 33 TSE industry indices (0040..0060 range inclusive)
      - STYLE_VALUE: value factor indices (e.g., 8100)
      - STYLE_GROWTH: growth factor indices (e.g., 8200)
      - SIZE: size buckets (e.g., Core30=0028, Small=002D)
      - REIT: REIT composite (0075) and segments (8501..8503)
      - OTHER: fallback

    Benchmark mapping:
      - SECTOR/STYLE/SIZE default to TOPIX (0000)
      - REIT segments benchmark to 0075
      - MARKET segments (0501/0502) benchmark to 0500
      - 0000 and 0075 have no benchmark (empty string)
    """
    def _family(code: str) -> str:
        c = code.upper() if code else ""
        if c == "0000" or c in {"0500", "0501", "0502"}:
            return "MARKET"
        if c == "0070":  # Mothers → Growth 250 (renamed 2023-11-06, code unchanged)
            return "MARKET"
        if "0040" <= c <= "0060":
            return "SECTOR"
        if ("0080" <= c <= "008F") or c == "0090":
            return "SECTOR17"
        if c in {"8100"}:
            return "STYLE_VALUE"
        if c in {"8200"}:
            return "STYLE_GROWTH"
        if c in {"0028", "002D", "002A", "002B", "002C"}:  # known size variants
            return "SIZE"
        if c == "0075" or ("8501" <= c <= "8503"):
            return "REIT"
        return "OTHER"

    def _bench(code: str) -> str:
        c = code.upper() if code else ""
        fam = _family(c)
        if c == "0000":
            return ""
        if c == "0075":
            return ""
        if fam == "REIT":
            return "0075"
        if fam == "MARKET":
            return "0500" if c in {"0501", "0502"} else ""
        # Default to TOPIX
        return "0000"

    if df.is_empty():
        return df
    out = df.with_columns([
        pl.col("Code").cast(pl.Utf8).alias("Code"),
    ])
    out = out.with_columns([
        pl.col("Code").map_elements(_family, return_dtype=pl.Utf8).alias("family"),
        pl.col("Code").map_elements(_bench, return_dtype=pl.Utf8).alias("bench_code"),
    ])
    return out


def build_per_index_features(indices: pl.DataFrame, *, mask_halt_day: bool = True) -> pl.DataFrame:
    """Compute per-index time-series features with per-Code windows.

    Output columns prefixed with idx_ to avoid collisions.
    """
    if indices.is_empty():
        return indices
    df = _ensure_types(indices).sort(["Code", "Date"])  # type: ignore
    eps = 1e-12

    # Previous close per code
    df = df.with_columns([
        pl.col("Close").shift(1).over("Code").alias("_prev_close"),
        pl.col("Open").cast(pl.Float64).alias("Open"),
        pl.col("High").cast(pl.Float64).alias("High"),
        pl.col("Low").cast(pl.Float64).alias("Low"),
        pl.col("Close").cast(pl.Float64).alias("Close"),
    ])

    # Returns (log)
    df = df.with_columns([
        (pl.col("Close") / (pl.col("_prev_close") + eps)).log().alias("idx_r_1d"),
        (pl.col("Close") / (pl.col("Close").shift(5).over("Code") + eps)).log().alias("idx_r_5d"),
        (pl.col("Close") / (pl.col("Close").shift(20).over("Code") + eps)).log().alias("idx_r_20d"),
    ])
    # Intraday/overnight decomposition
    if all(c in df.columns for c in ("Open", "Close")):
        df = df.with_columns([
            (pl.col("Close") / (pl.col("Open") + eps)).log().alias("idx_r_oc"),
            (pl.col("Open") / (pl.col("_prev_close") + eps)).log().alias("idx_r_co"),
        ])

    # True range and ATR14/NATR14
    if all(c in df.columns for c in ("High", "Low")):
        df = df.with_columns([
            pl.max_horizontal(
                pl.col("High") - pl.col("Low"),
                (pl.col("High") - pl.col("_prev_close")).abs(),
                (pl.col("Low") - pl.col("_prev_close")).abs(),
            ).alias("_TR"),
        ])
        df = df.with_columns([
            pl.col("_TR").rolling_mean(window_size=14, min_periods=5).over("Code").alias("idx_atr14"),
            (pl.col("_TR") / (pl.col("Close") + eps)).rolling_mean(window_size=14, min_periods=5).over("Code").alias("idx_natr14"),
        ])
        # Mask ATR/NATR on the known TSE halt date (2020-10-01) if enabled
        if mask_halt_day:
            df = df.with_columns([
                pl.when(pl.col("Date") == pl.date(2020, 10, 1)).then(None).otherwise(pl.col("idx_atr14")).alias("idx_atr14"),
                pl.when(pl.col("Date") == pl.date(2020, 10, 1)).then(None).otherwise(pl.col("idx_natr14")).alias("idx_natr14"),
            ])

    # Realized volatility (20d, annualized approx from log returns)
    df = df.with_columns([
        (pl.col("idx_r_1d").rolling_std(20, min_periods=10).over("Code") * (252 ** 0.5)).alias("idx_vol_20d"),
    ])

    # Daily Parkinson, Garman–Klass, Rogers–Satchell estimators and 20d annualized vols
    pl.element().log  # placeholder; we'll use pl.log in expressions directly
    eps = 1e-12
    if all(c in df.columns for c in ("High", "Low", "Open", "Close")):
        # Daily variances
        df = df.with_columns([
            # Parkinson variance
            (1.0 / (4.0 * pl.lit(pl.Series([2.0]).log()[0])) * (pl.col("High") / (pl.col("Low") + eps)).log().pow(2)).alias("_pk_var"),
            # Garman–Klass variance
            (0.5 * (pl.col("High") / (pl.col("Low") + eps)).log().pow(2) - (2.0 * pl.lit(pl.Series([2.0]).log()[0]) - 1.0) * (pl.col("Close") / (pl.col("Open") + eps)).log().pow(2)).alias("_gk_var"),
            # Rogers–Satchell variance (trend-robust)
            ((pl.col("High") / (pl.col("Close") + eps)).log() * (pl.col("High") / (pl.col("Open") + eps)).log() + (pl.col("Low") / (pl.col("Close") + eps)).log() * (pl.col("Low") / (pl.col("Open") + eps)).log()).alias("_rs_var"),
        ])
        # 20d annualized vols
        df = df.with_columns([
            pl.col("_pk_var").rolling_mean(20, min_periods=10).over("Code").mul(252).sqrt().alias("idx_pk_vol_20d"),
            pl.col("_gk_var").rolling_mean(20, min_periods=10).over("Code").mul(252).sqrt().alias("idx_gk_vol_20d"),
            pl.col("_rs_var").rolling_mean(20, min_periods=10).over("Code").mul(252).sqrt().alias("idx_rs_vol_20d"),
        ])
        df = df.drop(["_pk_var", "_gk_var", "_rs_var"])  # cleanup

    # SMA/STD bands and z-score
    df = df.with_columns([
        pl.col("Close").rolling_mean(20).over("Code").alias("_SMA20"),
        pl.col("Close").rolling_std(20).over("Code").alias("_STD20"),
    ])
    df = df.with_columns([
        ((pl.col("Close") - pl.col("_SMA20")) / (pl.col("_STD20") + eps)).alias("idx_z_close_20"),
        (pl.col("Close") / (pl.col("_SMA20") + eps)).alias("idx_price_to_sma20"),
    ])

    # 60d z-score of daily return (shock indicator)
    mu60 = pl.col("idx_r_1d").rolling_mean(60, min_periods=20).over("Code")
    sd60 = pl.col("idx_r_1d").rolling_std(60, min_periods=20).over("Code") + eps
    df = df.with_columns([((pl.col("idx_r_1d") - mu60) / sd60).alias("idx_z_r1d_60")])

    # Clean up temps
    drop_cols = [c for c in ("_prev_close", "_TR", "_SMA20", "_STD20") if c in df.columns]
    return df.drop(drop_cols)


def build_relative_to_benchmark(df: pl.DataFrame) -> pl.DataFrame:
    """Compute relative features vs benchmark per row using bench_code mapping.

    Adds:
      - idx_rel_r_5d = idx_r_5d - bench_r_5d
      - idx_rel_vol_20d = idx_vol_20d / bench_vol_20d
    """
    if df.is_empty() or "bench_code" not in df.columns:
        return df
    # Prepare benchmark series (rename columns with _bench suffix)
    bench = df.select([
        pl.col("Date"),
        pl.col("Code").alias("bench_code"),
        pl.col("idx_r_5d").alias("bench_r_5d"),
        pl.col("idx_vol_20d").alias("bench_vol_20d"),
    ])
    out = df.join(bench, on=["Date", "bench_code"], how="left")
    out = out.with_columns([
        (pl.col("idx_r_5d") - pl.col("bench_r_5d")).alias("idx_rel_r_5d"),
        (pl.col("idx_vol_20d") / (pl.col("bench_vol_20d") + 1e-12)).alias("idx_rel_vol_20d"),
    ])
    return out


def build_spread_series(df: pl.DataFrame) -> pl.DataFrame:
    """Build day-level spreads for key pairs.

    Output: Date + spread columns shared by all stocks on that date.
    Pairs: (Value-Growth), (Large-Small), (Prime-Standard), (Prime-Growth)
    """
    if df.is_empty():
        return pl.DataFrame()

    def _pair(col: str, a: str, b: str) -> pl.DataFrame:
        s1 = df.filter(pl.col("Code") == a).select(["Date", pl.col(col).alias(f"{col}_{a}")])
        s2 = df.filter(pl.col("Code") == b).select(["Date", pl.col(col).alias(f"{col}_{b}")])
        j = s1.join(s2, on="Date", how="inner")
        return j.select([pl.col("Date"), (pl.col(f"{col}_{a}") - pl.col(f"{col}_{b}")).alias(f"spread_{a}_{b}")])

    spreads: list[pl.DataFrame] = []
    # Default pairs
    default_pairs = [
        ("idx_r_5d", "8100", "8200"),  # Value - Growth
        ("idx_r_5d", "0028", "002D"),  # Large - Small (Core30 vs Small)
        ("idx_r_5d", "0500", "0501"),  # Prime - Standard
        ("idx_r_5d", "0500", "0502"),  # Prime - Growth Market
        # TOPIX-17 examples: (医薬品 - 素材・化学), (電機・精密 - 自動車・輸送機)
        ("idx_r_5d", "0084", "0083"),
        ("idx_r_5d", "0088", "0085"),
        # Additional Topix-17 examples
        ("idx_r_5d", "0089", "0081"),  # 情報通信・サービスその他 - エネルギー資源
        ("idx_r_5d", "0082", "0087"),  # 建設・資材 - 機械
        ("idx_r_5d", "008E", "008F"),  # 銀行 - 金融(除く銀行)
        ("idx_r_5d", "008C", "008D"),  # 商社・卸売 - 小売
        ("idx_r_5d", "008B", "008A"),  # 運輸・物流 - 電力・ガス
        # REIT segments
        ("idx_r_5d", "8501", "8502"),  # Office - Residential
        ("idx_r_5d", "8501", "8503"),  # Office - Commercial/Logistics
        ("idx_r_5d", "8502", "8503"),  # Residential - Commercial/Logistics
        # Size splits (TOPIX families)
        ("idx_r_5d", "002A", "002B"),  # TOPIX100 - Mid400
        ("idx_r_5d", "002E", "002F"),  # TOPIX1000 - Small500
        ("idx_r_5d", "002C", "002F"),  # TOPIX500 - Small500
        ("idx_r_5d", "0029", "002B"),  # Large70 - Mid400
        # Style within families
        ("idx_r_5d", "812C", "822C"),  # TOPIX500 Value - Growth
        ("idx_r_5d", "812D", "822D"),  # TOPIXSmall Value - Growth
    ]

    # Load overrides from configs/index_mappings/index_spreads.json
    def _load_spreads() -> list[tuple[str, str, str]]:
        root = Path(__file__).resolve().parents[2]
        path = root / "configs" / "index_mappings" / "index_spreads.json"
        pairs: list[tuple[str, str, str]] = []
        try:
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get("pairs"), list):
                    for item in data["pairs"]:
                        col = str(item.get("col", "idx_r_5d"))
                        a = str(item.get("a")) if item.get("a") is not None else None
                        b = str(item.get("b")) if item.get("b") is not None else None
                        if col and a and b:
                            pairs.append((col, a, b))
        except Exception:
            pass
        return pairs

    pairs = default_pairs + _load_spreads()
    for col, a, b in pairs:
        if a in df["Code"].to_list() and b in df["Code"].to_list() and col in df.columns:
            spreads.append(_pair(col, a, b))

    if not spreads:
        return pl.DataFrame()
    out = spreads[0]
    for s in spreads[1:]:
        out = out.join(s, on="Date", how="outer_coalesce")
    return out.sort("Date")


def build_breadth(df: pl.DataFrame) -> pl.DataFrame:
    """Compute breadth: share of sector indices above 50-day SMA."""
    if df.is_empty():
        return pl.DataFrame()
    # SMA50 per code
    tmp = df.with_columns([
        pl.col("Close").rolling_mean(50).over("Code").alias("_SMA50"),
        (pl.col("Close") > pl.col("Close").rolling_mean(50).over("Code")).cast(pl.Float64).alias("_gt_ma50"),
    ])
    # Shares by family where present
    if "family" in tmp.columns:
        br_sector = tmp.filter(pl.col("family") == "SECTOR").group_by("Date").agg([
            pl.col("_gt_ma50").mean().alias("breadth_sector_gt_ma50")
        ])
        br_t17 = tmp.filter(pl.col("family") == "SECTOR17").group_by("Date").agg([
            pl.col("_gt_ma50").mean().alias("breadth_t17_gt_ma50")
        ])
        out = br_sector.join(br_t17, on="Date", how="outer_coalesce")
        return out.sort("Date")
    else:
        br = tmp.group_by("Date").agg([pl.col("_gt_ma50").mean().alias("breadth_sector_gt_ma50")])
        return br.sort("Date")


def build_all_index_features(indices: pl.DataFrame, *, mask_halt_day: bool = True) -> tuple[pl.DataFrame, pl.DataFrame]:
    """High-level builder: returns (per-index features, day-level aggregates)."""
    if indices is None or indices.is_empty():
        return pl.DataFrame(), pl.DataFrame()
    df = assign_family_and_benchmark(_ensure_types(indices))
    p = build_per_index_features(df, mask_halt_day=mask_halt_day)
    p = build_relative_to_benchmark(p)
    # Daily aggregates
    spreads = build_spread_series(p)
    breadth = build_breadth(p)
    daily = None
    if not spreads.is_empty() and not breadth.is_empty():
        daily = spreads.join(breadth, on="Date", how="outer_coalesce")
    elif not spreads.is_empty():
        daily = spreads
    else:
        daily = breadth
    daily = daily if daily is not None else pl.DataFrame()
    # Add known market halt dummy (2020-10-01)
    if not p.is_empty():
        halt = p.select(["Date"]).unique().with_columns([
            (pl.col("Date") == pl.date(2020, 10, 1)).cast(pl.Int8).alias("is_halt_20201001")
        ])
        daily = halt if daily.is_empty() else daily.join(halt, on="Date", how="outer_coalesce")
    return p, daily


# ========= Sector mapping (33-industry → sector index code) =========
# Mapping is best-effort; extend via overrides or name-based fallback when available.
SECTOR_NAME_TO_INDEX: dict[str, str] = {
    "水産・農林業": "0040",
    "鉱業": "0041",
    "建設業": "0042",
    "食料品": "0043",
    "繊維製品": "0044",
    "パルプ・紙": "0045",
    "化学": "0046",
    "医薬品": "0047",
    "石油・石炭製品": "0048",
    "ゴム製品": "0049",
    "ガラス・土石製品": "004A",
    "鉄鋼": "004B",
    "非鉄金属": "004C",
    "金属製品": "004D",
    "機械": "004E",
    "電気機器": "004F",
    "輸送用機器": "0050",
    "精密機器": "0051",
    "その他製品": "0052",
    "電気・ガス業": "0053",
    "陸運業": "0054",
    "海運業": "0055",
    "空運業": "0056",
    "倉庫・運輸関連業": "0057",
    "情報・通信業": "0058",
    "卸売業": "0059",
    "小売業": "005A",
    "銀行業": "005B",
    "証券・商品先物取引業": "005C",
    "保険業": "005D",
    "その他金融業": "005E",
    "不動産業": "005F",
    "サービス業": "0060",
}

# Known Sector33Code → name hints seen in repo; extend as needed.
SECTOR33_HINTS: dict[str, str] = {
    "3200": "化学",
    "3300": "医薬品",
    "3400": "石油・石炭製品",
    "4200": "電気機器",
    "4300": "輸送用機器",
    "6050": "小売業",
    "7050": "銀行業",
    "7100": "証券・商品先物取引業",
}


def _load_json(path: Path) -> dict[str, str]:
    try:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def _load_overrides() -> None:
    """Load optional overrides for sector name/code mappings from configs.

    - configs/index_mappings/sector33_name_to_index.json: name → index code
    - configs/index_mappings/sector33_code_to_name.json: Sector33Code → name
    """
    root = Path(__file__).resolve().parents[2]
    base = root / "configs" / "index_mappings"
    name_to_idx = _load_json(base / "sector33_name_to_index.json")
    code_to_name = _load_json(base / "sector33_code_to_name.json")
    if name_to_idx:
        SECTOR_NAME_TO_INDEX.update(name_to_idx)
    if code_to_name:
        SECTOR33_HINTS.update(code_to_name)


# Attempt to load overrides at import (best effort)
_load_overrides()


def build_sector_index_mapping(listed_info_df: pl.DataFrame) -> pl.DataFrame:
    """Return mapping of equity Code → SectorIndexCode using Sector33 info.

    Falls back to name-based mapping when Sector33Name present; otherwise uses
    SECTOR33_HINTS. Unknown categories are left null and will be skipped at join.
    """
    if listed_info_df is None or listed_info_df.is_empty():
        return pl.DataFrame()
    cols = listed_info_df.columns
    df = listed_info_df
    if "Code" not in cols:
        return pl.DataFrame()
    df = df.with_columns(pl.col("Code").cast(pl.Utf8))

    # Derive sector name candidate
    name_expr = None
    if "Sector33Name" in cols:
        name_expr = pl.col("Sector33Name").cast(pl.Utf8)
    elif "SectorName" in cols:
        name_expr = pl.col("SectorName").cast(pl.Utf8)
    elif "Sector33Code" in cols:
        # Map via hints first
        name_expr = pl.col("Sector33Code").cast(pl.Utf8).map_elements(
            lambda x: SECTOR33_HINTS.get(x, None), return_dtype=pl.Utf8
        )

    out = df.select([
        pl.col("Code").alias("Code"),
        (pl.col("Sector33Code").cast(pl.Utf8) if "Sector33Code" in cols else pl.lit(None, dtype=pl.Utf8)).alias("Sector33Code"),
        (name_expr.alias("Sector33Name") if name_expr is not None else pl.lit(None, dtype=pl.Utf8).alias("Sector33Name")),
    ]).unique(maintain_order=True)

    # Map name → index code
    out = out.with_columns([
        pl.col("Sector33Name").map_elements(lambda s: SECTOR_NAME_TO_INDEX.get(s or "", None), return_dtype=pl.Utf8).alias("SectorIndexCode")
    ])

    # If still null, try a final best-effort mapping via code hints
    out = out.with_columns([
        pl.when(pl.col("SectorIndexCode").is_null() & pl.col("Sector33Code").is_not_null())
        .then(pl.col("Sector33Code").map_elements(lambda x: SECTOR_NAME_TO_INDEX.get(SECTOR33_HINTS.get(x or "", ""), None), return_dtype=pl.Utf8))
        .otherwise(pl.col("SectorIndexCode"))
        .alias("SectorIndexCode")
    ])

    return out.select(["Code", "SectorIndexCode"]).drop_nulls()


def attach_sector_index_features(
    quotes: pl.DataFrame,
    indices: pl.DataFrame,
    listed_info_df: pl.DataFrame,
    *,
    prefix: str = "sect_",
    mask_halt_day: bool = True,
) -> pl.DataFrame:
    """Attach sector index features to equities via (Date, SectorIndexCode).

    Uses listed_info to map equities to SectorIndexCode, then joins a subset of
    per-index features from the SECTOR family.
    """
    if quotes.is_empty() or indices.is_empty() or listed_info_df is None or listed_info_df.is_empty():
        return quotes
    # Build per-index features
    per_idx, _ = build_all_index_features(indices, mask_halt_day=mask_halt_day)
    if per_idx.is_empty():
        return quotes
    # Restrict to SECTOR indices
    if "family" not in per_idx.columns:
        per_idx = assign_family_and_benchmark(per_idx)
    sect = per_idx.filter(pl.col("family") == "SECTOR")
    if sect.is_empty():
        return quotes
    # Select and rename sector columns with prefix
    keep_cols = [
        "Date", "Code",  # join keys
        "idx_r_1d", "idx_r_5d", "idx_vol_20d", "idx_atr14", "idx_natr14",
        "idx_z_close_20", "idx_r_co", "idx_r_oc", "idx_rel_r_5d", "idx_rel_vol_20d", "idx_z_r1d_60",
    ]
    present = [c for c in keep_cols if c in sect.columns]
    sect_renamed = sect.select(present).rename({c: f"{prefix}{c.replace('idx_', '')}" for c in present if c not in ("Date", "Code")}).rename({"Code": "SectorIndexCode"})

    # Build equity → SectorIndexCode mapping
    mapping = build_sector_index_mapping(listed_info_df)
    if mapping.is_empty():
        return quotes
    q = quotes
    if q["Date"].dtype == pl.Utf8:
        q = q.with_columns(pl.col("Date").str.strptime(pl.Date, strict=False))
    q = q.join(mapping, on="Code", how="left")
    if "SectorIndexCode" not in q.columns:
        return quotes

    out = q.join(sect_renamed, on=["Date", "SectorIndexCode"], how="left")

    # Optional relative-to-sector feature if returns_5d present
    if "returns_5d" in out.columns and f"{prefix}r_5d" in out.columns:
        out = out.with_columns([
            (pl.col("returns_5d") - pl.col(f"{prefix}r_5d")).alias(f"{prefix}rel_to_sec_5d")
        ])
    return out


def attach_index_features_to_equity(
    quotes: pl.DataFrame,
    per_index: pl.DataFrame,
    daily_feats: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Attach day-level aggregates to equity panel (Date-join).

    Per-index features (Code x Date) are not directly joined to equities here;
    sector/style/size mapping-based joins can be done by callers when mapping
    data is available. This function attaches spreads/breadth shared across all
    equities for the same Date.
    """
    out = quotes
    if daily_feats is not None and not daily_feats.is_empty():
        # Avoid column collisions: only new columns
        new_cols = [c for c in daily_feats.columns if c != "Date" and c not in out.columns]
        if new_cols:
            out = out.join(daily_feats.select(["Date"] + new_cols), on="Date", how="left")
    return out
