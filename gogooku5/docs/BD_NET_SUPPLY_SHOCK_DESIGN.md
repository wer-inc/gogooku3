# BD Net / ADV60 / Supply-Shock – Design Memo

**Author**: Codex agent  
**Date**: 2025-11-17  
**Scope**: gogooku5 chunk builder + merged dataset tooling

## 0. Background & Motivation

- The current merged datasets (`output_g5/datasets/ml_dataset_2024_full.parquet`, `ml_dataset_2025_full_with_beta_bd.parquet`) still carry more than 500 all-null columns (see `gogooku5/docs/NULL_COLUMNS_REPORT.md`). Among them are the **bd_net_mc**, **buyback_flag**, **dilution_flag**, and all supply-shock related columns.
- `bd_net_value` (buy minus sell value) is already produced inside the chunk pipeline via the breakdown cache, but we never persist the liquidity denominators that the downstream ratios depend on:
  - `adv60_yen` (60 trading day average turnover) is missing, so `bd_net_adv60` evaluates to zero.
  - We do not retain a reliable market cap series built from FS shares + close price, therefore `bd_net_mc` also stays NULL.
  - Supply-shock flags require `fs_shares_outstanding` and/or free-float shares in a daily master table, which is not propagated.
- Goal: define clear specs so that (1) chunks contain the inputs we need, (2) derived ratios are calculated deterministically, and (3) rebuilt 2024–2025 datasets expose non-null columns suitable for training (Apex-Ranker, ATFT).

## 1. adv60_yen

### Definition
- `adv60_yen` = simple moving average of **daily turnover in yen** over the last 60 trading days (including the current day).
- Turnover base: `Close * Volume` aligned with the same timestamp convention that `dollar_volume` uses. We keep units in JPY to match `bd_net_value`.
- Window requirements:
  - Minimum 20 historical observations to emit a value; otherwise set NULL.
  - When fewer than 60 observations exist (e.g., IPOs), compute with the available window length but store `adv60_obs_count` (optional helper) so downstream consumers can gauge stability.

### Data Source & Storage
- Compute `turnover_yen` inside the chunk assembler right after prices join (same stage as `dollar_volume`).
- Maintain a **per code sorted frame** and use Polars rolling mean (`groupby_rolling`) so the computation is vectorized.
- Persist `adv60_yen` (and optional `adv60_obs_count`) on each chunk row before advanced features execute.
- No separate cache/master table: recompute on every chunk rebuild to avoid drift.

### Validation
- Sanity checks per chunk:
  - Coverage ratio (>=85% rows with non-null values outside IPO windows).
  - Quantile summary to ensure magnitudes line up with `dollar_volume` (median difference ≤5%).
- `check_dataset_quality.py` can assert `adv60_yen > 0` when non-null.

## 2. bd_net_adv60 & bd_net_mc

### Definitions
- `bd_net_value`: already available (buy minus sell value aggregated from daily breakdown cache).
- `bd_net_adv60 = bd_net_value / adv60_yen`
  - Unitless ratio; interpretable as **net buy flow expressed in “ADV multiples”**.
  - Guard rails: emit NULL when `adv60_yen` ≤ 0 or NULL.
- `bd_net_mc = bd_net_value / market_cap_ff`
  - `market_cap_ff`: preferred free-float market cap; fallback to total shares outstanding if free-float missing.
  - Provides a dilution-scale normalized flow signal.

### Market Cap Construction
- Inputs:
  - `Close` from price chunk (`gogooku5/data/src/builder/features/core/prices.py`).
  - `fs_shares_outstanding` and `fs_free_float_share` from FS tables (quarterly).
- Process:
  1. Build a **daily FS master** keyed by `Code` + `Date`:
     - Expand each financial statement’s shares_outstanding across days starting from `report_date` (or disclosure date) until the next report.
     - Forward fill up to **250 trading days**; beyond that keep NULL.
     - Provide two columns: `shares_total`, `shares_free_float`.
  2. Join this master before we compute advanced features.
  3. Compute:
     - `market_cap_total = Close * shares_total`
     - `market_cap_ff = Close * coalesce(shares_free_float, shares_total)`
  4. Persist these columns; they can be reused for other ratios later.

### Pipeline Hook
- After we produce `bd_net_value` from the breakdown join, inject:
  ```python
  combined_df = combined_df.with_columns([
      (pl.col("Close") * pl.col("shares_total")).alias("market_cap_total"),
      (pl.col("Close") * pl.coalesce([pl.col("shares_free_float"), pl.col("shares_total")])).alias("market_cap_ff"),
      (pl.col("bd_net_value") / pl.col("adv60_yen")).alias("bd_net_adv60"),
      (pl.col("bd_net_value") / pl.col("market_cap_ff")).alias("bd_net_mc"),
  ])
  ```
- Ensure we run this block **after** `adv60_yen` is materialized but **before** NULL-column filters or feature derivations.

### Validation & Monitoring
- Add histograms / summary stats in the build log (min, max, median).
- Extend dataset quality checker:
  - Fraction of non-null rows per chunk.
  - `abs(bd_net_adv60)` clipped at, say, 50 ADV to catch outliers (rare but indicates data issues).

## 3. Supply-Shock Stack (fs_shares_outstanding)

### Inputs & Master Table
- Source tables: FS fundamentals already staged under `output_g5/cache/fundamentals_*`.
- Build `shares_master.parquet` (Code, Date, shares_total, shares_free_float) via a dedicated tool (`data/tools/build_shares_master.py`).
- Forward fill rules:
  - Fill daily until the next filing or up to **250 trading days**.
  - When both free-float and total are missing for >250d, leave NULL.
  - Track `shares_source` (e.g., "fs_report", "ff_fill") for debugging.
  - Forward-fill horizon is configurable via `SHARE_FORWARD_FILL_DAYS` (default 250).

### Derived Features
1. `supply_shock_ratio = shares_total / shares_total_60d_lag - 1`
   - Use 60 trading day lag via rolling window on the master.
2. `buyback_flag = 1` when `supply_shock_ratio <= -0.02` (≥2% reduction) over any 60d span.
3. `dilution_flag = 1` when `supply_shock_ratio >= +0.02`.
4. Optional continuous columns:
   - `supply_shock_zscore` using 1-year rolling stats.
   - `shares_free_float_ratio = shares_free_float / shares_total`.

### Integration
- Join the shares master to the chunk frame alongside `adv60_yen`.
- After join, compute ratio/flag columns; ensure they are listed in the schema manifest so checkers expect them.
- For older chunks where shares data is sparse, columns should remain NULL rather than zeroed → the quality checker warnings remain informative but no schema mismatch occurs.

## 4. Implementation Order (Post-Design Sign-off)

1. **Chunk Layer Enhancements**
   - Persist `adv60_yen` (and optional obs count).
   - Join the daily shares master; expose `shares_total`, `shares_free_float`.
2. **Derived Ratios**
   - Compute `bd_net_adv60`, `bd_net_mc`, `market_cap_total`, `market_cap_ff`.
   - Generate supply-shock ratios/flags.
3. **Rebuild Scope**
   - Rebuild 2024Q1–2025Q4 chunks with `ENABLE_GRAPH_FEATURES=0` (baseline).
   - Merge 2024+2025 full datasets.
   - Re-run post-process scripts (`add_beta_alpha_bd_features_full.py` etc.) so merged artifacts align.
4. **Validation**
   - Update schema hash / manifest.
   - Run `check_dataset_quality.py` on 2024, 2025, and combined 2024-2025.
   - Produce NULL report diff to confirm the formerly NULL columns now carry data.

## 5. Impact Assessment

| Area | Expected Impact |
|------|-----------------|
| **Dataset Builder** | Additional columns increase schema size by ~6–8 columns; requires minor adjustments to manifest and logging but no control-flow change. |
| **Merge Scripts** | No logic changes; they concatenate chunks. Need to update the allowlist of columns if hard-coded, otherwise transparent. |
| **Quality Checker** | Must whitelist the new columns (non-null checks, warning thresholds). Adds regression coverage for ratios >|50|. |
| **Apex-Ranker & ATFT** | Feature configs can begin referencing `bd_net_adv60`, `bd_net_mc`, `supply_shock_ratio`, `buyback_flag`, `dilution_flag`. Until then, existing configs continue to work because new columns default to ignored. |
| **Storage & Runtime** | Rolling averages and share joins add modest cost (~+5% build time per chunk based on Dry run). Acceptable on the A100 host. |

No other modules read/write the affected columns today, so risk of “program breakage” is minimal once schema hashes are updated.

## 6. Open Questions / Next Steps

1. **adv60 definition**: The memo assumes a *simple* moving average. If we prefer EMA or dollar-volume-based ADV (USD), confirm before implementation.
2. **share splits handling**: Need to confirm whether `shares_total` from FS already reflects splits. If the upstream data lags, we may need to detect split events via prices.
3. **Historical Backfill**: Do we need to recompute these features for 2020–2023 as well, or is 2024–2025 sufficient for next training milestones?
4. **Downstream usage**: Apex-Ranker configs should enumerate which of the new columns participate in training so we can prioritize monitoring.

Once these items are acknowledged, we can proceed with Step 2 of the master plan (implementation inside `dataset_builder` and associated tools).
