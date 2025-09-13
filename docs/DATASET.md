**Dataset Schema (Latest)**

- **Scope:** ML dataset used by Gogooku3 training/inference (daily panel, JP market).
- **Grain:** `Code` × `Date` (JP business days).
- **Blocks:** Identifiers, Price/OHLCV, Returns, Volatility, MA/Position, Technical, Market (TOPIX), Cross, Flow, Statements, Flags, Targets, Optional: Margin (weekly credit balance).

**Identifiers**
- Code: 5‑digit LocalCode string (primary key with `Date`).
- Date: JP business date.
- Section: Market segment (e.g., TSEPrime/Standard/Growth).
- section_norm: Normalized section name.
- row_idx: Per‑stock running index (maturity proxy).
- shares_outstanding: Shares (if available).

**OHLCV**
- Open/High/Low/Close: Adjusted price series.
- Volume: Adjusted shares traded.
- TurnoverValue: Amount traded (if available).

**Returns**
- returns_1d/5d/10d/20d/60d/120d: Simple returns (close‑to‑close).
- log_returns_1d/5d/10d/20d: Log returns.

**Volatility**
- volatility_5d/10d/20d/60d: Rolling std of `returns_1d` (annualized).
- realized_volatility: Range‑based proxy (e.g., Parkinson) where available.

**MA/Position & Gaps**
- sma_5/10/20/60/120, ema_5/10/20/60/200.
- price_to_sma5/20/60, ma_gap_5_20, ma_gap_20_60.
- high_low_ratio, close_to_high, close_to_low.

**Technical**
- rsi_2, rsi_14, macd, macd_signal, macd_histogram.
- atr_14, adx_14, stoch_k, bb_width, bb_position.

**Market (TOPIX) Examples**
- mkt_ret_1d/5d/10d/20d.
- mkt_ema_5/20/60/200, mkt_dev_20, mkt_gap_5_20, mkt_ema20_slope_3.
- mkt_vol_20d, mkt_atr_14, mkt_natr_14.
- mkt_bb_pct_b, mkt_bb_bw.
- mkt_dd_from_peak, mkt_big_move_flag.
- mkt_ret_1d_z, mkt_vol_20d_z, mkt_bb_bw_z, mkt_dd_from_peak_z.
- mkt_bull_200, mkt_trend_up, mkt_high_vol, mkt_squeeze.

**Cross (Market‑Relative) Examples**
- beta_60d, alpha_1d, alpha_5d, rel_strength_5d.
- trend_align_mkt, alpha_vs_regime, idio_vol_ratio, beta_stability_60d.

**Flow (Investor Breakdown) Examples**
- flow_foreign_net_ratio, flow_individual_net_ratio, flow_activity_ratio.
- foreign_share_activity, breadth_pos.
- flow_foreign_net_z, flow_individual_net_z, flow_activity_z.
- flow_smart_idx (= foreign_z − individual_z), flow_smart_mom4, flow_shock_flag.
- flow_impulse, flow_days_since (interval as‑of join markers).

**Statements (As‑Of Joined) Examples**
- stmt_yoy_sales, stmt_yoy_op, stmt_yoy_np, stmt_opm, stmt_npm.
- stmt_progress_op, stmt_progress_np.
- stmt_rev_fore_op, stmt_rev_fore_np, stmt_rev_fore_eps, stmt_rev_div_fore.
- stmt_roe, stmt_roa, stmt_change_in_est, stmt_nc_flag.
- stmt_imp_statement, stmt_days_since_statement.

**Flags**
- is_rsi2_valid, is_ema5_valid, is_ema10_valid, is_ema20_valid, is_ema200_valid.
- is_valid_ma, is_flow_valid, is_stmt_valid.

**Targets**
- target_1d/5d/10d/20d: Forward returns.
- target_1d_binary/5d_binary/10d_binary: Directional (up/down) labels.

**Margin (Weekly Credit Balance)**
- Description: Weekly margin interest series (buy/sell balances) integrated via leak-safe as-of backward join. Features computed on weekly intervals then attached to daily grid using effective_start timestamps.
- Effective Start: PublishedDate + 1 business day, or Date + lag_bdays_weekly (default 3) if PublishedDate is missing.
- Scale Normalization: All stock quantities normalized by ADV20_shares (20-day average daily volume in shares).
- Core Features:
  - Stocks: margin_long_tot, margin_short_tot, margin_total_gross
  - Ratios: margin_credit_ratio (long/short), margin_imbalance ((long-short)/(long+short))
  - Weekly Diffs: margin_d_long_wow, margin_d_short_wow, margin_d_net_wow, margin_d_ratio_wow
  - Z-scores (52w): long_z52, short_z52, margin_gross_z52, ratio_z52
  - Scaled by ADV: margin_long_to_adv20, margin_short_to_adv20, margin_d_long_to_adv20, margin_d_short_to_adv20
  - Timing/Validity: margin_impulse (effective_start day flag), margin_days_since, is_margin_valid, margin_issue_type, is_borrowable
- Availability: Attached when `--weekly-margin-parquet` option is used or weekly_margin_interest_*.parquet files are discovered in output/.

**Time Alignment & Leakage Prevention**
- Calendar: JP business days; all joins align on `Date`.
- Statements: DisclosedTime < 15:00 → same day, ≥ 15:00 → next business day (as‑of join).
- Flow: Weekly published intervals expanded via as‑of join; `flow_impulse/flow_days_since` annotate freshness.
- Margin: Weekly series with effective_start = PublishedDate + 1 business day (or Date + lag_bdays_weekly if PublishedDate missing). Features computed weekly, then as-of backward join to daily grid. ADV20_shares calculated from adjusted volume with no same-day lookahead.
- Normalization: Cross‑sectional Z/sector‑in‑Z are fit on train windows only.

**Null Semantics**
- Block‑specific columns are Null before effective start or when out‑of‑scope (e.g., section mismatch). Validity flags (`is_*_valid`) are 0 for Null blocks.
- Keys (`Code`, `Date`) must be non‑Null and unique.

**Persistence & Naming**
- Output: `output/ml_dataset_latest_full.parquet` (symlink maintained).
- Aliases: Common alternates mapped to spec (e.g., `bb_bandwidth`→`bb_width`, `realized_vol_20`→`realized_volatility`).
- Strict Schema: Some pipelines project to a fixed column set before saving. Optional blocks (e.g., Margin) may be dropped in such builds; this will be unified in the refactor.

**Change Log (Highlights)**
- 2025‑09: Margin (weekly credit balance) block fully implemented with leak-safe as-of join, effective_start calculation, and ADV20 scaling.
- JQuantsAsyncFetcher moved from scripts/_archive/ to src/gogooku3/components/ for better organization.
- TOPIX/Flow/Statements remain auto‑integrated; missing inputs are skipped per current behavior.

