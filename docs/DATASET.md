**Dataset Schema (Latest)**

- **Scope:** ML dataset used by Gogooku3 training/inference (daily panel, JP market).
- **Grain:** `Code` × `Date` (JP business days).
- **Blocks:** Identifiers, Price/OHLCV, Returns, Volatility, MA/Position, Technical, Market (TOPIX), Cross, Flow, Statements, Flags, Targets; Optional: Margin (weekly/daily credit), Sector Short Selling (sector‑level), Sector Cross‑Sectional (relative), Graph (correlation network), Option Market Aggregates (NK225 options).

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

**Margin (Daily Credit Balance)**

- Description: Daily margin interest series (credit balances and regulatory flags) integrated via leak-safe as-of backward join to the daily panel. Column prefix is `dmi_` to distinguish from weekly (`margin_`).
- Effective Start: `effective_start = next_business_day(PublishedDate)` (T+1 rule). Values become valid starting on `effective_start` only (no same‑day lookahead).
- Join Strategy: `(Code, Date)` daily grid joined as‑of backward on `effective_start` per stock; latest correction per `(Code, ApplicationDate)` kept in the fetcher.
- Scale Normalization: Liquidity‑scaled features computed when `ADV20_shares` is available from quotes (20‑day rolling mean of adjusted volume).
- Core Features:
  - Stocks/ratios: `dmi_long`, `dmi_short`, `dmi_net`, `dmi_total`, `dmi_credit_ratio`, `dmi_imbalance`, `dmi_short_long_ratio`
  - Daily diffs/Z: `dmi_d_long_1d`, `dmi_d_short_1d`, `dmi_d_net_1d`, `dmi_d_ratio_1d`, `dmi_z26_long/short/total/d_short_1d`
  - ADV‑scaled: `dmi_long_to_adv20`, `dmi_short_to_adv20`, `dmi_total_to_adv20`, `dmi_d_long_to_adv1d`, `dmi_d_short_to_adv1d`, `dmi_d_net_to_adv1d`
  - Regulatory flags/count: `dmi_reason_restricted`, `dmi_reason_dailypublication`, `dmi_reason_monitoring`, `dmi_reason_restrictedbyjsf`, `dmi_reason_precautionbyjsf`, `dmi_reason_unclearorseconalert`, `dmi_reason_count`, `dmi_tse_reg_level`
  - Timing/Validity: `dmi_impulse` (1 on `effective_start`), `dmi_days_since_pub`, `dmi_days_since_app`, `is_dmi_valid`
- Enabling (pipeline): `scripts/pipelines/run_full_dataset.py`
  - Flags: `--enable-daily-margin`, `--daily-margin-parquet <path>`
  - Auto‑discover: `output/daily_margin_interest_*.parquet` when not specified.
- Leakage Prevention: T+1 effective start; as‑of backward join; latest correction kept per `(Code, ApplicationDate)`.

**Sector Short Selling (業種別空売り)**

- Description: Sector‑level short selling ratios and restriction ratios aggregated per 33‑sector, attached to stocks via leak‑safe as‑of join.
- Effective Start: `effective_date = next_business_day(Date)` (T+1 rule). Values become valid starting on `effective_date`.
- Sector Features (examples):
  - Sector: `ss_sec33_short_share`, `ss_sec33_restrict_share`, `ss_sec33_short_turnover`, daily diffs (`*_d1`), momentum/accel (`ss_sec33_short_mom5`, `ss_sec33_short_accel`), z‑scores (`ss_sec33_short_share_z60`, `ss_sec33_short_turnover_z252`).
  - Market aggregates: `ss_mkt_short_share`, `ss_mkt_restrict_share`, `ss_mkt_short_breadth_q80`.
  - Relative: `ss_rel_short_share`, `ss_rel_restrict_share`; conditional signals `ss_cond_pressure`, `ss_squeeze_setup`.
  - Validity: `is_ss_valid`.
- Enabling: `--enable-sector-short-selling` (pipeline); module `src/gogooku3/features/short_selling_sector.py`.

**Sector Cross‑Sectional (Sector‑Relative) Features**

- Description: Same‑day cross‑sectional statistics computed within sectors (e.g., 33‑sector) and joined T+0 (no lookahead).
- Core Features: deviations vs sector mean and z‑scores within sector.
  - Returns: `ret_1d_vs_sec`, `ret_1d_in_sec_z`, `ret_1d_rank_in_sec`; also `ret_5d_*`, `ret_10d_*` when present.
  - Volume/vol: `volume_in_sec_z`, `volume_rank_in_sec`, `rv20_in_sec_z`.
  - Technical interactions: `rsi14_in_sec_z`, `macd_slope_in_sec_z`, `vcm_in_sec_z`, `rsi_vol_in_sec_z`.
- Custom Include Cols: additional numeric columns can be specified to auto‑generate `<col>_vs_sec` and `<col>_in_sec_z`.
  - CLI: `--enable-sector-cs --sector-cs-cols "rsi_14,returns_10d"`
  - YAML: via `--config` (`sector_cs.include_cols`).
- Module: `src/gogooku3/features/sector_cross_sectional.py`.

**Graph (Correlation Network) Features**

- Description: For each Date, builds a correlation graph over past returns and attaches node‑level features. T+0 only uses past window.
- Core Features:
  - Topology/centrality: `graph_degree`, `graph_degree_z`, `graph_degree_z_in_comp`, `graph_clustering`, `graph_avg_neigh_deg`, `graph_core`, `graph_degree_centrality`, `graph_closeness`, `graph_pagerank`, `graph_pagerank_z_in_comp`.
  - Components/density: `graph_comp_size`, `graph_degree_in_comp`, `graph_local_density`, `graph_isolated`, `graph_pagerank_share_comp`.
  - Peer stats: `peer_corr_mean`, `peer_count`.
- Parameters (CLI): `--enable-graph-features --graph-window 60 --graph-threshold 0.3 --graph-max-k 10 --graph-cache-dir output/graph_cache`.
- YAML: via `--config` (`graph.window/threshold/max_k/cache_dir`).
- Module: `src/gogooku3/features/graph_features.py`.

**Option Market Aggregates (Nikkei225 Options)**

- Description: Builds daily market‑level aggregates from NK225 option contracts (per‑contract features built separately). Aggregates attached T+1 to equity panel.
- Aggregates: `opt_iv_cmat_30d`, `opt_iv_cmat_60d`, `opt_term_slope_30_60`, `opt_iv_atm_median`, `opt_oi_sum`, `opt_vol_sum`, `opt_dollar_vol_sum`.
- Effective Start: via `effective_date = next_business_day(Date)` when attaching; leak‑safe.
- Enabling attach: `--attach-nk225-option-market` (requires features/raw or API fetch).
- Modules: `src/gogooku3/features/index_option.py`, fetcher `get_index_option` in `src/gogooku3/components/jquants_async_fetcher.py`.

**Time Alignment & Leakage Prevention**
- Calendar: JP business days; all joins align on `Date`.
- Statements: DisclosedTime < 15:00 → same day, ≥ 15:00 → next business day (as‑of join).
- Flow: Weekly published intervals expanded via as‑of join; `flow_impulse/flow_days_since` annotate freshness.
- Margin: Weekly series with effective_start = PublishedDate + 1 business day (or Date + lag_bdays_weekly if PublishedDate missing). Features computed weekly, then as-of backward join to daily grid. ADV20_shares calculated from adjusted volume with no same-day lookahead.
- Sector Short Selling: Sector metrics have `effective_date = next_bday(Date)`, then as‑of backward join by sector; individual stocks receive sector values only on or after `effective_date`.
- Option Market Aggregates: Market aggregates are shifted to `effective_date = next_bday(Date)` and joined on Date (T+1) to avoid lookahead.
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
