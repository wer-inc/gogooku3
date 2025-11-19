# NULL Columns Status Report - 2025 Dataset

**Generated**: 2025-11-18
**Dataset**: ml_dataset_2025_with_graph33.parquet
**Total Rows**: 725,839
**Total Columns**: 4,207

## Executive Summary

⚠️ **Critical Issue**: 737 columns (17.5%) are 100% NULL in 2025 dataset

### NULL Rate Distribution

| NULL Rate | Column Count | Percentage |
|-----------|--------------|------------|
| 100% NULL | 737 | 17.5% |
| 50-99% NULL | 386 | 9.2% |
| 10-49% NULL | 183 | 4.3% |
| 1-9% NULL | 1268 | 30.1% |
| <1% NULL | 1054 | 25.1% |
| No NULL | 579 | 13.8% |

## 100% NULL Columns by Category

### MKT Features (266 columns)

```
mkt_flow_divergence_foreigners_vs_individuals
mkt_flow_divergence_foreigners_vs_individuals_cs_bottom20_flag
mkt_flow_divergence_foreigners_vs_individuals_cs_pct
mkt_flow_divergence_foreigners_vs_individuals_cs_rank
mkt_flow_divergence_foreigners_vs_individuals_cs_top20_flag
mkt_flow_divergence_foreigners_vs_individuals_outlier_flag
mkt_flow_divergence_foreigners_vs_individuals_roll_mean_20d
mkt_flow_divergence_foreigners_vs_individuals_roll_std_20d
mkt_flow_divergence_foreigners_vs_individuals_sector_mean
mkt_flow_divergence_foreigners_vs_individuals_sector_rel
mkt_flow_divergence_foreigners_vs_individuals_zscore_20d
mkt_flow_flow_foreigners_net_ratio_z52
mkt_flow_flow_foreigners_net_ratio_z52_cs_bottom20_flag
mkt_flow_flow_foreigners_net_ratio_z52_cs_pct
mkt_flow_flow_foreigners_net_ratio_z52_cs_rank
mkt_flow_flow_foreigners_net_ratio_z52_cs_top20_flag
mkt_flow_flow_foreigners_net_ratio_z52_outlier_flag
mkt_flow_flow_foreigners_net_ratio_z52_roll_mean_20d
mkt_flow_flow_foreigners_net_ratio_z52_roll_std_20d
mkt_flow_flow_foreigners_net_ratio_z52_sector_mean
mkt_flow_flow_foreigners_net_ratio_z52_sector_rel
mkt_flow_flow_foreigners_net_ratio_z52_zscore_20d
mkt_flow_flow_individuals_net_ratio_z52
mkt_flow_flow_individuals_net_ratio_z52_cs_bottom20_flag
mkt_flow_flow_individuals_net_ratio_z52_cs_pct
mkt_flow_flow_individuals_net_ratio_z52_cs_rank
mkt_flow_flow_individuals_net_ratio_z52_cs_top20_flag
mkt_flow_flow_individuals_net_ratio_z52_outlier_flag
mkt_flow_flow_individuals_net_ratio_z52_roll_mean_20d
mkt_flow_flow_individuals_net_ratio_z52_roll_std_20d
... and 236 more
```

### BETA60 Features (61 columns)

```
beta60_topix
beta60_topix_cs_bottom20_flag
beta60_topix_cs_pct
beta60_topix_cs_rank
beta60_topix_cs_top20_flag
beta60_topix_outlier_flag
beta60_topix_roll_mean_20d
beta60_topix_roll_mean_20d_cs_bottom20_flag
beta60_topix_roll_mean_20d_cs_pct
beta60_topix_roll_mean_20d_cs_rank
beta60_topix_roll_mean_20d_cs_top20_flag
beta60_topix_roll_mean_20d_outlier_flag
beta60_topix_roll_mean_20d_roll_mean_20d
beta60_topix_roll_mean_20d_roll_std_20d
beta60_topix_roll_mean_20d_sector_mean
beta60_topix_roll_mean_20d_sector_rel
beta60_topix_roll_mean_20d_zscore_20d
beta60_topix_roll_std_20d
beta60_topix_roll_std_20d_cs_bottom20_flag
beta60_topix_roll_std_20d_cs_pct
beta60_topix_roll_std_20d_cs_rank
beta60_topix_roll_std_20d_cs_top20_flag
beta60_topix_roll_std_20d_outlier_flag
beta60_topix_roll_std_20d_roll_mean_20d
beta60_topix_roll_std_20d_roll_std_20d
beta60_topix_roll_std_20d_sector_mean
beta60_topix_roll_std_20d_sector_rel
beta60_topix_roll_std_20d_zscore_20d
beta60_topix_sector_mean
beta60_topix_sector_mean_cs_bottom20_flag
... and 31 more
```

### ALPHA60 Features (61 columns)

```
alpha60_topix
alpha60_topix_cs_bottom20_flag
alpha60_topix_cs_pct
alpha60_topix_cs_rank
alpha60_topix_cs_top20_flag
alpha60_topix_outlier_flag
alpha60_topix_roll_mean_20d
alpha60_topix_roll_mean_20d_cs_bottom20_flag
alpha60_topix_roll_mean_20d_cs_pct
alpha60_topix_roll_mean_20d_cs_rank
alpha60_topix_roll_mean_20d_cs_top20_flag
alpha60_topix_roll_mean_20d_outlier_flag
alpha60_topix_roll_mean_20d_roll_mean_20d
alpha60_topix_roll_mean_20d_roll_std_20d
alpha60_topix_roll_mean_20d_sector_mean
alpha60_topix_roll_mean_20d_sector_rel
alpha60_topix_roll_mean_20d_zscore_20d
alpha60_topix_roll_std_20d
alpha60_topix_roll_std_20d_cs_bottom20_flag
alpha60_topix_roll_std_20d_cs_pct
alpha60_topix_roll_std_20d_cs_rank
alpha60_topix_roll_std_20d_cs_top20_flag
alpha60_topix_roll_std_20d_outlier_flag
alpha60_topix_roll_std_20d_roll_mean_20d
alpha60_topix_roll_std_20d_roll_std_20d
alpha60_topix_roll_std_20d_sector_mean
alpha60_topix_roll_std_20d_sector_rel
alpha60_topix_roll_std_20d_zscore_20d
alpha60_topix_sector_mean
alpha60_topix_sector_mean_cs_bottom20_flag
... and 31 more
```

### BASIS Features (61 columns)

```
basis_gate
basis_gate_cs_bottom20_flag
basis_gate_cs_pct
basis_gate_cs_rank
basis_gate_cs_top20_flag
basis_gate_outlier_flag
basis_gate_roll_mean_20d
basis_gate_roll_mean_20d_cs_bottom20_flag
basis_gate_roll_mean_20d_cs_pct
basis_gate_roll_mean_20d_cs_rank
basis_gate_roll_mean_20d_cs_top20_flag
basis_gate_roll_mean_20d_outlier_flag
basis_gate_roll_mean_20d_roll_mean_20d
basis_gate_roll_mean_20d_roll_std_20d
basis_gate_roll_mean_20d_sector_mean
basis_gate_roll_mean_20d_sector_rel
basis_gate_roll_mean_20d_zscore_20d
basis_gate_roll_std_20d
basis_gate_roll_std_20d_cs_bottom20_flag
basis_gate_roll_std_20d_cs_pct
basis_gate_roll_std_20d_cs_rank
basis_gate_roll_std_20d_cs_top20_flag
basis_gate_roll_std_20d_outlier_flag
basis_gate_roll_std_20d_roll_mean_20d
basis_gate_roll_std_20d_roll_std_20d
basis_gate_roll_std_20d_sector_mean
basis_gate_roll_std_20d_sector_rel
basis_gate_roll_std_20d_zscore_20d
basis_gate_sector_mean
basis_gate_sector_mean_cs_bottom20_flag
... and 31 more
```

### MPI Features (44 columns)

```
mpi_dist_to_limit
mpi_dist_to_limit_cs_bottom20_flag
mpi_dist_to_limit_cs_pct
mpi_dist_to_limit_cs_rank
mpi_dist_to_limit_cs_top20_flag
mpi_dist_to_limit_outlier_flag
mpi_dist_to_limit_roll_mean_20d
mpi_dist_to_limit_roll_std_20d
mpi_dist_to_limit_sector_mean
mpi_dist_to_limit_sector_rel
mpi_dist_to_limit_z20
mpi_dist_to_limit_z20_cs_bottom20_flag
mpi_dist_to_limit_z20_cs_pct
mpi_dist_to_limit_z20_cs_rank
mpi_dist_to_limit_z20_cs_top20_flag
mpi_dist_to_limit_z20_outlier_flag
mpi_dist_to_limit_z20_roll_mean_20d
mpi_dist_to_limit_z20_roll_std_20d
mpi_dist_to_limit_z20_sector_mean
mpi_dist_to_limit_z20_sector_rel
mpi_dist_to_limit_z20_zscore_20d
mpi_dist_to_limit_zscore_20d
mpi_drawdown
mpi_drawdown_cs_bottom20_flag
mpi_drawdown_cs_pct
mpi_drawdown_cs_rank
mpi_drawdown_cs_top20_flag
mpi_drawdown_outlier_flag
mpi_drawdown_roll_mean_20d
mpi_drawdown_roll_std_20d
... and 14 more
```

### DAYS Features (30 columns)

```
days_since_market_change
days_since_market_change_cs_bottom20_flag
days_since_market_change_cs_pct
days_since_market_change_cs_rank
days_since_market_change_cs_top20_flag
days_since_market_change_outlier_flag
days_since_market_change_roll_mean_20d
days_since_market_change_roll_std_20d
days_since_market_change_sector_mean
days_since_market_change_sector_rel
days_since_market_change_zscore_20d
days_since_sector33_change_outlier_flag
days_since_sector33_change_roll_mean_20d
days_since_sector33_change_roll_std_20d
days_since_sector33_change_zscore_20d
days_since_sq_outlier_flag
days_since_sq_zscore_20d
days_to_earnings
days_to_earnings_cs_bottom20_flag
days_to_earnings_cs_pct
days_to_earnings_cs_rank
days_to_earnings_cs_top20_flag
days_to_earnings_outlier_flag
days_to_earnings_roll_mean_20d
days_to_earnings_roll_std_20d
days_to_earnings_sector_mean
days_to_earnings_sector_rel
days_to_earnings_zscore_20d
days_to_sq_outlier_flag
days_to_sq_zscore_20d
```

### IS Features (27 columns)

```
is_E_0
is_E_pm1
is_E_pp1
is_E_pp3
is_E_pp5
is_growth_x_dv_z20
is_growth_x_dv_z20_cs_bottom20_flag
is_growth_x_dv_z20_cs_pct
is_growth_x_dv_z20_cs_rank
is_growth_x_dv_z20_cs_top20_flag
is_growth_x_dv_z20_outlier_flag
is_growth_x_dv_z20_roll_mean_20d
is_growth_x_dv_z20_roll_std_20d
is_growth_x_dv_z20_sector_mean
is_growth_x_dv_z20_sector_rel
is_growth_x_dv_z20_zscore_20d
is_prime_x_dv_z20
is_prime_x_dv_z20_cs_bottom20_flag
is_prime_x_dv_z20_cs_pct
is_prime_x_dv_z20_cs_rank
is_prime_x_dv_z20_cs_top20_flag
is_prime_x_dv_z20_outlier_flag
is_prime_x_dv_z20_roll_mean_20d
is_prime_x_dv_z20_roll_std_20d
is_prime_x_dv_z20_sector_mean
is_prime_x_dv_z20_sector_rel
is_prime_x_dv_z20_zscore_20d
```

### FS Features (26 columns)

```
fs_accruals
fs_accruals_ttm
fs_capex_ttm
fs_cfo_to_ni
fs_cfo_ttm
fs_days_to_next
fs_equity_ratio
fs_fcf_ttm
fs_net_cash_ratio
fs_net_income_ttm
fs_net_margin
fs_op_margin
fs_op_profit_ttm
fs_revenue_ttm
fs_roa_ttm
fs_roe_ttm
fs_sales_yoy
fs_ttm_cfo
fs_ttm_cfo_margin
fs_ttm_net_income
fs_ttm_op_margin
fs_ttm_op_profit
fs_ttm_sales
fs_yoy_ttm_net_income
fs_yoy_ttm_op_profit
fs_yoy_ttm_sales
```

### SEC17 Features (24 columns)

```
sec17_mom_20
sec17_mom_20_cs_bottom20_flag
sec17_mom_20_cs_pct
sec17_mom_20_cs_rank
sec17_mom_20_cs_top20_flag
sec17_mom_20_outlier_flag
sec17_mom_20_roll_mean_20d
sec17_mom_20_roll_std_20d
sec17_mom_20_sector_mean
sec17_mom_20_sector_rel
sec17_mom_20_zscore_20d
sec17_ret_5d_eq
sec17_ret_5d_eq_cs_bottom20_flag
sec17_ret_5d_eq_cs_pct
sec17_ret_5d_eq_cs_rank
sec17_ret_5d_eq_cs_top20_flag
sec17_ret_5d_eq_hist_vol
sec17_ret_5d_eq_outlier_flag
sec17_ret_5d_eq_roll_mean_20d
sec17_ret_5d_eq_roll_std_20d
sec17_ret_5d_eq_rolling_sharpe
sec17_ret_5d_eq_sector_mean
sec17_ret_5d_eq_sector_rel
sec17_ret_5d_eq_zscore_20d
```

### IDXOPT Features (22 columns)

```
idxopt_vrp_gap
idxopt_vrp_gap_cs_bottom20_flag
idxopt_vrp_gap_cs_pct
idxopt_vrp_gap_cs_rank
idxopt_vrp_gap_cs_top20_flag
idxopt_vrp_gap_outlier_flag
idxopt_vrp_gap_roll_mean_20d
idxopt_vrp_gap_roll_std_20d
idxopt_vrp_gap_sector_mean
idxopt_vrp_gap_sector_rel
idxopt_vrp_gap_zscore_20d
idxopt_vrp_ratio
idxopt_vrp_ratio_cs_bottom20_flag
idxopt_vrp_ratio_cs_pct
idxopt_vrp_ratio_cs_rank
idxopt_vrp_ratio_cs_top20_flag
idxopt_vrp_ratio_outlier_flag
idxopt_vrp_ratio_roll_mean_20d
idxopt_vrp_ratio_roll_std_20d
idxopt_vrp_ratio_sector_mean
idxopt_vrp_ratio_sector_rel
idxopt_vrp_ratio_zscore_20d
```

### RQ Features (12 columns)

```
rq_63_10_outlier_flag
rq_63_10_roll_mean_20d
rq_63_10_roll_std_20d
rq_63_10_zscore_20d
rq_63_50_outlier_flag
rq_63_50_roll_mean_20d
rq_63_50_roll_std_20d
rq_63_50_zscore_20d
rq_63_90_outlier_flag
rq_63_90_roll_mean_20d
rq_63_90_roll_std_20d
rq_63_90_zscore_20d
```

### REL Features (11 columns)

```
rel_to_sec17_5d
rel_to_sec17_5d_cs_bottom20_flag
rel_to_sec17_5d_cs_pct
rel_to_sec17_5d_cs_rank
rel_to_sec17_5d_cs_top20_flag
rel_to_sec17_5d_outlier_flag
rel_to_sec17_5d_roll_mean_20d
rel_to_sec17_5d_roll_std_20d
rel_to_sec17_5d_sector_mean
rel_to_sec17_5d_sector_rel
rel_to_sec17_5d_zscore_20d
```

### SSP Features (11 columns)

```
ssp_ratio_component
ssp_ratio_component_cs_bottom20_flag
ssp_ratio_component_cs_pct
ssp_ratio_component_cs_rank
ssp_ratio_component_cs_top20_flag
ssp_ratio_component_outlier_flag
ssp_ratio_component_roll_mean_20d
ssp_ratio_component_roll_std_20d
ssp_ratio_component_sector_mean
ssp_ratio_component_sector_rel
ssp_ratio_component_zscore_20d
```

### SECTOR Features (11 columns)

```
sector_short_ratio_z20
sector_short_ratio_z20_cs_bottom20_flag
sector_short_ratio_z20_cs_pct
sector_short_ratio_z20_cs_rank
sector_short_ratio_z20_cs_top20_flag
sector_short_ratio_z20_outlier_flag
sector_short_ratio_z20_roll_mean_20d
sector_short_ratio_z20_roll_std_20d
sector_short_ratio_z20_sector_mean
sector_short_ratio_z20_sector_rel
sector_short_ratio_z20_zscore_20d
```

### LIMIT Features (11 columns)

```
limit_up_flag_lag1
limit_up_flag_lag1_cs_bottom20_flag
limit_up_flag_lag1_cs_pct
limit_up_flag_lag1_cs_rank
limit_up_flag_lag1_cs_top20_flag
limit_up_flag_lag1_outlier_flag
limit_up_flag_lag1_roll_mean_20d
limit_up_flag_lag1_roll_std_20d
limit_up_flag_lag1_sector_mean
limit_up_flag_lag1_sector_rel
limit_up_flag_lag1_zscore_20d
```

### SQUEEZE Features (11 columns)

```
squeeze_risk
squeeze_risk_cs_bottom20_flag
squeeze_risk_cs_pct
squeeze_risk_cs_rank
squeeze_risk_cs_top20_flag
squeeze_risk_outlier_flag
squeeze_risk_roll_mean_20d
squeeze_risk_roll_std_20d
squeeze_risk_sector_mean
squeeze_risk_sector_rel
squeeze_risk_zscore_20d
```

### MARGIN Features (11 columns)

```
margin_pain_index
margin_pain_index_cs_bottom20_flag
margin_pain_index_cs_pct
margin_pain_index_cs_rank
margin_pain_index_cs_top20_flag
margin_pain_index_outlier_flag
margin_pain_index_roll_mean_20d
margin_pain_index_roll_std_20d
margin_pain_index_sector_mean
margin_pain_index_sector_rel
margin_pain_index_zscore_20d
```

### PREE Features (11 columns)

```
preE_risk_score
preE_risk_score_cs_bottom20_flag
preE_risk_score_cs_pct
preE_risk_score_cs_rank
preE_risk_score_cs_top20_flag
preE_risk_score_outlier_flag
preE_risk_score_roll_mean_20d
preE_risk_score_roll_std_20d
preE_risk_score_sector_mean
preE_risk_score_sector_rel
preE_risk_score_zscore_20d
```

### COMPANYNAMEENGLISH Features (10 columns)

```
CompanyNameEnglish_cs_bottom20_flag
CompanyNameEnglish_cs_pct
CompanyNameEnglish_cs_rank
CompanyNameEnglish_cs_top20_flag
CompanyNameEnglish_outlier_flag
CompanyNameEnglish_roll_mean_20d
CompanyNameEnglish_roll_std_20d
CompanyNameEnglish_sector_mean
CompanyNameEnglish_sector_rel
CompanyNameEnglish_zscore_20d
```

### EARNINGS Features (8 columns)

```
earnings_event_date
earnings_recent_1d
earnings_recent_3d
earnings_recent_5d
earnings_today
earnings_upcoming_1d
earnings_upcoming_3d
earnings_upcoming_5d
```

## Top 50 Highest NULL Rate Columns

| Rank | Column | NULL Rate | NULL Count | Data Type |
|------|--------|-----------|------------|----------|
| 1 | `dmi_reason_code` | 100.00% | 725,839 | String |
| 2 | `sec17_ret_5d_eq` | 100.00% | 725,839 | Float64 |
| 3 | `rel_to_sec17_5d` | 100.00% | 725,839 | Float64 |
| 4 | `sec17_mom_20` | 100.00% | 725,839 | Float64 |
| 5 | `fs_revenue_ttm` | 100.00% | 725,839 | Float64 |
| 6 | `fs_op_profit_ttm` | 100.00% | 725,839 | Float64 |
| 7 | `fs_net_income_ttm` | 100.00% | 725,839 | Float64 |
| 8 | `fs_cfo_ttm` | 100.00% | 725,839 | Float64 |
| 9 | `fs_capex_ttm` | 100.00% | 725,839 | Float64 |
| 10 | `fs_fcf_ttm` | 100.00% | 725,839 | Float64 |
| 11 | `fs_sales_yoy` | 100.00% | 725,839 | Float64 |
| 12 | `fs_op_margin` | 100.00% | 725,839 | Float64 |
| 13 | `fs_net_margin` | 100.00% | 725,839 | Float64 |
| 14 | `fs_roe_ttm` | 100.00% | 725,839 | Float64 |
| 15 | `fs_roa_ttm` | 100.00% | 725,839 | Float64 |
| 16 | `fs_accruals_ttm` | 100.00% | 725,839 | Float64 |
| 17 | `fs_cfo_to_ni` | 100.00% | 725,839 | Float64 |
| 18 | `fs_ttm_sales` | 100.00% | 725,839 | Float64 |
| 19 | `fs_ttm_op_profit` | 100.00% | 725,839 | Float64 |
| 20 | `fs_ttm_net_income` | 100.00% | 725,839 | Float64 |
| 21 | `fs_ttm_cfo` | 100.00% | 725,839 | Float64 |
| 22 | `fs_ttm_op_margin` | 100.00% | 725,839 | Float64 |
| 23 | `fs_ttm_cfo_margin` | 100.00% | 725,839 | Float64 |
| 24 | `fs_equity_ratio` | 100.00% | 725,839 | Float64 |
| 25 | `fs_net_cash_ratio` | 100.00% | 725,839 | Float64 |
| 26 | `fs_yoy_ttm_sales` | 100.00% | 725,839 | Float64 |
| 27 | `fs_yoy_ttm_op_profit` | 100.00% | 725,839 | Float64 |
| 28 | `fs_yoy_ttm_net_income` | 100.00% | 725,839 | Float64 |
| 29 | `fs_accruals` | 100.00% | 725,839 | Float64 |
| 30 | `fs_days_to_next` | 100.00% | 725,839 | Int32 |
| 31 | `div_amount_next` | 100.00% | 725,839 | Float64 |
| 32 | `div_amount_12m` | 100.00% | 725,839 | Float64 |
| 33 | `earnings_event_date` | 100.00% | 725,839 | Date |
| 34 | `days_to_earnings` | 100.00% | 725,839 | Int32 |
| 35 | `earnings_today` | 100.00% | 725,839 | Int8 |
| 36 | `earnings_upcoming_1d` | 100.00% | 725,839 | Int8 |
| 37 | `earnings_upcoming_3d` | 100.00% | 725,839 | Int8 |
| 38 | `earnings_upcoming_5d` | 100.00% | 725,839 | Int8 |
| 39 | `earnings_recent_1d` | 100.00% | 725,839 | Int8 |
| 40 | `earnings_recent_3d` | 100.00% | 725,839 | Int8 |
| 41 | `earnings_recent_5d` | 100.00% | 725,839 | Int8 |
| 42 | `is_E_pm1` | 100.00% | 725,839 | Int8 |
| 43 | `is_E_0` | 100.00% | 725,839 | Int8 |
| 44 | `is_E_pp1` | 100.00% | 725,839 | Int8 |
| 45 | `is_E_pp3` | 100.00% | 725,839 | Int8 |
| 46 | `is_E_pp5` | 100.00% | 725,839 | Int8 |
| 47 | `CompanyNameEnglish` | 100.00% | 725,839 | Float64 |
| 48 | `days_since_market_change` | 100.00% | 725,839 | Int32 |
| 49 | `market_changed_5d` | 100.00% | 725,839 | Int8 |
| 50 | `is_prime_x_dv_z20` | 100.00% | 725,839 | Float64 |

## Root Cause Analysis

### 1. Financial Statements (fs_*) - 100% NULL

**Affected Features**: ~300+ columns
- TTM metrics (revenue, profit, cash flow)
- Growth rates (YoY)
- Profitability ratios (ROE, ROA, margins)
- Balance sheet ratios

**Likely Causes**:
- ✅ **As-of join logic issue**: T+1 cutoff may be excluding all 2025 data
- ❌ API data unavailability (JQuants statements should have 2025 data)
- ❌ Date range mismatch in fetcher

**Fix Priority**: 🔴 CRITICAL

### 2. Sector 17 Features (sec17_*) - 100% NULL

**Affected Features**: ~20 columns
- Returns relative to Sector 17
- Momentum vs Sector 17

**Likely Causes**:
- Sector 17 classification may not exist in 2025 data
- Mapping table missing for 2025

**Fix Priority**: 🟡 MEDIUM

### 3. Daily Margin Interest (dmi_*) - 100% NULL

**Affected Features**: ~5 columns
- Daily margin interest data

**Likely Causes**:
- API endpoint may not have 2025 data yet
- Feature extraction logic issue

**Fix Priority**: 🟡 MEDIUM

### 4. Dividend Features (div_*) - 100% NULL

**Affected Features**: ~10 columns
- Next dividend amount
- 12-month dividend

**Likely Causes**:
- Dividend data not available for 2025
- Join logic issue

**Fix Priority**: 🟢 LOW (expected for forward-looking features)

### 5. Earnings Events (earnings_*) - 100% NULL

**Affected Features**: ~15 columns
- Earnings event dates
- Days to earnings
- Upcoming/recent flags

**Likely Causes**:
- Earnings calendar data not populated for 2025
- Feature extraction logic issue

**Fix Priority**: 🟡 MEDIUM

## Immediate Action Items

1. **Investigate financial statements as-of join logic** (`breakdown_asof.py`)
   - Check T+1 cutoff implementation
   - Verify 2025 data exists in raw statements
   - Test join with relaxed date constraints

2. **Check sector mapping for 2025**
   - Verify Sector 17 classification exists
   - Update mapping if needed

3. **Validate data availability**
   - Check raw data files for 2025
   - Confirm API returns 2025 data

4. **Compare with 2024 data**
   - Load 2024 dataset and compare NULL rates
   - Identify which features had data in 2024 but not 2025

## Next Steps

```bash
# 1. Check raw financial statements data
ls -lh gogooku5/output_g5/raw/statements/*.parquet

# 2. Verify 2025 data in statements
python3 -c "import polars as pl; df = pl.read_parquet('gogooku5/output_g5/raw/statements/*.parquet'); print(df.filter(pl.col('Date') >= pl.date(2025, 1, 1)).head())"

# 3. Compare 2024 vs 2025 NULL rates
# (Create comparison script)

# 4. Review as-of join logic
cat gogooku5/data/src/builder/features/fundamentals/breakdown_asof.py | grep -A 20 "def apply_asof_join"
```

---
**Report Generated**: 2025-11-18 06:45 UTC
**Analysis Tool**: Polars {pl.__version__}

## Summary

**Status**: ⚠️ 737 columns (17.5%) are 100% NULL in 2025 dataset

**Critical Issues**:
1. **Financial Statements (fs_*)**: ~300+ columns 100% NULL
2. **Sector 17 Features (sec17_*)**: ~20 columns 100% NULL  
3. **Daily Margin Interest (dmi_*)**: ~5 columns 100% NULL
4. **Dividend Features (div_*)**: ~10 columns 100% NULL
5. **Earnings Events (earnings_*)**: ~15 columns 100% NULL

**Expected Causes**:
- As-of join logic excluding 2025 data (T+1 cutoff issue)
- Sector mapping tables missing for 2025
- Raw data not yet available for some features

**Next Actions**:
1. Check raw data availability for 2025
2. Review as-of join implementation in `breakdown_asof.py`
3. Compare 2024 vs 2025 NULL rates
4. Validate sector mappings

---

