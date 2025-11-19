# NULL Columns Report - gogooku5

**Generated**: 2025-11-17 09:28:29

## Summary

| Dataset | Total Rows | Total Cols | All-NULL (100%) | High-NULL (>95%) |
|---------|------------|------------|-----------------|------------------|
| 2024 Dataset | 940,889 | 2,775 | 529 | 41 |
| 2025 Dataset | 809,306 | 2,775 | 564 | 34 |
| 2024-2025 Combined (APEX) | 1,750,195 | 2,775 | 526 | 32 |

## Common All-NULL Columns Across All Datasets

**Count**: 526 columns


### COMPANYNAMEENGLISH Features (7 columns)

- `CompanyNameEnglish`
- `CompanyNameEnglish_outlier_flag`
- `CompanyNameEnglish_roll_mean_20d`
- `CompanyNameEnglish_roll_std_20d`
- `CompanyNameEnglish_sector_mean`
- `CompanyNameEnglish_sector_rel`
- `CompanyNameEnglish_zscore_20d`

### DISCLOSEDDATE Features (1 columns)

- `DisclosedDate`

### ALPHA60 Features (6 columns)

- `alpha60_topix_outlier_flag`
- `alpha60_topix_roll_mean_20d`
- `alpha60_topix_roll_std_20d`
- `alpha60_topix_sector_mean`
- `alpha60_topix_sector_rel`
- `alpha60_topix_zscore_20d`

### BASIS Features (7 columns)

- `basis_gate`
- `basis_gate_outlier_flag`
- `basis_gate_roll_mean_20d`
- `basis_gate_roll_std_20d`
- `basis_gate_sector_mean`
- `basis_gate_sector_rel`
- `basis_gate_zscore_20d`

### BD Features (1 columns)

- `bd_net_mc`

### BETA60 Features (6 columns)

- `beta60_topix_outlier_flag`
- `beta60_topix_roll_mean_20d`
- `beta60_topix_roll_std_20d`
- `beta60_topix_sector_mean`
- `beta60_topix_sector_rel`
- `beta60_topix_zscore_20d`

### BUYBACK Features (1 columns)

- `buyback_flag`

### CROWDING Features (7 columns)

- `crowding_score`
- `crowding_score_outlier_flag`
- `crowding_score_roll_mean_20d`
- `crowding_score_roll_std_20d`
- `crowding_score_sector_mean`
- `crowding_score_sector_rel`
- `crowding_score_zscore_20d`

### DAYS Features (22 columns)

- `days_since_market_change`
- `days_since_market_change_outlier_flag`
- `days_since_market_change_roll_mean_20d`
- `days_since_market_change_roll_std_20d`
- `days_since_market_change_sector_mean`
- `days_since_market_change_sector_rel`
- `days_since_market_change_zscore_20d`
- `days_since_sector33_change_outlier_flag`
- `days_since_sector33_change_roll_mean_20d`
- `days_since_sector33_change_roll_std_20d`
- `days_since_sector33_change_zscore_20d`
- `days_since_sq_outlier_flag`
- `days_since_sq_zscore_20d`
- `days_to_earnings`
- `days_to_earnings_outlier_flag`
- `days_to_earnings_roll_mean_20d`
- `days_to_earnings_roll_std_20d`
- `days_to_earnings_sector_mean`
- `days_to_earnings_sector_rel`
- `days_to_earnings_zscore_20d`
- `days_to_sq_outlier_flag`
- `days_to_sq_zscore_20d`

### DILUTION Features (1 columns)

- `dilution_flag`

### DIV Features (13 columns)

- `div_amount_12m`
- `div_amount_next`
- `div_days_since_ex`
- `div_days_to_ex`
- `div_dy_12m`
- `div_ex_cycle_z`
- `div_ex_drop_expected`
- `div_ex_gap_miss`
- `div_ex_gap_theo`
- `div_staleness_bd`
- `div_staleness_days`
- `div_yield_12m`
- `div_yield_ttm`

### DMI Features (1 columns)

- `dmi_reason_code`

### EARNINGS Features (8 columns)

- `earnings_event_date`
- `earnings_recent_1d`
- `earnings_recent_3d`
- `earnings_recent_5d`
- `earnings_today`
- `earnings_upcoming_1d`
- `earnings_upcoming_3d`
- `earnings_upcoming_5d`

### FLOAT Features (21 columns)

- `float_turnover_pct`
- `float_turnover_pct_outlier_flag`
- `float_turnover_pct_roc_5d`
- `float_turnover_pct_roc_5d_outlier_flag`
- `float_turnover_pct_roc_5d_roll_mean_20d`
- `float_turnover_pct_roc_5d_roll_std_20d`
- `float_turnover_pct_roc_5d_sector_mean`
- `float_turnover_pct_roc_5d_sector_rel`
- `float_turnover_pct_roc_5d_zscore_20d`
- `float_turnover_pct_roll_mean_20d`
- `float_turnover_pct_roll_std_20d`
- `float_turnover_pct_sector_mean`
- `float_turnover_pct_sector_rel`
- `float_turnover_pct_z20`
- `float_turnover_pct_z20_outlier_flag`
- `float_turnover_pct_z20_roll_mean_20d`
- `float_turnover_pct_z20_roll_std_20d`
- `float_turnover_pct_z20_sector_mean`
- `float_turnover_pct_z20_sector_rel`
- `float_turnover_pct_z20_zscore_20d`
- `float_turnover_pct_zscore_20d`

### FS Features (47 columns)

- `fs_accruals`
- `fs_accruals_ttm`
- `fs_average_shares`
- `fs_capex_ttm`
- `fs_cfo_to_ni`
- `fs_cfo_ttm`
- `fs_consolidated_flag`
- `fs_days_since`
- `fs_days_to_next`
- `fs_doc_family_1Q`
- `fs_doc_family_2Q`
- `fs_doc_family_3Q`
- `fs_doc_family_FY`
- `fs_equity_ratio`
- `fs_fcf_ttm`
- `fs_guidance_revision_flag`
- `fs_is_valid`
- `fs_lag_days`
- `fs_net_cash_ratio`
- `fs_net_income_ttm`
- `fs_net_margin`
- `fs_observation_count`
- `fs_op_margin`
- `fs_op_profit_ttm`
- `fs_revenue_ttm`
- `fs_roa_ttm`
- `fs_roe_ttm`
- `fs_sales_yoy`
- `fs_shares_outstanding`
- `fs_staleness_bd`
- `fs_standard_Foreign`
- `fs_standard_IFRS`
- `fs_standard_JGAAP`
- `fs_standard_JMIS`
- `fs_standard_US`
- `fs_ttm_cfo`
- `fs_ttm_cfo_margin`
- `fs_ttm_net_income`
- `fs_ttm_op_margin`
- `fs_ttm_op_profit`
- `fs_ttm_sales`
- `fs_window_e_pm1`
- `fs_window_e_pp3`
- `fs_window_e_pp5`
- `fs_yoy_ttm_net_income`
- `fs_yoy_ttm_op_profit`
- `fs_yoy_ttm_sales`

### GAP Features (2 columns)

- `gap_atr`
- `gap_predictor`

### IDXOPT Features (14 columns)

- `idxopt_vrp_gap`
- `idxopt_vrp_gap_outlier_flag`
- `idxopt_vrp_gap_roll_mean_20d`
- `idxopt_vrp_gap_roll_std_20d`
- `idxopt_vrp_gap_sector_mean`
- `idxopt_vrp_gap_sector_rel`
- `idxopt_vrp_gap_zscore_20d`
- `idxopt_vrp_ratio`
- `idxopt_vrp_ratio_outlier_flag`
- `idxopt_vrp_ratio_roll_mean_20d`
- `idxopt_vrp_ratio_roll_std_20d`
- `idxopt_vrp_ratio_sector_mean`
- `idxopt_vrp_ratio_sector_rel`
- `idxopt_vrp_ratio_zscore_20d`

### IS Features (20 columns)

- `is_E_0`
- `is_E_pm1`
- `is_E_pp1`
- `is_E_pp3`
- `is_E_pp5`
- `is_fs_valid`
- `is_growth_x_dv_z20`
- `is_growth_x_dv_z20_outlier_flag`
- `is_growth_x_dv_z20_roll_mean_20d`
- `is_growth_x_dv_z20_roll_std_20d`
- `is_growth_x_dv_z20_sector_mean`
- `is_growth_x_dv_z20_sector_rel`
- `is_growth_x_dv_z20_zscore_20d`
- `is_prime_x_dv_z20`
- `is_prime_x_dv_z20_outlier_flag`
- `is_prime_x_dv_z20_roll_mean_20d`
- `is_prime_x_dv_z20_roll_std_20d`
- `is_prime_x_dv_z20_sector_mean`
- `is_prime_x_dv_z20_sector_rel`
- `is_prime_x_dv_z20_zscore_20d`

### LIMIT Features (7 columns)

- `limit_up_flag_lag1`
- `limit_up_flag_lag1_outlier_flag`
- `limit_up_flag_lag1_roll_mean_20d`
- `limit_up_flag_lag1_roll_std_20d`
- `limit_up_flag_lag1_sector_mean`
- `limit_up_flag_lag1_sector_rel`
- `limit_up_flag_lag1_zscore_20d`

### MARGIN Features (28 columns)

- `margin_long_pct_float`
- `margin_long_pct_float_outlier_flag`
- `margin_long_pct_float_roc_5d`
- `margin_long_pct_float_roc_5d_outlier_flag`
- `margin_long_pct_float_roc_5d_roll_mean_20d`
- `margin_long_pct_float_roc_5d_roll_std_20d`
- `margin_long_pct_float_roc_5d_sector_mean`
- `margin_long_pct_float_roc_5d_sector_rel`
- `margin_long_pct_float_roc_5d_zscore_20d`
- `margin_long_pct_float_roll_mean_20d`
- `margin_long_pct_float_roll_std_20d`
- `margin_long_pct_float_sector_mean`
- `margin_long_pct_float_sector_rel`
- `margin_long_pct_float_z20`
- `margin_long_pct_float_z20_outlier_flag`
- `margin_long_pct_float_z20_roll_mean_20d`
- `margin_long_pct_float_z20_roll_std_20d`
- `margin_long_pct_float_z20_sector_mean`
- `margin_long_pct_float_z20_sector_rel`
- `margin_long_pct_float_z20_zscore_20d`
- `margin_long_pct_float_zscore_20d`
- `margin_pain_index`
- `margin_pain_index_outlier_flag`
- `margin_pain_index_roll_mean_20d`
- `margin_pain_index_roll_std_20d`
- `margin_pain_index_sector_mean`
- `margin_pain_index_sector_rel`
- `margin_pain_index_zscore_20d`

### MARKET Features (1 columns)

- `market_changed_5d`

### MKT Features (170 columns)

- `mkt_flow_divergence_foreigners_vs_individuals`
- `mkt_flow_divergence_foreigners_vs_individuals_outlier_flag`
- `mkt_flow_divergence_foreigners_vs_individuals_roll_mean_20d`
- `mkt_flow_divergence_foreigners_vs_individuals_roll_std_20d`
- `mkt_flow_divergence_foreigners_vs_individuals_sector_mean`
- `mkt_flow_divergence_foreigners_vs_individuals_sector_rel`
- `mkt_flow_divergence_foreigners_vs_individuals_zscore_20d`
- `mkt_flow_flow_foreigners_net_ratio_z52`
- `mkt_flow_flow_foreigners_net_ratio_z52_outlier_flag`
- `mkt_flow_flow_foreigners_net_ratio_z52_roll_mean_20d`
- `mkt_flow_flow_foreigners_net_ratio_z52_roll_std_20d`
- `mkt_flow_flow_foreigners_net_ratio_z52_sector_mean`
- `mkt_flow_flow_foreigners_net_ratio_z52_sector_rel`
- `mkt_flow_flow_foreigners_net_ratio_z52_zscore_20d`
- `mkt_flow_flow_individuals_net_ratio_z52`
- `mkt_flow_flow_individuals_net_ratio_z52_outlier_flag`
- `mkt_flow_flow_individuals_net_ratio_z52_roll_mean_20d`
- `mkt_flow_flow_individuals_net_ratio_z52_roll_std_20d`
- `mkt_flow_flow_individuals_net_ratio_z52_sector_mean`
- `mkt_flow_flow_individuals_net_ratio_z52_sector_rel`
- `mkt_flow_flow_individuals_net_ratio_z52_zscore_20d`
- `mkt_flow_flow_investment_trusts_net_ratio_z52`
- `mkt_flow_flow_investment_trusts_net_ratio_z52_outlier_flag`
- `mkt_flow_flow_investment_trusts_net_ratio_z52_roll_mean_20d`
- `mkt_flow_flow_investment_trusts_net_ratio_z52_roll_std_20d`
- `mkt_flow_flow_investment_trusts_net_ratio_z52_sector_mean`
- `mkt_flow_flow_investment_trusts_net_ratio_z52_sector_rel`
- `mkt_flow_flow_investment_trusts_net_ratio_z52_zscore_20d`
- `mkt_flow_flow_trust_banks_net_ratio_z52`
- `mkt_flow_flow_trust_banks_net_ratio_z52_outlier_flag`
- `mkt_flow_flow_trust_banks_net_ratio_z52_roll_mean_20d`
- `mkt_flow_flow_trust_banks_net_ratio_z52_roll_std_20d`
- `mkt_flow_flow_trust_banks_net_ratio_z52_sector_mean`
- `mkt_flow_flow_trust_banks_net_ratio_z52_sector_rel`
- `mkt_flow_flow_trust_banks_net_ratio_z52_zscore_20d`
- `mkt_flow_foreigners_net`
- `mkt_flow_foreigners_net_outlier_flag`
- `mkt_flow_foreigners_net_ratio`
- `mkt_flow_foreigners_net_ratio_outlier_flag`
- `mkt_flow_foreigners_net_ratio_roll_mean_20d`
- `mkt_flow_foreigners_net_ratio_roll_std_20d`
- `mkt_flow_foreigners_net_ratio_sector_mean`
- `mkt_flow_foreigners_net_ratio_sector_rel`
- `mkt_flow_foreigners_net_ratio_turn_flag`
- `mkt_flow_foreigners_net_ratio_wow`
- `mkt_flow_foreigners_net_ratio_wow_outlier_flag`
- `mkt_flow_foreigners_net_ratio_wow_roll_mean_20d`
- `mkt_flow_foreigners_net_ratio_wow_roll_std_20d`
- `mkt_flow_foreigners_net_ratio_wow_sector_mean`
- `mkt_flow_foreigners_net_ratio_wow_sector_rel`
- `mkt_flow_foreigners_net_ratio_wow_zscore_20d`
- `mkt_flow_foreigners_net_ratio_z13`
- `mkt_flow_foreigners_net_ratio_z13_outlier_flag`
- `mkt_flow_foreigners_net_ratio_z13_roll_mean_20d`
- `mkt_flow_foreigners_net_ratio_z13_roll_std_20d`
- `mkt_flow_foreigners_net_ratio_z13_sector_mean`
- `mkt_flow_foreigners_net_ratio_z13_sector_rel`
- `mkt_flow_foreigners_net_ratio_z13_zscore_20d`
- `mkt_flow_foreigners_net_ratio_z52`
- `mkt_flow_foreigners_net_ratio_z52_outlier_flag`
- `mkt_flow_foreigners_net_ratio_z52_roll_mean_20d`
- `mkt_flow_foreigners_net_ratio_z52_roll_std_20d`
- `mkt_flow_foreigners_net_ratio_z52_sector_mean`
- `mkt_flow_foreigners_net_ratio_z52_sector_rel`
- `mkt_flow_foreigners_net_ratio_z52_zscore_20d`
- `mkt_flow_foreigners_net_ratio_zscore_20d`
- `mkt_flow_foreigners_net_roll_mean_20d`
- `mkt_flow_foreigners_net_roll_std_20d`
- `mkt_flow_foreigners_net_sector_mean`
- `mkt_flow_foreigners_net_sector_rel`
- `mkt_flow_foreigners_net_zscore_20d`
- `mkt_flow_individuals_net`
- `mkt_flow_individuals_net_outlier_flag`
- `mkt_flow_individuals_net_ratio`
- `mkt_flow_individuals_net_ratio_outlier_flag`
- `mkt_flow_individuals_net_ratio_roll_mean_20d`
- `mkt_flow_individuals_net_ratio_roll_std_20d`
- `mkt_flow_individuals_net_ratio_sector_mean`
- `mkt_flow_individuals_net_ratio_sector_rel`
- `mkt_flow_individuals_net_ratio_turn_flag`
- `mkt_flow_individuals_net_ratio_wow`
- `mkt_flow_individuals_net_ratio_wow_outlier_flag`
- `mkt_flow_individuals_net_ratio_wow_roll_mean_20d`
- `mkt_flow_individuals_net_ratio_wow_roll_std_20d`
- `mkt_flow_individuals_net_ratio_wow_sector_mean`
- `mkt_flow_individuals_net_ratio_wow_sector_rel`
- `mkt_flow_individuals_net_ratio_wow_zscore_20d`
- `mkt_flow_individuals_net_ratio_z13`
- `mkt_flow_individuals_net_ratio_z13_outlier_flag`
- `mkt_flow_individuals_net_ratio_z13_roll_mean_20d`
- `mkt_flow_individuals_net_ratio_z13_roll_std_20d`
- `mkt_flow_individuals_net_ratio_z13_sector_mean`
- `mkt_flow_individuals_net_ratio_z13_sector_rel`
- `mkt_flow_individuals_net_ratio_z13_zscore_20d`
- `mkt_flow_individuals_net_ratio_z52`
- `mkt_flow_individuals_net_ratio_z52_outlier_flag`
- `mkt_flow_individuals_net_ratio_z52_roll_mean_20d`
- `mkt_flow_individuals_net_ratio_z52_roll_std_20d`
- `mkt_flow_individuals_net_ratio_z52_sector_mean`
- `mkt_flow_individuals_net_ratio_z52_sector_rel`
- `mkt_flow_individuals_net_ratio_z52_zscore_20d`
- `mkt_flow_individuals_net_ratio_zscore_20d`
- `mkt_flow_individuals_net_roll_mean_20d`
- `mkt_flow_individuals_net_roll_std_20d`
- `mkt_flow_individuals_net_sector_mean`
- `mkt_flow_individuals_net_sector_rel`
- `mkt_flow_individuals_net_zscore_20d`
- `mkt_flow_investment_trusts_net`
- `mkt_flow_investment_trusts_net_outlier_flag`
- `mkt_flow_investment_trusts_net_ratio`
- `mkt_flow_investment_trusts_net_ratio_outlier_flag`
- `mkt_flow_investment_trusts_net_ratio_roll_mean_20d`
- `mkt_flow_investment_trusts_net_ratio_roll_std_20d`
- `mkt_flow_investment_trusts_net_ratio_sector_mean`
- `mkt_flow_investment_trusts_net_ratio_sector_rel`
- `mkt_flow_investment_trusts_net_ratio_z13`
- `mkt_flow_investment_trusts_net_ratio_z13_outlier_flag`
- `mkt_flow_investment_trusts_net_ratio_z13_roll_mean_20d`
- `mkt_flow_investment_trusts_net_ratio_z13_roll_std_20d`
- `mkt_flow_investment_trusts_net_ratio_z13_sector_mean`
- `mkt_flow_investment_trusts_net_ratio_z13_sector_rel`
- `mkt_flow_investment_trusts_net_ratio_z13_zscore_20d`
- `mkt_flow_investment_trusts_net_ratio_z52`
- `mkt_flow_investment_trusts_net_ratio_z52_outlier_flag`
- `mkt_flow_investment_trusts_net_ratio_z52_roll_mean_20d`
- `mkt_flow_investment_trusts_net_ratio_z52_roll_std_20d`
- `mkt_flow_investment_trusts_net_ratio_z52_sector_mean`
- `mkt_flow_investment_trusts_net_ratio_z52_sector_rel`
- `mkt_flow_investment_trusts_net_ratio_z52_zscore_20d`
- `mkt_flow_investment_trusts_net_ratio_zscore_20d`
- `mkt_flow_investment_trusts_net_roll_mean_20d`
- `mkt_flow_investment_trusts_net_roll_std_20d`
- `mkt_flow_investment_trusts_net_sector_mean`
- `mkt_flow_investment_trusts_net_sector_rel`
- `mkt_flow_investment_trusts_net_zscore_20d`
- `mkt_flow_total_net`
- `mkt_flow_total_net_outlier_flag`
- `mkt_flow_total_net_roll_mean_20d`
- `mkt_flow_total_net_roll_std_20d`
- `mkt_flow_total_net_sector_mean`
- `mkt_flow_total_net_sector_rel`
- `mkt_flow_total_net_zscore_20d`
- `mkt_flow_trust_banks_net`
- `mkt_flow_trust_banks_net_outlier_flag`
- `mkt_flow_trust_banks_net_ratio`
- `mkt_flow_trust_banks_net_ratio_outlier_flag`
- `mkt_flow_trust_banks_net_ratio_roll_mean_20d`
- `mkt_flow_trust_banks_net_ratio_roll_std_20d`
- `mkt_flow_trust_banks_net_ratio_sector_mean`
- `mkt_flow_trust_banks_net_ratio_sector_rel`
- `mkt_flow_trust_banks_net_ratio_z13`
- `mkt_flow_trust_banks_net_ratio_z13_outlier_flag`
- `mkt_flow_trust_banks_net_ratio_z13_roll_mean_20d`
- `mkt_flow_trust_banks_net_ratio_z13_roll_std_20d`
- `mkt_flow_trust_banks_net_ratio_z13_sector_mean`
- `mkt_flow_trust_banks_net_ratio_z13_sector_rel`
- `mkt_flow_trust_banks_net_ratio_z13_zscore_20d`
- `mkt_flow_trust_banks_net_ratio_z52`
- `mkt_flow_trust_banks_net_ratio_z52_outlier_flag`
- `mkt_flow_trust_banks_net_ratio_z52_roll_mean_20d`
- `mkt_flow_trust_banks_net_ratio_z52_roll_std_20d`
- `mkt_flow_trust_banks_net_ratio_z52_sector_mean`
- `mkt_flow_trust_banks_net_ratio_z52_sector_rel`
- `mkt_flow_trust_banks_net_ratio_z52_zscore_20d`
- `mkt_flow_trust_banks_net_ratio_zscore_20d`
- `mkt_flow_trust_banks_net_roll_mean_20d`
- `mkt_flow_trust_banks_net_roll_std_20d`
- `mkt_flow_trust_banks_net_sector_mean`
- `mkt_flow_trust_banks_net_sector_rel`
- `mkt_flow_trust_banks_net_zscore_20d`

### MPI Features (28 columns)

- `mpi_dist_to_limit`
- `mpi_dist_to_limit_outlier_flag`
- `mpi_dist_to_limit_roll_mean_20d`
- `mpi_dist_to_limit_roll_std_20d`
- `mpi_dist_to_limit_sector_mean`
- `mpi_dist_to_limit_sector_rel`
- `mpi_dist_to_limit_z20`
- `mpi_dist_to_limit_z20_outlier_flag`
- `mpi_dist_to_limit_z20_roll_mean_20d`
- `mpi_dist_to_limit_z20_roll_std_20d`
- `mpi_dist_to_limit_z20_sector_mean`
- `mpi_dist_to_limit_z20_sector_rel`
- `mpi_dist_to_limit_z20_zscore_20d`
- `mpi_dist_to_limit_zscore_20d`
- `mpi_drawdown`
- `mpi_drawdown_outlier_flag`
- `mpi_drawdown_roll_mean_20d`
- `mpi_drawdown_roll_std_20d`
- `mpi_drawdown_sector_mean`
- `mpi_drawdown_sector_rel`
- `mpi_drawdown_z20`
- `mpi_drawdown_z20_outlier_flag`
- `mpi_drawdown_z20_roll_mean_20d`
- `mpi_drawdown_z20_roll_std_20d`
- `mpi_drawdown_z20_sector_mean`
- `mpi_drawdown_z20_sector_rel`
- `mpi_drawdown_z20_zscore_20d`
- `mpi_drawdown_zscore_20d`

### PREE Features (21 columns)

- `preE_margin_diff`
- `preE_margin_diff_outlier_flag`
- `preE_margin_diff_roll_mean_20d`
- `preE_margin_diff_roll_std_20d`
- `preE_margin_diff_sector_mean`
- `preE_margin_diff_sector_rel`
- `preE_margin_diff_z20`
- `preE_margin_diff_z20_outlier_flag`
- `preE_margin_diff_z20_roll_mean_20d`
- `preE_margin_diff_z20_roll_std_20d`
- `preE_margin_diff_z20_sector_mean`
- `preE_margin_diff_z20_sector_rel`
- `preE_margin_diff_z20_zscore_20d`
- `preE_margin_diff_zscore_20d`
- `preE_risk_score`
- `preE_risk_score_outlier_flag`
- `preE_risk_score_roll_mean_20d`
- `preE_risk_score_roll_std_20d`
- `preE_risk_score_sector_mean`
- `preE_risk_score_sector_rel`
- `preE_risk_score_zscore_20d`

### REL Features (7 columns)

- `rel_to_sec17_5d`
- `rel_to_sec17_5d_outlier_flag`
- `rel_to_sec17_5d_roll_mean_20d`
- `rel_to_sec17_5d_roll_std_20d`
- `rel_to_sec17_5d_sector_mean`
- `rel_to_sec17_5d_sector_rel`
- `rel_to_sec17_5d_zscore_20d`

### RET Features (1 columns)

- `ret_prev_120d`

### RQ Features (12 columns)

- `rq_63_10_outlier_flag`
- `rq_63_10_roll_mean_20d`
- `rq_63_10_roll_std_20d`
- `rq_63_10_zscore_20d`
- `rq_63_50_outlier_flag`
- `rq_63_50_roll_mean_20d`
- `rq_63_50_roll_std_20d`
- `rq_63_50_zscore_20d`
- `rq_63_90_outlier_flag`
- `rq_63_90_roll_mean_20d`
- `rq_63_90_roll_std_20d`
- `rq_63_90_zscore_20d`

### SEC17 Features (16 columns)

- `sec17_mom_20`
- `sec17_mom_20_outlier_flag`
- `sec17_mom_20_roll_mean_20d`
- `sec17_mom_20_roll_std_20d`
- `sec17_mom_20_sector_mean`
- `sec17_mom_20_sector_rel`
- `sec17_mom_20_zscore_20d`
- `sec17_ret_5d_eq`
- `sec17_ret_5d_eq_hist_vol`
- `sec17_ret_5d_eq_outlier_flag`
- `sec17_ret_5d_eq_roll_mean_20d`
- `sec17_ret_5d_eq_roll_std_20d`
- `sec17_ret_5d_eq_rolling_sharpe`
- `sec17_ret_5d_eq_sector_mean`
- `sec17_ret_5d_eq_sector_rel`
- `sec17_ret_5d_eq_zscore_20d`

### SECTOR Features (7 columns)

- `sector_short_ratio_z20`
- `sector_short_ratio_z20_outlier_flag`
- `sector_short_ratio_z20_roll_mean_20d`
- `sector_short_ratio_z20_roll_std_20d`
- `sector_short_ratio_z20_sector_mean`
- `sector_short_ratio_z20_sector_rel`
- `sector_short_ratio_z20_zscore_20d`

### SHARES Features (7 columns)

- `shares_out_delta_pct`
- `shares_out_delta_pct_outlier_flag`
- `shares_out_delta_pct_roll_mean_20d`
- `shares_out_delta_pct_roll_std_20d`
- `shares_out_delta_pct_sector_mean`
- `shares_out_delta_pct_sector_rel`
- `shares_out_delta_pct_zscore_20d`

### SQUEEZE Features (7 columns)

- `squeeze_risk`
- `squeeze_risk_outlier_flag`
- `squeeze_risk_roll_mean_20d`
- `squeeze_risk_roll_std_20d`
- `squeeze_risk_sector_mean`
- `squeeze_risk_sector_rel`
- `squeeze_risk_zscore_20d`

### SSP Features (7 columns)

- `ssp_ratio_component`
- `ssp_ratio_component_outlier_flag`
- `ssp_ratio_component_roll_mean_20d`
- `ssp_ratio_component_roll_std_20d`
- `ssp_ratio_component_sector_mean`
- `ssp_ratio_component_sector_rel`
- `ssp_ratio_component_zscore_20d`

### SUPPLY Features (1 columns)

- `supply_shock`

### WEEKLY Features (21 columns)

- `weekly_margin_long_pct_float`
- `weekly_margin_long_pct_float_outlier_flag`
- `weekly_margin_long_pct_float_roc_5d`
- `weekly_margin_long_pct_float_roc_5d_outlier_flag`
- `weekly_margin_long_pct_float_roc_5d_roll_mean_20d`
- `weekly_margin_long_pct_float_roc_5d_roll_std_20d`
- `weekly_margin_long_pct_float_roc_5d_sector_mean`
- `weekly_margin_long_pct_float_roc_5d_sector_rel`
- `weekly_margin_long_pct_float_roc_5d_zscore_20d`
- `weekly_margin_long_pct_float_roll_mean_20d`
- `weekly_margin_long_pct_float_roll_std_20d`
- `weekly_margin_long_pct_float_sector_mean`
- `weekly_margin_long_pct_float_sector_rel`
- `weekly_margin_long_pct_float_z20`
- `weekly_margin_long_pct_float_z20_outlier_flag`
- `weekly_margin_long_pct_float_z20_roll_mean_20d`
- `weekly_margin_long_pct_float_z20_roll_std_20d`
- `weekly_margin_long_pct_float_z20_sector_mean`
- `weekly_margin_long_pct_float_z20_sector_rel`
- `weekly_margin_long_pct_float_z20_zscore_20d`
- `weekly_margin_long_pct_float_zscore_20d`

---

## Detailed Analysis by Dataset


### 2024 Dataset

- **Total Rows**: 940,889
- **Total Columns**: 2,775
- **All-NULL Columns**: 529
- **High-NULL Columns (>95%)**: 41

#### All-NULL Columns by Feature Group

**COMPANYNAMEENGLISH** (7 columns):
- `CompanyNameEnglish`
- `CompanyNameEnglish_outlier_flag`
- `CompanyNameEnglish_roll_mean_20d`
- `CompanyNameEnglish_roll_std_20d`
- `CompanyNameEnglish_sector_mean`
- `CompanyNameEnglish_sector_rel`
- `CompanyNameEnglish_zscore_20d`

**DISCLOSEDDATE** (1 columns):
- `DisclosedDate`

**ALPHA60** (7 columns):
- `alpha60_topix`
- `alpha60_topix_outlier_flag`
- `alpha60_topix_roll_mean_20d`
- `alpha60_topix_roll_std_20d`
- `alpha60_topix_sector_mean`
- `alpha60_topix_sector_rel`
- `alpha60_topix_zscore_20d`

**BASIS** (7 columns):
- `basis_gate`
- `basis_gate_outlier_flag`
- `basis_gate_roll_mean_20d`
- `basis_gate_roll_std_20d`
- `basis_gate_sector_mean`
- `basis_gate_sector_rel`
- `basis_gate_zscore_20d`

**BD** (2 columns):
- `bd_net_adv60`
- `bd_net_mc`

**BETA60** (7 columns):
- `beta60_topix`
- `beta60_topix_outlier_flag`
- `beta60_topix_roll_mean_20d`
- `beta60_topix_roll_std_20d`
- `beta60_topix_sector_mean`
- `beta60_topix_sector_rel`
- `beta60_topix_zscore_20d`

**BUYBACK** (1 columns):
- `buyback_flag`

**CROWDING** (7 columns):
- `crowding_score`
- `crowding_score_outlier_flag`
- `crowding_score_roll_mean_20d`
- `crowding_score_roll_std_20d`
- `crowding_score_sector_mean`
- `crowding_score_sector_rel`
- `crowding_score_zscore_20d`

**DAYS** (22 columns):
- `days_since_market_change`
- `days_since_market_change_outlier_flag`
- `days_since_market_change_roll_mean_20d`
- `days_since_market_change_roll_std_20d`
- `days_since_market_change_sector_mean`
- `days_since_market_change_sector_rel`
- `days_since_market_change_zscore_20d`
- `days_since_sector33_change_outlier_flag`
- `days_since_sector33_change_roll_mean_20d`
- `days_since_sector33_change_roll_std_20d`
- `days_since_sector33_change_zscore_20d`
- `days_since_sq_outlier_flag`
- `days_since_sq_zscore_20d`
- `days_to_earnings`
- `days_to_earnings_outlier_flag`
- `days_to_earnings_roll_mean_20d`
- `days_to_earnings_roll_std_20d`
- `days_to_earnings_sector_mean`
- `days_to_earnings_sector_rel`
- `days_to_earnings_zscore_20d`
- `days_to_sq_outlier_flag`
- `days_to_sq_zscore_20d`

**DILUTION** (1 columns):
- `dilution_flag`

**DIV** (13 columns):
- `div_amount_12m`
- `div_amount_next`
- `div_days_since_ex`
- `div_days_to_ex`
- `div_dy_12m`
- `div_ex_cycle_z`
- `div_ex_drop_expected`
- `div_ex_gap_miss`
- `div_ex_gap_theo`
- `div_staleness_bd`
- `div_staleness_days`
- `div_yield_12m`
- `div_yield_ttm`

**DMI** (1 columns):
- `dmi_reason_code`

**EARNINGS** (8 columns):
- `earnings_event_date`
- `earnings_recent_1d`
- `earnings_recent_3d`
- `earnings_recent_5d`
- `earnings_today`
- `earnings_upcoming_1d`
- `earnings_upcoming_3d`
- `earnings_upcoming_5d`

**FLOAT** (21 columns):
- `float_turnover_pct`
- `float_turnover_pct_outlier_flag`
- `float_turnover_pct_roc_5d`
- `float_turnover_pct_roc_5d_outlier_flag`
- `float_turnover_pct_roc_5d_roll_mean_20d`
- `float_turnover_pct_roc_5d_roll_std_20d`
- `float_turnover_pct_roc_5d_sector_mean`
- `float_turnover_pct_roc_5d_sector_rel`
- `float_turnover_pct_roc_5d_zscore_20d`
- `float_turnover_pct_roll_mean_20d`
- `float_turnover_pct_roll_std_20d`
- `float_turnover_pct_sector_mean`
- `float_turnover_pct_sector_rel`
- `float_turnover_pct_z20`
- `float_turnover_pct_z20_outlier_flag`
- `float_turnover_pct_z20_roll_mean_20d`
- `float_turnover_pct_z20_roll_std_20d`
- `float_turnover_pct_z20_sector_mean`
- `float_turnover_pct_z20_sector_rel`
- `float_turnover_pct_z20_zscore_20d`
- `float_turnover_pct_zscore_20d`

**FS** (47 columns):
- `fs_accruals`
- `fs_accruals_ttm`
- `fs_average_shares`
- `fs_capex_ttm`
- `fs_cfo_to_ni`
- `fs_cfo_ttm`
- `fs_consolidated_flag`
- `fs_days_since`
- `fs_days_to_next`
- `fs_doc_family_1Q`
- `fs_doc_family_2Q`
- `fs_doc_family_3Q`
- `fs_doc_family_FY`
- `fs_equity_ratio`
- `fs_fcf_ttm`
- `fs_guidance_revision_flag`
- `fs_is_valid`
- `fs_lag_days`
- `fs_net_cash_ratio`
- `fs_net_income_ttm`
- `fs_net_margin`
- `fs_observation_count`
- `fs_op_margin`
- `fs_op_profit_ttm`
- `fs_revenue_ttm`
- `fs_roa_ttm`
- `fs_roe_ttm`
- `fs_sales_yoy`
- `fs_shares_outstanding`
- `fs_staleness_bd`
- `fs_standard_Foreign`
- `fs_standard_IFRS`
- `fs_standard_JGAAP`
- `fs_standard_JMIS`
- `fs_standard_US`
- `fs_ttm_cfo`
- `fs_ttm_cfo_margin`
- `fs_ttm_net_income`
- `fs_ttm_op_margin`
- `fs_ttm_op_profit`
- `fs_ttm_sales`
- `fs_window_e_pm1`
- `fs_window_e_pp3`
- `fs_window_e_pp5`
- `fs_yoy_ttm_net_income`
- `fs_yoy_ttm_op_profit`
- `fs_yoy_ttm_sales`

**GAP** (2 columns):
- `gap_atr`
- `gap_predictor`

**IDXOPT** (14 columns):
- `idxopt_vrp_gap`
- `idxopt_vrp_gap_outlier_flag`
- `idxopt_vrp_gap_roll_mean_20d`
- `idxopt_vrp_gap_roll_std_20d`
- `idxopt_vrp_gap_sector_mean`
- `idxopt_vrp_gap_sector_rel`
- `idxopt_vrp_gap_zscore_20d`
- `idxopt_vrp_ratio`
- `idxopt_vrp_ratio_outlier_flag`
- `idxopt_vrp_ratio_roll_mean_20d`
- `idxopt_vrp_ratio_roll_std_20d`
- `idxopt_vrp_ratio_sector_mean`
- `idxopt_vrp_ratio_sector_rel`
- `idxopt_vrp_ratio_zscore_20d`

**IS** (20 columns):
- `is_E_0`
- `is_E_pm1`
- `is_E_pp1`
- `is_E_pp3`
- `is_E_pp5`
- `is_fs_valid`
- `is_growth_x_dv_z20`
- `is_growth_x_dv_z20_outlier_flag`
- `is_growth_x_dv_z20_roll_mean_20d`
- `is_growth_x_dv_z20_roll_std_20d`
- `is_growth_x_dv_z20_sector_mean`
- `is_growth_x_dv_z20_sector_rel`
- `is_growth_x_dv_z20_zscore_20d`
- `is_prime_x_dv_z20`
- `is_prime_x_dv_z20_outlier_flag`
- `is_prime_x_dv_z20_roll_mean_20d`
- `is_prime_x_dv_z20_roll_std_20d`
- `is_prime_x_dv_z20_sector_mean`
- `is_prime_x_dv_z20_sector_rel`
- `is_prime_x_dv_z20_zscore_20d`

**LIMIT** (7 columns):
- `limit_up_flag_lag1`
- `limit_up_flag_lag1_outlier_flag`
- `limit_up_flag_lag1_roll_mean_20d`
- `limit_up_flag_lag1_roll_std_20d`
- `limit_up_flag_lag1_sector_mean`
- `limit_up_flag_lag1_sector_rel`
- `limit_up_flag_lag1_zscore_20d`

**MARGIN** (28 columns):
- `margin_long_pct_float`
- `margin_long_pct_float_outlier_flag`
- `margin_long_pct_float_roc_5d`
- `margin_long_pct_float_roc_5d_outlier_flag`
- `margin_long_pct_float_roc_5d_roll_mean_20d`
- `margin_long_pct_float_roc_5d_roll_std_20d`
- `margin_long_pct_float_roc_5d_sector_mean`
- `margin_long_pct_float_roc_5d_sector_rel`
- `margin_long_pct_float_roc_5d_zscore_20d`
- `margin_long_pct_float_roll_mean_20d`
- `margin_long_pct_float_roll_std_20d`
- `margin_long_pct_float_sector_mean`
- `margin_long_pct_float_sector_rel`
- `margin_long_pct_float_z20`
- `margin_long_pct_float_z20_outlier_flag`
- `margin_long_pct_float_z20_roll_mean_20d`
- `margin_long_pct_float_z20_roll_std_20d`
- `margin_long_pct_float_z20_sector_mean`
- `margin_long_pct_float_z20_sector_rel`
- `margin_long_pct_float_z20_zscore_20d`
- `margin_long_pct_float_zscore_20d`
- `margin_pain_index`
- `margin_pain_index_outlier_flag`
- `margin_pain_index_roll_mean_20d`
- `margin_pain_index_roll_std_20d`
- `margin_pain_index_sector_mean`
- `margin_pain_index_sector_rel`
- `margin_pain_index_zscore_20d`

**MARKET** (1 columns):
- `market_changed_5d`

**MKT** (170 columns):
- `mkt_flow_divergence_foreigners_vs_individuals`
- `mkt_flow_divergence_foreigners_vs_individuals_outlier_flag`
- `mkt_flow_divergence_foreigners_vs_individuals_roll_mean_20d`
- `mkt_flow_divergence_foreigners_vs_individuals_roll_std_20d`
- `mkt_flow_divergence_foreigners_vs_individuals_sector_mean`
- `mkt_flow_divergence_foreigners_vs_individuals_sector_rel`
- `mkt_flow_divergence_foreigners_vs_individuals_zscore_20d`
- `mkt_flow_flow_foreigners_net_ratio_z52`
- `mkt_flow_flow_foreigners_net_ratio_z52_outlier_flag`
- `mkt_flow_flow_foreigners_net_ratio_z52_roll_mean_20d`
- `mkt_flow_flow_foreigners_net_ratio_z52_roll_std_20d`
- `mkt_flow_flow_foreigners_net_ratio_z52_sector_mean`
- `mkt_flow_flow_foreigners_net_ratio_z52_sector_rel`
- `mkt_flow_flow_foreigners_net_ratio_z52_zscore_20d`
- `mkt_flow_flow_individuals_net_ratio_z52`
- `mkt_flow_flow_individuals_net_ratio_z52_outlier_flag`
- `mkt_flow_flow_individuals_net_ratio_z52_roll_mean_20d`
- `mkt_flow_flow_individuals_net_ratio_z52_roll_std_20d`
- `mkt_flow_flow_individuals_net_ratio_z52_sector_mean`
- `mkt_flow_flow_individuals_net_ratio_z52_sector_rel`
- `mkt_flow_flow_individuals_net_ratio_z52_zscore_20d`
- `mkt_flow_flow_investment_trusts_net_ratio_z52`
- `mkt_flow_flow_investment_trusts_net_ratio_z52_outlier_flag`
- `mkt_flow_flow_investment_trusts_net_ratio_z52_roll_mean_20d`
- `mkt_flow_flow_investment_trusts_net_ratio_z52_roll_std_20d`
- `mkt_flow_flow_investment_trusts_net_ratio_z52_sector_mean`
- `mkt_flow_flow_investment_trusts_net_ratio_z52_sector_rel`
- `mkt_flow_flow_investment_trusts_net_ratio_z52_zscore_20d`
- `mkt_flow_flow_trust_banks_net_ratio_z52`
- `mkt_flow_flow_trust_banks_net_ratio_z52_outlier_flag`
- `mkt_flow_flow_trust_banks_net_ratio_z52_roll_mean_20d`
- `mkt_flow_flow_trust_banks_net_ratio_z52_roll_std_20d`
- `mkt_flow_flow_trust_banks_net_ratio_z52_sector_mean`
- `mkt_flow_flow_trust_banks_net_ratio_z52_sector_rel`
- `mkt_flow_flow_trust_banks_net_ratio_z52_zscore_20d`
- `mkt_flow_foreigners_net`
- `mkt_flow_foreigners_net_outlier_flag`
- `mkt_flow_foreigners_net_ratio`
- `mkt_flow_foreigners_net_ratio_outlier_flag`
- `mkt_flow_foreigners_net_ratio_roll_mean_20d`
- `mkt_flow_foreigners_net_ratio_roll_std_20d`
- `mkt_flow_foreigners_net_ratio_sector_mean`
- `mkt_flow_foreigners_net_ratio_sector_rel`
- `mkt_flow_foreigners_net_ratio_turn_flag`
- `mkt_flow_foreigners_net_ratio_wow`
- `mkt_flow_foreigners_net_ratio_wow_outlier_flag`
- `mkt_flow_foreigners_net_ratio_wow_roll_mean_20d`
- `mkt_flow_foreigners_net_ratio_wow_roll_std_20d`
- `mkt_flow_foreigners_net_ratio_wow_sector_mean`
- `mkt_flow_foreigners_net_ratio_wow_sector_rel`
- `mkt_flow_foreigners_net_ratio_wow_zscore_20d`
- `mkt_flow_foreigners_net_ratio_z13`
- `mkt_flow_foreigners_net_ratio_z13_outlier_flag`
- `mkt_flow_foreigners_net_ratio_z13_roll_mean_20d`
- `mkt_flow_foreigners_net_ratio_z13_roll_std_20d`
- `mkt_flow_foreigners_net_ratio_z13_sector_mean`
- `mkt_flow_foreigners_net_ratio_z13_sector_rel`
- `mkt_flow_foreigners_net_ratio_z13_zscore_20d`
- `mkt_flow_foreigners_net_ratio_z52`
- `mkt_flow_foreigners_net_ratio_z52_outlier_flag`
- `mkt_flow_foreigners_net_ratio_z52_roll_mean_20d`
- `mkt_flow_foreigners_net_ratio_z52_roll_std_20d`
- `mkt_flow_foreigners_net_ratio_z52_sector_mean`
- `mkt_flow_foreigners_net_ratio_z52_sector_rel`
- `mkt_flow_foreigners_net_ratio_z52_zscore_20d`
- `mkt_flow_foreigners_net_ratio_zscore_20d`
- `mkt_flow_foreigners_net_roll_mean_20d`
- `mkt_flow_foreigners_net_roll_std_20d`
- `mkt_flow_foreigners_net_sector_mean`
- `mkt_flow_foreigners_net_sector_rel`
- `mkt_flow_foreigners_net_zscore_20d`
- `mkt_flow_individuals_net`
- `mkt_flow_individuals_net_outlier_flag`
- `mkt_flow_individuals_net_ratio`
- `mkt_flow_individuals_net_ratio_outlier_flag`
- `mkt_flow_individuals_net_ratio_roll_mean_20d`
- `mkt_flow_individuals_net_ratio_roll_std_20d`
- `mkt_flow_individuals_net_ratio_sector_mean`
- `mkt_flow_individuals_net_ratio_sector_rel`
- `mkt_flow_individuals_net_ratio_turn_flag`
- `mkt_flow_individuals_net_ratio_wow`
- `mkt_flow_individuals_net_ratio_wow_outlier_flag`
- `mkt_flow_individuals_net_ratio_wow_roll_mean_20d`
- `mkt_flow_individuals_net_ratio_wow_roll_std_20d`
- `mkt_flow_individuals_net_ratio_wow_sector_mean`
- `mkt_flow_individuals_net_ratio_wow_sector_rel`
- `mkt_flow_individuals_net_ratio_wow_zscore_20d`
- `mkt_flow_individuals_net_ratio_z13`
- `mkt_flow_individuals_net_ratio_z13_outlier_flag`
- `mkt_flow_individuals_net_ratio_z13_roll_mean_20d`
- `mkt_flow_individuals_net_ratio_z13_roll_std_20d`
- `mkt_flow_individuals_net_ratio_z13_sector_mean`
- `mkt_flow_individuals_net_ratio_z13_sector_rel`
- `mkt_flow_individuals_net_ratio_z13_zscore_20d`
- `mkt_flow_individuals_net_ratio_z52`
- `mkt_flow_individuals_net_ratio_z52_outlier_flag`
- `mkt_flow_individuals_net_ratio_z52_roll_mean_20d`
- `mkt_flow_individuals_net_ratio_z52_roll_std_20d`
- `mkt_flow_individuals_net_ratio_z52_sector_mean`
- `mkt_flow_individuals_net_ratio_z52_sector_rel`
- `mkt_flow_individuals_net_ratio_z52_zscore_20d`
- `mkt_flow_individuals_net_ratio_zscore_20d`
- `mkt_flow_individuals_net_roll_mean_20d`
- `mkt_flow_individuals_net_roll_std_20d`
- `mkt_flow_individuals_net_sector_mean`
- `mkt_flow_individuals_net_sector_rel`
- `mkt_flow_individuals_net_zscore_20d`
- `mkt_flow_investment_trusts_net`
- `mkt_flow_investment_trusts_net_outlier_flag`
- `mkt_flow_investment_trusts_net_ratio`
- `mkt_flow_investment_trusts_net_ratio_outlier_flag`
- `mkt_flow_investment_trusts_net_ratio_roll_mean_20d`
- `mkt_flow_investment_trusts_net_ratio_roll_std_20d`
- `mkt_flow_investment_trusts_net_ratio_sector_mean`
- `mkt_flow_investment_trusts_net_ratio_sector_rel`
- `mkt_flow_investment_trusts_net_ratio_z13`
- `mkt_flow_investment_trusts_net_ratio_z13_outlier_flag`
- `mkt_flow_investment_trusts_net_ratio_z13_roll_mean_20d`
- `mkt_flow_investment_trusts_net_ratio_z13_roll_std_20d`
- `mkt_flow_investment_trusts_net_ratio_z13_sector_mean`
- `mkt_flow_investment_trusts_net_ratio_z13_sector_rel`
- `mkt_flow_investment_trusts_net_ratio_z13_zscore_20d`
- `mkt_flow_investment_trusts_net_ratio_z52`
- `mkt_flow_investment_trusts_net_ratio_z52_outlier_flag`
- `mkt_flow_investment_trusts_net_ratio_z52_roll_mean_20d`
- `mkt_flow_investment_trusts_net_ratio_z52_roll_std_20d`
- `mkt_flow_investment_trusts_net_ratio_z52_sector_mean`
- `mkt_flow_investment_trusts_net_ratio_z52_sector_rel`
- `mkt_flow_investment_trusts_net_ratio_z52_zscore_20d`
- `mkt_flow_investment_trusts_net_ratio_zscore_20d`
- `mkt_flow_investment_trusts_net_roll_mean_20d`
- `mkt_flow_investment_trusts_net_roll_std_20d`
- `mkt_flow_investment_trusts_net_sector_mean`
- `mkt_flow_investment_trusts_net_sector_rel`
- `mkt_flow_investment_trusts_net_zscore_20d`
- `mkt_flow_total_net`
- `mkt_flow_total_net_outlier_flag`
- `mkt_flow_total_net_roll_mean_20d`
- `mkt_flow_total_net_roll_std_20d`
- `mkt_flow_total_net_sector_mean`
- `mkt_flow_total_net_sector_rel`
- `mkt_flow_total_net_zscore_20d`
- `mkt_flow_trust_banks_net`
- `mkt_flow_trust_banks_net_outlier_flag`
- `mkt_flow_trust_banks_net_ratio`
- `mkt_flow_trust_banks_net_ratio_outlier_flag`
- `mkt_flow_trust_banks_net_ratio_roll_mean_20d`
- `mkt_flow_trust_banks_net_ratio_roll_std_20d`
- `mkt_flow_trust_banks_net_ratio_sector_mean`
- `mkt_flow_trust_banks_net_ratio_sector_rel`
- `mkt_flow_trust_banks_net_ratio_z13`
- `mkt_flow_trust_banks_net_ratio_z13_outlier_flag`
- `mkt_flow_trust_banks_net_ratio_z13_roll_mean_20d`
- `mkt_flow_trust_banks_net_ratio_z13_roll_std_20d`
- `mkt_flow_trust_banks_net_ratio_z13_sector_mean`
- `mkt_flow_trust_banks_net_ratio_z13_sector_rel`
- `mkt_flow_trust_banks_net_ratio_z13_zscore_20d`
- `mkt_flow_trust_banks_net_ratio_z52`
- `mkt_flow_trust_banks_net_ratio_z52_outlier_flag`
- `mkt_flow_trust_banks_net_ratio_z52_roll_mean_20d`
- `mkt_flow_trust_banks_net_ratio_z52_roll_std_20d`
- `mkt_flow_trust_banks_net_ratio_z52_sector_mean`
- `mkt_flow_trust_banks_net_ratio_z52_sector_rel`
- `mkt_flow_trust_banks_net_ratio_z52_zscore_20d`
- `mkt_flow_trust_banks_net_ratio_zscore_20d`
- `mkt_flow_trust_banks_net_roll_mean_20d`
- `mkt_flow_trust_banks_net_roll_std_20d`
- `mkt_flow_trust_banks_net_sector_mean`
- `mkt_flow_trust_banks_net_sector_rel`
- `mkt_flow_trust_banks_net_zscore_20d`

**MPI** (28 columns):
- `mpi_dist_to_limit`
- `mpi_dist_to_limit_outlier_flag`
- `mpi_dist_to_limit_roll_mean_20d`
- `mpi_dist_to_limit_roll_std_20d`
- `mpi_dist_to_limit_sector_mean`
- `mpi_dist_to_limit_sector_rel`
- `mpi_dist_to_limit_z20`
- `mpi_dist_to_limit_z20_outlier_flag`
- `mpi_dist_to_limit_z20_roll_mean_20d`
- `mpi_dist_to_limit_z20_roll_std_20d`
- `mpi_dist_to_limit_z20_sector_mean`
- `mpi_dist_to_limit_z20_sector_rel`
- `mpi_dist_to_limit_z20_zscore_20d`
- `mpi_dist_to_limit_zscore_20d`
- `mpi_drawdown`
- `mpi_drawdown_outlier_flag`
- `mpi_drawdown_roll_mean_20d`
- `mpi_drawdown_roll_std_20d`
- `mpi_drawdown_sector_mean`
- `mpi_drawdown_sector_rel`
- `mpi_drawdown_z20`
- `mpi_drawdown_z20_outlier_flag`
- `mpi_drawdown_z20_roll_mean_20d`
- `mpi_drawdown_z20_roll_std_20d`
- `mpi_drawdown_z20_sector_mean`
- `mpi_drawdown_z20_sector_rel`
- `mpi_drawdown_z20_zscore_20d`
- `mpi_drawdown_zscore_20d`

**PREE** (21 columns):
- `preE_margin_diff`
- `preE_margin_diff_outlier_flag`
- `preE_margin_diff_roll_mean_20d`
- `preE_margin_diff_roll_std_20d`
- `preE_margin_diff_sector_mean`
- `preE_margin_diff_sector_rel`
- `preE_margin_diff_z20`
- `preE_margin_diff_z20_outlier_flag`
- `preE_margin_diff_z20_roll_mean_20d`
- `preE_margin_diff_z20_roll_std_20d`
- `preE_margin_diff_z20_sector_mean`
- `preE_margin_diff_z20_sector_rel`
- `preE_margin_diff_z20_zscore_20d`
- `preE_margin_diff_zscore_20d`
- `preE_risk_score`
- `preE_risk_score_outlier_flag`
- `preE_risk_score_roll_mean_20d`
- `preE_risk_score_roll_std_20d`
- `preE_risk_score_sector_mean`
- `preE_risk_score_sector_rel`
- `preE_risk_score_zscore_20d`

**REL** (7 columns):
- `rel_to_sec17_5d`
- `rel_to_sec17_5d_outlier_flag`
- `rel_to_sec17_5d_roll_mean_20d`
- `rel_to_sec17_5d_roll_std_20d`
- `rel_to_sec17_5d_sector_mean`
- `rel_to_sec17_5d_sector_rel`
- `rel_to_sec17_5d_zscore_20d`

**RET** (1 columns):
- `ret_prev_120d`

**RQ** (12 columns):
- `rq_63_10_outlier_flag`
- `rq_63_10_roll_mean_20d`
- `rq_63_10_roll_std_20d`
- `rq_63_10_zscore_20d`
- `rq_63_50_outlier_flag`
- `rq_63_50_roll_mean_20d`
- `rq_63_50_roll_std_20d`
- `rq_63_50_zscore_20d`
- `rq_63_90_outlier_flag`
- `rq_63_90_roll_mean_20d`
- `rq_63_90_roll_std_20d`
- `rq_63_90_zscore_20d`

**SEC17** (16 columns):
- `sec17_mom_20`
- `sec17_mom_20_outlier_flag`
- `sec17_mom_20_roll_mean_20d`
- `sec17_mom_20_roll_std_20d`
- `sec17_mom_20_sector_mean`
- `sec17_mom_20_sector_rel`
- `sec17_mom_20_zscore_20d`
- `sec17_ret_5d_eq`
- `sec17_ret_5d_eq_hist_vol`
- `sec17_ret_5d_eq_outlier_flag`
- `sec17_ret_5d_eq_roll_mean_20d`
- `sec17_ret_5d_eq_roll_std_20d`
- `sec17_ret_5d_eq_rolling_sharpe`
- `sec17_ret_5d_eq_sector_mean`
- `sec17_ret_5d_eq_sector_rel`
- `sec17_ret_5d_eq_zscore_20d`

**SECTOR** (7 columns):
- `sector_short_ratio_z20`
- `sector_short_ratio_z20_outlier_flag`
- `sector_short_ratio_z20_roll_mean_20d`
- `sector_short_ratio_z20_roll_std_20d`
- `sector_short_ratio_z20_sector_mean`
- `sector_short_ratio_z20_sector_rel`
- `sector_short_ratio_z20_zscore_20d`

**SHARES** (7 columns):
- `shares_out_delta_pct`
- `shares_out_delta_pct_outlier_flag`
- `shares_out_delta_pct_roll_mean_20d`
- `shares_out_delta_pct_roll_std_20d`
- `shares_out_delta_pct_sector_mean`
- `shares_out_delta_pct_sector_rel`
- `shares_out_delta_pct_zscore_20d`

**SQUEEZE** (7 columns):
- `squeeze_risk`
- `squeeze_risk_outlier_flag`
- `squeeze_risk_roll_mean_20d`
- `squeeze_risk_roll_std_20d`
- `squeeze_risk_sector_mean`
- `squeeze_risk_sector_rel`
- `squeeze_risk_zscore_20d`

**SSP** (7 columns):
- `ssp_ratio_component`
- `ssp_ratio_component_outlier_flag`
- `ssp_ratio_component_roll_mean_20d`
- `ssp_ratio_component_roll_std_20d`
- `ssp_ratio_component_sector_mean`
- `ssp_ratio_component_sector_rel`
- `ssp_ratio_component_zscore_20d`

**SUPPLY** (1 columns):
- `supply_shock`

**WEEKLY** (21 columns):
- `weekly_margin_long_pct_float`
- `weekly_margin_long_pct_float_outlier_flag`
- `weekly_margin_long_pct_float_roc_5d`
- `weekly_margin_long_pct_float_roc_5d_outlier_flag`
- `weekly_margin_long_pct_float_roc_5d_roll_mean_20d`
- `weekly_margin_long_pct_float_roc_5d_roll_std_20d`
- `weekly_margin_long_pct_float_roc_5d_sector_mean`
- `weekly_margin_long_pct_float_roc_5d_sector_rel`
- `weekly_margin_long_pct_float_roc_5d_zscore_20d`
- `weekly_margin_long_pct_float_roll_mean_20d`
- `weekly_margin_long_pct_float_roll_std_20d`
- `weekly_margin_long_pct_float_sector_mean`
- `weekly_margin_long_pct_float_sector_rel`
- `weekly_margin_long_pct_float_z20`
- `weekly_margin_long_pct_float_z20_outlier_flag`
- `weekly_margin_long_pct_float_z20_roll_mean_20d`
- `weekly_margin_long_pct_float_z20_roll_std_20d`
- `weekly_margin_long_pct_float_z20_sector_mean`
- `weekly_margin_long_pct_float_z20_sector_rel`
- `weekly_margin_long_pct_float_z20_zscore_20d`
- `weekly_margin_long_pct_float_zscore_20d`

#### High-NULL Columns (>95% but <100%)

| Column | NULL % |
|--------|--------|
| `concentrated_flow_signal_outlier_flag` | 99.99% |
| `concentrated_flow_signal_zscore_20d` | 99.99% |
| `flow_concentration_outlier_flag` | 99.99% |
| `flow_concentration_zscore_20d` | 99.99% |
| `foreign_domestic_divergence_outlier_flag` | 99.99% |
| `foreign_domestic_divergence_zscore_20d` | 99.99% |
| `foreign_persistence_outlier_flag` | 99.99% |
| `foreign_persistence_zscore_20d` | 99.99% |
| `foreign_sentiment_outlier_flag` | 99.99% |
| `foreign_sentiment_zscore_20d` | 99.99% |
| `institutional_accumulation_outlier_flag` | 99.99% |
| `institutional_accumulation_zscore_20d` | 99.99% |
| `institutional_persistence_outlier_flag` | 99.99% |
| `institutional_persistence_zscore_20d` | 99.99% |
| `retail_institutional_divergence_outlier_flag` | 99.99% |
| `retail_institutional_divergence_zscore_20d` | 99.99% |
| `smart_flow_indicator_outlier_flag` | 99.99% |
| `smart_flow_indicator_zscore_20d` | 99.99% |
| `rq_63_10` | 99.97% |
| `rq_63_10_sector_rel` | 99.97% |
| `rq_63_50` | 99.97% |
| `rq_63_50_sector_rel` | 99.97% |
| `rq_63_90` | 99.97% |
| `rq_63_90_sector_rel` | 99.97% |
| `days_since_sector33_change` | 99.90% |
| `days_since_sector33_change_sector_rel` | 99.90% |
| `sector33_changed_5d` | 99.90% |
| `rq_63_10_sector_mean` | 99.69% |
| `rq_63_50_sector_mean` | 99.69% |
| `rq_63_90_sector_mean` | 99.69% |
| `ret_prev_60d` | 97.19% |
| `days_since_sq` | 95.10% |
| `days_since_sq_sector_mean` | 95.10% |
| `days_since_sq_sector_rel` | 95.10% |
| `days_to_sq` | 95.10% |
| `days_to_sq_sector_mean` | 95.10% |
| `days_to_sq_sector_rel` | 95.10% |
| `days_since_sq_roll_mean_20d` | 95.10% |
| `days_since_sq_roll_std_20d` | 95.10% |
| `days_to_sq_roll_mean_20d` | 95.10% |
| `days_to_sq_roll_std_20d` | 95.10% |


### 2025 Dataset

- **Total Rows**: 809,306
- **Total Columns**: 2,775
- **All-NULL Columns**: 564
- **High-NULL Columns (>95%)**: 34

#### All-NULL Columns by Feature Group

**COMPANYNAMEENGLISH** (7 columns):
- `CompanyNameEnglish`
- `CompanyNameEnglish_outlier_flag`
- `CompanyNameEnglish_roll_mean_20d`
- `CompanyNameEnglish_roll_std_20d`
- `CompanyNameEnglish_sector_mean`
- `CompanyNameEnglish_sector_rel`
- `CompanyNameEnglish_zscore_20d`

**DISCLOSEDDATE** (1 columns):
- `DisclosedDate`

**ALPHA60** (7 columns):
- `alpha60_topix`
- `alpha60_topix_outlier_flag`
- `alpha60_topix_roll_mean_20d`
- `alpha60_topix_roll_std_20d`
- `alpha60_topix_sector_mean`
- `alpha60_topix_sector_rel`
- `alpha60_topix_zscore_20d`

**BASIS** (7 columns):
- `basis_gate`
- `basis_gate_outlier_flag`
- `basis_gate_roll_mean_20d`
- `basis_gate_roll_std_20d`
- `basis_gate_sector_mean`
- `basis_gate_sector_rel`
- `basis_gate_zscore_20d`

**BD** (2 columns):
- `bd_net_adv60`
- `bd_net_mc`

**BETA60** (7 columns):
- `beta60_topix`
- `beta60_topix_outlier_flag`
- `beta60_topix_roll_mean_20d`
- `beta60_topix_roll_std_20d`
- `beta60_topix_sector_mean`
- `beta60_topix_sector_rel`
- `beta60_topix_zscore_20d`

**BUYBACK** (1 columns):
- `buyback_flag`

**CROWDING** (7 columns):
- `crowding_score`
- `crowding_score_outlier_flag`
- `crowding_score_roll_mean_20d`
- `crowding_score_roll_std_20d`
- `crowding_score_sector_mean`
- `crowding_score_sector_rel`
- `crowding_score_zscore_20d`

**DAYS** (22 columns):
- `days_since_market_change`
- `days_since_market_change_outlier_flag`
- `days_since_market_change_roll_mean_20d`
- `days_since_market_change_roll_std_20d`
- `days_since_market_change_sector_mean`
- `days_since_market_change_sector_rel`
- `days_since_market_change_zscore_20d`
- `days_since_sector33_change_outlier_flag`
- `days_since_sector33_change_roll_mean_20d`
- `days_since_sector33_change_roll_std_20d`
- `days_since_sector33_change_zscore_20d`
- `days_since_sq_outlier_flag`
- `days_since_sq_zscore_20d`
- `days_to_earnings`
- `days_to_earnings_outlier_flag`
- `days_to_earnings_roll_mean_20d`
- `days_to_earnings_roll_std_20d`
- `days_to_earnings_sector_mean`
- `days_to_earnings_sector_rel`
- `days_to_earnings_zscore_20d`
- `days_to_sq_outlier_flag`
- `days_to_sq_zscore_20d`

**DILUTION** (1 columns):
- `dilution_flag`

**DIV** (13 columns):
- `div_amount_12m`
- `div_amount_next`
- `div_days_since_ex`
- `div_days_to_ex`
- `div_dy_12m`
- `div_ex_cycle_z`
- `div_ex_drop_expected`
- `div_ex_gap_miss`
- `div_ex_gap_theo`
- `div_staleness_bd`
- `div_staleness_days`
- `div_yield_12m`
- `div_yield_ttm`

**DMI** (1 columns):
- `dmi_reason_code`

**EARNINGS** (8 columns):
- `earnings_event_date`
- `earnings_recent_1d`
- `earnings_recent_3d`
- `earnings_recent_5d`
- `earnings_today`
- `earnings_upcoming_1d`
- `earnings_upcoming_3d`
- `earnings_upcoming_5d`

**FLOAT** (21 columns):
- `float_turnover_pct`
- `float_turnover_pct_outlier_flag`
- `float_turnover_pct_roc_5d`
- `float_turnover_pct_roc_5d_outlier_flag`
- `float_turnover_pct_roc_5d_roll_mean_20d`
- `float_turnover_pct_roc_5d_roll_std_20d`
- `float_turnover_pct_roc_5d_sector_mean`
- `float_turnover_pct_roc_5d_sector_rel`
- `float_turnover_pct_roc_5d_zscore_20d`
- `float_turnover_pct_roll_mean_20d`
- `float_turnover_pct_roll_std_20d`
- `float_turnover_pct_sector_mean`
- `float_turnover_pct_sector_rel`
- `float_turnover_pct_z20`
- `float_turnover_pct_z20_outlier_flag`
- `float_turnover_pct_z20_roll_mean_20d`
- `float_turnover_pct_z20_roll_std_20d`
- `float_turnover_pct_z20_sector_mean`
- `float_turnover_pct_z20_sector_rel`
- `float_turnover_pct_z20_zscore_20d`
- `float_turnover_pct_zscore_20d`

**FS** (47 columns):
- `fs_accruals`
- `fs_accruals_ttm`
- `fs_average_shares`
- `fs_capex_ttm`
- `fs_cfo_to_ni`
- `fs_cfo_ttm`
- `fs_consolidated_flag`
- `fs_days_since`
- `fs_days_to_next`
- `fs_doc_family_1Q`
- `fs_doc_family_2Q`
- `fs_doc_family_3Q`
- `fs_doc_family_FY`
- `fs_equity_ratio`
- `fs_fcf_ttm`
- `fs_guidance_revision_flag`
- `fs_is_valid`
- `fs_lag_days`
- `fs_net_cash_ratio`
- `fs_net_income_ttm`
- `fs_net_margin`
- `fs_observation_count`
- `fs_op_margin`
- `fs_op_profit_ttm`
- `fs_revenue_ttm`
- `fs_roa_ttm`
- `fs_roe_ttm`
- `fs_sales_yoy`
- `fs_shares_outstanding`
- `fs_staleness_bd`
- `fs_standard_Foreign`
- `fs_standard_IFRS`
- `fs_standard_JGAAP`
- `fs_standard_JMIS`
- `fs_standard_US`
- `fs_ttm_cfo`
- `fs_ttm_cfo_margin`
- `fs_ttm_net_income`
- `fs_ttm_op_margin`
- `fs_ttm_op_profit`
- `fs_ttm_sales`
- `fs_window_e_pm1`
- `fs_window_e_pp3`
- `fs_window_e_pp5`
- `fs_yoy_ttm_net_income`
- `fs_yoy_ttm_op_profit`
- `fs_yoy_ttm_sales`

**GAP** (2 columns):
- `gap_atr`
- `gap_predictor`

**IDXOPT** (14 columns):
- `idxopt_vrp_gap`
- `idxopt_vrp_gap_outlier_flag`
- `idxopt_vrp_gap_roll_mean_20d`
- `idxopt_vrp_gap_roll_std_20d`
- `idxopt_vrp_gap_sector_mean`
- `idxopt_vrp_gap_sector_rel`
- `idxopt_vrp_gap_zscore_20d`
- `idxopt_vrp_ratio`
- `idxopt_vrp_ratio_outlier_flag`
- `idxopt_vrp_ratio_roll_mean_20d`
- `idxopt_vrp_ratio_roll_std_20d`
- `idxopt_vrp_ratio_sector_mean`
- `idxopt_vrp_ratio_sector_rel`
- `idxopt_vrp_ratio_zscore_20d`

**IS** (20 columns):
- `is_E_0`
- `is_E_pm1`
- `is_E_pp1`
- `is_E_pp3`
- `is_E_pp5`
- `is_fs_valid`
- `is_growth_x_dv_z20`
- `is_growth_x_dv_z20_outlier_flag`
- `is_growth_x_dv_z20_roll_mean_20d`
- `is_growth_x_dv_z20_roll_std_20d`
- `is_growth_x_dv_z20_sector_mean`
- `is_growth_x_dv_z20_sector_rel`
- `is_growth_x_dv_z20_zscore_20d`
- `is_prime_x_dv_z20`
- `is_prime_x_dv_z20_outlier_flag`
- `is_prime_x_dv_z20_roll_mean_20d`
- `is_prime_x_dv_z20_roll_std_20d`
- `is_prime_x_dv_z20_sector_mean`
- `is_prime_x_dv_z20_sector_rel`
- `is_prime_x_dv_z20_zscore_20d`

**LIMIT** (7 columns):
- `limit_up_flag_lag1`
- `limit_up_flag_lag1_outlier_flag`
- `limit_up_flag_lag1_roll_mean_20d`
- `limit_up_flag_lag1_roll_std_20d`
- `limit_up_flag_lag1_sector_mean`
- `limit_up_flag_lag1_sector_rel`
- `limit_up_flag_lag1_zscore_20d`

**MARGIN** (28 columns):
- `margin_long_pct_float`
- `margin_long_pct_float_outlier_flag`
- `margin_long_pct_float_roc_5d`
- `margin_long_pct_float_roc_5d_outlier_flag`
- `margin_long_pct_float_roc_5d_roll_mean_20d`
- `margin_long_pct_float_roc_5d_roll_std_20d`
- `margin_long_pct_float_roc_5d_sector_mean`
- `margin_long_pct_float_roc_5d_sector_rel`
- `margin_long_pct_float_roc_5d_zscore_20d`
- `margin_long_pct_float_roll_mean_20d`
- `margin_long_pct_float_roll_std_20d`
- `margin_long_pct_float_sector_mean`
- `margin_long_pct_float_sector_rel`
- `margin_long_pct_float_z20`
- `margin_long_pct_float_z20_outlier_flag`
- `margin_long_pct_float_z20_roll_mean_20d`
- `margin_long_pct_float_z20_roll_std_20d`
- `margin_long_pct_float_z20_sector_mean`
- `margin_long_pct_float_z20_sector_rel`
- `margin_long_pct_float_z20_zscore_20d`
- `margin_long_pct_float_zscore_20d`
- `margin_pain_index`
- `margin_pain_index_outlier_flag`
- `margin_pain_index_roll_mean_20d`
- `margin_pain_index_roll_std_20d`
- `margin_pain_index_sector_mean`
- `margin_pain_index_sector_rel`
- `margin_pain_index_zscore_20d`

**MARKET** (1 columns):
- `market_changed_5d`

**MKT** (170 columns):
- `mkt_flow_divergence_foreigners_vs_individuals`
- `mkt_flow_divergence_foreigners_vs_individuals_outlier_flag`
- `mkt_flow_divergence_foreigners_vs_individuals_roll_mean_20d`
- `mkt_flow_divergence_foreigners_vs_individuals_roll_std_20d`
- `mkt_flow_divergence_foreigners_vs_individuals_sector_mean`
- `mkt_flow_divergence_foreigners_vs_individuals_sector_rel`
- `mkt_flow_divergence_foreigners_vs_individuals_zscore_20d`
- `mkt_flow_flow_foreigners_net_ratio_z52`
- `mkt_flow_flow_foreigners_net_ratio_z52_outlier_flag`
- `mkt_flow_flow_foreigners_net_ratio_z52_roll_mean_20d`
- `mkt_flow_flow_foreigners_net_ratio_z52_roll_std_20d`
- `mkt_flow_flow_foreigners_net_ratio_z52_sector_mean`
- `mkt_flow_flow_foreigners_net_ratio_z52_sector_rel`
- `mkt_flow_flow_foreigners_net_ratio_z52_zscore_20d`
- `mkt_flow_flow_individuals_net_ratio_z52`
- `mkt_flow_flow_individuals_net_ratio_z52_outlier_flag`
- `mkt_flow_flow_individuals_net_ratio_z52_roll_mean_20d`
- `mkt_flow_flow_individuals_net_ratio_z52_roll_std_20d`
- `mkt_flow_flow_individuals_net_ratio_z52_sector_mean`
- `mkt_flow_flow_individuals_net_ratio_z52_sector_rel`
- `mkt_flow_flow_individuals_net_ratio_z52_zscore_20d`
- `mkt_flow_flow_investment_trusts_net_ratio_z52`
- `mkt_flow_flow_investment_trusts_net_ratio_z52_outlier_flag`
- `mkt_flow_flow_investment_trusts_net_ratio_z52_roll_mean_20d`
- `mkt_flow_flow_investment_trusts_net_ratio_z52_roll_std_20d`
- `mkt_flow_flow_investment_trusts_net_ratio_z52_sector_mean`
- `mkt_flow_flow_investment_trusts_net_ratio_z52_sector_rel`
- `mkt_flow_flow_investment_trusts_net_ratio_z52_zscore_20d`
- `mkt_flow_flow_trust_banks_net_ratio_z52`
- `mkt_flow_flow_trust_banks_net_ratio_z52_outlier_flag`
- `mkt_flow_flow_trust_banks_net_ratio_z52_roll_mean_20d`
- `mkt_flow_flow_trust_banks_net_ratio_z52_roll_std_20d`
- `mkt_flow_flow_trust_banks_net_ratio_z52_sector_mean`
- `mkt_flow_flow_trust_banks_net_ratio_z52_sector_rel`
- `mkt_flow_flow_trust_banks_net_ratio_z52_zscore_20d`
- `mkt_flow_foreigners_net`
- `mkt_flow_foreigners_net_outlier_flag`
- `mkt_flow_foreigners_net_ratio`
- `mkt_flow_foreigners_net_ratio_outlier_flag`
- `mkt_flow_foreigners_net_ratio_roll_mean_20d`
- `mkt_flow_foreigners_net_ratio_roll_std_20d`
- `mkt_flow_foreigners_net_ratio_sector_mean`
- `mkt_flow_foreigners_net_ratio_sector_rel`
- `mkt_flow_foreigners_net_ratio_turn_flag`
- `mkt_flow_foreigners_net_ratio_wow`
- `mkt_flow_foreigners_net_ratio_wow_outlier_flag`
- `mkt_flow_foreigners_net_ratio_wow_roll_mean_20d`
- `mkt_flow_foreigners_net_ratio_wow_roll_std_20d`
- `mkt_flow_foreigners_net_ratio_wow_sector_mean`
- `mkt_flow_foreigners_net_ratio_wow_sector_rel`
- `mkt_flow_foreigners_net_ratio_wow_zscore_20d`
- `mkt_flow_foreigners_net_ratio_z13`
- `mkt_flow_foreigners_net_ratio_z13_outlier_flag`
- `mkt_flow_foreigners_net_ratio_z13_roll_mean_20d`
- `mkt_flow_foreigners_net_ratio_z13_roll_std_20d`
- `mkt_flow_foreigners_net_ratio_z13_sector_mean`
- `mkt_flow_foreigners_net_ratio_z13_sector_rel`
- `mkt_flow_foreigners_net_ratio_z13_zscore_20d`
- `mkt_flow_foreigners_net_ratio_z52`
- `mkt_flow_foreigners_net_ratio_z52_outlier_flag`
- `mkt_flow_foreigners_net_ratio_z52_roll_mean_20d`
- `mkt_flow_foreigners_net_ratio_z52_roll_std_20d`
- `mkt_flow_foreigners_net_ratio_z52_sector_mean`
- `mkt_flow_foreigners_net_ratio_z52_sector_rel`
- `mkt_flow_foreigners_net_ratio_z52_zscore_20d`
- `mkt_flow_foreigners_net_ratio_zscore_20d`
- `mkt_flow_foreigners_net_roll_mean_20d`
- `mkt_flow_foreigners_net_roll_std_20d`
- `mkt_flow_foreigners_net_sector_mean`
- `mkt_flow_foreigners_net_sector_rel`
- `mkt_flow_foreigners_net_zscore_20d`
- `mkt_flow_individuals_net`
- `mkt_flow_individuals_net_outlier_flag`
- `mkt_flow_individuals_net_ratio`
- `mkt_flow_individuals_net_ratio_outlier_flag`
- `mkt_flow_individuals_net_ratio_roll_mean_20d`
- `mkt_flow_individuals_net_ratio_roll_std_20d`
- `mkt_flow_individuals_net_ratio_sector_mean`
- `mkt_flow_individuals_net_ratio_sector_rel`
- `mkt_flow_individuals_net_ratio_turn_flag`
- `mkt_flow_individuals_net_ratio_wow`
- `mkt_flow_individuals_net_ratio_wow_outlier_flag`
- `mkt_flow_individuals_net_ratio_wow_roll_mean_20d`
- `mkt_flow_individuals_net_ratio_wow_roll_std_20d`
- `mkt_flow_individuals_net_ratio_wow_sector_mean`
- `mkt_flow_individuals_net_ratio_wow_sector_rel`
- `mkt_flow_individuals_net_ratio_wow_zscore_20d`
- `mkt_flow_individuals_net_ratio_z13`
- `mkt_flow_individuals_net_ratio_z13_outlier_flag`
- `mkt_flow_individuals_net_ratio_z13_roll_mean_20d`
- `mkt_flow_individuals_net_ratio_z13_roll_std_20d`
- `mkt_flow_individuals_net_ratio_z13_sector_mean`
- `mkt_flow_individuals_net_ratio_z13_sector_rel`
- `mkt_flow_individuals_net_ratio_z13_zscore_20d`
- `mkt_flow_individuals_net_ratio_z52`
- `mkt_flow_individuals_net_ratio_z52_outlier_flag`
- `mkt_flow_individuals_net_ratio_z52_roll_mean_20d`
- `mkt_flow_individuals_net_ratio_z52_roll_std_20d`
- `mkt_flow_individuals_net_ratio_z52_sector_mean`
- `mkt_flow_individuals_net_ratio_z52_sector_rel`
- `mkt_flow_individuals_net_ratio_z52_zscore_20d`
- `mkt_flow_individuals_net_ratio_zscore_20d`
- `mkt_flow_individuals_net_roll_mean_20d`
- `mkt_flow_individuals_net_roll_std_20d`
- `mkt_flow_individuals_net_sector_mean`
- `mkt_flow_individuals_net_sector_rel`
- `mkt_flow_individuals_net_zscore_20d`
- `mkt_flow_investment_trusts_net`
- `mkt_flow_investment_trusts_net_outlier_flag`
- `mkt_flow_investment_trusts_net_ratio`
- `mkt_flow_investment_trusts_net_ratio_outlier_flag`
- `mkt_flow_investment_trusts_net_ratio_roll_mean_20d`
- `mkt_flow_investment_trusts_net_ratio_roll_std_20d`
- `mkt_flow_investment_trusts_net_ratio_sector_mean`
- `mkt_flow_investment_trusts_net_ratio_sector_rel`
- `mkt_flow_investment_trusts_net_ratio_z13`
- `mkt_flow_investment_trusts_net_ratio_z13_outlier_flag`
- `mkt_flow_investment_trusts_net_ratio_z13_roll_mean_20d`
- `mkt_flow_investment_trusts_net_ratio_z13_roll_std_20d`
- `mkt_flow_investment_trusts_net_ratio_z13_sector_mean`
- `mkt_flow_investment_trusts_net_ratio_z13_sector_rel`
- `mkt_flow_investment_trusts_net_ratio_z13_zscore_20d`
- `mkt_flow_investment_trusts_net_ratio_z52`
- `mkt_flow_investment_trusts_net_ratio_z52_outlier_flag`
- `mkt_flow_investment_trusts_net_ratio_z52_roll_mean_20d`
- `mkt_flow_investment_trusts_net_ratio_z52_roll_std_20d`
- `mkt_flow_investment_trusts_net_ratio_z52_sector_mean`
- `mkt_flow_investment_trusts_net_ratio_z52_sector_rel`
- `mkt_flow_investment_trusts_net_ratio_z52_zscore_20d`
- `mkt_flow_investment_trusts_net_ratio_zscore_20d`
- `mkt_flow_investment_trusts_net_roll_mean_20d`
- `mkt_flow_investment_trusts_net_roll_std_20d`
- `mkt_flow_investment_trusts_net_sector_mean`
- `mkt_flow_investment_trusts_net_sector_rel`
- `mkt_flow_investment_trusts_net_zscore_20d`
- `mkt_flow_total_net`
- `mkt_flow_total_net_outlier_flag`
- `mkt_flow_total_net_roll_mean_20d`
- `mkt_flow_total_net_roll_std_20d`
- `mkt_flow_total_net_sector_mean`
- `mkt_flow_total_net_sector_rel`
- `mkt_flow_total_net_zscore_20d`
- `mkt_flow_trust_banks_net`
- `mkt_flow_trust_banks_net_outlier_flag`
- `mkt_flow_trust_banks_net_ratio`
- `mkt_flow_trust_banks_net_ratio_outlier_flag`
- `mkt_flow_trust_banks_net_ratio_roll_mean_20d`
- `mkt_flow_trust_banks_net_ratio_roll_std_20d`
- `mkt_flow_trust_banks_net_ratio_sector_mean`
- `mkt_flow_trust_banks_net_ratio_sector_rel`
- `mkt_flow_trust_banks_net_ratio_z13`
- `mkt_flow_trust_banks_net_ratio_z13_outlier_flag`
- `mkt_flow_trust_banks_net_ratio_z13_roll_mean_20d`
- `mkt_flow_trust_banks_net_ratio_z13_roll_std_20d`
- `mkt_flow_trust_banks_net_ratio_z13_sector_mean`
- `mkt_flow_trust_banks_net_ratio_z13_sector_rel`
- `mkt_flow_trust_banks_net_ratio_z13_zscore_20d`
- `mkt_flow_trust_banks_net_ratio_z52`
- `mkt_flow_trust_banks_net_ratio_z52_outlier_flag`
- `mkt_flow_trust_banks_net_ratio_z52_roll_mean_20d`
- `mkt_flow_trust_banks_net_ratio_z52_roll_std_20d`
- `mkt_flow_trust_banks_net_ratio_z52_sector_mean`
- `mkt_flow_trust_banks_net_ratio_z52_sector_rel`
- `mkt_flow_trust_banks_net_ratio_z52_zscore_20d`
- `mkt_flow_trust_banks_net_ratio_zscore_20d`
- `mkt_flow_trust_banks_net_roll_mean_20d`
- `mkt_flow_trust_banks_net_roll_std_20d`
- `mkt_flow_trust_banks_net_sector_mean`
- `mkt_flow_trust_banks_net_sector_rel`
- `mkt_flow_trust_banks_net_zscore_20d`

**MPI** (28 columns):
- `mpi_dist_to_limit`
- `mpi_dist_to_limit_outlier_flag`
- `mpi_dist_to_limit_roll_mean_20d`
- `mpi_dist_to_limit_roll_std_20d`
- `mpi_dist_to_limit_sector_mean`
- `mpi_dist_to_limit_sector_rel`
- `mpi_dist_to_limit_z20`
- `mpi_dist_to_limit_z20_outlier_flag`
- `mpi_dist_to_limit_z20_roll_mean_20d`
- `mpi_dist_to_limit_z20_roll_std_20d`
- `mpi_dist_to_limit_z20_sector_mean`
- `mpi_dist_to_limit_z20_sector_rel`
- `mpi_dist_to_limit_z20_zscore_20d`
- `mpi_dist_to_limit_zscore_20d`
- `mpi_drawdown`
- `mpi_drawdown_outlier_flag`
- `mpi_drawdown_roll_mean_20d`
- `mpi_drawdown_roll_std_20d`
- `mpi_drawdown_sector_mean`
- `mpi_drawdown_sector_rel`
- `mpi_drawdown_z20`
- `mpi_drawdown_z20_outlier_flag`
- `mpi_drawdown_z20_roll_mean_20d`
- `mpi_drawdown_z20_roll_std_20d`
- `mpi_drawdown_z20_sector_mean`
- `mpi_drawdown_z20_sector_rel`
- `mpi_drawdown_z20_zscore_20d`
- `mpi_drawdown_zscore_20d`

**PREE** (35 columns):
- `preE_margin_diff`
- `preE_margin_diff_outlier_flag`
- `preE_margin_diff_roll_mean_20d`
- `preE_margin_diff_roll_std_20d`
- `preE_margin_diff_sector_mean`
- `preE_margin_diff_sector_rel`
- `preE_margin_diff_z20`
- `preE_margin_diff_z20_outlier_flag`
- `preE_margin_diff_z20_roll_mean_20d`
- `preE_margin_diff_z20_roll_std_20d`
- `preE_margin_diff_z20_sector_mean`
- `preE_margin_diff_z20_sector_rel`
- `preE_margin_diff_z20_zscore_20d`
- `preE_margin_diff_zscore_20d`
- `preE_risk_score`
- `preE_risk_score_outlier_flag`
- `preE_risk_score_roll_mean_20d`
- `preE_risk_score_roll_std_20d`
- `preE_risk_score_sector_mean`
- `preE_risk_score_sector_rel`
- `preE_risk_score_zscore_20d`
- `preE_short_ratio_diff`
- `preE_short_ratio_diff_outlier_flag`
- `preE_short_ratio_diff_roll_mean_20d`
- `preE_short_ratio_diff_roll_std_20d`
- `preE_short_ratio_diff_sector_mean`
- `preE_short_ratio_diff_sector_rel`
- `preE_short_ratio_diff_z20`
- `preE_short_ratio_diff_z20_outlier_flag`
- `preE_short_ratio_diff_z20_roll_mean_20d`
- `preE_short_ratio_diff_z20_roll_std_20d`
- `preE_short_ratio_diff_z20_sector_mean`
- `preE_short_ratio_diff_z20_sector_rel`
- `preE_short_ratio_diff_z20_zscore_20d`
- `preE_short_ratio_diff_zscore_20d`

**REL** (7 columns):
- `rel_to_sec17_5d`
- `rel_to_sec17_5d_outlier_flag`
- `rel_to_sec17_5d_roll_mean_20d`
- `rel_to_sec17_5d_roll_std_20d`
- `rel_to_sec17_5d_sector_mean`
- `rel_to_sec17_5d_sector_rel`
- `rel_to_sec17_5d_zscore_20d`

**RET** (1 columns):
- `ret_prev_120d`

**RQ** (12 columns):
- `rq_63_10_outlier_flag`
- `rq_63_10_roll_mean_20d`
- `rq_63_10_roll_std_20d`
- `rq_63_10_zscore_20d`
- `rq_63_50_outlier_flag`
- `rq_63_50_roll_mean_20d`
- `rq_63_50_roll_std_20d`
- `rq_63_50_zscore_20d`
- `rq_63_90_outlier_flag`
- `rq_63_90_roll_mean_20d`
- `rq_63_90_roll_std_20d`
- `rq_63_90_zscore_20d`

**SEC17** (16 columns):
- `sec17_mom_20`
- `sec17_mom_20_outlier_flag`
- `sec17_mom_20_roll_mean_20d`
- `sec17_mom_20_roll_std_20d`
- `sec17_mom_20_sector_mean`
- `sec17_mom_20_sector_rel`
- `sec17_mom_20_zscore_20d`
- `sec17_ret_5d_eq`
- `sec17_ret_5d_eq_hist_vol`
- `sec17_ret_5d_eq_outlier_flag`
- `sec17_ret_5d_eq_roll_mean_20d`
- `sec17_ret_5d_eq_roll_std_20d`
- `sec17_ret_5d_eq_rolling_sharpe`
- `sec17_ret_5d_eq_sector_mean`
- `sec17_ret_5d_eq_sector_rel`
- `sec17_ret_5d_eq_zscore_20d`

**SECTOR** (7 columns):
- `sector_short_ratio_z20`
- `sector_short_ratio_z20_outlier_flag`
- `sector_short_ratio_z20_roll_mean_20d`
- `sector_short_ratio_z20_roll_std_20d`
- `sector_short_ratio_z20_sector_mean`
- `sector_short_ratio_z20_sector_rel`
- `sector_short_ratio_z20_zscore_20d`

**SHARES** (7 columns):
- `shares_out_delta_pct`
- `shares_out_delta_pct_outlier_flag`
- `shares_out_delta_pct_roll_mean_20d`
- `shares_out_delta_pct_roll_std_20d`
- `shares_out_delta_pct_sector_mean`
- `shares_out_delta_pct_sector_rel`
- `shares_out_delta_pct_zscore_20d`

**SHORT** (21 columns):
- `short_selling_ratio_market`
- `short_selling_ratio_market_outlier_flag`
- `short_selling_ratio_market_roll_mean_20d`
- `short_selling_ratio_market_roll_std_20d`
- `short_selling_ratio_market_sector_mean`
- `short_selling_ratio_market_sector_rel`
- `short_selling_ratio_market_zscore_20d`
- `short_selling_with_restrictions_ratio`
- `short_selling_with_restrictions_ratio_outlier_flag`
- `short_selling_with_restrictions_ratio_roll_mean_20d`
- `short_selling_with_restrictions_ratio_roll_std_20d`
- `short_selling_with_restrictions_ratio_sector_mean`
- `short_selling_with_restrictions_ratio_sector_rel`
- `short_selling_with_restrictions_ratio_zscore_20d`
- `short_selling_without_restrictions_ratio`
- `short_selling_without_restrictions_ratio_outlier_flag`
- `short_selling_without_restrictions_ratio_roll_mean_20d`
- `short_selling_without_restrictions_ratio_roll_std_20d`
- `short_selling_without_restrictions_ratio_sector_mean`
- `short_selling_without_restrictions_ratio_sector_rel`
- `short_selling_without_restrictions_ratio_zscore_20d`

**SQUEEZE** (7 columns):
- `squeeze_risk`
- `squeeze_risk_outlier_flag`
- `squeeze_risk_roll_mean_20d`
- `squeeze_risk_roll_std_20d`
- `squeeze_risk_sector_mean`
- `squeeze_risk_sector_rel`
- `squeeze_risk_zscore_20d`

**SSP** (7 columns):
- `ssp_ratio_component`
- `ssp_ratio_component_outlier_flag`
- `ssp_ratio_component_roll_mean_20d`
- `ssp_ratio_component_roll_std_20d`
- `ssp_ratio_component_sector_mean`
- `ssp_ratio_component_sector_rel`
- `ssp_ratio_component_zscore_20d`

**SUPPLY** (1 columns):
- `supply_shock`

**WEEKLY** (21 columns):
- `weekly_margin_long_pct_float`
- `weekly_margin_long_pct_float_outlier_flag`
- `weekly_margin_long_pct_float_roc_5d`
- `weekly_margin_long_pct_float_roc_5d_outlier_flag`
- `weekly_margin_long_pct_float_roc_5d_roll_mean_20d`
- `weekly_margin_long_pct_float_roc_5d_roll_std_20d`
- `weekly_margin_long_pct_float_roc_5d_sector_mean`
- `weekly_margin_long_pct_float_roc_5d_sector_rel`
- `weekly_margin_long_pct_float_roc_5d_zscore_20d`
- `weekly_margin_long_pct_float_roll_mean_20d`
- `weekly_margin_long_pct_float_roll_std_20d`
- `weekly_margin_long_pct_float_sector_mean`
- `weekly_margin_long_pct_float_sector_rel`
- `weekly_margin_long_pct_float_z20`
- `weekly_margin_long_pct_float_z20_outlier_flag`
- `weekly_margin_long_pct_float_z20_roll_mean_20d`
- `weekly_margin_long_pct_float_z20_roll_std_20d`
- `weekly_margin_long_pct_float_z20_sector_mean`
- `weekly_margin_long_pct_float_z20_sector_rel`
- `weekly_margin_long_pct_float_z20_zscore_20d`
- `weekly_margin_long_pct_float_zscore_20d`

#### High-NULL Columns (>95% but <100%)

| Column | NULL % |
|--------|--------|
| `concentrated_flow_signal_outlier_flag` | 100.00% |
| `concentrated_flow_signal_zscore_20d` | 100.00% |
| `flow_concentration_outlier_flag` | 100.00% |
| `flow_concentration_zscore_20d` | 100.00% |
| `foreign_domestic_divergence_outlier_flag` | 100.00% |
| `foreign_domestic_divergence_zscore_20d` | 100.00% |
| `foreign_persistence_outlier_flag` | 100.00% |
| `foreign_persistence_zscore_20d` | 100.00% |
| `foreign_sentiment_outlier_flag` | 100.00% |
| `foreign_sentiment_zscore_20d` | 100.00% |
| `institutional_accumulation_outlier_flag` | 100.00% |
| `institutional_accumulation_zscore_20d` | 100.00% |
| `institutional_persistence_outlier_flag` | 100.00% |
| `institutional_persistence_zscore_20d` | 100.00% |
| `retail_institutional_divergence_outlier_flag` | 100.00% |
| `retail_institutional_divergence_zscore_20d` | 100.00% |
| `smart_flow_indicator_outlier_flag` | 100.00% |
| `smart_flow_indicator_zscore_20d` | 100.00% |
| `rq_63_10` | 99.97% |
| `rq_63_10_sector_rel` | 99.97% |
| `rq_63_50` | 99.97% |
| `rq_63_50_sector_rel` | 99.97% |
| `rq_63_90` | 99.97% |
| `rq_63_90_sector_rel` | 99.97% |
| `days_since_sector33_change` | 99.95% |
| `days_since_sector33_change_sector_rel` | 99.95% |
| `sector33_changed_5d` | 99.95% |
| `ret_prev_60d` | 98.15% |
| `days_since_limit_outlier_flag` | 96.20% |
| `days_since_limit_zscore_20d` | 96.20% |
| `days_since_sq_roll_mean_20d` | 95.28% |
| `days_since_sq_roll_std_20d` | 95.28% |
| `days_to_sq_roll_mean_20d` | 95.28% |
| `days_to_sq_roll_std_20d` | 95.28% |


### 2024-2025 Combined (APEX)

- **Total Rows**: 1,750,195
- **Total Columns**: 2,775
- **All-NULL Columns**: 526
- **High-NULL Columns (>95%)**: 32

#### All-NULL Columns by Feature Group

**COMPANYNAMEENGLISH** (7 columns):
- `CompanyNameEnglish`
- `CompanyNameEnglish_outlier_flag`
- `CompanyNameEnglish_roll_mean_20d`
- `CompanyNameEnglish_roll_std_20d`
- `CompanyNameEnglish_sector_mean`
- `CompanyNameEnglish_sector_rel`
- `CompanyNameEnglish_zscore_20d`

**DISCLOSEDDATE** (1 columns):
- `DisclosedDate`

**ALPHA60** (6 columns):
- `alpha60_topix_outlier_flag`
- `alpha60_topix_roll_mean_20d`
- `alpha60_topix_roll_std_20d`
- `alpha60_topix_sector_mean`
- `alpha60_topix_sector_rel`
- `alpha60_topix_zscore_20d`

**BASIS** (7 columns):
- `basis_gate`
- `basis_gate_outlier_flag`
- `basis_gate_roll_mean_20d`
- `basis_gate_roll_std_20d`
- `basis_gate_sector_mean`
- `basis_gate_sector_rel`
- `basis_gate_zscore_20d`

**BD** (1 columns):
- `bd_net_mc`

**BETA60** (6 columns):
- `beta60_topix_outlier_flag`
- `beta60_topix_roll_mean_20d`
- `beta60_topix_roll_std_20d`
- `beta60_topix_sector_mean`
- `beta60_topix_sector_rel`
- `beta60_topix_zscore_20d`

**BUYBACK** (1 columns):
- `buyback_flag`

**CROWDING** (7 columns):
- `crowding_score`
- `crowding_score_outlier_flag`
- `crowding_score_roll_mean_20d`
- `crowding_score_roll_std_20d`
- `crowding_score_sector_mean`
- `crowding_score_sector_rel`
- `crowding_score_zscore_20d`

**DAYS** (22 columns):
- `days_since_market_change`
- `days_since_market_change_outlier_flag`
- `days_since_market_change_roll_mean_20d`
- `days_since_market_change_roll_std_20d`
- `days_since_market_change_sector_mean`
- `days_since_market_change_sector_rel`
- `days_since_market_change_zscore_20d`
- `days_since_sector33_change_outlier_flag`
- `days_since_sector33_change_roll_mean_20d`
- `days_since_sector33_change_roll_std_20d`
- `days_since_sector33_change_zscore_20d`
- `days_since_sq_outlier_flag`
- `days_since_sq_zscore_20d`
- `days_to_earnings`
- `days_to_earnings_outlier_flag`
- `days_to_earnings_roll_mean_20d`
- `days_to_earnings_roll_std_20d`
- `days_to_earnings_sector_mean`
- `days_to_earnings_sector_rel`
- `days_to_earnings_zscore_20d`
- `days_to_sq_outlier_flag`
- `days_to_sq_zscore_20d`

**DILUTION** (1 columns):
- `dilution_flag`

**DIV** (13 columns):
- `div_amount_12m`
- `div_amount_next`
- `div_days_since_ex`
- `div_days_to_ex`
- `div_dy_12m`
- `div_ex_cycle_z`
- `div_ex_drop_expected`
- `div_ex_gap_miss`
- `div_ex_gap_theo`
- `div_staleness_bd`
- `div_staleness_days`
- `div_yield_12m`
- `div_yield_ttm`

**DMI** (1 columns):
- `dmi_reason_code`

**EARNINGS** (8 columns):
- `earnings_event_date`
- `earnings_recent_1d`
- `earnings_recent_3d`
- `earnings_recent_5d`
- `earnings_today`
- `earnings_upcoming_1d`
- `earnings_upcoming_3d`
- `earnings_upcoming_5d`

**FLOAT** (21 columns):
- `float_turnover_pct`
- `float_turnover_pct_outlier_flag`
- `float_turnover_pct_roc_5d`
- `float_turnover_pct_roc_5d_outlier_flag`
- `float_turnover_pct_roc_5d_roll_mean_20d`
- `float_turnover_pct_roc_5d_roll_std_20d`
- `float_turnover_pct_roc_5d_sector_mean`
- `float_turnover_pct_roc_5d_sector_rel`
- `float_turnover_pct_roc_5d_zscore_20d`
- `float_turnover_pct_roll_mean_20d`
- `float_turnover_pct_roll_std_20d`
- `float_turnover_pct_sector_mean`
- `float_turnover_pct_sector_rel`
- `float_turnover_pct_z20`
- `float_turnover_pct_z20_outlier_flag`
- `float_turnover_pct_z20_roll_mean_20d`
- `float_turnover_pct_z20_roll_std_20d`
- `float_turnover_pct_z20_sector_mean`
- `float_turnover_pct_z20_sector_rel`
- `float_turnover_pct_z20_zscore_20d`
- `float_turnover_pct_zscore_20d`

**FS** (47 columns):
- `fs_accruals`
- `fs_accruals_ttm`
- `fs_average_shares`
- `fs_capex_ttm`
- `fs_cfo_to_ni`
- `fs_cfo_ttm`
- `fs_consolidated_flag`
- `fs_days_since`
- `fs_days_to_next`
- `fs_doc_family_1Q`
- `fs_doc_family_2Q`
- `fs_doc_family_3Q`
- `fs_doc_family_FY`
- `fs_equity_ratio`
- `fs_fcf_ttm`
- `fs_guidance_revision_flag`
- `fs_is_valid`
- `fs_lag_days`
- `fs_net_cash_ratio`
- `fs_net_income_ttm`
- `fs_net_margin`
- `fs_observation_count`
- `fs_op_margin`
- `fs_op_profit_ttm`
- `fs_revenue_ttm`
- `fs_roa_ttm`
- `fs_roe_ttm`
- `fs_sales_yoy`
- `fs_shares_outstanding`
- `fs_staleness_bd`
- `fs_standard_Foreign`
- `fs_standard_IFRS`
- `fs_standard_JGAAP`
- `fs_standard_JMIS`
- `fs_standard_US`
- `fs_ttm_cfo`
- `fs_ttm_cfo_margin`
- `fs_ttm_net_income`
- `fs_ttm_op_margin`
- `fs_ttm_op_profit`
- `fs_ttm_sales`
- `fs_window_e_pm1`
- `fs_window_e_pp3`
- `fs_window_e_pp5`
- `fs_yoy_ttm_net_income`
- `fs_yoy_ttm_op_profit`
- `fs_yoy_ttm_sales`

**GAP** (2 columns):
- `gap_atr`
- `gap_predictor`

**IDXOPT** (14 columns):
- `idxopt_vrp_gap`
- `idxopt_vrp_gap_outlier_flag`
- `idxopt_vrp_gap_roll_mean_20d`
- `idxopt_vrp_gap_roll_std_20d`
- `idxopt_vrp_gap_sector_mean`
- `idxopt_vrp_gap_sector_rel`
- `idxopt_vrp_gap_zscore_20d`
- `idxopt_vrp_ratio`
- `idxopt_vrp_ratio_outlier_flag`
- `idxopt_vrp_ratio_roll_mean_20d`
- `idxopt_vrp_ratio_roll_std_20d`
- `idxopt_vrp_ratio_sector_mean`
- `idxopt_vrp_ratio_sector_rel`
- `idxopt_vrp_ratio_zscore_20d`

**IS** (20 columns):
- `is_E_0`
- `is_E_pm1`
- `is_E_pp1`
- `is_E_pp3`
- `is_E_pp5`
- `is_fs_valid`
- `is_growth_x_dv_z20`
- `is_growth_x_dv_z20_outlier_flag`
- `is_growth_x_dv_z20_roll_mean_20d`
- `is_growth_x_dv_z20_roll_std_20d`
- `is_growth_x_dv_z20_sector_mean`
- `is_growth_x_dv_z20_sector_rel`
- `is_growth_x_dv_z20_zscore_20d`
- `is_prime_x_dv_z20`
- `is_prime_x_dv_z20_outlier_flag`
- `is_prime_x_dv_z20_roll_mean_20d`
- `is_prime_x_dv_z20_roll_std_20d`
- `is_prime_x_dv_z20_sector_mean`
- `is_prime_x_dv_z20_sector_rel`
- `is_prime_x_dv_z20_zscore_20d`

**LIMIT** (7 columns):
- `limit_up_flag_lag1`
- `limit_up_flag_lag1_outlier_flag`
- `limit_up_flag_lag1_roll_mean_20d`
- `limit_up_flag_lag1_roll_std_20d`
- `limit_up_flag_lag1_sector_mean`
- `limit_up_flag_lag1_sector_rel`
- `limit_up_flag_lag1_zscore_20d`

**MARGIN** (28 columns):
- `margin_long_pct_float`
- `margin_long_pct_float_outlier_flag`
- `margin_long_pct_float_roc_5d`
- `margin_long_pct_float_roc_5d_outlier_flag`
- `margin_long_pct_float_roc_5d_roll_mean_20d`
- `margin_long_pct_float_roc_5d_roll_std_20d`
- `margin_long_pct_float_roc_5d_sector_mean`
- `margin_long_pct_float_roc_5d_sector_rel`
- `margin_long_pct_float_roc_5d_zscore_20d`
- `margin_long_pct_float_roll_mean_20d`
- `margin_long_pct_float_roll_std_20d`
- `margin_long_pct_float_sector_mean`
- `margin_long_pct_float_sector_rel`
- `margin_long_pct_float_z20`
- `margin_long_pct_float_z20_outlier_flag`
- `margin_long_pct_float_z20_roll_mean_20d`
- `margin_long_pct_float_z20_roll_std_20d`
- `margin_long_pct_float_z20_sector_mean`
- `margin_long_pct_float_z20_sector_rel`
- `margin_long_pct_float_z20_zscore_20d`
- `margin_long_pct_float_zscore_20d`
- `margin_pain_index`
- `margin_pain_index_outlier_flag`
- `margin_pain_index_roll_mean_20d`
- `margin_pain_index_roll_std_20d`
- `margin_pain_index_sector_mean`
- `margin_pain_index_sector_rel`
- `margin_pain_index_zscore_20d`

**MARKET** (1 columns):
- `market_changed_5d`

**MKT** (170 columns):
- `mkt_flow_divergence_foreigners_vs_individuals`
- `mkt_flow_divergence_foreigners_vs_individuals_outlier_flag`
- `mkt_flow_divergence_foreigners_vs_individuals_roll_mean_20d`
- `mkt_flow_divergence_foreigners_vs_individuals_roll_std_20d`
- `mkt_flow_divergence_foreigners_vs_individuals_sector_mean`
- `mkt_flow_divergence_foreigners_vs_individuals_sector_rel`
- `mkt_flow_divergence_foreigners_vs_individuals_zscore_20d`
- `mkt_flow_flow_foreigners_net_ratio_z52`
- `mkt_flow_flow_foreigners_net_ratio_z52_outlier_flag`
- `mkt_flow_flow_foreigners_net_ratio_z52_roll_mean_20d`
- `mkt_flow_flow_foreigners_net_ratio_z52_roll_std_20d`
- `mkt_flow_flow_foreigners_net_ratio_z52_sector_mean`
- `mkt_flow_flow_foreigners_net_ratio_z52_sector_rel`
- `mkt_flow_flow_foreigners_net_ratio_z52_zscore_20d`
- `mkt_flow_flow_individuals_net_ratio_z52`
- `mkt_flow_flow_individuals_net_ratio_z52_outlier_flag`
- `mkt_flow_flow_individuals_net_ratio_z52_roll_mean_20d`
- `mkt_flow_flow_individuals_net_ratio_z52_roll_std_20d`
- `mkt_flow_flow_individuals_net_ratio_z52_sector_mean`
- `mkt_flow_flow_individuals_net_ratio_z52_sector_rel`
- `mkt_flow_flow_individuals_net_ratio_z52_zscore_20d`
- `mkt_flow_flow_investment_trusts_net_ratio_z52`
- `mkt_flow_flow_investment_trusts_net_ratio_z52_outlier_flag`
- `mkt_flow_flow_investment_trusts_net_ratio_z52_roll_mean_20d`
- `mkt_flow_flow_investment_trusts_net_ratio_z52_roll_std_20d`
- `mkt_flow_flow_investment_trusts_net_ratio_z52_sector_mean`
- `mkt_flow_flow_investment_trusts_net_ratio_z52_sector_rel`
- `mkt_flow_flow_investment_trusts_net_ratio_z52_zscore_20d`
- `mkt_flow_flow_trust_banks_net_ratio_z52`
- `mkt_flow_flow_trust_banks_net_ratio_z52_outlier_flag`
- `mkt_flow_flow_trust_banks_net_ratio_z52_roll_mean_20d`
- `mkt_flow_flow_trust_banks_net_ratio_z52_roll_std_20d`
- `mkt_flow_flow_trust_banks_net_ratio_z52_sector_mean`
- `mkt_flow_flow_trust_banks_net_ratio_z52_sector_rel`
- `mkt_flow_flow_trust_banks_net_ratio_z52_zscore_20d`
- `mkt_flow_foreigners_net`
- `mkt_flow_foreigners_net_outlier_flag`
- `mkt_flow_foreigners_net_ratio`
- `mkt_flow_foreigners_net_ratio_outlier_flag`
- `mkt_flow_foreigners_net_ratio_roll_mean_20d`
- `mkt_flow_foreigners_net_ratio_roll_std_20d`
- `mkt_flow_foreigners_net_ratio_sector_mean`
- `mkt_flow_foreigners_net_ratio_sector_rel`
- `mkt_flow_foreigners_net_ratio_turn_flag`
- `mkt_flow_foreigners_net_ratio_wow`
- `mkt_flow_foreigners_net_ratio_wow_outlier_flag`
- `mkt_flow_foreigners_net_ratio_wow_roll_mean_20d`
- `mkt_flow_foreigners_net_ratio_wow_roll_std_20d`
- `mkt_flow_foreigners_net_ratio_wow_sector_mean`
- `mkt_flow_foreigners_net_ratio_wow_sector_rel`
- `mkt_flow_foreigners_net_ratio_wow_zscore_20d`
- `mkt_flow_foreigners_net_ratio_z13`
- `mkt_flow_foreigners_net_ratio_z13_outlier_flag`
- `mkt_flow_foreigners_net_ratio_z13_roll_mean_20d`
- `mkt_flow_foreigners_net_ratio_z13_roll_std_20d`
- `mkt_flow_foreigners_net_ratio_z13_sector_mean`
- `mkt_flow_foreigners_net_ratio_z13_sector_rel`
- `mkt_flow_foreigners_net_ratio_z13_zscore_20d`
- `mkt_flow_foreigners_net_ratio_z52`
- `mkt_flow_foreigners_net_ratio_z52_outlier_flag`
- `mkt_flow_foreigners_net_ratio_z52_roll_mean_20d`
- `mkt_flow_foreigners_net_ratio_z52_roll_std_20d`
- `mkt_flow_foreigners_net_ratio_z52_sector_mean`
- `mkt_flow_foreigners_net_ratio_z52_sector_rel`
- `mkt_flow_foreigners_net_ratio_z52_zscore_20d`
- `mkt_flow_foreigners_net_ratio_zscore_20d`
- `mkt_flow_foreigners_net_roll_mean_20d`
- `mkt_flow_foreigners_net_roll_std_20d`
- `mkt_flow_foreigners_net_sector_mean`
- `mkt_flow_foreigners_net_sector_rel`
- `mkt_flow_foreigners_net_zscore_20d`
- `mkt_flow_individuals_net`
- `mkt_flow_individuals_net_outlier_flag`
- `mkt_flow_individuals_net_ratio`
- `mkt_flow_individuals_net_ratio_outlier_flag`
- `mkt_flow_individuals_net_ratio_roll_mean_20d`
- `mkt_flow_individuals_net_ratio_roll_std_20d`
- `mkt_flow_individuals_net_ratio_sector_mean`
- `mkt_flow_individuals_net_ratio_sector_rel`
- `mkt_flow_individuals_net_ratio_turn_flag`
- `mkt_flow_individuals_net_ratio_wow`
- `mkt_flow_individuals_net_ratio_wow_outlier_flag`
- `mkt_flow_individuals_net_ratio_wow_roll_mean_20d`
- `mkt_flow_individuals_net_ratio_wow_roll_std_20d`
- `mkt_flow_individuals_net_ratio_wow_sector_mean`
- `mkt_flow_individuals_net_ratio_wow_sector_rel`
- `mkt_flow_individuals_net_ratio_wow_zscore_20d`
- `mkt_flow_individuals_net_ratio_z13`
- `mkt_flow_individuals_net_ratio_z13_outlier_flag`
- `mkt_flow_individuals_net_ratio_z13_roll_mean_20d`
- `mkt_flow_individuals_net_ratio_z13_roll_std_20d`
- `mkt_flow_individuals_net_ratio_z13_sector_mean`
- `mkt_flow_individuals_net_ratio_z13_sector_rel`
- `mkt_flow_individuals_net_ratio_z13_zscore_20d`
- `mkt_flow_individuals_net_ratio_z52`
- `mkt_flow_individuals_net_ratio_z52_outlier_flag`
- `mkt_flow_individuals_net_ratio_z52_roll_mean_20d`
- `mkt_flow_individuals_net_ratio_z52_roll_std_20d`
- `mkt_flow_individuals_net_ratio_z52_sector_mean`
- `mkt_flow_individuals_net_ratio_z52_sector_rel`
- `mkt_flow_individuals_net_ratio_z52_zscore_20d`
- `mkt_flow_individuals_net_ratio_zscore_20d`
- `mkt_flow_individuals_net_roll_mean_20d`
- `mkt_flow_individuals_net_roll_std_20d`
- `mkt_flow_individuals_net_sector_mean`
- `mkt_flow_individuals_net_sector_rel`
- `mkt_flow_individuals_net_zscore_20d`
- `mkt_flow_investment_trusts_net`
- `mkt_flow_investment_trusts_net_outlier_flag`
- `mkt_flow_investment_trusts_net_ratio`
- `mkt_flow_investment_trusts_net_ratio_outlier_flag`
- `mkt_flow_investment_trusts_net_ratio_roll_mean_20d`
- `mkt_flow_investment_trusts_net_ratio_roll_std_20d`
- `mkt_flow_investment_trusts_net_ratio_sector_mean`
- `mkt_flow_investment_trusts_net_ratio_sector_rel`
- `mkt_flow_investment_trusts_net_ratio_z13`
- `mkt_flow_investment_trusts_net_ratio_z13_outlier_flag`
- `mkt_flow_investment_trusts_net_ratio_z13_roll_mean_20d`
- `mkt_flow_investment_trusts_net_ratio_z13_roll_std_20d`
- `mkt_flow_investment_trusts_net_ratio_z13_sector_mean`
- `mkt_flow_investment_trusts_net_ratio_z13_sector_rel`
- `mkt_flow_investment_trusts_net_ratio_z13_zscore_20d`
- `mkt_flow_investment_trusts_net_ratio_z52`
- `mkt_flow_investment_trusts_net_ratio_z52_outlier_flag`
- `mkt_flow_investment_trusts_net_ratio_z52_roll_mean_20d`
- `mkt_flow_investment_trusts_net_ratio_z52_roll_std_20d`
- `mkt_flow_investment_trusts_net_ratio_z52_sector_mean`
- `mkt_flow_investment_trusts_net_ratio_z52_sector_rel`
- `mkt_flow_investment_trusts_net_ratio_z52_zscore_20d`
- `mkt_flow_investment_trusts_net_ratio_zscore_20d`
- `mkt_flow_investment_trusts_net_roll_mean_20d`
- `mkt_flow_investment_trusts_net_roll_std_20d`
- `mkt_flow_investment_trusts_net_sector_mean`
- `mkt_flow_investment_trusts_net_sector_rel`
- `mkt_flow_investment_trusts_net_zscore_20d`
- `mkt_flow_total_net`
- `mkt_flow_total_net_outlier_flag`
- `mkt_flow_total_net_roll_mean_20d`
- `mkt_flow_total_net_roll_std_20d`
- `mkt_flow_total_net_sector_mean`
- `mkt_flow_total_net_sector_rel`
- `mkt_flow_total_net_zscore_20d`
- `mkt_flow_trust_banks_net`
- `mkt_flow_trust_banks_net_outlier_flag`
- `mkt_flow_trust_banks_net_ratio`
- `mkt_flow_trust_banks_net_ratio_outlier_flag`
- `mkt_flow_trust_banks_net_ratio_roll_mean_20d`
- `mkt_flow_trust_banks_net_ratio_roll_std_20d`
- `mkt_flow_trust_banks_net_ratio_sector_mean`
- `mkt_flow_trust_banks_net_ratio_sector_rel`
- `mkt_flow_trust_banks_net_ratio_z13`
- `mkt_flow_trust_banks_net_ratio_z13_outlier_flag`
- `mkt_flow_trust_banks_net_ratio_z13_roll_mean_20d`
- `mkt_flow_trust_banks_net_ratio_z13_roll_std_20d`
- `mkt_flow_trust_banks_net_ratio_z13_sector_mean`
- `mkt_flow_trust_banks_net_ratio_z13_sector_rel`
- `mkt_flow_trust_banks_net_ratio_z13_zscore_20d`
- `mkt_flow_trust_banks_net_ratio_z52`
- `mkt_flow_trust_banks_net_ratio_z52_outlier_flag`
- `mkt_flow_trust_banks_net_ratio_z52_roll_mean_20d`
- `mkt_flow_trust_banks_net_ratio_z52_roll_std_20d`
- `mkt_flow_trust_banks_net_ratio_z52_sector_mean`
- `mkt_flow_trust_banks_net_ratio_z52_sector_rel`
- `mkt_flow_trust_banks_net_ratio_z52_zscore_20d`
- `mkt_flow_trust_banks_net_ratio_zscore_20d`
- `mkt_flow_trust_banks_net_roll_mean_20d`
- `mkt_flow_trust_banks_net_roll_std_20d`
- `mkt_flow_trust_banks_net_sector_mean`
- `mkt_flow_trust_banks_net_sector_rel`
- `mkt_flow_trust_banks_net_zscore_20d`

**MPI** (28 columns):
- `mpi_dist_to_limit`
- `mpi_dist_to_limit_outlier_flag`
- `mpi_dist_to_limit_roll_mean_20d`
- `mpi_dist_to_limit_roll_std_20d`
- `mpi_dist_to_limit_sector_mean`
- `mpi_dist_to_limit_sector_rel`
- `mpi_dist_to_limit_z20`
- `mpi_dist_to_limit_z20_outlier_flag`
- `mpi_dist_to_limit_z20_roll_mean_20d`
- `mpi_dist_to_limit_z20_roll_std_20d`
- `mpi_dist_to_limit_z20_sector_mean`
- `mpi_dist_to_limit_z20_sector_rel`
- `mpi_dist_to_limit_z20_zscore_20d`
- `mpi_dist_to_limit_zscore_20d`
- `mpi_drawdown`
- `mpi_drawdown_outlier_flag`
- `mpi_drawdown_roll_mean_20d`
- `mpi_drawdown_roll_std_20d`
- `mpi_drawdown_sector_mean`
- `mpi_drawdown_sector_rel`
- `mpi_drawdown_z20`
- `mpi_drawdown_z20_outlier_flag`
- `mpi_drawdown_z20_roll_mean_20d`
- `mpi_drawdown_z20_roll_std_20d`
- `mpi_drawdown_z20_sector_mean`
- `mpi_drawdown_z20_sector_rel`
- `mpi_drawdown_z20_zscore_20d`
- `mpi_drawdown_zscore_20d`

**PREE** (21 columns):
- `preE_margin_diff`
- `preE_margin_diff_outlier_flag`
- `preE_margin_diff_roll_mean_20d`
- `preE_margin_diff_roll_std_20d`
- `preE_margin_diff_sector_mean`
- `preE_margin_diff_sector_rel`
- `preE_margin_diff_z20`
- `preE_margin_diff_z20_outlier_flag`
- `preE_margin_diff_z20_roll_mean_20d`
- `preE_margin_diff_z20_roll_std_20d`
- `preE_margin_diff_z20_sector_mean`
- `preE_margin_diff_z20_sector_rel`
- `preE_margin_diff_z20_zscore_20d`
- `preE_margin_diff_zscore_20d`
- `preE_risk_score`
- `preE_risk_score_outlier_flag`
- `preE_risk_score_roll_mean_20d`
- `preE_risk_score_roll_std_20d`
- `preE_risk_score_sector_mean`
- `preE_risk_score_sector_rel`
- `preE_risk_score_zscore_20d`

**REL** (7 columns):
- `rel_to_sec17_5d`
- `rel_to_sec17_5d_outlier_flag`
- `rel_to_sec17_5d_roll_mean_20d`
- `rel_to_sec17_5d_roll_std_20d`
- `rel_to_sec17_5d_sector_mean`
- `rel_to_sec17_5d_sector_rel`
- `rel_to_sec17_5d_zscore_20d`

**RET** (1 columns):
- `ret_prev_120d`

**RQ** (12 columns):
- `rq_63_10_outlier_flag`
- `rq_63_10_roll_mean_20d`
- `rq_63_10_roll_std_20d`
- `rq_63_10_zscore_20d`
- `rq_63_50_outlier_flag`
- `rq_63_50_roll_mean_20d`
- `rq_63_50_roll_std_20d`
- `rq_63_50_zscore_20d`
- `rq_63_90_outlier_flag`
- `rq_63_90_roll_mean_20d`
- `rq_63_90_roll_std_20d`
- `rq_63_90_zscore_20d`

**SEC17** (16 columns):
- `sec17_mom_20`
- `sec17_mom_20_outlier_flag`
- `sec17_mom_20_roll_mean_20d`
- `sec17_mom_20_roll_std_20d`
- `sec17_mom_20_sector_mean`
- `sec17_mom_20_sector_rel`
- `sec17_mom_20_zscore_20d`
- `sec17_ret_5d_eq`
- `sec17_ret_5d_eq_hist_vol`
- `sec17_ret_5d_eq_outlier_flag`
- `sec17_ret_5d_eq_roll_mean_20d`
- `sec17_ret_5d_eq_roll_std_20d`
- `sec17_ret_5d_eq_rolling_sharpe`
- `sec17_ret_5d_eq_sector_mean`
- `sec17_ret_5d_eq_sector_rel`
- `sec17_ret_5d_eq_zscore_20d`

**SECTOR** (7 columns):
- `sector_short_ratio_z20`
- `sector_short_ratio_z20_outlier_flag`
- `sector_short_ratio_z20_roll_mean_20d`
- `sector_short_ratio_z20_roll_std_20d`
- `sector_short_ratio_z20_sector_mean`
- `sector_short_ratio_z20_sector_rel`
- `sector_short_ratio_z20_zscore_20d`

**SHARES** (7 columns):
- `shares_out_delta_pct`
- `shares_out_delta_pct_outlier_flag`
- `shares_out_delta_pct_roll_mean_20d`
- `shares_out_delta_pct_roll_std_20d`
- `shares_out_delta_pct_sector_mean`
- `shares_out_delta_pct_sector_rel`
- `shares_out_delta_pct_zscore_20d`

**SQUEEZE** (7 columns):
- `squeeze_risk`
- `squeeze_risk_outlier_flag`
- `squeeze_risk_roll_mean_20d`
- `squeeze_risk_roll_std_20d`
- `squeeze_risk_sector_mean`
- `squeeze_risk_sector_rel`
- `squeeze_risk_zscore_20d`

**SSP** (7 columns):
- `ssp_ratio_component`
- `ssp_ratio_component_outlier_flag`
- `ssp_ratio_component_roll_mean_20d`
- `ssp_ratio_component_roll_std_20d`
- `ssp_ratio_component_sector_mean`
- `ssp_ratio_component_sector_rel`
- `ssp_ratio_component_zscore_20d`

**SUPPLY** (1 columns):
- `supply_shock`

**WEEKLY** (21 columns):
- `weekly_margin_long_pct_float`
- `weekly_margin_long_pct_float_outlier_flag`
- `weekly_margin_long_pct_float_roc_5d`
- `weekly_margin_long_pct_float_roc_5d_outlier_flag`
- `weekly_margin_long_pct_float_roc_5d_roll_mean_20d`
- `weekly_margin_long_pct_float_roc_5d_roll_std_20d`
- `weekly_margin_long_pct_float_roc_5d_sector_mean`
- `weekly_margin_long_pct_float_roc_5d_sector_rel`
- `weekly_margin_long_pct_float_roc_5d_zscore_20d`
- `weekly_margin_long_pct_float_roll_mean_20d`
- `weekly_margin_long_pct_float_roll_std_20d`
- `weekly_margin_long_pct_float_sector_mean`
- `weekly_margin_long_pct_float_sector_rel`
- `weekly_margin_long_pct_float_z20`
- `weekly_margin_long_pct_float_z20_outlier_flag`
- `weekly_margin_long_pct_float_z20_roll_mean_20d`
- `weekly_margin_long_pct_float_z20_roll_std_20d`
- `weekly_margin_long_pct_float_z20_sector_mean`
- `weekly_margin_long_pct_float_z20_sector_rel`
- `weekly_margin_long_pct_float_z20_zscore_20d`
- `weekly_margin_long_pct_float_zscore_20d`

#### High-NULL Columns (>95% but <100%)

| Column | NULL % |
|--------|--------|
| `concentrated_flow_signal_outlier_flag` | 100.00% |
| `concentrated_flow_signal_zscore_20d` | 100.00% |
| `flow_concentration_outlier_flag` | 100.00% |
| `flow_concentration_zscore_20d` | 100.00% |
| `foreign_domestic_divergence_outlier_flag` | 100.00% |
| `foreign_domestic_divergence_zscore_20d` | 100.00% |
| `foreign_persistence_outlier_flag` | 100.00% |
| `foreign_persistence_zscore_20d` | 100.00% |
| `foreign_sentiment_outlier_flag` | 100.00% |
| `foreign_sentiment_zscore_20d` | 100.00% |
| `institutional_accumulation_outlier_flag` | 100.00% |
| `institutional_accumulation_zscore_20d` | 100.00% |
| `institutional_persistence_outlier_flag` | 100.00% |
| `institutional_persistence_zscore_20d` | 100.00% |
| `retail_institutional_divergence_outlier_flag` | 100.00% |
| `retail_institutional_divergence_zscore_20d` | 100.00% |
| `smart_flow_indicator_outlier_flag` | 100.00% |
| `smart_flow_indicator_zscore_20d` | 100.00% |
| `rq_63_10` | 99.97% |
| `rq_63_10_sector_rel` | 99.97% |
| `rq_63_50` | 99.97% |
| `rq_63_50_sector_rel` | 99.97% |
| `rq_63_90` | 99.97% |
| `rq_63_90_sector_rel` | 99.97% |
| `days_since_sector33_change` | 99.92% |
| `days_since_sector33_change_sector_rel` | 99.92% |
| `sector33_changed_5d` | 99.92% |
| `ret_prev_60d` | 97.64% |
| `days_since_sq_roll_mean_20d` | 95.18% |
| `days_since_sq_roll_std_20d` | 95.18% |
| `days_to_sq_roll_mean_20d` | 95.18% |
| `days_to_sq_roll_std_20d` | 95.18% |

