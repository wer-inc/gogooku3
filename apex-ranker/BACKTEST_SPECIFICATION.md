# APEX-Ranker Phase 3 Backtest Specification

**Version**: 1.0
**Date**: 2025-10-29
**Status**: Design Document

---

## Objectives

Phase 3 validates APEX-Ranker models over **2-3 years** with:
1. Walk-forward validation (avoid look-ahead bias)
2. Realistic transaction costs
3. Out-of-sample performance metrics
4. Model comparison (pruned vs enhanced)

**Success Criteria**:
- Annualized Sharpe Ratio > 1.5 (after costs)
- Maximum Drawdown < 20%
- Positive returns in >70% of rolling 3-month periods

---

## Walk-Forward Framework

### Training/Validation Split

**Rolling Window Approach**:
```
Train Window: 252 days (1 year)
Validation Window: 63 days (3 months)
Step Size: 21 days (1 month)

Timeline:
[-------Train 1-------][--Val 1--]
    [-------Train 2-------][--Val 2--]
        [-------Train 3-------][--Val 3--]
                ...
```

**Example** (2023-01-01 to 2025-10-29, ~2.8 years):
- Total days: ~700 trading days
- Walk-forward folds: ~30 folds
- Each fold: 252 days train + 63 days validation

### No Retraining (Static Models)

**Phase 3 Scope**: Use pre-trained models (v0_pruned, v0_enhanced)
- **Rationale**: Validate existing models, not retrain
- **Benefit**: Faster execution, focus on strategy evaluation
- **Future**: Phase 4 can add online learning / model updates

---

## Backtest Input/Output

### Input

**Required**:
1. **Model Checkpoint**: `models/apex_ranker_v0_{pruned,enhanced}.pt`
2. **Config File**: `apex-ranker/configs/v0_{pruned,base}.yaml`
3. **Dataset**: `output/ml_dataset_latest_full.parquet`
4. **Date Range**: Start date, end date (e.g., 2023-01-01 to 2025-10-29)

**Optional**:
- Top-K: Number of stocks to hold (default: 50)
- Initial capital: Starting portfolio value (default: ¥100M)
- Rebalance frequency: Daily, weekly, monthly (default: daily)
- Transaction cost parameters (use defaults from TRANSACTION_COST_MODEL.md)

### Output

**Primary Output**: `results/backtest_{model_name}_{date_range}.json`

```json
{
  "metadata": {
    "model": "apex_ranker_v0_pruned",
    "config": "apex-ranker/configs/v0_pruned.yaml",
    "start_date": "2023-01-01",
    "end_date": "2025-10-29",
    "initial_capital": 100000000,
    "top_k": 50,
    "rebalance_frequency": "daily"
  },
  "performance": {
    "total_return": 85.3,
    "annualized_return": 28.4,
    "sharpe_ratio": 2.15,
    "sortino_ratio": 3.42,
    "max_drawdown": -12.5,
    "calmar_ratio": 2.27,
    "win_rate": 0.687,
    "avg_turnover": 0.28,
    "total_trades": 35000,
    "transaction_costs": {
      "total_cost": 8500000,
      "cost_pct_of_pv": 8.5,
      "avg_daily_cost_bps": 3.4
    }
  },
  "timeseries": [
    {
      "date": "2023-01-04",
      "portfolio_value": 100000000,
      "daily_return": 0.0,
      "daily_pnl": 0,
      "positions": 50,
      "turnover": 1.0,
      "transaction_cost": 133000
    },
    ...
  ],
  "walk_forward_folds": [
    {
      "fold_id": 1,
      "train_start": "2023-01-04",
      "train_end": "2023-12-29",
      "val_start": "2024-01-04",
      "val_end": "2024-03-29",
      "val_sharpe": 2.5,
      "val_return": 12.3
    },
    ...
  ]
}
```

**Secondary Outputs**:
- **Daily positions CSV**: `results/positions_{model}_{date}.csv`
- **Daily metrics CSV**: `results/daily_metrics_{model}_{date}.csv`
- **Trade log CSV**: `results/trades_{model}_{date}.csv`

---

## Strategy Logic

### Daily Workflow

```python
# Pseudocode for single day backtest
def backtest_single_day(date, portfolio, model, dataset):
    # 1. Skip non-trading days
    if not is_trading_day(date):
        return

    # 2. Generate predictions at market close (T)
    predictions = inference_v0.generate_rankings(
        model=model,
        date=date,
        dataset=dataset,
        top_k=50,
        horizon=20  # 20-day forward
    )

    # 3. Get current portfolio
    current_positions = portfolio.get_positions(date)
    current_weights = portfolio.get_weights(date)

    # 4. Calculate target weights (equal-weight Top-K)
    target_codes = predictions["codes"][:50]
    target_weights = {code: 1.0 / 50 for code in target_codes}

    # 5. Identify trades (buys, sells, holds)
    trades = calculate_trades(current_weights, target_weights)

    # 6. Calculate transaction costs
    costs = 0.0
    for trade in trades:
        volume = dataset.get_volume(trade["code"], date)
        cost = calculate_total_cost(
            trade["value"],
            volume,
            direction=trade["direction"]
        )
        costs += cost

    # 7. Execute trades at closing prices
    execution_prices = dataset.get_close_prices(target_codes, date)
    portfolio.rebalance(target_weights, execution_prices, costs)

    # 8. Log trade details
    log_trades(date, trades, costs)

    # 9. Mark-to-market at next day's close (T+1)
    next_date = get_next_trading_day(date)
    next_prices = dataset.get_close_prices(portfolio.get_codes(), next_date)
    portfolio.update_value(next_prices)
    daily_return = portfolio.calculate_daily_return()

    # 10. Log daily metrics
    log_daily_metrics(next_date, portfolio, daily_return)

    return portfolio
```

### Portfolio Construction

**Equal-Weight Top-K**:
```python
target_weight_i = 1.0 / K  # for i in Top-K
target_weight_i = 0.0      # for i not in Top-K
```

**No Leverage**: Sum of weights = 1.0 (100% invested, no borrowing)

**No Short Selling**: All weights ≥ 0 (long-only strategy)

---

## Performance Metrics

### Returns
- **Total Return**: (Final Value / Initial Value - 1) × 100%
- **Annualized Return**: ((Final Value / Initial Value)^(252/N_days) - 1) × 100%
- **Daily Returns**: (Value_t / Value_{t-1} - 1)

### Risk
- **Sharpe Ratio**: (Annualized Return - Risk-Free Rate) / Annualized Volatility
  - Risk-Free Rate: 0% (Japanese short-term rate ~0%)
- **Sortino Ratio**: (Annualized Return) / Downside Deviation
  - Downside Deviation: Std of negative returns only
- **Max Drawdown**: Max((Peak - Trough) / Peak) over period
- **Calmar Ratio**: Annualized Return / Max Drawdown

### Trading Activity
- **Turnover**: (Sum|Δweight_i|) / 2 per day
- **Average Turnover**: Mean daily turnover
- **Total Trades**: Count of buy + sell transactions
- **Win Rate**: Fraction of profitable trades

### Cost Analysis
- **Total Transaction Cost**: Sum of all costs (JPY)
- **Cost as % of Portfolio**: Total Cost / Average Portfolio Value
- **Avg Daily Cost (bps)**: Mean daily cost in basis points

---

## Script Interface

### Command-Line Interface

```bash
# Basic usage
python apex-ranker/scripts/backtest_walkforward_v0.py \
  --model models/apex_ranker_v0_pruned.pt \
  --config apex-ranker/configs/v0_pruned.yaml \
  --start-date 2023-01-01 \
  --end-date 2025-10-29 \
  --output results/backtest_pruned_2023_2025.json

# Compare multiple models
python apex-ranker/scripts/backtest_walkforward_v0.py \
  --models models/apex_ranker_v0_pruned.pt models/apex_ranker_v0_enhanced.pt \
  --configs apex-ranker/configs/v0_pruned.yaml apex-ranker/configs/v0_base.yaml \
  --start-date 2023-01-01 \
  --end-date 2025-10-29 \
  --output results/backtest_comparison.json

# Custom parameters
python apex-ranker/scripts/backtest_walkforward_v0.py \
  --model models/apex_ranker_v0_pruned.pt \
  --config apex-ranker/configs/v0_pruned.yaml \
  --start-date 2023-01-01 \
  --end-date 2025-10-29 \
  --initial-capital 500000000 \
  --top-k 30 \
  --rebalance-freq weekly \
  --output results/backtest_custom.json
```

### Python API

```python
from apex_ranker.backtest import WalkForwardBacktest, Portfolio

# Initialize backtest
backtest = WalkForwardBacktest(
    model_path="models/apex_ranker_v0_pruned.pt",
    config_path="apex-ranker/configs/v0_pruned.yaml",
    dataset_path="output/ml_dataset_latest_full.parquet",
    start_date="2023-01-01",
    end_date="2025-10-29",
    initial_capital=100_000_000,
    top_k=50,
)

# Run backtest
results = backtest.run()

# Access results
print(f"Sharpe Ratio: {results['performance']['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['performance']['max_drawdown']:.2f}%")

# Save results
results.save("results/backtest_pruned.json")
```

---

## Implementation Tasks

### Task Breakdown

**Phase 3.1: Core Infrastructure** (2-3 days)
1. ✅ Transaction cost model (TRANSACTION_COST_MODEL.md)
2. ⏳ Portfolio class with position tracking
3. ⏳ Walk-forward split generator
4. ⏳ Daily metrics logger

**Phase 3.2: Backtest Engine** (2-3 days)
5. ⏳ Main backtest loop
6. ⏳ Integration with inference_v0.py
7. ⏳ Cost calculation module
8. ⏳ Trade execution simulator

**Phase 3.3: Metrics & Reporting** (1-2 days)
9. ⏳ Performance metrics calculation
10. ⏳ JSON/CSV output formatters
11. ⏳ Summary report generator

**Phase 3.4: Validation & Testing** (1 day)
12. ⏳ Sanity checks (no look-ahead, realistic costs)
13. ⏳ Comparison with existing backtest_v0.py results
14. ⏳ Edge case testing (missing data, holidays)

**Total Estimated Time**: 6-9 days

---

## Data Requirements

### Dataset Columns (from `output/ml_dataset_latest_full.parquet`)

**Required**:
- `Date`: Trading date
- `Code`: Stock code (4-digit)
- `close`: Closing price
- `volume` or `turnover_value`: Daily trading volume
- Feature columns (64 for pruned, 89 for enhanced)

**Validation**:
```python
# Check data availability
import polars as pl
df = pl.read_parquet("output/ml_dataset_latest_full.parquet")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Unique stocks: {df['Code'].n_unique()}")
print(f"Total rows: {len(df):,}")

# Check for missing data
missing_dates = df.group_by("Date").agg(pl.col("Code").count().alias("n_stocks"))
print(missing_dates.filter(pl.col("n_stocks") < 3000))  # Flag sparse dates
```

---

## Known Limitations & Assumptions

### Simplifications
1. **No intraday execution**: All trades at closing price
2. **No partial fills**: Assume full execution at target weight
3. **No market closure handling**: Assume continuous trading
4. **No dividend/corporate actions**: Not modeled in Phase 3
5. **No borrowing costs**: Long-only, no margin

### Data Limitations
1. **Survivorship bias**: Dataset may exclude delisted stocks
2. **Point-in-time accuracy**: Feature availability may have look-ahead
3. **Volume data quality**: May have missing/incorrect values

### Mitigation
- Document assumptions clearly
- Perform sensitivity analysis
- Compare with simpler baseline (buy-and-hold, equal-weight universe)

---

## Validation Checks

### Pre-Backtest Validation
- [ ] Model loads successfully
- [ ] Dataset covers full date range
- [ ] All required columns present
- [ ] No data leakage (features only use past data)

### During-Backtest Validation
- [ ] Portfolio value never negative
- [ ] Sum of weights = 1.0 ± 1e-6
- [ ] All trades have valid prices
- [ ] Transaction costs reasonable (<1% per day)

### Post-Backtest Validation
- [ ] Sharpe ratio in plausible range (-2 to 10)
- [ ] Max drawdown < 100%
- [ ] Total return matches sum of daily returns
- [ ] Final portfolio value matches calculated value

---

## Comparison Baseline

### Benchmark Strategies

1. **Buy-and-Hold Top-50** (Static)
   - Select Top-50 on day 1, hold entire period
   - Minimal transaction costs

2. **Equal-Weight Universe** (Dynamic)
   - Hold all stocks equally, rebalance monthly
   - Compare diversification benefit

3. **Momentum Strategy** (Simple)
   - Select Top-50 by past 20-day return
   - Compare model alpha vs simple momentum

---

## Next Steps

1. **Immediate**: Implement Portfolio class
2. **Next**: Implement walk-forward split generator
3. **Then**: Build main backtest loop
4. **Finally**: Add metrics & reporting

---

**Generated**: 2025-10-29
**Author**: Claude Code (Autonomous Development Agent)
