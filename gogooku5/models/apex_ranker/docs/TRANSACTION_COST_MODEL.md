# Transaction Cost Model for APEX-Ranker Backtest

**Version**: 1.0
**Date**: 2025-10-29
**Status**: Phase 3 Specification

---

## Overview

This document defines the transaction cost model and market microstructure assumptions for APEX-Ranker's Phase 3 long-term backtest validation.

## Market Microstructure Assumptions

### Trading Venue
- **Primary Exchange**: Tokyo Stock Exchange (TSE)
- **Trading Hours**: 9:00-11:30 (morning session), 12:30-15:00 (afternoon session)
- **Settlement**: T+2 (trade date + 2 business days)

### Order Execution
- **Execution Time**: Market open (9:00-9:30) or market close (14:30-15:00)
- **Order Type**: Market orders (immediate execution at prevailing price)
- **Liquidity Assumption**: Sufficient liquidity for Top-50 stocks (no partial fills)

### Price Formation
- **Reference Price**: Daily closing price from dataset
- **Execution Price**: Closing price + slippage
- **No Look-Ahead Bias**: Only use data available at T (no T+1 information)

---

## Transaction Cost Components

### 1. Commission (固定手数料)

**Model**: Tiered commission based on trade value

```python
def calculate_commission(trade_value_jpy: float) -> float:
    """
    Calculate commission based on Japanese broker fee structure.

    Typical online broker fees (SBI Securities, Rakuten Securities):
    - ¥5M以下: 0.055% (min ¥55, max ¥550)
    - ¥5M-¥10M: 0.033% (max ¥1,650)
    - ¥10M-¥20M: 0.022% (max ¥3,300)
    - ¥20M超: 0.011% (max ¥11,000)
    """
    if trade_value_jpy <= 5_000_000:
        commission = trade_value_jpy * 0.00055
        return max(55, min(commission, 550))
    elif trade_value_jpy <= 10_000_000:
        commission = trade_value_jpy * 0.00033
        return min(commission, 1650)
    elif trade_value_jpy <= 20_000_000:
        commission = trade_value_jpy * 0.00022
        return min(commission, 3300)
    else:
        commission = trade_value_jpy * 0.00011
        return min(commission, 11000)
```

**Rationale**:
- Based on actual Japanese online broker fee schedules
- Conservative estimate (mid-tier broker)
- Includes both buy and sell commissions

**Effective Rate**:
- Small trades (<¥5M): ~0.055%
- Medium trades (¥5M-¥20M): ~0.022-0.033%
- Large trades (>¥20M): ~0.011%
- **Average for Top-50 portfolio**: ~0.02-0.03%

---

### 2. Slippage (市場インパクト)

**Model**: Linear market impact model

```python
def calculate_slippage(
    trade_value_jpy: float,
    daily_volume_jpy: float,
    direction: str,  # "buy" or "sell"
) -> float:
    """
    Calculate slippage based on trade size relative to daily volume.

    Slippage = base_spread + market_impact

    Parameters:
    - trade_value_jpy: Trade value in JPY
    - daily_volume_jpy: Daily trading volume in JPY
    - direction: "buy" (positive slippage) or "sell" (negative slippage)

    Returns:
    - Slippage in basis points (bps)
    """
    # Bid-ask spread (TSE Prime stocks)
    base_spread_bps = 5.0  # ~0.05% for liquid stocks

    # Market impact: Linear in participation rate
    participation_rate = trade_value_jpy / daily_volume_jpy
    market_impact_bps = participation_rate * 100  # 1% participation = 1 bps

    # Total slippage
    total_slippage_bps = base_spread_bps + market_impact_bps

    # Direction adjustment
    sign = 1 if direction == "buy" else -1

    return sign * total_slippage_bps
```

**Parameters**:
- **Base Spread**: 5 bps (0.05%) for TSE Prime liquid stocks
  - Rationale: Typical bid-ask spread for large-cap Japanese equities
- **Market Impact**: Linear function of participation rate
  - 1% of daily volume → 1 bps additional impact
  - 5% of daily volume → 5 bps additional impact
  - Cap at 20 bps (0.2%) for extreme cases

**Participation Rate Assumptions**:
- **Portfolio Value**: ¥100M - ¥1B (configurable)
- **Top-K stocks**: 50
- **Per-stock allocation**: ¥2M - ¥20M
- **Typical daily volume**: ¥500M - ¥5B for Top-50 stocks
- **Typical participation**: 0.1% - 2% (low impact)

**Effective Slippage**:
- Buy orders: +5 to +10 bps (0.05% - 0.10%)
- Sell orders: -5 to -10 bps (0.05% - 0.10%)
- **Average round-trip**: ~10-20 bps (0.10% - 0.20%)

---

### 3. Total Transaction Cost

**Round-Trip Cost** (Buy + Hold + Sell):

```python
def calculate_total_cost(
    position_value_jpy: float,
    daily_volume_jpy: float,
) -> dict:
    """
    Calculate total round-trip transaction cost.

    Returns:
    - buy_cost: Cost to enter position (commission + slippage)
    - sell_cost: Cost to exit position (commission + slippage)
    - total_cost: Total round-trip cost
    - total_cost_bps: Total cost in basis points
    """
    # Buy transaction
    buy_commission = calculate_commission(position_value_jpy)
    buy_slippage_bps = calculate_slippage(position_value_jpy, daily_volume_jpy, "buy")
    buy_slippage_jpy = position_value_jpy * (buy_slippage_bps / 10000)
    buy_cost = buy_commission + buy_slippage_jpy

    # Sell transaction
    sell_commission = calculate_commission(position_value_jpy)
    sell_slippage_bps = calculate_slippage(position_value_jpy, daily_volume_jpy, "sell")
    sell_slippage_jpy = position_value_jpy * (sell_slippage_bps / 10000)
    sell_cost = sell_commission + abs(sell_slippage_jpy)

    # Total
    total_cost = buy_cost + sell_cost
    total_cost_bps = (total_cost / position_value_jpy) * 10000

    return {
        "buy_cost": buy_cost,
        "sell_cost": sell_cost,
        "total_cost": total_cost,
        "total_cost_bps": total_cost_bps,
    }
```

**Typical Values** (for ¥10M position with ¥1B daily volume):
- Buy commission: ¥1,650 (0.0165%)
- Buy slippage: ¥5,000 (0.05%)
- Sell commission: ¥1,650 (0.0165%)
- Sell slippage: ¥5,000 (0.05%)
- **Total round-trip**: ¥13,300 (0.133% or 13.3 bps)

---

## Portfolio Rebalancing Cost

### Turnover Calculation

**Definition**: Fraction of portfolio value that is traded per period.

```python
turnover = (sum(|new_weight_i - old_weight_i|) / 2)
```

**Expected Turnover** (for Top-50 daily ranking):
- **Full rebalance**: 100% turnover (all positions changed)
- **Partial rebalance**: 20-40% turnover (some positions held)
- **Assumption for backtest**: 30% daily turnover (conservative)

### Portfolio-Level Transaction Cost

```python
def calculate_portfolio_cost(
    portfolio_value: float,
    turnover: float,
    avg_cost_bps: float = 13.3,  # From example above
) -> float:
    """
    Calculate portfolio-level transaction cost.

    Parameters:
    - portfolio_value: Total portfolio value (JPY)
    - turnover: Fraction of portfolio traded (0.0 - 1.0)
    - avg_cost_bps: Average round-trip cost per position (bps)

    Returns:
    - Total transaction cost (JPY)
    """
    traded_value = portfolio_value * turnover
    cost = traded_value * (avg_cost_bps / 10000)
    return cost
```

**Example** (¥100M portfolio, 30% turnover):
- Traded value: ¥30M
- Transaction cost: ¥30M × 0.00133 = ¥39,900
- **Daily cost**: 0.04% of portfolio value
- **Annualized cost**: ~10% of portfolio value (0.04% × 250 days)

---

## Market Impact Mitigation Strategies

### 1. Execution Algorithm
**Not Implemented in Phase 3**, but noted for future:
- TWAP (Time-Weighted Average Price) execution
- VWAP (Volume-Weighted Average Price) execution
- Iceberg orders (隠し注文)

### 2. Order Timing
**Phase 3 Assumption**: Single execution at market close (15:00)
- Pro: Simple, reproducible
- Con: No intraday optimization

### 3. Position Sizing
**Volume-Based Limits**:
```python
max_participation_rate = 0.05  # Max 5% of daily volume
max_position_value = daily_volume_jpy * max_participation_rate
```

---

## Data Requirements

### For Accurate Cost Calculation

**Required Fields** (from dataset):
1. **Daily Volume**: `volume` or `turnover_value` column
2. **Daily Close Price**: `close` column
3. **Stock Code**: `Code` column
4. **Date**: `Date` column

**Volume Data Availability**:
- ✅ Available in JQuants API (`/prices/daily_quotes`)
- ✅ Cached in `output/ml_dataset_latest_full.parquet`

---

## Backtest Implementation

### Cost Calculation Workflow

```python
# Pseudocode for backtest loop
for date in backtest_dates:
    # 1. Generate predictions at T
    predictions = inference_v0.generate_rankings(date, model)

    # 2. Get current portfolio weights at T
    current_weights = portfolio.get_weights(date)

    # 3. Calculate target weights (Top-K equal-weight)
    target_weights = calculate_target_weights(predictions, top_k=50)

    # 4. Calculate turnover
    turnover = calculate_turnover(current_weights, target_weights)

    # 5. Calculate transaction costs
    for stock in rebalanced_stocks:
        volume = get_daily_volume(stock, date)
        trade_value = abs(target_weight - current_weight) * portfolio_value
        cost = calculate_total_cost(trade_value, volume)
        total_cost += cost

    # 6. Execute trades at closing price (T)
    portfolio.rebalance(target_weights, execution_price="close")

    # 7. Apply costs to portfolio value
    portfolio.apply_costs(total_cost)

    # 8. Mark-to-market at T+1
    portfolio.update_value(date + 1)
```

---

## Sensitivity Analysis

### Key Parameters to Test

1. **Commission Rate**: 0.01% - 0.05%
2. **Base Spread**: 3 bps - 10 bps
3. **Market Impact**: Linear vs Square-root model
4. **Participation Rate**: 0.5% - 5% cap
5. **Turnover**: 10% - 100%

### Expected Impact on Sharpe Ratio

**Baseline** (no costs):
- Sharpe Ratio: ~8.65 (from previous backtest)

**With Costs** (30% turnover, 13.3 bps round-trip):
- Daily cost: ~0.04%
- Annualized cost: ~10%
- Expected Sharpe: ~4-5 (50% reduction)

**Breakeven Turnover**:
- If annual return = 40%, annual cost = 10%
- Net return = 30%
- Sharpe degradation tolerable if IC remains strong

---

## Validation & Sanity Checks

### 1. Cost Reasonableness
- Round-trip cost: 10-20 bps (✅ typical for liquid Japanese stocks)
- Daily cost: <0.1% of portfolio (✅ reasonable)
- Annual cost: 5-15% (✅ in line with high-frequency strategies)

### 2. Volume Feasibility
- Max participation: <5% of daily volume (✅ achievable)
- No single stock >10% of portfolio (✅ diversified)

### 3. Historical Precedents
- Quantitative equity strategies: 2-5% annual cost (✅ our model higher due to daily rebalance)
- Market-making strategies: 10-30% annual cost (✅ we're in lower end)

---

## References

### Japanese Market Structure
- TSE Trading Regulations: https://www.jpx.co.jp/english/
- Bid-Ask Spread Studies: Typical 3-10 bps for Prime stocks

### Transaction Cost Models
- Almgren-Chriss (2000): Optimal execution with market impact
- Grinold-Kahn (2000): Active Portfolio Management (Chapter 17)
- Kissell-Glantz (2003): Optimal Trading Strategies

### Broker Fee Schedules
- SBI Securities: https://www.sbisec.co.jp/ETGate/WPLETmgR001Control?...
- Rakuten Securities: https://www.rakuten-sec.co.jp/web/domestic/...

---

## Implementation Checklist

Phase 3 Backtest Requirements:
- [ ] Commission calculation function
- [ ] Slippage calculation function
- [ ] Portfolio turnover tracking
- [ ] Daily volume data integration
- [ ] Cost logging and reporting
- [ ] Sensitivity analysis runner
- [ ] Net return (gross return - costs) calculation

---

**Generated**: 2025-10-29
**Author**: Claude Code (Autonomous Development Agent)
