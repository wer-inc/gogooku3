# Missing 40 Macro Columns - Root Cause Analysis & Fix

## Investigation Summary
**Date**: 2025-11-15  
**Issue**: All chunks (2020Q1, 2025Q1-Q4) failed schema validation with 40 missing columns  
**User Request**: "2. Option B: 40列の欠落が意図的でないと判断し、ビルドを再調査"  

---

## Root Cause

### Primary Issue: yfinance Module Not Installed
**Impact**: 40 macro/market feature columns not generated  

**Evidence**:
```
[2025-11-15 06:30:11,580] WARNING builder.features.macro.vix - yfinance not available; VIX history unavailable
[2025-11-15 06:30:11,584] WARNING builder.features.macro.global_regime - yfinance not available; global regime data unavailable
```

**Missing Features** (40 columns):
1. **VIX Features** (10 columns):
   - macro_vix_close, macro_vix_log_close
   - macro_vix_ret_1d, macro_vix_ret_5d, macro_vix_ret_10d, macro_vix_ret_20d
   - macro_vix_sma_ratio_5_20, macro_vix_spike
   - macro_vix_vol_20, macro_vix_vol_z

2. **VVMD Global Regime Features** (30 columns):
   - **Volatility (4)**: macro_vvmd_vol_spy_rv20, macro_vvmd_vol_spy_drv_20_63, macro_vvmd_vol_qqq_rv20, macro_vvmd_vol_vix_z_252d
   - **Volume (2)**: macro_vvmd_vlm_spy_surge20, macro_vvmd_vlm_qqq_surge20
   - **Momentum/Trend (13)**: SPY/QQQ momentum, MA gaps, breakout positions, term structures
   - **Demand (3)**: DXY, BTC relative momentum, BTC volatility
   - **Cross-market (8)**: VRP, credit spreads, rates term, VIX term, SPY overnight/intraday, FX USDJPY

**Code Location**:
- `/workspace/gogooku3/gogooku5/data/src/builder/features/macro/vix.py:43-46`
- `/workspace/gogooku3/gogooku5/data/src/builder/features/macro/global_regime.py:86-89`
- Fallback behavior: `raise_on_missing=False` → Returns empty DataFrame

---

### Secondary Issue: Categorical Type Conversion
**Impact**: 2 type mismatches in schema validation  

**Evidence**:
```
[2025-11-15 06:41:50,177] INFO builder.utils.lazy_io - [CATEGORICAL] Converting 3 columns to Categorical: ['Code', 'SectorCode', 'MarketCode']

ERROR: Type mismatches (2):
   SectorCode: expected String, got Categorical
   MarketCode: expected String, got Categorical
```

**Note**: `Code` is already Categorical in manifest, so only `SectorCode` and `MarketCode` are problematic.

---

## Fix Applied

### 1. Install yfinance
```bash
pip install yfinance
```

**Verification**:
```bash
python3 -c "import yfinance; print(f'✅ yfinance version: {yfinance.__version__}')"
```

### 2. Next Steps
- Delete existing incomplete chunks (2020Q1, 2025Q1-Q4)
- Rebuild chunks with yfinance available
- Verify all 40 macro columns are generated
- Address Categorical type conversion if needed

---

## Technical Details

### Why yfinance was missing:
- Not included in `gogooku5/data/pyproject.toml` or `requirements*.txt`
- Optional dependency with graceful degradation (no hard error)
- Feature modules check availability with `get_yfinance_module(raise_on_missing=False)`
- When missing, returns empty DataFrame → 40 columns omitted

### Data Sources (via yfinance):
- **VIX**: `^VIX` (CBOE Volatility Index)
- **US Equities**: `SPY` (S&P 500), `QQQ` (Nasdaq-100)
- **FX**: `JPY=X` (USD/JPY), `DX-Y.NYB` (US Dollar Index)
- **Crypto**: `BTC-USD` (Bitcoin)
- **Credit**: `HYG` (High Yield), `LQD` (Investment Grade)
- **Rates**: `TLT` (20Y Treasury), `IEF` (7-10Y Treasury)
- **VIX Term**: `^VIX9D`, `^VIX3M`

### Cache Behavior:
- First run: Fetches from Yahoo Finance API → Saves to `output/macro/`
- Subsequent runs: Reads from Parquet cache (3-5x faster with IPC)

---

## Validation Plan

After rebuilding chunks:
1. ✅ Check column count: 2767 columns (not 2727)
2. ✅ Verify schema hash: `2951c76cdc446355` (not `f5900870d0582812`)
3. ✅ Confirm macro features present: `macro_vix_*` and `macro_vvmd_*`
4. ✅ Run `scripts/check_chunk_status.py` → All chunks "completed"
5. ⚠️  Address Categorical type issue if validation still fails

