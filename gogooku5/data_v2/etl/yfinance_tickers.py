"""Default yfinance ticker sets (aligned with data macro features)."""

from __future__ import annotations

# Global regime set (mirrors data/src/builder/features/macro/global_regime.py)
MACRO_TICKERS: list[str] = [
    "SPY",
    "QQQ",
    "^VIX",
    "DX-Y.NYB",  # US Dollar Index
    "BTC-USD",
    "JPY=X",
    "HYG",
    "LQD",
    "TLT",
    "IEF",
    "^VIX9D",
    "^VIX3M",
    "GC=F",  # Gold futures (COMEX)
    "CL=F",  # WTI crude oil futures
    "FEZ",  # Euro STOXX 50 ETF
    "EEM",  # MSCI Emerging Markets ETF
    "EMB",  # EM USD sovereign bond ETF
    "DBC",  # Broad commodity basket
    "JNK",  # High yield (alt to HYG)
]

# Fallbacks: if primary fails/empty, try fallback ticker but store under original name
MACRO_FALLBACKS: dict[str, str] = {
    "DX-Y.NYB": "UUP",  # DXY fallback
}
