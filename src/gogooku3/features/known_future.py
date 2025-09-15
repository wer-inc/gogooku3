from __future__ import annotations

"""Known-future feature helpers: holidays and earnings/event flags.

These helpers enrich the observation DataFrame with known-in-advance signals.
If a separate known-future DataFrame is provided, left-join it as well.
"""

import jpholiday
import pandas as pd


def add_jp_holiday_features(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    d = df.copy()
    d[ts_col] = pd.to_datetime(d[ts_col])  # type: ignore[assignment]
    d["weekday"] = d[ts_col].dt.weekday
    d["month"] = d[ts_col].dt.month
    d["is_month_end"] = d[ts_col].dt.is_month_end.astype(int)
    d["is_quarter_end"] = d[ts_col].dt.is_quarter_end.astype(int)
    # Japanese holiday flag (1/0)
    d["holiday"] = d[ts_col].dt.date.map(lambda x: 1 if jpholiday.is_holiday(x) else 0)
    return d


def normalize_static(df_static: pd.DataFrame) -> pd.DataFrame:
    """Ensure static features include size buckets if market_cap present."""
    s = df_static.copy()
    if "market_cap" in s.columns and "size_bucket" not in s.columns:
        # Tertiles S/M/L by id
        q = s["market_cap"].rank(pct=True)
        bucket = pd.Series(index=s.index, dtype=object)
        bucket[q <= 1/3] = "S"; bucket[(q>1/3) & (q<=2/3)] = "M"; bucket[q>2/3] = "L"
        s["size_bucket"] = bucket.astype(str)
    # Liquidity bucket if ADV20 or avg_volume exists
    adv_col = None
    for c in ("adv20", "ADV20", "avg_volume"):
        if c in s.columns:
            adv_col = c
            break
    if adv_col and "liquidity_bucket" not in s.columns:
        q = s[adv_col].rank(pct=True)
        lb = pd.Series(index=s.index, dtype=object)
        lb[q <= 1/3] = "LQ_LOW"; lb[(q>1/3) & (q<=2/3)] = "LQ_MED"; lb[q>2/3] = "LQ_HIGH"
        s["liquidity_bucket"] = lb.astype(str)
    return s


def add_event_flags(df: pd.DataFrame, df_events: pd.DataFrame) -> pd.DataFrame:
    """Left-join event flags (earnings, etc.).

    df_events supports wildcard id='*'. Known columns are copied (e.g., 'event_earnings', 'event_fop_expiry').
    """
    if df_events is None or df_events.empty:
        return df
    ev = df_events.copy()
    ev["ts"] = pd.to_datetime(ev["ts"])  # type: ignore[assignment]
    d = df.copy()
    d["ts"] = pd.to_datetime(d["ts"])  # type: ignore[assignment]
    # Broadcast
    broadcast = ev[ev.get("id", "").astype(str) == "*"] if "id" in ev.columns else ev.iloc[0:0]
    specific = ev if broadcast.empty else ev[ev.get("id", "").astype(str) != "*"]
    if not specific.empty:
        d = d.merge(specific, on=["id", "ts"], how="left")
    if not broadcast.empty:
        d = d.merge(broadcast.drop(columns=["id"]), on=["ts"], how="left")
    # Fill NaNs in boolean-like event columns with 0
    for c in d.columns:
        if c.startswith("event_") or c.endswith("_event"):
            d[c] = d[c].fillna(0).astype(int)
    return d
