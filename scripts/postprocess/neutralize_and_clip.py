#!/usr/bin/env python3
"""
Neutralize and clip predictions for better live performance.

Input: a parquet/csv with at least columns: Date, Code, predicted_return
Optional columns: market, sector

Steps per Date:
- z-score standardize predicted_return cross-sectionally
- neutralize by market/sector (subtract group mean if columns present)
- clip to percentile (e.g., 99.5)
- scale to target gross exposure and cap per-name weight

Output: parquet with columns: Date, Code, score, weight
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def neutralize(df: pd.DataFrame, by_cols: list[str]) -> pd.Series:
    if not by_cols:
        return df["score"]
    s = df["score"].copy()
    try:
        for c in by_cols:
            if c in df.columns:
                s = s - df.groupby(c)["score"].transform("mean")
        return s
    except Exception:
        return s


def per_date_process(g: pd.DataFrame, by_cols: list[str], clip_q: float, cap: float, gross: float) -> pd.DataFrame:
    out = g.copy()
    # z-score
    mu = out["predicted_return"].mean()
    sd = out["predicted_return"].std() + 1e-8
    out["score"] = (out["predicted_return"] - mu) / sd
    # neutralize
    out["score"] = neutralize(out, by_cols)
    # clip
    lo = out["score"].quantile(1 - clip_q)
    hi = out["score"].quantile(clip_q)
    out["score"] = out["score"].clip(lo, hi)
    # weights: proportional to score, zero-sum
    s = out["score"]
    if s.abs().sum() > 0:
        w = s / (s.abs().sum() + 1e-12) * gross
    else:
        w = np.zeros(len(out))
    # cap
    w = np.clip(w, -cap, cap)
    # re-normalize gross exposure after cap
    if np.sum(np.abs(w)) > 0:
        w = w / np.sum(np.abs(w)) * gross
    out["weight"] = w
    return out[["Date", "Code", "score", "weight"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="predictions parquet/csv with Date,Code,predicted_return")
    ap.add_argument("--output", required=True, help="output parquet path")
    ap.add_argument("--by", default="market,sector", help="neutralize by columns (comma) if present")
    ap.add_argument("--clip-quantile", type=float, default=0.995, help="two-sided clipping quantile")
    ap.add_argument("--cap", type=float, default=0.01, help="per-name cap (as fraction)")
    ap.add_argument("--gross", type=float, default=1.0, help="target gross exposure (sum abs weights)")
    args = ap.parse_args()

    inp = Path(args.input)
    if inp.suffix.lower() == ".csv":
        df = pd.read_csv(inp)
    else:
        df = pd.read_parquet(inp)
    # basic checks
    for c in ("Date", "Code", "predicted_return"):
        if c not in df.columns:
            raise ValueError(f"missing required column: {c}")
    df["Date"] = pd.to_datetime(df["Date"])

    by_cols = [c.strip() for c in args.by.split(",") if c.strip()]
    out = (
        df.groupby("Date", group_keys=False)
        .apply(lambda g: per_date_process(g, by_cols, args.clip_quantile, args.cap, args.gross))
        .reset_index(drop=True)
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"✅ wrote: {args.output} rows={len(out)}")


if __name__ == "__main__":
    main()

