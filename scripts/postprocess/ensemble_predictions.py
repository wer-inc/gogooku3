#!/usr/bin/env python3
"""
Average-ensemble multiple prediction files.

Each input parquet/csv must have columns: Date, Code, predicted_return.
Outputs a single parquet with the same schema and predicted_return = mean across inputs.
"""

import argparse
from pathlib import Path

import pandas as pd


def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="input prediction files (parquet/csv)")
    ap.add_argument("--output", required=True, help="output parquet path")
    args = ap.parse_args()

    paths = [Path(p) for p in args.inputs]
    dfs = [read_any(p)[["Date", "Code", "predicted_return"]] for p in paths]

    # sequential join on Date+Code and average
    from functools import reduce
    def join_two(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
        return a.merge(b, on=["Date", "Code"], how="inner", suffixes=("", "_b"))

    merged = reduce(join_two, dfs)
    # collect predicted_return columns
    pr_cols = [c for c in merged.columns if c.startswith("predicted_return")]
    merged["predicted_return"] = merged[pr_cols].mean(axis=1)
    out = merged[["Date", "Code", "predicted_return"]]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"✅ wrote ensemble predictions: {args.output} rows={len(out)}")


if __name__ == "__main__":
    main()

