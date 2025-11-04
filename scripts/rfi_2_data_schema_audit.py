"""
RFI-2: Data Schema & Missing Audit
生列→採用列→除外理由の完全監査
"""
import json
import os
import sys
from pathlib import Path

# Polarsで読み込み（PyArrowより高速）
try:
    import polars as pl
    USE_POLARS = True
except ImportError:
    import pyarrow.parquet as pq
    USE_POLARS = False

def audit_dataset(parquet_path: str, sample_size: int = 200_000):
    """データセットの完全監査"""
    print(f"Loading dataset: {parquet_path}")
    print(f"Sample size: {sample_size:,}")

    # Load data
    if USE_POLARS:
        df = pl.read_parquet(parquet_path)
        if len(df) > sample_size:
            df = df.head(sample_size)
        # Convert to pandas for unified processing
        df_pd = df.to_pandas()
    else:
        table = pq.read_table(parquet_path)
        df_pd = table.to_pandas()
        if len(df_pd) > sample_size:
            df_pd = df_pd.head(sample_size)

    n_rows = len(df_pd)
    n_cols = len(df_pd.columns)

    print(f"Loaded: {n_rows:,} rows × {n_cols} columns")

    # Schema (column names + types)
    schema = {c: str(df_pd[c].dtype) for c in df_pd.columns}

    # Missing rate per column
    missing = {c: float(df_pd[c].isna().mean()) for c in df_pd.columns}

    # Feature patterns (from existing code)
    patterns = [
        "return_", "rsi", "adx", "atr", "ema", "macd", "turnover", "volume",
        "stmt_", "dmi_", "x_", "peer_", "flow_", "margin_", "short_",
        "sector_", "macro_", "index_", "option_", "graph_"
    ]

    # Adoption/exclusion decision
    adopt, drop = [], []
    reasons = {}

    HIGH_MISSING_THRESHOLD = 0.98

    for c in df_pd.columns:
        rs = []

        # Exclusion reason 1: High missing rate
        miss_rate = df_pd[c].isna().mean()
        if miss_rate >= HIGH_MISSING_THRESHOLD:
            rs.append(f"high_missing>={HIGH_MISSING_THRESHOLD}")

        # Exclusion reason 2: Constant (single unique value)
        try:
            n_unique = df_pd[c].nunique(dropna=True)
            if n_unique <= 1:
                rs.append("constant")
        except (TypeError, AttributeError):
            # Skip unhashable types (dict, list, etc.)
            pass

        # Exclusion reason 3: All NaN
        if df_pd[c].isna().all():
            rs.append("all_nan")

        if rs:
            drop.append(c)
            reasons[c] = rs
        else:
            adopt.append(c)

    # Top missing columns
    top_missing = sorted(missing.items(), key=lambda x: -x[1])[:40]

    # Sample drop reasons
    drop_reasons_sample = dict(list(reasons.items())[:40])

    # Pattern matching analysis
    pattern_matched = {}
    for pattern in patterns:
        matched = [c for c in df_pd.columns if pattern in c.lower()]
        if matched:
            pattern_matched[pattern] = len(matched)

    # Statistics summary
    stats = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "adopt_n": len(adopt),
        "drop_n": len(drop),
        "schema": schema,
        "top_missing": [(c, round(rate, 4)) for c, rate in top_missing],
        "drop_reasons_sample": drop_reasons_sample,
        "pattern_matched": pattern_matched,
        "adopt_sample": adopt[:50],  # First 50 adopted columns
        "drop_sample": drop[:50],    # First 50 dropped columns
    }

    return stats


def main():
    # Dataset path priority
    candidates = [
        "output/ml_dataset_latest_clean_final.parquet",
        "output/ml_dataset_latest_clean_with_adv.parquet",
        "output/ml_dataset_latest_clean.parquet",
        "output/ml_dataset_latest_full.parquet",
    ]

    parquet_path = os.environ.get("PARQUET_DIR", None)

    if not parquet_path:
        # Find first existing file
        for candidate in candidates:
            if os.path.exists(candidate):
                parquet_path = candidate
                break

    if not parquet_path or not os.path.exists(parquet_path):
        print("ERROR: No dataset found. Tried:")
        for c in candidates:
            print(f"  - {c}")
        sys.exit(1)

    # Run audit
    stats = audit_dataset(parquet_path, sample_size=200_000)

    # Save results
    output_dir = Path("output/reports/diag_bundle")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "data_schema_and_missing.json"
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*80}")
    print("RFI-2: Data Schema & Missing Audit Complete")
    print(f"{'='*80}")
    print(f"Dataset: {parquet_path}")
    print(f"Rows sampled: {stats['n_rows']:,}")
    print(f"Total columns: {stats['n_cols']}")
    print(f"Adopted columns: {stats['adopt_n']} ({stats['adopt_n']/stats['n_cols']*100:.1f}%)")
    print(f"Dropped columns: {stats['drop_n']} ({stats['drop_n']/stats['n_cols']*100:.1f}%)")
    print("\nTop drop reasons:")
    reason_counts = {}
    for col, reasons_list in stats['drop_reasons_sample'].items():
        for reason in reasons_list:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  - {reason}: {count} columns")

    print("\nPattern matching:")
    for pattern, count in sorted(stats['pattern_matched'].items(), key=lambda x: -x[1])[:10]:
        print(f"  - {pattern}: {count} columns")

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
