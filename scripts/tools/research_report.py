#!/usr/bin/env python3
from __future__ import annotations

"""
Generate a concise research report (Markdown) summarizing dataset snapshot
and baseline RankIC/HitRate for a list of factors across horizons.

Usage:
  python scripts/tools/research_report.py \
    --dataset output/ml_dataset_latest_full.parquet \
    --factors returns_5d,ret_1d_vs_sec,rank_ret_1d,macd_hist_slope,graph_degree \
    --horizons 1,5,10,20 \
    --out reports/research_report.md
"""

import argparse
import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import polars as pl


def snapshot(df: pl.DataFrame) -> dict:
    n_rows = df.height
    n_cols = len(df.columns)
    codes = df.select("Code").unique().height if "Code" in df.columns else None
    dmin = df.select(pl.col("Date").min()).item() if "Date" in df.columns else None
    dmax = df.select(pl.col("Date").max()).item() if "Date" in df.columns else None
    return {
        "rows": n_rows,
        "cols": n_cols,
        "codes": codes,
        "date_min": dmin,
        "date_max": dmax,
    }


def ranks(expr: pl.Expr, by: str) -> pl.Expr:
    return expr.rank(method="average").over(by)


def rankic_stats(df: pl.DataFrame, factor: str, label: str) -> tuple[float | None, float | None, int]:
    if not {"Date", factor, label}.issubset(df.columns):
        return None, None, 0
    d = df.select(["Date", factor, label]).drop_nulls([factor, label])
    if d.is_empty():
        return None, None, 0
    d = d.with_columns([
        ranks(pl.col(factor), by="Date").alias("_rf"),
        ranks(pl.col(label), by="Date").alias("_ry"),
    ])
    daily = d.group_by("Date").agg(pl.pearson_corr(pl.col("_rf"), pl.col("_ry")).alias("rho")).drop_nulls()
    if daily.is_empty():
        return None, None, 0
    vals = daily["rho"].to_list()
    n = len(vals)
    mu = float(sum(vals) / n)
    # simple std and normal-approx CI
    if n > 1:
        var = float(sum((x - mu) ** 2 for x in vals) / (n - 1))
        sd = var ** 0.5
        return mu, 1.96 * sd / (n ** 0.5), n
    return mu, None, n


def hitrate(df: pl.DataFrame, factor: str, label: str) -> float | None:
    if not {factor, label}.issubset(df.columns):
        return None
    d = df.select([factor, label]).drop_nulls([factor, label])
    if d.is_empty():
        return None
    return float(d.select(((pl.sign(pl.col(factor)) == pl.sign(pl.col(label))).cast(pl.Int8).mean()).alias("hit"))["hit"][0])


def _compute_fold_rankic(df: pl.DataFrame, splits: list[dict], factors: list[str], horizons: list[int]) -> pl.DataFrame:
    """Compute per-fold RankIC for each factor × horizon using provided splits.

    Splits are a list of dicts with keys: fold, test_start, test_end.
    """
    label_cols = [f"feat_ret_{h}d" for h in horizons]
    rows: list[dict] = []
    for s in splits:
        fold = s.get("fold")
        t0 = s.get("test_start")
        t1 = s.get("test_end")
        if not t0 or not t1:
            continue
        dsub = df.filter((pl.col("Date") >= pl.lit(pl.Date(t0))) & (pl.col("Date") <= pl.lit(pl.Date(t1))))
        if dsub.is_empty():
            continue
        for fcol in factors:
            if fcol not in dsub.columns:
                continue
            for hcol in label_cols:
                if hcol not in dsub.columns:
                    continue
                ric, _, n = rankic_stats(dsub, fcol, hcol)
                rows.append({
                    "fold": fold,
                    "factor": fcol,
                    "horizon": int(hcol.split("_")[2][:-1]),
                    "rankic": ric,
                    "n_days": n,
                })
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate research report")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--factors", type=str, default="returns_5d,ret_1d_vs_sec,rank_ret_1d,macd_hist_slope,graph_degree")
    ap.add_argument("--horizons", type=str, default="1,5,10,20")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--csv", type=Path, default=None, help="Optional CSV to save per-horizon metrics")
    ap.add_argument("--splits-json", type=Path, default=None, help="Optional splits JSON to add fold-level top factors section")
    ap.add_argument("--topk", type=int, default=5, help="Top K factors per horizon for fold-level section")
    args = ap.parse_args()

    if not args.dataset.exists():
        print(f"ERROR: dataset not found: {args.dataset}")
        return 1

    cols = ["Code", "Date"]
    factors = [s.strip() for s in args.factors.split(",") if s.strip()]
    horizons: Iterable[int] = (int(x.strip()) for x in args.horizons.split(",") if x.strip())
    label_cols = [f"feat_ret_{h}d" for h in horizons]
    cols.extend(factors)
    cols.extend(label_cols)
    df = pl.read_parquet(args.dataset, columns=[c for c in cols if c])
    # Ensure stable ordering for any windowed/rank ops
    if {"Code", "Date"}.issubset(df.columns):
        df = df.sort(["Code", "Date"])  # type: ignore[arg-type]

    meta = snapshot(df)

    lines = []
    lines.append("# Research Report\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("\n## Dataset Snapshot\n")
    lines.append(f"- Rows: {meta['rows']:,}")
    lines.append(f"- Cols: {meta['cols']:,}")
    lines.append(f"- Codes: {meta['codes']}")
    lines.append(f"- Date range: {meta['date_min']} → {meta['date_max']}\n")

    lines.append("## Baseline Metrics (RankIC / HitRate)\n")
    rows_csv = []
    for f in factors:
        lines.append(f"### Factor: `{f}`\n")
        # Coverage (% non-null)
        if f in df.columns:
            cov = float(df.select((pl.col(f).is_not_null().mean() * 100).alias("cov"))["cov"][0])
            lines.append(f"- coverage: {cov:.1f}%\n")
        for h in horizons:
            label = f"feat_ret_{h}d"
            ric, ci, n = rankic_stats(df, f, label)
            hr = hitrate(df, f, label)
            if ric is None:
                ric_s = "n/a"
            else:
                if ci is not None:
                    ric_s = f"{ric:.4f} ± {ci:.4f} (n={n})"
                else:
                    ric_s = f"{ric:.4f} (n={n})"
            hr_s = "n/a" if hr is None else f"{hr*100:.1f}%"
            lines.append(f"- h={h}d: RankIC={ric_s}, HitRate={hr_s}")
            rows_csv.append({
                "factor": f,
                "horizon": h,
                "rankic": ric,
                "rankic_ci95": ci,
                "n_days": n,
                "hitrate": hr,
                "coverage": (float(df.select((pl.col(f).is_not_null().mean()).alias("c"))["c"][0]) if f in df.columns else None),
            })
        lines.append("")

    out = args.out or Path("reports") / f"research_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved report: {out}")
    # Top factor summary (by horizon)
    try:
        by_h = {}
        for r in rows_csv:
            h = r["horizon"]
            by_h.setdefault(h, []).append(r)
        tf = out.parent / (out.stem + "_top.txt")
        with open(tf, "w", encoding="utf-8") as g:
            for h in sorted(by_h.keys()):
                items = [x for x in by_h[h] if x["rankic"] is not None]
                items.sort(key=lambda x: x["rankic"], reverse=True)
                g.write(f"Top factors (h={h}d):\n")
                for x in items[:5]:
                    if x.get('coverage') is None:
                        cov_str = ''
                    else:
                        cov_str = f"{x['coverage']*100:.1f}%"
                    g.write(f"  - {x['factor']}: RankIC={x['rankic']:.4f} (cov={cov_str})\n")
                g.write("\n")
        print(f"Saved top factors summary: {tf}")
    except Exception:
        pass
    # Optional fold-level section
    if args.splits_json is not None:
        try:
            if args.splits_json.exists():
                with open(args.splits_json, encoding="utf-8") as f:
                    splits = json.load(f)
                if isinstance(splits, list) and splits:
                    fm = _compute_fold_rankic(df, splits, factors, list(horizons))
                    if not fm.is_empty():
                        lines2 = []
                        lines2.append("## Fold-Level Top Factors\n")
                        # summarize per factor × horizon across folds
                        summary = (
                            fm.group_by(["factor", "horizon"]).agg([
                                pl.col("rankic").mean().alias("rankic_mean"),
                                pl.col("rankic").std().alias("rankic_sd"),
                                pl.count().alias("n_folds"),
                            ])
                        )
                        # write per-horizon tables
                        for h in sorted(set(summary["horizon"].to_list())):
                            sub = summary.filter(pl.col("horizon") == h).sort("rankic_mean", descending=True)
                            top = sub.head(args.topk)
                            if top.is_empty():
                                continue
                            lines2.append(f"### Horizon {h}d\n")
                            lines2.append("Factor | RankIC (mean ± CI95) | Folds")
                            lines2.append("--- | --- | ---")
                            for r in top.iter_rows(named=True):
                                mu = r["rankic_mean"] if r["rankic_mean"] is not None else 0.0
                                sd = r["rankic_sd"] if r["rankic_sd"] is not None else 0.0
                                k = int(r["n_folds"]) if r["n_folds"] is not None else 0
                                ci = (1.96 * sd / (k ** 0.5)) if (k and k > 1 and sd is not None) else None
                                ci_s = f" ± {ci:.4f}" if ci is not None else ""
                                lines2.append(f"`{r['factor']}` | {mu:.4f}{ci_s} | {k}")
                            lines2.append("")
                        with open(out, "a", encoding="utf-8") as f:
                            f.write("\n" + "\n".join(lines2))
                        print("Appended fold-level top factors to report")
            else:
                print(f"Splits JSON not found: {args.splits_json}")
        except Exception as e:
            print(f"Fold-level section skipped: {e}")

    if args.csv:
        import csv
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["factor","horizon","rankic","rankic_ci95","n_days","hitrate","coverage"])
            writer.writeheader()
            for r in rows_csv:
                writer.writerow(r)
        print(f"Saved CSV: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
