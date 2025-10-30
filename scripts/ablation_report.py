from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from gogooku3.training.cv_purged import purged_kfold_indices
from gogooku3.training.datamodule import PanelDataModule
from gogooku3.training.losses import HuberMultiHorizon
from gogooku3.training.metrics import rank_ic, sharpe_ratio
from gogooku3.training.model_multihead import MultiHeadRegressor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Run ablation and produce a markdown report")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--target", type=str, default="target_1d")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--out", type=Path, default=Path("output/ablation_report.md"))
    # Universe/time-slice validation (optional)
    p.add_argument("--group-col", type=str, default=None, help="Categorical column to define universes (e.g., sector33_id)")
    p.add_argument("--group-topk", type=int, default=8, help="Top-K most frequent groups to evaluate")
    p.add_argument("--time-slices", type=str, default="auto", help="Comma-separated start:end date pairs or 'auto' for halves")
    p.add_argument("--slice-variant", choices=["final", "all"], default="final", help="Which variant(s) to use in slice validation")
    return p.parse_args()


def numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c, dt in zip(df.columns, df.dtypes, strict=False) if pl.datatypes.is_numeric_dtype(dt)]


def ensure_date_col(df: pl.DataFrame, name: str = "Date") -> pl.DataFrame:
    if name not in df.columns:
        return df
    if df.schema[name] in (pl.Date, pl.Datetime):
        return df
    try:
        return df.with_columns(pl.col(name).str.strptime(pl.Date, strict=False, exact=False).alias(name))
    except Exception:
        return df


def feature_set_for_variant(df: pl.DataFrame, base: list[str], variant: str) -> tuple[list[str], list[str]]:
    """Return (feature_cols, outlier_cols) for a given ablation variant."""
    # Outlier columns default list
    winsor_cols = [c for c in ["returns_1d", "returns_5d", "rel_to_sec_5d", "dmi_long_to_adv20", "stmt_rev_fore_op"] if c in df.columns]
    feats = [c for c in base if c not in {"Date", "Code", "sector33_id"}]
    # Remove target and obvious leakage columns
    feats = [c for c in feats if not c.startswith("returns_")]
    # Base excludes new engineered columns by pattern
    if variant == "base":
        feats = [c for c in feats if (not c.startswith("x_")) and c != "sec_ret_1d_eq_loo" and (not c.endswith("_to_adv20")) and (not c.endswith("_z260"))]
        return feats, []  # no winsor in base
    if variant == "+loo":
        if "sec_ret_1d_eq_loo" in df.columns:
            feats = feats + ["sec_ret_1d_eq_loo"]
        return feats, []
    if variant == "+scale":
        add = [c for c in ["margin_long_to_adv20", "margin_long_z260", "dmi_long_to_adv20", "dmi_long_z260"] if c in df.columns]
        feats = list(dict.fromkeys(feats + add))
        return feats, []
    if variant == "+winsor":
        return feats, winsor_cols
    if variant == "+interactions":
        add = [c for c in df.columns if c.startswith("x_")]
        feats = list(dict.fromkeys(feats + add))
        return feats, winsor_cols
    return feats, winsor_cols


def run_variant(
    df: pl.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    outlier_cols: Sequence[str],
    *,
    epochs: int,
    batch_size: int,
) -> tuple[float, float]:
    dates = df["Date"].to_numpy()
    folds = purged_kfold_indices(dates, n_splits=3, embargo_days=20)
    dm = PanelDataModule(
        df,
        feature_cols=feature_cols,
        target_col=target_col,
        date_col="Date",
        by_cols=["sector33_id"] if "sector33_id" in df.columns else None,
        outlier_cols=list(outlier_cols),
        vol_col="volatility_20d" if "volatility_20d" in df.columns else None,
    )
    model = MultiHeadRegressor(in_dim=len(feature_cols), hidden=256, groups=None, out_heads=(1, 1, 1, 1, 1))
    crit = HuberMultiHorizon()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ics: list[float] = []
    sharpes: list[float] = []
    for f in folds:
        train_ds, val_ds, _, val_df_t = dm.setup_fold(f)
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        for _ in range(max(1, epochs)):
            model.train()
            for X, y, vol in loader:
                opt.zero_grad(); outs = model(X)
                loss = crit([o.squeeze(-1) for o in outs], [y for _ in outs], vol20=vol)
                loss.backward(); opt.step()
        # Predict validation and compute RankIC
        model.eval()
        with torch.no_grad():
            Xv = val_ds.X
            yv = val_ds.y
            pred = model(Xv)[0].squeeze(-1).cpu().numpy()
        cols = ["Date"] + (["Code"] if "Code" in val_df_t.columns else [])
        dfp = val_df_t.select(cols).with_columns(
            pl.Series("pred_1d", pred),
            pl.Series(target_col, yv.cpu().numpy()),
        )
        ic = rank_ic(dfp, pred_col="pred_1d", target_col=target_col, date_col="Date")
        ics.append(ic)
        # Long-Short decile spread per date → Sharpe
        if "Code" in dfp.columns:
            # assign deciles per date
            with_dec = dfp.with_columns(
                pl.col("pred_1d").rank(method="average").over("Date").alias("_rk")
            )
            with_dec = with_dec.with_columns(
                (pl.col("_rk") / pl.col("_rk").max().over("Date") * 10).floor().cast(pl.Int32).alias("_dec")
            )
            top = (
                with_dec.filter(pl.col("_dec") >= 9)
                .group_by("Date")
                .agg(pl.col(target_col).mean().alias("top"))
            )
            bot = (
                with_dec.filter(pl.col("_dec") <= 0)
                .group_by("Date")
                .agg(pl.col(target_col).mean().alias("bot"))
            )
            ls = top.join(bot, on="Date", how="inner").with_columns((pl.col("top") - pl.col("bot")).alias("ls"))
            sr = sharpe_ratio(ls["ls"].to_numpy()) if ls.height > 3 else 0.0
            sharpes.append(sr)
        else:
            sharpes.append(0.0)
    return float(np.mean(np.array(ics))), float(np.mean(np.array(sharpes)))


def main() -> None:
    a = parse_args()
    df = pl.read_parquet(str(a.data))
    df = ensure_date_col(df, "Date")
    base = numeric_cols(df)
    variants = ["base", "+loo", "+scale", "+winsor", "+interactions"]
    rows: list[tuple[str, float, float]] = []
    for v in variants:
        feats, outs = feature_set_for_variant(df, base, v)
        ric, sr = run_variant(df, feats, a.target, outs, epochs=a.epochs, batch_size=a.batch_size)
        rows.append((v, ric, sr))
        print({"variant": v, "RankIC": ric, "Sharpe": sr})

    # Save markdown report
    a.out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Ablation Report", "", "| Variant | RankIC | Sharpe |", "|---|---:|---:|"]
    for v, ric, sr in rows:
        lines.append(f"| {v} | {ric:.5f} | {sr:.3f} |")
    # Optional: Time-slice validation
    def _variant_list() -> list[str]:
        return variants if a.slice_variant == "all" else [variants[-1]]

    if a.time_slices:
        if a.time_slices == "auto":
            dmin = df["Date"].min(); dmax = df["Date"].max()
            slices = []
            if dmin is not None and dmax is not None and dmin != dmax:
                mid = dmin + (dmax - dmin) // 2
                slices = [(dmin, mid), (mid, dmax)]
        else:
            slices = []
            for pr in [p.strip() for p in a.time_slices.split(",") if p.strip()]:
                try:
                    s, e = [np.datetime64(x.strip()) for x in pr.split(":", 1)]
                    slices.append((s, e))
                except Exception:
                    continue
        if slices:
            lines += ["", "## Time-slice Validation", "| Slice | Variant | RankIC | Sharpe |", "|---|---|---:|---:|"]
            for (s, e) in slices:
                sub = df.filter((pl.col("Date") >= s) & (pl.col("Date") <= e))
                if sub.height < 100:
                    continue
                for v in _variant_list():
                    feats, outs = feature_set_for_variant(sub, numeric_cols(sub), v)
                    ric, sr = run_variant(sub, feats, a.target, outs, epochs=a.epochs, batch_size=a.batch_size)
                    lines.append(f"| {str(s)[:10]}–{str(e)[:10]} | {v} | {ric:.5f} | {sr:.3f} |")

    # Optional: Universe validation
    if a.group_col and a.group_col in df.columns:
        counts = df.group_by(a.group_col).len().sort("len", descending=True)
        values = counts[a.group_col].to_list()[: max(1, a.group_topk)]
        lines += ["", f"## Universe Validation by `{a.group_col}`", "| Group | Variant | RankIC | Sharpe |", "|---|---|---:|---:|"]
        for gv in values:
            sub = df.filter(pl.col(a.group_col) == gv)
            if sub.height < 100:
                continue
            for v in _variant_list():
                feats, outs = feature_set_for_variant(sub, numeric_cols(sub), v)
                ric, sr = run_variant(sub, feats, a.target, outs, epochs=a.epochs, batch_size=a.batch_size)
                lines.append(f"| {gv} | {v} | {ric:.5f} | {sr:.3f} |")

    a.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Ablation report saved: {a.out}")


if __name__ == "__main__":
    main()
