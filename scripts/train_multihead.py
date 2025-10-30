from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import DataLoader

from gogooku3.training.cv_purged import purged_kfold_indices
from gogooku3.training.datamodule import PanelDataModule
from gogooku3.training.feature_groups import resolve_groups_from_prefixes
from gogooku3.training.losses import HuberMultiHorizon
from gogooku3.training.model_multihead import MultiHeadRegressor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train multi-head regressor on extended dataset")
    p.add_argument("--data", type=Path, required=True, help="Panel parquet with features + target")
    p.add_argument("--target", type=str, default="target_1d", help="Target column name")
    p.add_argument(
        "--features",
        type=str,
        nargs="+",
        required=False,
        help="Explicit feature columns; defaults to *_cs_z if present else numeric",
    )
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--feature-groups", type=Path, default=Path("configs/feature_groups.yaml"))
    p.add_argument("--vol-col", type=str, default="volatility_20d")
    p.add_argument("--outlier-cols", type=str, default="returns_1d,returns_5d")
    p.add_argument("--pred-out", type=Path, default=None, help="Write validation predictions parquet for eval")
    return p.parse_args()


def default_feature_cols(df: pl.DataFrame) -> list[str]:
    cs = [c for c in df.columns if c.endswith("_cs_z")]
    if cs:
        return cs
    # Fallback: simple numeric columns excluding target
    return [c for c, dt in zip(df.columns, df.dtypes, strict=False) if pl.datatypes.is_numeric_dtype(dt)]


def train_one_fold(model: MultiHeadRegressor, loader: DataLoader, crit: HuberMultiHorizon, opt: torch.optim.Optimizer) -> float:
    model.train()
    total = 0.0
    for X, y, vol in loader:
        opt.zero_grad()
        outs = model(X)
        # Share the same target across horizons for demo purposes
        loss = crit([o.squeeze(-1) for o in outs], [y for _ in outs], vol20=vol)
        loss.backward()
        opt.step()
        total += float(loss.item())
    return total / max(1, len(loader))


def main() -> None:
    args = parse_args()
    df = pl.read_parquet(str(args.data))
    feature_cols: Sequence[str] = args.features or default_feature_cols(df)

    # Build folds from dates
    dates = df["Date"].to_numpy()
    folds = purged_kfold_indices(dates, n_splits=3, embargo_days=20)

    outlier_cols = [c.strip() for c in (args.outlier_cols or "").split(",") if c.strip() and c.strip() in df.columns]
    dm = PanelDataModule(
        df,
        feature_cols=feature_cols,
        target_col=args.target,
        date_col="Date",
        by_cols=["sector33_id"] if "sector33_id" in df.columns else None,
        outlier_cols=outlier_cols,
        vol_col=args.vol_col if args.vol_col in df.columns else None,
    )

    in_dim = len(feature_cols)
    groups = resolve_groups_from_prefixes([f"{c}_cs_z" for c in feature_cols], args.feature_groups)
    model = MultiHeadRegressor(in_dim=in_dim, hidden=256, groups=groups or None, out_heads=(1, 1, 1, 1, 1))
    crit = HuberMultiHorizon()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for fi, f in enumerate(folds):
        train_ds, val_ds, train_df_t, val_df_t = dm.setup_fold(f)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        # one short epoch by default to keep runtimes low in CI
        loss = train_one_fold(model, train_loader, crit, opt)
        print(f"fold={fi} train_loss={loss:.6f}")
        # Optional validation predictions
        if args.pred_out:
            model.eval()
            with torch.no_grad():
                Xv, yv, _ = val_ds.X, val_ds.y, val_ds.vol20
                outs = model(Xv)
                pred = outs[0].squeeze(-1).cpu().numpy()
                pdv = val_df_t.select(["Date"]).with_columns(pl.Series("pred_1d", pred), pl.Series("target_1d", yv.cpu().numpy()))
            if fi == 0:
                pdv.write_parquet(str(args.pred_out))
            else:
                # append by reading and vertical stacking (small validation scales in CI)
                existing = pl.read_parquet(str(args.pred_out))
                existing.vstack(pdv, in_place=True)
                existing.write_parquet(str(args.pred_out))


if __name__ == "__main__":
    main()
