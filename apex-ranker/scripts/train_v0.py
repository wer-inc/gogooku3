#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from apex_ranker.data import (
    FeatureSelector,
    add_cross_sectional_zscores,
    build_panel_cache,
    collate_day_batch,
    DayPanelDataset,
)
from apex_ranker.losses import CompositeLoss
from apex_ranker.models import APEXRankerV0
from apex_ranker.utils import load_config, precision_at_k, spearman_ic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train APEX-Ranker v0 baseline.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration.")
    parser.add_argument("--output", default=None, help="Optional path to save model state dict.")
    return parser.parse_args()


def to_day_int(dates: Iterable[np.datetime64]) -> list[int]:
    days = np.asarray(dates, dtype="datetime64[D]").astype("int64")
    return [int(x) for x in days.tolist()]


def load_dataset(
    cfg: dict,
    feature_selector: FeatureSelector,
    *,
    include_plus30: bool = False,
):
    data_cfg = cfg["data"]
    date_col = data_cfg["date_column"]
    code_col = data_cfg["code_column"]
    parquet_path = Path(data_cfg["parquet_path"])
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet dataset not found: {parquet_path}")

    groups = list(data_cfg.get("feature_groups", []))
    optional_groups = list(data_cfg.get("optional_groups", []))
    if include_plus30 or data_cfg.get("use_plus30"):
        groups = groups + ["plus30"]

    selection = feature_selector.select(
        groups=groups,
        optional_groups=optional_groups,
        metadata_path=data_cfg.get("metadata_path"),
    )

    target_map = data_cfg["target_columns"]
    horizons = cfg["train"]["horizons"]

    def resolve_target(h: int) -> str:
        if isinstance(target_map, dict):
            if str(h) in target_map:
                return target_map[str(h)]
            if h in target_map:
                return target_map[h]
        raise KeyError(f"target column for horizon {h} is not defined in config")

    target_cols = [resolve_target(int(h)) for h in horizons]

    required_columns = (
        [date_col, code_col]
        + selection.features
        + selection.masks
        + target_cols
    )
    required_columns = list(dict.fromkeys(required_columns))

    frame = pl.read_parquet(parquet_path, columns=required_columns)
    frame = add_cross_sectional_zscores(
        frame,
        columns=selection.features,
        date_col=date_col,
        clip_sigma=cfg.get("normalization", {}).get("clip_sigma", 5.0),
    )

    z_features = [f"{col}_cs_z" for col in selection.features]

    cache = build_panel_cache(
        frame,
        feature_cols=z_features,
        target_cols=target_cols,
        mask_cols=selection.masks,
        date_col=date_col,
        code_col=code_col,
        lookback=data_cfg["lookback"],
        min_stocks_per_day=data_cfg["min_stocks_per_day"],
    )

    return cache, z_features, target_cols, selection.masks


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg = cfg["data"]
    norm_cfg = cfg.get("normalization", {})
    train_cfg = cfg["train"]
    loss_cfg = cfg["loss"]
    model_cfg = cfg["model"]

    feature_selector = FeatureSelector(data_cfg["feature_groups_config"])
    cache, feature_cols, target_cols, mask_cols = load_dataset(cfg, feature_selector)

    dates = cache.date_ints
    val_days = max(1, min(train_cfg.get("val_days", 60), len(dates) // 5))
    train_dates = dates[:-val_days] if len(dates) > val_days else dates
    val_dates = dates[-val_days:] if len(dates) > val_days else dates[-1:]

    train_dataset = DayPanelDataset(
        cache,
        feature_cols=feature_cols,
        mask_cols=mask_cols,
        target_cols=target_cols,
        dates_subset=train_dates,
    )
    val_dataset = DayPanelDataset(
        cache,
        feature_cols=feature_cols,
        mask_cols=mask_cols,
        target_cols=target_cols,
        dates_subset=val_dates,
    )

    num_workers = int(train_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_day_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_day_batch,
    )

    horizons = [int(h) for h in train_cfg["horizons"]]
    loss_module = CompositeLoss(
        listnet_weight=loss_cfg["listnet"]["weight"],
        listnet_tau=loss_cfg["listnet"]["tau"],
        listnet_topk=loss_cfg["listnet"].get("topk"),
        ranknet_weight=loss_cfg["ranknet"]["weight"],
        ranknet_neg_sample=loss_cfg["ranknet"].get("neg_sample"),
        mse_weight=loss_cfg["mse"]["weight"],
    )

    model = APEXRankerV0(
        in_features=len(feature_cols),
        horizons=horizons,
        d_model=model_cfg["d_model"],
        depth=model_cfg["depth"],
        patch_len=model_cfg["patch_len"],
        stride=model_cfg["stride"],
        n_heads=model_cfg["n_heads"],
        dropout=model_cfg["dropout"],
        loss_fn=loss_module,
    )

    device = train_cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.get("amp", True) and device.startswith("cuda"))

    def to_device(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device, non_blocking=True)

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for step, batch in enumerate(train_loader, start=1):
            if batch is None:
                continue

            X = to_device(batch["X"].squeeze(0))
            y = to_device(batch["y"].squeeze(0))

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                scores = model(X)
                loss = model.compute_loss(scores, y)

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            if grad_clip := train_cfg.get("grad_clip"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            batch_count += 1

            if step % train_cfg.get("log_interval", 10) == 0:
                avg_loss = running_loss / max(1, batch_count)
                print(f"[Epoch {epoch} | Step {step}] loss={avg_loss:.4f}")

        scheduler.step()
        if batch_count:
            print(f"Epoch {epoch} training loss: {running_loss / batch_count:.4f}")

        # Validation
        model.eval()
        metrics = {h: {"ic": [], "p@k": []} for h in horizons}
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                X = to_device(batch["X"].squeeze(0))
                y = to_device(batch["y"].squeeze(0))
                scores = model(X)

                for idx, horizon in enumerate(horizons):
                    if idx >= y.shape[1]:
                        continue
                    s = scores[horizon]
                    target = y[:, idx]
                    if torch.std(target) < 1e-8:
                        continue
                    metrics[horizon]["ic"].append(spearman_ic(s, target))
                    k = max(1, min(50, target.numel() // 10))
                    metrics[horizon]["p@k"].append(precision_at_k(s, target, k))

        for horizon in horizons:
            icvals = metrics[horizon]["ic"]
            pkvals = metrics[horizon]["p@k"]
            if not icvals:
                continue
            ic_mean = sum(icvals) / len(icvals)
            pk_mean = sum(pkvals) / len(pkvals) if pkvals else float("nan")
            print(f"[Val] h={horizon:>2}d RankIC={ic_mean:.4f}  P@K={pk_mean:.4f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_path)
        print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
