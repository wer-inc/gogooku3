#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
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


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving.

    Args:
        patience: Number of epochs to wait after last improvement
        mode: 'max' to maximize metric, 'min' to minimize
        min_delta: Minimum change to qualify as improvement
    """
    def __init__(self, patience: int = 3, mode: str = "max", min_delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False
        self.best_state = None

    def __call__(self, score: float, epoch: int, model_state: dict) -> bool:
        """Check if training should stop.

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.best_state = model_state
            return False

        if self.mode == "max":
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.best_state = model_state
            self.counter = 0
            print(f"[EarlyStopping] New best: {score:.4f} at epoch {epoch}")
        else:
            self.counter += 1
            print(f"[EarlyStopping] No improvement for {self.counter}/{self.patience} epochs (best: {self.best_score:.4f} at epoch {self.best_epoch})")

            if self.counter >= self.patience:
                self.early_stop = True
                print(f"[EarlyStopping] Stopping training! Best epoch: {self.best_epoch}, Best score: {self.best_score:.4f}")
                return True

        return False


def clone_state_dict(state_dict: dict) -> dict:
    """Create a detached CPU clone of a model state dict."""
    cloned: dict = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cloned[key] = value.detach().cpu().clone()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train APEX-Ranker v0 baseline.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration.")
    parser.add_argument("--output", default=None, help="Optional path to save model state dict.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Optional cap on training epochs.")
    parser.add_argument("--max-train-days", type=int, default=None, help="Limit number of training days (recent).")
    parser.add_argument("--max-val-days", type=int, default=None, help="Limit number of validation days (recent).")
    parser.add_argument("--max-train-steps", type=int, default=None, help="Stop each epoch after N optimisation steps.")
    parser.add_argument("--log-interval", type=int, default=None, help="Override log interval for training loss prints.")
    parser.add_argument("--early-stopping-patience", type=int, default=None, help="Enable early stopping with patience N (epochs).")
    parser.add_argument(
        "--early-stopping-metric",
        default=None,
        help="Metric to monitor for early stopping (e.g., 20d_pak). Defaults to config value when omitted.",
    )
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

    # Feature exclusion support (pruned configs)
    exclude_features = data_cfg.get("exclude_features", None)
    if exclude_features:
        print(f"[Config] Excluding {len(exclude_features)} features from selection")

    selection = feature_selector.select(
        groups=groups,
        optional_groups=optional_groups,
        exclude_features=exclude_features,
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

    mask_columns = list(selection.masks)
    if mask_columns:
        coverage_exprs = [
            pl.col(mask).fill_null(0).gt(0.5).sum().alias(mask) for mask in mask_columns
        ]
        coverage = frame.select(coverage_exprs)
        active_masks: list[str] = []
        dropped_masks: list[str] = []
        for mask in mask_columns:
            positive = int(coverage[0, mask])
            if positive == 0:
                dropped_masks.append(mask)
            else:
                active_masks.append(mask)
        if dropped_masks:
            dropped_str = ", ".join(dropped_masks)
            print(f"[WARN] Dropping mask columns with zero coverage: {dropped_str}")
        mask_columns = active_masks

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
        mask_cols=mask_columns,
        date_col=date_col,
        code_col=code_col,
        lookback=data_cfg["lookback"],
        min_stocks_per_day=data_cfg["min_stocks_per_day"],
    )

    return cache, z_features, target_cols, mask_columns


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg = cfg["data"]
    norm_cfg = cfg.get("normalization", {})
    train_cfg = cfg["train"]
    loss_cfg = cfg["loss"]
    model_cfg = cfg["model"]

    if args.max_epochs is not None:
        train_cfg["epochs"] = max(1, int(args.max_epochs))
    if args.log_interval is not None:
        train_cfg["log_interval"] = max(1, int(args.log_interval))

    feature_selector = FeatureSelector(data_cfg["feature_groups_config"])
    cache, feature_cols, target_cols, mask_cols = load_dataset(cfg, feature_selector)

    dates = cache.date_ints
    val_days = max(1, min(train_cfg.get("val_days", 60), len(dates) // 5))
    train_dates = dates[:-val_days] if len(dates) > val_days else dates
    val_dates = dates[-val_days:] if len(dates) > val_days else dates[-1:]

    if args.max_train_days is not None and train_dates:
        train_dates = train_dates[-int(args.max_train_days) :]
    if args.max_val_days is not None and val_dates:
        val_dates = val_dates[-int(args.max_val_days) :]

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

    total_epochs = int(train_cfg["epochs"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    warmup_epochs = max(0, int(train_cfg.get("warmup_epochs", 0)))
    warmup_start_factor = float(train_cfg.get("warmup_start_factor", 0.1))
    if warmup_epochs >= total_epochs:
        warmup_epochs = max(0, total_epochs - 1)

    if warmup_epochs > 0:
        cosine_epochs = max(1, total_epochs - warmup_epochs)
        warmup_start_factor = min(1.0, max(1e-3, warmup_start_factor))
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
        print(f"[INFO] LR warmup enabled: epochs={warmup_epochs}, start_factor={warmup_start_factor}")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
        )
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    amp_enabled = bool(train_cfg.get("amp", True) and device_type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            scaler = torch.amp.GradScaler(device_type=device_type, enabled=amp_enabled)
            autocast_kwargs = {"device_type": device_type, "enabled": amp_enabled}
            autocast_ctx = torch.autocast
        except TypeError:
            scaler = torch.amp.GradScaler(enabled=amp_enabled)
            autocast_kwargs = {"device_type": device_type, "enabled": amp_enabled}
            autocast_ctx = torch.autocast
    else:
        from torch.cuda.amp import GradScaler as CudaGradScaler, autocast as cuda_autocast

        scaler = CudaGradScaler(enabled=amp_enabled)
        autocast_kwargs = {"enabled": amp_enabled}
        autocast_ctx = cuda_autocast

    def to_device(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device, non_blocking=True)

    print(f"[INFO] Training device: {device}")
    print(f"[INFO] Train days: {len(train_dataset)}, Val days: {len(val_dataset)}")
    print(f"[INFO] Active features: {len(feature_cols)}, Active masks: {len(mask_cols)}")
    print(f"[INFO] Cached trading days available: {len(cache.date_ints)}")
    if not train_dataset:
        print("[ERROR] No training days available after masking; aborting.")
        return

    # Initialize early stopping if requested (config or CLI)
    early_stopping = None
    early_cfg = train_cfg.get("early_stopping", {}) or {}
    patience = args.early_stopping_patience if args.early_stopping_patience is not None else early_cfg.get("patience")
    monitor_metric = args.early_stopping_metric or early_cfg.get("metric", "20d_pak")
    min_delta = float(early_cfg.get("min_delta", 0.0))
    if patience is not None:
        early_stopping = EarlyStopping(patience=int(patience), mode="max", min_delta=min_delta)
        print(
            "[INFO] Early stopping enabled: patience=%s, metric=%s, min_delta=%.4f"
            % (patience, monitor_metric, min_delta)
        )

    for epoch in range(1, train_cfg["epochs"] + 1):
        print(f"[INFO] Starting epoch {epoch}/{train_cfg['epochs']}")
        model.train()
        running_loss = 0.0
        batch_count = 0

        for step, batch in enumerate(train_loader, start=1):
            if batch is None:
                continue

            X = to_device(batch["X"].squeeze(0))
            y = to_device(batch["y"].squeeze(0))

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx(**autocast_kwargs):
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

            if args.max_train_steps is not None and batch_count >= args.max_train_steps:
                print(f"[INFO] Reached max_train_steps={args.max_train_steps}; stopping epoch early.")
                break

        if batch_count:
            scheduler.step()
            print(f"Epoch {epoch} training loss: {running_loss / batch_count:.4f}")
        else:
            print(f"[WARN] Epoch {epoch} produced no training batches; check data coverage.")
            continue

        # Validation
        model.eval()
        metrics = {h: {"ic": [], "p@k": []} for h in horizons}
        val_panel_count = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                X = to_device(batch["X"].squeeze(0))
                y = to_device(batch["y"].squeeze(0))
                scores = model(X)
                val_panel_count += 1

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

        # Compute and print validation metrics
        metric_results = {}
        for horizon in horizons:
            icvals = metrics[horizon]["ic"]
            pkvals = metrics[horizon]["p@k"]
            if not icvals:
                continue
            ic_mean = sum(icvals) / len(icvals)
            pk_mean = sum(pkvals) / len(pkvals) if pkvals else float("nan")
            metric_results[f"{horizon}d_ic"] = ic_mean
            metric_results[f"{horizon}d_pak"] = pk_mean
            print(
                "[Val] h=%2dd RankIC=%.4f  P@K=%.4f  (panels=%d)"
                % (horizon, ic_mean, pk_mean, len(icvals))
            )
        if val_panel_count == 0:
            print("[WARN] Validation loader yielded no panels; verify mask coverage and min_stocks_per_day.")
        else:
            print(f"[INFO] Processed {val_panel_count} validation panels.")

        # Check early stopping
        if early_stopping is not None and metric_results:
            if monitor_metric in metric_results:
                score = metric_results[monitor_metric]
                model_state = clone_state_dict(model.state_dict())
                should_stop = early_stopping(score, epoch, model_state)

                if should_stop:
                    print(f"[INFO] Restoring best model from epoch {early_stopping.best_epoch}")
                    model.load_state_dict(early_stopping.best_state)
                    break
            else:
                print(f"[WARN] Early stopping metric '{monitor_metric}' not found in results. Available: {list(metric_results.keys())}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_path)
        print(f"Saved model to {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - debugging aid
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        raise
