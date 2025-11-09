#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import math
from collections.abc import Iterable
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch
from apex_ranker.backtest.splitter import PurgedKFoldSplitter, PurgeParams
from gogooku5.data.src.builder.utils.lazy_io import lazy_load
from apex_ranker.data import (
    DayPanelDataset,
    FeatureSelector,
    add_cross_sectional_zscores,
    build_panel_cache,
    collate_day_batch,
)
from apex_ranker.losses import CompositeLoss
from apex_ranker.models import APEXRankerV0
from apex_ranker.utils import (
    k_from_ratio,
    load_config,
    ndcg_at_k,
    ndcg_random_baseline,
    precision_at_k_pos,
    spearman_ic,
    top_bottom_spread,
    topk_overlap,
    wil_at_k,
)
from torch.utils.data import DataLoader


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
        self.best_extra = None

    def __call__(
        self,
        score: float,
        epoch: int,
        model_state: dict,
        extra_state: dict | None = None,
    ) -> bool:
        """Check if training should stop.

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.best_state = model_state
            self.best_extra = extra_state
            return False

        if self.mode == "max":
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.best_state = model_state
            self.best_extra = extra_state
            self.counter = 0
            print(f"[EarlyStopping] New best: {score:.4f} at epoch {epoch}")
        else:
            self.counter += 1
            print(
                f"[EarlyStopping] No improvement for {self.counter}/{self.patience} epochs (best: {self.best_score:.4f} at epoch {self.best_epoch})"
            )

            if self.counter >= self.patience:
                self.early_stop = True
                print(
                    f"[EarlyStopping] Stopping training! Best epoch: {self.best_epoch}, Best score: {self.best_score:.4f}"
                )
                return True

        return False


class ExponentialMovingAverage:
    """Maintain an exponential moving average of model parameters."""

    def __init__(self, model: torch.nn.Module, beta: float = 0.999) -> None:
        self.beta = float(beta)
        self.shadow: dict[str, torch.Tensor] = {
            name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad
        }
        self.backup: dict[str, torch.Tensor] | None = None

    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            shadow_param = self.shadow[name]
            shadow_param.mul_(self.beta).add_(param.detach(), alpha=1.0 - self.beta)

    def apply(self, model: torch.nn.Module) -> None:
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.backup[name] = param.detach().clone()
            param.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module) -> None:
        if self.backup is None:
            return
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param.data.copy_(self.backup[name])
        self.backup = None

    def clone_shadow(self) -> dict[str, torch.Tensor]:
        return {name: tensor.detach().clone() for name, tensor in self.shadow.items()}

    def load_shadow(self, shadow_state: dict[str, torch.Tensor]) -> None:
        for name, tensor in shadow_state.items():
            if name in self.shadow:
                self.shadow[name].data.copy_(tensor)


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
    parser.add_argument(
        "--max-train-days",
        type=int,
        default=None,
        help="Limit number of training days (recent).",
    )
    parser.add_argument(
        "--max-val-days",
        type=int,
        default=None,
        help="Limit number of validation days (recent).",
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=None,
        help="Stop each epoch after N optimisation steps.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=None,
        help="Override log interval for training loss prints.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Enable early stopping with patience N (epochs).",
    )
    parser.add_argument(
        "--early-stopping-metric",
        default=None,
        help="Metric to monitor for early stopping (e.g., 20d_pak). Defaults to config value when omitted.",
    )
    parser.add_argument(
        "--train-start-date",
        default=None,
        help="Optional training window start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--train-end-date",
        default=None,
        help="Optional training window end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--val-start-date",
        default=None,
        help="Optional validation window start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--val-end-date",
        default=None,
        help="Optional validation window end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--ema-snapshot-epochs",
        type=int,
        nargs="+",
        default=None,
        help="Save EMA-weight snapshots at the given 1-indexed epochs (requires --output).",
    )
    parser.add_argument(
        "--cv-type",
        choices=["walk_forward", "purged_kfold"],
        default=None,
        help="Override cross-validation splitter (defaults to config).",
    )
    parser.add_argument(
        "--cv-n-splits",
        type=int,
        default=None,
        help="Number of folds when using Purged K-Fold (defaults to config).",
    )
    parser.add_argument(
        "--cv-fold",
        type=int,
        default=None,
        help="Purged K-Fold index (1-based). Defaults to config/1 when purged_kfold is used.",
    )
    parser.add_argument(
        "--embargo-days",
        type=int,
        default=None,
        help="Embargo days appended to purge gaps (defaults to config).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; load --checkpoint and run validation only.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model state_dict for eval-only mode.",
    )
    parser.add_argument(
        "--eval-k-ratios",
        type=str,
        default=None,
        help="Comma-separated override(s) for eval.k_ratio (e.g. '0.05,0.10'). Used in eval-only mode.",
    )
    return parser.parse_args()


def to_day_int(dates: Iterable[np.datetime64]) -> list[int]:
    days = np.asarray(dates, dtype="datetime64[D]").astype("int64")
    return [int(x) for x in days.tolist()]


def date_str_to_int(date_str: str) -> int:
    """Convert YYYY-MM-DD string to integer day representation."""
    day = np.datetime64(date_str, "D")
    return int(day.astype("int64"))


DATE_EPOCH = Date(1970, 1, 1)


def day_int_to_date(day_int: int) -> Date:
    """Convert cached day integer to ``datetime.date``."""
    return DATE_EPOCH + timedelta(days=int(day_int))


def date_to_day_int(date_obj: Date) -> int:
    """Convert ``datetime.date`` back to integer offset."""
    return (date_obj - DATE_EPOCH).days


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

    # Load ALL columns first (aliases may not exist in dataset yet)
    # Use lazy_load for IPC cache support (3-5x faster reads)
    frame = lazy_load(parquet_path, prefer_ipc=True)

    # Apply feature aliases if configured (creates computed columns like dmi_z26_net)
    aliases_yaml = data_cfg.get("feature_aliases_yaml")
    if aliases_yaml:
        from apex_ranker.data.loader import apply_feature_aliases

        frame = apply_feature_aliases(frame, aliases_yaml)

    # Now select only required columns (aliases are now available)
    required_columns = [date_col, code_col] + selection.features + selection.masks + target_cols
    required_columns = list(dict.fromkeys(required_columns))
    frame = frame.select(required_columns)

    mask_columns = list(selection.masks)
    if mask_columns:
        coverage_exprs = [pl.col(mask).fill_null(0).gt(0.5).sum().alias(mask) for mask in mask_columns]
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
    snapshot_epochs = {int(epoch) for epoch in args.ema_snapshot_epochs or [] if int(epoch) > 0}
    warned_snapshot_no_output = False
    eval_cfg = cfg.get("eval", {}) or {}
    eval_k_value = eval_cfg.get("k")
    if eval_k_value is not None:
        eval_k_value = int(eval_k_value)
    eval_ratio_value = eval_cfg.get("k_ratio", 0.1)
    if eval_ratio_value is not None:
        eval_ratio_value = float(eval_ratio_value)
    ndcg_beta_value = float(eval_cfg.get("ndcg_beta", 1.0))
    es_weights_cfg = eval_cfg.get("es_weights", "0.5,0.35,0.15")
    es_weights: dict[int, float] = {}
    try:
        w1, w2, w3 = (float(x.strip()) for x in es_weights_cfg.split(","))
        es_weights = {5: w1, 10: w2, 20: w3}
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("eval.es_weights must be a comma-separated string for horizons 5,10,20") from exc

    data_cfg = cfg["data"]
    cfg.get("normalization", {})
    train_cfg = cfg["train"]
    loss_cfg = cfg["loss"]
    model_cfg = cfg["model"]

    if args.max_epochs is not None:
        train_cfg["epochs"] = max(1, int(args.max_epochs))
    if args.log_interval is not None:
        train_cfg["log_interval"] = max(1, int(args.log_interval))

    feature_selector = FeatureSelector(data_cfg["feature_groups_config"])
    cache, feature_cols, target_cols, mask_cols = load_dataset(cfg, feature_selector)

    dates = sorted(int(d) for d in cache.date_ints)

    evaluation_cfg = train_cfg.get("evaluation", {}) or {}
    cfg_cv_type = evaluation_cfg.get("splitter", "walk_forward").lower()
    cv_type = args.cv_type.lower() if args.cv_type else cfg_cv_type
    purged_cfg = evaluation_cfg.get("purged_kfold", {}) or {}

    default_n_splits = int(purged_cfg.get("n_splits", 5))
    default_embargo = int(purged_cfg.get("embargo_days", 0))

    n_splits = args.cv_n_splits if args.cv_n_splits is not None else default_n_splits
    embargo_days = args.embargo_days if args.embargo_days is not None else default_embargo
    cv_fold = int(args.cv_fold if args.cv_fold is not None else purged_cfg.get("fold", 1) or 1)

    horizons = [int(h) for h in train_cfg["horizons"]]
    short_term_horizons = [h for h in (5, 10, 20) if h in horizons]
    lookback_days = int(data_cfg.get("lookback", data_cfg.get("lookback_days", 180)))
    max_horizon = max(horizons) if horizons else 0

    train_dates: list[int]
    val_dates: list[int]

    if cv_type == "purged_kfold":
        params = PurgeParams(
            lookback_days=lookback_days,
            max_horizon_days=max_horizon,
            embargo_days=int(embargo_days),
        )
        splitter = PurgedKFoldSplitter(n_splits=int(n_splits), params=params)
        index_array = np.arange(len(dates))
        try:
            train_idx, val_idx = next(splitter.split(index_array, fold_index=cv_fold))
        except StopIteration:  # pragma: no cover - defensive
            raise RuntimeError("PurgedKFoldSplitter produced no folds.")

        train_dates = [dates[i] for i in train_idx.tolist()]
        val_dates = [dates[i] for i in val_idx.tolist()]

        if not train_dates:
            raise ValueError("PurgedKFold produced empty training set.")
        if not val_dates:
            raise ValueError("PurgedKFold produced empty validation set.")

        train_start = day_int_to_date(train_dates[0])
        train_end = day_int_to_date(train_dates[-1])
        val_start = day_int_to_date(val_dates[0])
        val_end = day_int_to_date(val_dates[-1])

        print(
            "[INFO] PurgedKFold split "
            f"{cv_fold}/{n_splits}: train {train_start} → {train_end} ({len(train_dates)} days) | "
            f"val {val_start} → {val_end} ({len(val_dates)} days) | "
            f"purge_days={params.purge_days} (lookback={params.lookback_days}, max_h={params.max_horizon_days}, embargo={params.embargo_days})"
        )
    else:
        if args.cv_type and args.cv_type.lower() == "purged_kfold":
            raise ValueError("Purged K-Fold requested but evaluation config is missing.")
        val_days = max(1, min(train_cfg.get("val_days", 60), len(dates) // 5))
        train_dates = dates[:-val_days] if len(dates) > val_days else dates
        val_dates = dates[-val_days:] if len(dates) > val_days else dates[-1:]
        if args.cv_fold is not None:
            print("[WARN] --cv-fold specified but purged_kfold is not active; ignoring.")

    if args.max_train_days is not None and train_dates:
        train_dates = train_dates[-int(args.max_train_days) :]
    if args.max_val_days is not None and val_dates:
        val_dates = val_dates[-int(args.max_val_days) :]

    if args.train_start_date:
        start_int = date_str_to_int(args.train_start_date)
        train_dates = [d for d in train_dates if d >= start_int]
    if args.train_end_date:
        end_int = date_str_to_int(args.train_end_date)
        train_dates = [d for d in train_dates if d <= end_int]
    if args.val_start_date:
        start_int = date_str_to_int(args.val_start_date)
        val_dates = [d for d in val_dates if d >= start_int]
    if args.val_end_date:
        end_int = date_str_to_int(args.val_end_date)
        val_dates = [d for d in val_dates if d <= end_int]

    if not train_dates:
        raise ValueError("No training dates available after applying filters.")
    if not val_dates:
        fallback = train_dates[-1]
        print(f"[WARN] No validation dates after filtering; falling back to the last training day {fallback}")
        val_dates = [fallback]

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
    train_total_steps = len(train_loader) if len(train_loader) > 0 else 0

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

    use_ema = bool(train_cfg.get("use_ema", True))
    ema_beta = float(train_cfg.get("ema_beta", 0.999))
    ema = ExponentialMovingAverage(model, beta=ema_beta) if use_ema else None

    # ========== EVAL-ONLY MODE ==========
    if args.eval_only:
        import sys

        if not args.checkpoint:
            raise ValueError("--eval-only requires --checkpoint to be specified")

        print(f"[EVAL-ONLY] Loading checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state, strict=True)
        model.eval()

        # Parse eval.k_ratios (default to config value)
        if args.eval_k_ratios:
            eval_ratios = [float(r.strip()) for r in args.eval_k_ratios.split(",")]
        else:
            eval_ratios = [eval_ratio_value]

        # Run validation for each ratio
        for eval_ratio_override in eval_ratios:
            print(f"\n[EVAL-ONLY] Running validation with eval.k_ratio={eval_ratio_override}")

            # Validation logic (copied from main training loop)
            if ema is not None:
                ema.apply(model)
            model.eval()

            per_horizon: dict[int, dict[str, list[float]]] = {
                h: {
                    "date_int": [],
                    "rank_ic": [],
                    "topk_overlap": [],
                    "p_at_k_pos": [],
                    "p_at_k_pos_rand": [],
                    "ndcg": [],
                    "ndcg_rand": [],
                    "spread": [],
                    "k_over_n": [],
                    "wil": [],
                }
                for h in horizons
            }
            val_panel_count = 0
            ndcg_beta = ndcg_beta_value
            eval_k = eval_k_value
            eval_ratio = eval_ratio_override  # USE OVERRIDE

            # Helper function for device transfer
            def to_device(tensor: torch.Tensor) -> torch.Tensor:
                return tensor.to(device, non_blocking=True)

            with torch.no_grad():
                for batch in val_loader:
                    if batch is None:
                        continue
                    date_int = int(batch.get("date_int", 0))
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
                        score_vec = s.detach().float().cpu()
                        label_vec = target.detach().float().cpu()
                        n = label_vec.numel()
                        if n == 0:
                            continue
                        k = k_from_ratio(n, k=eval_k, ratio=eval_ratio)
                        ic = spearman_ic(score_vec, label_vec)
                        overlap = topk_overlap(score_vec, label_vec, k)
                        p_at_k, p_rand = precision_at_k_pos(score_vec, label_vec, k)
                        nd_val = ndcg_at_k(score_vec, label_vec, k, beta=ndcg_beta)
                        nd_rand = ndcg_random_baseline(label_vec, k, beta=ndcg_beta)
                        spread_val = top_bottom_spread(score_vec, label_vec, k)
                        wil_val = wil_at_k(score_vec, label_vec, k)

                        stats = per_horizon[horizon]
                        stats["date_int"].append(date_int)
                        stats["rank_ic"].append(ic)
                        stats["topk_overlap"].append(overlap)
                        stats["p_at_k_pos"].append(p_at_k)
                        stats["p_at_k_pos_rand"].append(p_rand)
                        stats["ndcg"].append(nd_val)
                        stats["ndcg_rand"].append(nd_rand)
                        stats["spread"].append(spread_val)
                        stats["k_over_n"].append(k / max(1, n))
                        stats["wil"].append(wil_val)

            # Compute and print validation metrics
            summary: dict[int, dict[str, float]] = {}
            for horizon in horizons:
                stats = per_horizon[horizon]
                if not stats["rank_ic"]:
                    continue
                arrays = {
                    key: np.asarray(values, dtype=np.float64) for key, values in stats.items() if key != "date_int"
                }
                rank_ic_mean = float(arrays["rank_ic"].mean())
                overlap_mean = float(arrays["topk_overlap"].mean())
                p_at_k_mean = float(arrays["p_at_k_pos"].mean())
                p_rand_mean = float(arrays["p_at_k_pos_rand"].mean())
                delta_p = float((arrays["p_at_k_pos"] - arrays["p_at_k_pos_rand"]).mean())
                ndcg_mean = float(arrays["ndcg"].mean())
                ndcg_rand_mean = float(arrays["ndcg_rand"].mean())
                delta_ndcg = float((arrays["ndcg"] - arrays["ndcg_rand"]).mean())
                spread_mean = float(arrays["spread"].mean())
                k_over_n_mean = float(arrays["k_over_n"].mean())
                wil_mean = float(arrays["wil"].mean())

                summary[horizon] = {
                    "rank_ic": rank_ic_mean,
                    "topk_overlap": overlap_mean,
                    "p_at_k_pos": p_at_k_mean,
                    "p_at_k_pos_rand": p_rand_mean,
                    "delta_p_at_k_pos": delta_p,
                    "ndcg": ndcg_mean,
                    "ndcg_rand": ndcg_rand_mean,
                    "delta_ndcg": delta_ndcg,
                    "spread": spread_mean,
                    "k_over_n": k_over_n_mean,
                    "wil": wil_mean,
                    "n_days": int(arrays["rank_ic"].size),
                }

                print(
                    "[Val] h=%2dd RankIC=%.4f ΔP@K=%.4f (P@K=%.4f | rand=%.4f) "
                    "ΔNDCG=%.4f (NDCG=%.4f | rand=%.4f) "
                    "Overlap=%.4f Spread=%.4f WIL=%.4f (panels=%d, K/N=%.3f)"
                    % (
                        horizon,
                        rank_ic_mean,
                        delta_p,
                        p_at_k_mean,
                        p_rand_mean,
                        delta_ndcg,
                        ndcg_mean,
                        ndcg_rand_mean,
                        overlap_mean,
                        spread_mean,
                        wil_mean,
                        arrays["rank_ic"].size,
                        k_over_n_mean,
                    )
                )

            if val_panel_count == 0:
                print("[WARN] Validation loader yielded no panels; verify mask coverage and min_stocks_per_day.")
            else:
                print(f"[INFO] Processed {val_panel_count} validation panels.")

            # Save per-day metrics with ratio in filename
            if args.output:
                output_path = Path(args.output)
                ratio_int = int(round(eval_ratio_override * 100))
                npz_path = output_path.with_name(f"{output_path.stem}_reval_k{ratio_int:02d}_val_perday.npz")
                payload: dict[str, np.ndarray] = {}
                for horizon, stats in per_horizon.items():
                    prefix = f"h{horizon}_"
                    payload[prefix + "date_int"] = np.asarray(stats["date_int"], dtype=np.int64)
                    for key, values in stats.items():
                        if key == "date_int":
                            continue
                        payload[prefix + key] = np.asarray(values, dtype=np.float64)
                if payload:
                    np.savez_compressed(npz_path, **payload)
                    print(f"[EVAL-ONLY] Saved validation per-day metrics to {npz_path}")

            if ema is not None:
                ema.restore(model)

        print("\n[EVAL-ONLY] Completed. Exiting.")
        sys.exit(0)
    # ========== END EVAL-ONLY MODE ==========

    time_decay_tau = float(train_cfg.get("time_decay_tau_days", 0.0))

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
        from torch.cuda.amp import GradScaler as CudaGradScaler
        from torch.cuda.amp import autocast as cuda_autocast

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
    monitor_metric = args.early_stopping_metric or early_cfg.get("metric")
    if not monitor_metric:
        monitor_metric = "short_term_score"
    min_delta = float(early_cfg.get("min_delta", 0.0))
    if patience is not None:
        early_stopping = EarlyStopping(patience=int(patience), mode="max", min_delta=min_delta)
        print(f"[INFO] Early stopping enabled: patience={patience}, metric={monitor_metric}, min_delta={min_delta:.4f}")

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

            # GRADIENT-SAFE: Skip if all horizons were invalid (empty aggregation)
            if loss is None:
                if step % train_cfg.get("log_interval", 10) == 0:
                    print(f"[WARN] Step {step}: No valid horizon loss (skipping)")
                optimizer.zero_grad(set_to_none=True)
                continue

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            # GRADIENT-SAFE: Verify loss has gradient before backward
            if not loss.requires_grad:
                raise RuntimeError(
                    f"Loss has no grad at step={step}. "
                    "Check CompositeLoss/compute_loss: avoid .item()/detach/empty aggregation"
                )

            if time_decay_tau > 0 and train_total_steps > 0:
                age = train_total_steps - step
                weight = math.exp(-float(age) / max(time_decay_tau, 1e-6))
                loss = loss * weight

            scaler.scale(loss).backward()
            if grad_clip := train_cfg.get("grad_clip"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)

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

        # Validation (use EMA weights when enabled)
        if ema is not None:
            ema.apply(model)
        model.eval()
        per_horizon: dict[int, dict[str, list[float]]] = {
            h: {
                "date_int": [],
                "rank_ic": [],
                "topk_overlap": [],
                "p_at_k_pos": [],
                "p_at_k_pos_rand": [],
                "ndcg": [],
                "ndcg_rand": [],
                "spread": [],
                "k_over_n": [],
                "wil": [],
            }
            for h in horizons
        }
        val_panel_count = 0
        ndcg_beta = ndcg_beta_value
        eval_k = eval_k_value
        eval_ratio = eval_ratio_value
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                date_int = int(batch.get("date_int", 0))
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
                    score_vec = s.detach().float().cpu()
                    label_vec = target.detach().float().cpu()
                    n = label_vec.numel()
                    if n == 0:
                        continue
                    k = k_from_ratio(n, k=eval_k, ratio=eval_ratio)
                    ic = spearman_ic(score_vec, label_vec)
                    overlap = topk_overlap(score_vec, label_vec, k)
                    p_at_k, p_rand = precision_at_k_pos(score_vec, label_vec, k)
                    nd_val = ndcg_at_k(score_vec, label_vec, k, beta=ndcg_beta)
                    nd_rand = ndcg_random_baseline(label_vec, k, beta=ndcg_beta)
                    spread_val = top_bottom_spread(score_vec, label_vec, k)
                    wil_val = wil_at_k(score_vec, label_vec, k)

                    stats = per_horizon[horizon]
                    stats["date_int"].append(date_int)
                    stats["rank_ic"].append(ic)
                    stats["topk_overlap"].append(overlap)
                    stats["p_at_k_pos"].append(p_at_k)
                    stats["p_at_k_pos_rand"].append(p_rand)
                    stats["ndcg"].append(nd_val)
                    stats["ndcg_rand"].append(nd_rand)
                    stats["spread"].append(spread_val)
                    stats["k_over_n"].append(k / max(1, n))
                    stats["wil"].append(wil_val)

        # Compute and print validation metrics
        summary: dict[int, dict[str, float]] = {}
        for horizon in horizons:
            stats = per_horizon[horizon]
            if not stats["rank_ic"]:
                continue
            arrays = {key: np.asarray(values, dtype=np.float64) for key, values in stats.items() if key != "date_int"}
            rank_ic_mean = float(arrays["rank_ic"].mean())
            overlap_mean = float(arrays["topk_overlap"].mean())
            p_at_k_mean = float(arrays["p_at_k_pos"].mean())
            p_rand_mean = float(arrays["p_at_k_pos_rand"].mean())
            delta_p = float((arrays["p_at_k_pos"] - arrays["p_at_k_pos_rand"]).mean())
            ndcg_mean = float(arrays["ndcg"].mean())
            ndcg_rand_mean = float(arrays["ndcg_rand"].mean())
            delta_ndcg = float((arrays["ndcg"] - arrays["ndcg_rand"]).mean())
            spread_mean = float(arrays["spread"].mean())
            k_over_n_mean = float(arrays["k_over_n"].mean())
            wil_mean = float(arrays["wil"].mean())

            summary[horizon] = {
                "rank_ic": rank_ic_mean,
                "topk_overlap": overlap_mean,
                "p_at_k_pos": p_at_k_mean,
                "p_at_k_pos_rand": p_rand_mean,
                "delta_p_at_k_pos": delta_p,
                "ndcg": ndcg_mean,
                "ndcg_rand": ndcg_rand_mean,
                "delta_ndcg": delta_ndcg,
                "spread": spread_mean,
                "k_over_n": k_over_n_mean,
                "wil": wil_mean,
                "n_days": int(arrays["rank_ic"].size),
            }

            print(
                "[Val] h=%2dd RankIC=%.4f ΔP@K=%.4f (P@K=%.4f | rand=%.4f) "
                "ΔNDCG=%.4f (NDCG=%.4f | rand=%.4f) "
                "Overlap=%.4f Spread=%.4f WIL=%.4f (panels=%d, K/N=%.3f)"
                % (
                    horizon,
                    rank_ic_mean,
                    delta_p,
                    p_at_k_mean,
                    p_rand_mean,
                    delta_ndcg,
                    ndcg_mean,
                    ndcg_rand_mean,
                    overlap_mean,
                    spread_mean,
                    wil_mean,
                    arrays["rank_ic"].size,
                    k_over_n_mean,
                )
            )
        if val_panel_count == 0:
            print("[WARN] Validation loader yielded no panels; verify mask coverage and min_stocks_per_day.")
        else:
            print(f"[INFO] Processed {val_panel_count} validation panels.")

        metric_results: dict[str, float] = {}
        for horizon, stats in summary.items():
            metric_results[f"{horizon}d_ic"] = stats["rank_ic"]
            metric_results[f"{horizon}d_pak"] = stats["p_at_k_pos"]
            metric_results[f"{horizon}d_delta_pak"] = stats["delta_p_at_k_pos"]
            metric_results[f"{horizon}d_ndcg"] = stats["ndcg"]
            metric_results[f"{horizon}d_delta_ndcg"] = stats["delta_ndcg"]
            metric_results[f"{horizon}d_overlap"] = stats["topk_overlap"]
            metric_results[f"{horizon}d_spread"] = stats["spread"]
            metric_results[f"{horizon}d_wil"] = stats["wil"]

        short_term_score = 0.0
        total_weight = 0.0
        for horizon in short_term_horizons:
            weight = es_weights.get(horizon, 0.0)
            if weight <= 0 or horizon not in summary:
                continue
            short_term_score += weight * summary[horizon]["p_at_k_pos"]
            total_weight += weight
        if total_weight > 0:
            print(f"[Val] Short-term score (5/10/20d weighted) = {short_term_score:.4f}")
            metric_results["short_term_score"] = short_term_score

        if args.output:
            output_path = Path(args.output)
            npz_path = output_path.with_name(output_path.stem + "_val_perday.npz")
            payload: dict[str, np.ndarray] = {}
            for horizon, stats in per_horizon.items():
                prefix = f"h{horizon}_"
                payload[prefix + "date_int"] = np.asarray(stats["date_int"], dtype=np.int64)
                for key, values in stats.items():
                    if key == "date_int":
                        continue
                    payload[prefix + key] = np.asarray(values, dtype=np.float64)
            if payload:
                np.savez_compressed(npz_path, **payload)
                print(f"[INFO] Saved validation per-day metrics to {npz_path}")

        if ema is not None and snapshot_epochs and epoch in snapshot_epochs:
            if args.output:
                snapshot_path = Path(args.output).with_name(f"{Path(args.output).stem}_ema_epoch{epoch}.pt")
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), snapshot_path)
                print(f"[Snapshot] Saved EMA weights to {snapshot_path}")
                snapshot_epochs.discard(epoch)
            elif not warned_snapshot_no_output:
                print("[WARN] --ema-snapshot-epochs provided but --output missing; skipping snapshot exports.")
                warned_snapshot_no_output = True
                snapshot_epochs.discard(epoch)

        if ema is not None:
            ema.restore(model)

        # Check early stopping
        if early_stopping is not None and metric_results:
            if monitor_metric in metric_results:
                score = metric_results[monitor_metric]
                model_state = clone_state_dict(model.state_dict())
                ema_state = ema.clone_shadow() if ema is not None else None
                should_stop = early_stopping(score, epoch, model_state, ema_state)

                if should_stop:
                    print(f"[INFO] Restoring best model from epoch {early_stopping.best_epoch}")
                    model.load_state_dict(early_stopping.best_state)
                    break
            else:
                print(
                    f"[WARN] Early stopping metric '{monitor_metric}' not found in results. Available: {list(metric_results.keys())}"
                )

    if early_stopping is not None and early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)
        if ema is not None and early_stopping.best_extra is not None:
            ema.load_shadow(early_stopping.best_extra)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if ema is not None:
            ema.apply(model)
            torch.save(model.state_dict(), out_path)
            ema.restore(model)
        else:
            torch.save(model.state_dict(), out_path)
        print(f"Saved model to {out_path}")
    else:
        print("[WARN] --output not set; best model weights were not saved to disk.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - debugging aid
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        raise
