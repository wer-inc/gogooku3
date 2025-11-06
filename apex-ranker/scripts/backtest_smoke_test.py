#!/usr/bin/env python3
"""
APEX-Ranker Phase 3 backtest driver.

Integrates trained model inference, realistic transaction costs, and daily
reporting to validate ranking models over arbitrary horizons. Supports both
mock predictions (for lightweight smoke tests) and real inference using the
Phase 2 pipeline components.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import date as Date
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
from apex_ranker.backtest import (
    CostCalculator,
    OptimizationConfig,
    Portfolio,
    Trade,
    generate_target_weights,
    normalise_frequency,
    select_by_percentile,
    should_rebalance,
)
from apex_ranker.backtest.enhanced_inference import (
    hysteresis_selection,
    risk_neutralize,
)
from apex_ranker.backtest.inference import (
    BacktestInferenceEngine,
    compute_weight_turnover,
    ensure_date,
)
from apex_ranker.data import FeatureSelector
from apex_ranker.data.loader import load_backtest_frame
from apex_ranker.utils import (
    block_bootstrap_ci,
    deflated_sharpe_ratio,
    load_config,
    ndcg_at_k,
    ndcg_random_baseline,
    precision_at_k_pos,
    spearman_ic,
    top_bottom_spread,
    topk_overlap,
    wil_at_k,
)

DATE_EPOCH = Date(1970, 1, 1)


def get_feature_columns(config: dict) -> list[str]:
    """Determine active feature list from config (respecting exclusions)."""
    data_cfg = config["data"]
    selector = FeatureSelector(data_cfg["feature_groups_config"])
    groups = list(data_cfg.get("feature_groups", []))
    if data_cfg.get("use_plus30"):
        groups.append("plus30")
    selection = selector.select(
        groups=groups,
        optional_groups=data_cfg.get("optional_groups", []),
        exclude_features=data_cfg.get("exclude_features"),
    )
    return list(selection.features)


def build_daily_lookup(frame: pl.DataFrame) -> dict[Date, pl.DataFrame]:
    """Partition dataset by date for fast lookup."""
    daily_frames: dict[Date, pl.DataFrame] = {}
    for day_frame in frame.partition_by("Date", maintain_order=True):
        day = ensure_date(day_frame[0, "Date"])
        daily_frames[day] = day_frame
    return daily_frames


def trade_to_dict(trade: Trade) -> dict[str, float | str]:
    """Convert ``Trade`` dataclass to dictionary with JSON-friendly values."""
    record = asdict(trade)
    record["date"] = str(trade.date)
    return record


def generate_mock_predictions(
    date_frame: pl.DataFrame,
    top_k: int = 50,
) -> dict[str, float]:
    """
    Generate heuristic predictions for smoke testing.

    Uses 5-day forward returns as a proxy ranking signal.
    """
    if date_frame.is_empty():
        return {}

    predictions = (
        date_frame.select(["Code", "returns_5d"])
        .drop_nulls("returns_5d")
        .sort("returns_5d", descending=True)
        .head(top_k)
    )

    return {row["Code"]: float(row["returns_5d"]) for row in predictions.iter_rows(named=True)}


def run_backtest_smoke_test(
    data_path: Path,
    start_date: str | None,
    end_date: str | None,
    initial_capital: float = 10_000_000,
    top_k: int = 50,
    output_path: Path | None = None,
    *,
    model_path: Path | None = None,
    config_path: Path | None = None,
    horizon: int = 20,
    device: str = "auto",
    use_mock: bool = False,
    daily_metrics_path: Path | None = None,
    trades_path: Path | None = None,
    rebalance_freq: str = "weekly",
    optimization_target_top_k: int | None = None,
    min_position_weight: float = 0.02,
    turnover_limit: float = 0.35,
    cost_penalty: float = 1.0,
    candidate_multiplier: float = 2.0,
    min_alpha: float = 0.1,
    panel_cache_dir: Path | None = None,
    selection_k_ratio: float | None = None,
    selection_k_min: int | None = None,
    selection_sign: int | None = None,
    panel_cache_salt: str | None = None,
    use_enhanced_inference: bool = False,
    ei_hysteresis_entry_k: int = 35,
    ei_hysteresis_exit_k: int = 60,
    ei_neutralize_risk: bool = False,
    ei_risk_factors: list[str] | None = None,
    ei_neutralize_gamma: float = 0.3,
    ei_ridge_alpha: float = 10.0,
    infer_add_csz: bool = False,
    infer_csz_eps: float = 1e-6,
    infer_csz_clip: float = 5.0,
    features_mode: str = "drop-missing",
) -> dict:
    """
    Execute backtest using either mock predictions or model inference.

    Returns:
        Backtest summary dictionary (also written to disk when requested).
    """
    if selection_sign is not None and selection_sign not in (-1, 1):
        raise ValueError("selection_sign must be either +1 or -1 when provided.")

    print("\n" + "=" * 80)
    print("Phase 3: Backtest Driver")
    print("=" * 80)

    rebalance_mode = normalise_frequency(rebalance_freq)

    config: dict | None = None
    feature_cols: list[str] | None = None
    lookback = 0

    aliases_yaml: str | None = None
    if config_path is not None and config_path.exists():
        config = load_config(str(config_path))
        # Fix ①: CLI --data priority override (take precedence over config default)
        if data_path is not None:
            config.setdefault("data", {})["parquet_path"] = str(data_path)
            print(f"[override] data.parquet_path = {config['data']['parquet_path']}")
        feature_cols = get_feature_columns(config)
        lookback = config["data"]["lookback"]
        # Check for feature aliases configuration
        aliases_yaml = config.get("data", {}).get("feature_aliases_yaml")
        print(f"[Backtest] Loaded config: {config_path}")
    elif model_path is not None and not use_mock:
        raise FileNotFoundError("Model inference requested but config file was not provided or found.")

    # Prepare passthrough columns for Enhanced Inference (A.4)
    passthrough_cols = []
    if use_enhanced_inference and ei_neutralize_risk and ei_risk_factors:
        passthrough_cols = ei_risk_factors
        print(f"[A.4] Requesting passthrough columns for risk neutralization: {passthrough_cols}")

    frame = load_backtest_frame(
        data_path=data_path,
        start_date=start_date,
        end_date=end_date,
        feature_cols=feature_cols or [],
        lookback=lookback,
        aliases_yaml=aliases_yaml,
        features_mode=features_mode,
        passthrough_cols=passthrough_cols,
    )

    # Extract risk factors from the loaded frame (now included via passthrough_cols)
    # These are NOT in feature_cols but needed for risk neutralization
    risk_factors_frame = None
    if use_enhanced_inference and ei_neutralize_risk and passthrough_cols:
        # Risk factors are now in the main frame thanks to passthrough_cols
        available_risk_cols = ["Code", "Date"] + [c for c in passthrough_cols if c in frame.columns]
        risk_factors_frame = frame.select(available_risk_cols)
        loaded_factors = [c for c in available_risk_cols if c not in ["Code", "Date"]]
        print(f"[A.4] Risk factors extracted from frame: {loaded_factors}")

    daily_frames = build_daily_lookup(frame)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else None
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else None

    trading_dates = [
        day
        for day in sorted(daily_frames.keys())
        if (start_dt is None or day >= start_dt) and (end_dt is None or day <= end_dt)
    ]

    if len(trading_dates) < 2:
        raise ValueError("Not enough trading days in the specified window.")

    inference_engine: BacktestInferenceEngine | None = None
    prediction_dates: set[Date] = set()
    cache_directory = Path(panel_cache_dir).expanduser() if panel_cache_dir is not None else Path("cache/panel")

    # Print Enhanced Inference configuration
    if use_enhanced_inference:
        print("\n" + "=" * 80)
        print("Enhanced Inference (EI) Configuration")
        print("=" * 80)
        if ei_neutralize_risk:
            print("[A.4] Risk Neutralization: ENABLED")
            factors_str = ", ".join(ei_risk_factors) if ei_risk_factors else "sector33_code, volatility_60d"
            print(f"      Factors: {factors_str}")
            print(f"      Gamma (partial neutralization): {ei_neutralize_gamma}")
            print(f"      Ridge alpha: {ei_ridge_alpha}")
        else:
            print("[A.4] Risk Neutralization: DISABLED")
        print("[A.3] Hysteresis Selection: ENABLED")
        print(f"      Entry threshold: {ei_hysteresis_entry_k}")
        print(f"      Exit threshold: {ei_hysteresis_exit_k}")
        print("=" * 80 + "\n")

    if model_path is not None and not use_mock:
        inference_engine = BacktestInferenceEngine(
            model_path=model_path,
            config=config,
            frame=frame,
            feature_cols=feature_cols or [],
            device=device,
            dataset_path=data_path,
            panel_cache_dir=cache_directory,
            cache_salt=panel_cache_salt,
            aliases_yaml=aliases_yaml,
            add_csz=infer_add_csz,
            csz_eps=infer_csz_eps,
            csz_clip=infer_csz_clip,
        )
        prediction_dates = inference_engine.available_dates()
        print(f"[Backtest] Inference ready on {len(prediction_dates)} dates " f"(device={inference_engine.device})")
    else:
        print("[Backtest] Using mock predictions (returns_5d proxy)")

    portfolio = Portfolio(initial_capital)
    cost_calculator = CostCalculator()

    resolved_top_k = max(1, int(top_k))
    target_top_k_value = optimization_target_top_k or resolved_top_k
    target_top_k_value = max(1, min(int(target_top_k_value), resolved_top_k))
    optimization_config = OptimizationConfig(
        target_top_k=target_top_k_value,
        candidate_multiplier=candidate_multiplier,
        min_weight=max(1e-3, float(min_position_weight)),
        turnover_limit=max(0.0, float(turnover_limit)),
        cost_penalty=max(0.0, float(cost_penalty)),
        min_alpha=max(0.0, min(1.0, float(min_alpha))),
    )

    print(f"[Backtest] Initial capital: ¥{initial_capital:,.0f}")
    print(f"[Backtest] Candidate Top-K: {resolved_top_k}")
    print(
        f"[Backtest] Optimised holdings target: {optimization_config.target_top_k} "
        f"(min_weight={optimization_config.min_weight:.4f}, "
        f"turnover_limit={optimization_config.turnover_limit:.2f}, "
        f"cost_penalty={optimization_config.cost_penalty:.2f})"
    )
    print(f"[Backtest] Horizon: {horizon}d")
    print(f"[Backtest] Panel cache directory: {cache_directory}")

    selection_cfg = config.get("selection", {}) if config else {}
    default_selection = selection_cfg.get("default", {}) if isinstance(selection_cfg, dict) else {}
    horizon_selection = (
        selection_cfg.get("horizons", {}).get(str(horizon), {}) if isinstance(selection_cfg, dict) else {}
    )
    effective_k_ratio = (
        selection_k_ratio
        if selection_k_ratio is not None
        else horizon_selection.get("k_ratio", default_selection.get("k_ratio"))
    )
    effective_k_min = (
        selection_k_min
        if selection_k_min is not None
        else horizon_selection.get("k_min", default_selection.get("k_min"))
    )
    effective_sign = (
        selection_sign if selection_sign is not None else horizon_selection.get("sign", default_selection.get("sign"))
    )
    if effective_sign is None:
        effective_sign = 1
    effective_sign = int(effective_sign)
    if effective_sign not in (-1, 1):
        raise ValueError("selection_sign must resolve to either +1 or -1.")

    candidate_request = max(
        resolved_top_k,
        int(resolved_top_k * max(candidate_multiplier, 1.0)),
        optimization_config.target_top_k * 2,
    )

    daily_results: list[dict] = []
    cross_section_metrics: list[dict[str, object]] = []
    rebalance_count = 0
    last_rebalance_date: Date | None = None
    last_predictions: dict[str, float] | None = None
    last_prediction_source: str | None = None

    for idx, current_date in enumerate(trading_dates[:-1]):
        next_date = trading_dates[idx + 1]
        current_frame = daily_frames.get(current_date)
        next_frame = daily_frames.get(next_date)

        if current_frame is None or next_frame is None:
            continue

        candidate_pool_size: int | None = None
        gate_candidate_total: int | None = None
        gate_threshold_value: float | None = None
        gate_ratio_used: float | None = None
        gate_fallback_used = False

        # Build price/volume maps
        price_map = {
            code: float(price)
            for code, price in zip(
                current_frame["Code"].to_list(),
                current_frame["Close"].to_list(),
                strict=False,
            )
            if price is not None
        }

        volume_map: dict[str, float] = {}
        turnover_values = current_frame["TurnoverValue"].to_list()
        volumes = current_frame["Volume"].to_list()
        closes = current_frame["Close"].to_list()
        codes = current_frame["Code"].to_list()
        for code, turnover_value, volume, close in zip(
            codes,
            turnover_values,
            volumes,
            closes,
            strict=False,
        ):
            if turnover_value is not None and turnover_value > 0:
                volume_map[code] = float(turnover_value)
            elif volume is not None and close is not None:
                volume_map[code] = float(volume) * float(close)

        prediction_source = last_prediction_source or (
            "model" if inference_engine is not None and not use_mock else "mock"
        )
        predictions: dict[str, float] | None = None
        trades: list[Trade] = []
        daily_cost = 0.0
        did_rebalance = False
        target_weights: dict[str, float] = {}
        optimization_summary: dict[str, object] | None = None

        if should_rebalance(current_date, last_rebalance_date, rebalance_mode):
            if inference_engine is not None and not use_mock:
                if current_date not in prediction_dates:
                    print(f"[Backtest] {current_date}: insufficient lookback, skipping " "rebalance attempt")
                else:
                    rankings = inference_engine.predict(
                        target_date=current_date,
                        horizon=horizon,
                        top_k=candidate_request,
                    )

                    if rankings.is_empty():
                        print(f"[Backtest] {current_date}: model produced no candidates")
                    else:
                        available_codes = set(price_map.keys())
                        filtered = rankings.filter(pl.col("Code").is_in(list(available_codes))).sort("Rank")
                        if filtered.is_empty():
                            print(f"[Backtest] {current_date}: " "no overlap between predictions and price data")
                        else:
                            pool_limit = optimization_config.candidate_count(filtered.height)
                            if pool_limit <= 0:
                                print(f"[Backtest] {current_date}: candidate pool exhausted")
                            else:
                                filtered = filtered.head(pool_limit)
                                gate_candidate_total = filtered.height
                                scores_tensor = torch.tensor(filtered["Score"].to_numpy(), dtype=torch.float32)
                                gate_candidate_total = int(scores_tensor.numel())
                                if scores_tensor.numel() == 0:
                                    print(f"[Backtest] {current_date}: no usable scores after filtering")
                                else:
                                    gate_ratio_used = (
                                        effective_k_ratio
                                        if effective_k_ratio is not None
                                        else optimization_config.target_top_k / max(1, scores_tensor.numel())
                                    )
                                    gate_ratio_used = float(max(0.0, min(1.0, gate_ratio_used)))
                                    gate_minimum = (
                                        effective_k_min
                                        if effective_k_min is not None
                                        else optimization_config.target_top_k
                                    )
                                    gate_minimum = max(1, int(gate_minimum))

                                    # ----- A.4: Risk Neutralization (if enabled) -----
                                    if use_enhanced_inference and ei_neutralize_risk:
                                        # Extract risk factors from current_frame
                                        # SAFE defaults: Start with sector+vol only (per user spec)
                                        # NOTE: Use lowercase names to match dataset columns
                                        default_factors = ["sector33_code", "volatility_60d"]
                                        risk_factors_to_use = ei_risk_factors or default_factors

                                        # Get risk factor data from preloaded risk_factors_frame
                                        if risk_factors_frame is not None:
                                            # Filter to current date and codes
                                            codes_in_order = filtered["Code"].to_list()
                                            df_risk = risk_factors_frame.filter(
                                                (pl.col("Date") == current_date)
                                                & (pl.col("Code").is_in(codes_in_order))
                                            )
                                            # CRITICAL: Reorder df_risk to match codes_in_order (not alphabetically)
                                            # This ensures alignment between scores_tensor and risk factors
                                            # Create a mapping from code to index in codes_in_order
                                            code_to_index = {code: idx for idx, code in enumerate(codes_in_order)}
                                            df_risk = (
                                                df_risk.with_columns(
                                                    pl.col("Code")
                                                    .map_elements(
                                                        lambda c: code_to_index.get(c, len(codes_in_order)),
                                                        return_dtype=pl.Int64,
                                                    )
                                                    .alias("_sort_order")
                                                )
                                                .sort("_sort_order")
                                                .drop("_sort_order")
                                            )

                                            # Verify alignment: df_risk codes should match codes_in_order
                                            df_risk_codes = df_risk["Code"].to_list()
                                            codes_in_risk = set(df_risk_codes)
                                            codes_in_scores = set(codes_in_order)

                                            # Find codes that are missing in either direction
                                            missing_in_risk = codes_in_scores - codes_in_risk
                                            missing_in_scores = codes_in_risk - codes_in_scores

                                            if missing_in_risk or missing_in_scores:
                                                print(
                                                    f"[A.4] WARNING: Code mismatch detected. "
                                                    f"Missing in risk factors: {missing_in_risk if missing_in_risk else 'none'}, "
                                                    f"Extra in risk factors: {missing_in_scores if missing_in_scores else 'none'}"
                                                )

                                            # Filter to only codes that exist in both
                                            common_codes = list(codes_in_risk & codes_in_scores)
                                            original_count = len(codes_in_order)
                                            if len(common_codes) != len(codes_in_order):
                                                # Reorder common codes to match codes_in_order
                                                common_codes = [c for c in codes_in_order if c in common_codes]
                                                # Filter df_risk to common codes and maintain order
                                                df_risk = df_risk.filter(pl.col("Code").is_in(common_codes))
                                                # Re-sort to match common_codes order
                                                code_to_index = {code: idx for idx, code in enumerate(common_codes)}
                                                df_risk = (
                                                    df_risk.with_columns(
                                                        pl.col("Code")
                                                        .map_elements(
                                                            lambda c: code_to_index.get(c, len(common_codes)),
                                                            return_dtype=pl.Int64,
                                                        )
                                                        .alias("_sort_order")
                                                    )
                                                    .sort("_sort_order")
                                                    .drop("_sort_order")
                                                )
                                                # Filter scores_tensor to match common codes
                                                scores_indices = [codes_in_order.index(c) for c in common_codes]
                                                scores_tensor = scores_tensor[scores_indices]
                                                codes_in_order = common_codes
                                                print(
                                                    f"[A.4] Filtered to {len(common_codes)} common codes "
                                                    f"(removed {original_count - len(common_codes)} codes)"
                                                )
                                            elif df_risk_codes != codes_in_order:
                                                # Log warning if order mismatch detected (should not happen after fix)
                                                mismatched = sum(
                                                    1 for a, b in zip(df_risk_codes, codes_in_order) if a != b
                                                )
                                                if mismatched > 0:
                                                    print(
                                                        f"[A.4] WARNING: {mismatched} codes misaligned between "
                                                        f"scores and risk factors. This may cause incorrect neutralization."
                                                    )

                                            # Case-insensitive factor matching
                                            cols_lower = {c.lower(): c for c in df_risk.columns}
                                            available_factors = []
                                            missing_factors = []
                                            for f in risk_factors_to_use:
                                                f_lower = f.lower()
                                                if f_lower in cols_lower:
                                                    available_factors.append(cols_lower[f_lower])
                                                else:
                                                    missing_factors.append(f)

                                            if missing_factors:
                                                print(f"[A.4] WARNING: Risk factors not found: {missing_factors}")

                                            # Convert to Pandas (risk_neutralize expects pd.DataFrame)
                                            df_risk_pd = (
                                                df_risk.select(available_factors).to_pandas()
                                                if available_factors
                                                else None
                                            )
                                        else:
                                            print("[A.4] ERROR: risk_factors_frame not loaded")
                                            df_risk_pd = None
                                            available_factors = []

                                        # Apply risk neutralization (SAFE VERSION with gamma)
                                        if df_risk_pd is not None and len(available_factors) > 0:
                                            try:
                                                print(
                                                    f"[A.4] {current_date}: Applying risk neutralization "
                                                    f"(n={len(scores_tensor)}, factors={available_factors}, "
                                                    f"gamma={ei_neutralize_gamma}, alpha={ei_ridge_alpha})"
                                                )
                                                scores_neutralized = risk_neutralize(
                                                    scores_tensor.numpy(),
                                                    df_risk_pd,
                                                    factors=available_factors,
                                                    alpha=ei_ridge_alpha,
                                                    gamma=ei_neutralize_gamma,
                                                )
                                                # Convert back to tensor
                                                scores_tensor = torch.from_numpy(scores_neutralized).to(
                                                    dtype=torch.float32
                                                )
                                                print(f"[A.4] {current_date}: Risk neutralization applied successfully")
                                            except Exception as e:
                                                print(
                                                    f"[Backtest] {current_date}: "
                                                    f"Risk neutralization failed ({e}), using raw scores"
                                                )
                                        else:
                                            print(
                                                f"[Backtest] {current_date}: "
                                                f"No risk factors available, skipping neutralization"
                                            )

                                    # ----- A.3: Hysteresis Selection (if enabled) -----
                                    if use_enhanced_inference:
                                        # Get previous holdings as Code→index mapping
                                        prev_codes = list(portfolio.positions.keys())
                                        code_to_idx = {code: i for i, code in enumerate(filtered["Code"].to_list())}
                                        prev_indices = [code_to_idx[code] for code in prev_codes if code in code_to_idx]

                                        # Apply hysteresis selection
                                        selected_indices = hysteresis_selection(
                                            scores_tensor.numpy(),
                                            current_holdings=prev_indices if prev_indices else None,
                                            entry_k=ei_hysteresis_entry_k,
                                            exit_k=ei_hysteresis_exit_k,
                                        )

                                        # Convert to tensor
                                        idx_tensor = torch.tensor(selected_indices, dtype=torch.long)

                                        # Threshold is minimum score (for logging)
                                        if len(selected_indices) > 0:
                                            threshold_signed = float(scores_tensor[selected_indices[-1]].item())
                                        else:
                                            threshold_signed = float("nan")

                                        # Fallback flag (hysteresis doesn't use fallback)
                                        gate_fallback_used = False
                                    else:
                                        # Existing logic (select_by_percentile)
                                        idx_tensor, threshold_signed, gate_fallback_used = select_by_percentile(
                                            scores_tensor,
                                            k_ratio=gate_ratio_used,
                                            k_min=gate_minimum,
                                            sign=effective_sign,
                                        )
                                    if idx_tensor.numel() == 0:
                                        gate_fallback_used = True
                                        fallback_count = min(scores_tensor.numel(), gate_minimum)
                                        if fallback_count == 0:
                                            idx_tensor = torch.empty(0, dtype=torch.long)
                                            filtered = filtered.head(0)
                                        else:
                                            signed_scores = scores_tensor * float(effective_sign)
                                            idx_tensor = torch.topk(
                                                signed_scores,
                                                fallback_count,
                                                sorted=True,
                                            ).indices
                                            threshold_signed = float(signed_scores[idx_tensor[-1]].item())
                                    if idx_tensor.numel() > 0:
                                        idx_tensor = torch.unique(idx_tensor, sorted=True)
                                        idx_list = idx_tensor.tolist()
                                        if idx_list:
                                            selected_rows = [filtered.row(i) for i in idx_list]
                                            filtered = pl.DataFrame(
                                                selected_rows,
                                                schema=filtered.schema,
                                                orient="row",
                                            )
                                        else:
                                            filtered = filtered.head(0)
                                    candidate_pool_size = filtered.height
                                    if idx_tensor.numel() > 0:
                                        gate_threshold_value = float(threshold_signed * float(effective_sign))
                                    else:
                                        gate_threshold_value = None
                                    predictions = {
                                        row["Code"]: float(row["Score"]) for row in filtered.iter_rows(named=True)
                                    }
                                    if candidate_pool_size is None:
                                        candidate_pool_size = len(predictions)
                                    if gate_ratio_used is not None:
                                        threshold_display = (
                                            gate_threshold_value if gate_threshold_value is not None else float("nan")
                                        )
                                        print(
                                            "[Backtest] "
                                            f"{current_date}: gate_ratio={gate_ratio_used:.3f} "
                                            f"threshold={threshold_display:.6f} "
                                            f"fallback={int(gate_fallback_used)} "
                                            f"candidate_total={gate_candidate_total} "
                                            f"candidate_kept={candidate_pool_size} "
                                            f"sign={effective_sign}"
                                        )
                                    prediction_source = "model"
            else:
                predictions = generate_mock_predictions(current_frame, candidate_request)
                prediction_source = "mock"
                candidate_pool_size = len(predictions)

            if predictions:
                opt_weights, opt_result = generate_target_weights(
                    predictions,
                    portfolio.weights,
                    portfolio_value=portfolio.total_value,
                    config=optimization_config,
                    cost_calculator=cost_calculator,
                    volumes=volume_map,
                )

                if not opt_weights and optimization_config.turnover_limit > 0.0:
                    relaxed_config = replace(
                        optimization_config,
                        turnover_limit=1.0,
                        min_alpha=min(optimization_config.min_alpha, 0.05),
                    )
                    opt_weights, opt_result = generate_target_weights(
                        predictions,
                        portfolio.weights,
                        portfolio_value=portfolio.total_value,
                        config=relaxed_config,
                        cost_calculator=cost_calculator,
                        volumes=volume_map,
                    )
                    opt_result.notes.append("turnover_constraint_relaxed")

                if not opt_weights:
                    fallback_codes = list(predictions.keys())[: optimization_config.target_top_k]
                    if fallback_codes:
                        weight = 1.0 / len(fallback_codes)
                        opt_weights = dict.fromkeys(fallback_codes, weight)
                        opt_result.selected_codes = list(opt_weights.keys())
                        fallback_turnover = compute_weight_turnover(portfolio.weights, opt_weights)
                        opt_result.unconstrained_turnover = fallback_turnover
                        opt_result.constrained_turnover = fallback_turnover
                        opt_result.applied_alpha = 1.0
                        opt_result.notes.append("fallback_equal_weights")

                if opt_result:
                    optimization_summary = opt_result.to_dict()

                if opt_weights:
                    target_weights = opt_weights
                    trades = portfolio.rebalance(
                        target_weights=target_weights,
                        prices=price_map,
                        date=current_date,
                        cost_calculator=cost_calculator,
                        volumes=volume_map,
                    )

                    daily_cost = sum(trade.total_cost for trade in trades)
                    last_rebalance_date = current_date
                    last_predictions = predictions
                    last_prediction_source = prediction_source
                    rebalance_count += 1
                    did_rebalance = True
                else:
                    predictions = last_predictions
            else:
                predictions = last_predictions

        if predictions is None:
            predictions = last_predictions

        active_predictions = predictions or {}
        daily_turnover = portfolio.calculate_turnover(trades)

        cross_metrics_record: dict[str, object] | None = None
        label_col = f"returns_{horizon}d"
        if active_predictions and label_col in current_frame.columns:
            codes = current_frame["Code"].to_list()
            label_values = current_frame[label_col].to_list()
            label_map = {
                code: float(value)
                for code, value in zip(codes, label_values, strict=False)
                if value is not None and np.isfinite(value)
            }

            scored_codes: list[str] = []
            score_buffer: list[float] = []
            label_buffer: list[float] = []

            for code, score in active_predictions.items():
                label_value = label_map.get(code)
                if label_value is None or not np.isfinite(label_value):
                    continue
                scored_codes.append(code)
                score_buffer.append(float(score))
                label_buffer.append(float(label_value))

            if score_buffer:
                score_tensor = torch.tensor(score_buffer, dtype=torch.float32)
                label_tensor = torch.tensor(label_buffer, dtype=torch.float32)
                k_eval = max(1, min(resolved_top_k, score_tensor.numel()))
                p_at_k, p_rand = precision_at_k_pos(score_tensor, label_tensor, k_eval)
                ndcg_val = ndcg_at_k(score_tensor, label_tensor, k_eval)
                ndcg_rand = ndcg_random_baseline(label_tensor, k_eval)
                overlap_val = topk_overlap(score_tensor, label_tensor, k_eval)
                spread_val = top_bottom_spread(score_tensor, label_tensor, k_eval)
                wil_val = wil_at_k(score_tensor, label_tensor, k_eval)
                cross_metrics_record = {
                    "prediction_date": str(current_date),
                    "prediction_source": prediction_source,
                    "cross_section_total": int(
                        sum(1 for value in label_values if value is not None and np.isfinite(value))
                    ),
                    "scored_count": int(score_tensor.numel()),
                    "k_eval": int(k_eval),
                    "rank_ic": float(spearman_ic(score_tensor, label_tensor)),
                    "topk_overlap": float(overlap_val),
                    "p_at_k_pos": float(p_at_k),
                    "p_at_k_pos_rand": float(p_rand),
                    "delta_p_at_k_pos": float(p_at_k - p_rand),
                    "ndcg_at_k": float(ndcg_val),
                    "ndcg_rand": float(ndcg_rand),
                    "delta_ndcg": float(ndcg_val - ndcg_rand),
                    "wil_at_k": float(wil_val),
                    "spread": float(spread_val),
                    "k_over_n": float(k_eval / max(1, score_tensor.numel())),
                    "mean_label": float(np.mean(label_buffer)) if label_buffer else float("nan"),
                }
                cross_metrics_record["selection_threshold"] = (
                    float(gate_threshold_value) if gate_threshold_value is not None else None
                )
                cross_metrics_record["selection_gate_fallback"] = (
                    int(gate_fallback_used) if gate_candidate_total is not None else None
                )
                cross_metrics_record["selection_k_ratio"] = (
                    float(gate_ratio_used) if gate_ratio_used is not None else None
                )
                cross_metrics_record["selection_candidates_total"] = (
                    int(gate_candidate_total) if gate_candidate_total is not None else None
                )
                cross_metrics_record["selection_candidates_kept"] = (
                    int(candidate_pool_size) if candidate_pool_size is not None else None
                )
                cross_metrics_record["selection_k_min"] = int(effective_k_min) if effective_k_min is not None else None
                cross_metrics_record["selection_sign"] = effective_sign

        next_prices = {
            code: float(price)
            for code, price in zip(
                next_frame["Code"].to_list(),
                next_frame["Close"].to_list(),
                strict=False,
            )
            if price is not None
        }

        portfolio.update_prices(next_prices, next_date)

        state = portfolio.log_state(
            next_date,
            turnover=daily_turnover,
            transaction_cost=daily_cost,
        )
        state["prediction_date"] = str(current_date)
        state["prediction_source"] = last_prediction_source if last_prediction_source else prediction_source
        state["selection_count"] = len(portfolio.positions)
        state["optimized_top_k"] = len(target_weights) if target_weights else len(portfolio.positions)
        effective_candidate_count = candidate_pool_size if candidate_pool_size is not None else len(active_predictions)
        state["candidate_count"] = int(effective_candidate_count)
        state["k_over_n"] = (
            float(len(target_weights) / max(1, effective_candidate_count))
            if target_weights and effective_candidate_count
            else None
        )
        state["selection_threshold"] = (
            float(gate_threshold_value) if gate_ratio_used is not None and gate_threshold_value is not None else None
        )
        state["selection_gate_fallback"] = int(gate_fallback_used) if gate_candidate_total is not None else None
        state["selection_k_ratio"] = (
            float(gate_ratio_used)
            if gate_ratio_used is not None
            else (float(effective_k_ratio) if effective_k_ratio is not None else None)
        )
        state["selection_candidates_total"] = int(gate_candidate_total) if gate_candidate_total is not None else None
        state["selection_k_min"] = int(effective_k_min) if effective_k_min is not None else None
        state["selection_sign"] = effective_sign
        if cross_metrics_record:
            state["cs_cross_section"] = cross_metrics_record["cross_section_total"]
            state["cs_scored_count"] = cross_metrics_record["scored_count"]
            state["cs_k_eval"] = cross_metrics_record["k_eval"]
            state["cs_rank_ic"] = cross_metrics_record["rank_ic"]
            state["cs_topk_overlap"] = cross_metrics_record["topk_overlap"]
            state["cs_p_at_k_pos"] = cross_metrics_record["p_at_k_pos"]
            state["cs_p_at_k_pos_rand"] = cross_metrics_record["p_at_k_pos_rand"]
            state["cs_delta_p_at_k_pos"] = cross_metrics_record["delta_p_at_k_pos"]
            state["cs_ndcg_at_k"] = cross_metrics_record["ndcg_at_k"]
            state["cs_ndcg_rand"] = cross_metrics_record["ndcg_rand"]
            state["cs_delta_ndcg"] = cross_metrics_record["delta_ndcg"]
            state["cs_wil_at_k"] = cross_metrics_record["wil_at_k"]
            state["cs_spread"] = cross_metrics_record["spread"]
            state["cs_k_over_n"] = cross_metrics_record["k_over_n"]
            state["cs_mean_label"] = cross_metrics_record["mean_label"]
        if target_weights:
            state["target_weights"] = {code: float(weight) for code, weight in target_weights.items()}
        if optimization_summary is not None:
            state["optimization"] = optimization_summary
        if portfolio.positions:
            state["selected_codes"] = ",".join(sorted(portfolio.positions.keys()))
        else:
            state["selected_codes"] = ""
        state["avg_prediction_score"] = (
            float(np.mean(list(active_predictions.values()))) if active_predictions else None
        )
        state["num_trades"] = len(trades)
        state["rebalanced"] = did_rebalance
        state["last_rebalance_date"] = str(last_rebalance_date) if last_rebalance_date else None
        daily_results.append(state)

        if cross_metrics_record is not None:
            cross_metrics_record.update(
                {
                    "result_date": str(next_date),
                    "daily_return_pct": state.get("daily_return"),
                    "turnover": state.get("turnover"),
                    "transaction_cost": state.get("transaction_cost"),
                }
            )
            cross_section_metrics.append(cross_metrics_record)

        if idx % 5 == 0:
            print(
                f"[Backtest] {next_date}: "
                f"PV=¥{state['portfolio_value']:,.0f}, "
                f"Return={state['daily_return']:.2f}%, "
                f"Turnover={daily_turnover:.2%}, "
                f"Cost=¥{daily_cost:,.0f}"
            )

    metrics = portfolio.calculate_metrics()
    total_trades = len(portfolio.get_trades())

    cost_cfg = cost_calculator.config
    cost_model_info = {
        "base_spread_bps": cost_cfg.base_spread_bps,
        "market_impact_factor": cost_cfg.market_impact_factor,
        "max_slippage_bps": cost_cfg.max_slippage_bps,
        "commission_tiers": cost_cfg.commission_tiers,
    }

    print("\n" + "=" * 80)
    print("Backtest Results")
    print("=" * 80)
    print(f"  Prediction mode: {'Model' if inference_engine else 'Mock'}")
    print(f"  Rebalance frequency: {rebalance_mode}")
    print(f"  Trading days simulated: {len(daily_results)}")
    print(f"  Rebalances executed: {rebalance_count}")
    print(f"  Total trades: {total_trades}")
    print(f"  Total return: {metrics.get('total_return', 0.0):.2f}%")
    print(f"  Annualized return: {metrics.get('annualized_return', 0.0):.2f}%")
    print(f"  Sharpe ratio: {metrics.get('sharpe_ratio', 0.0):.3f}")
    print(f"  Sortino ratio: {metrics.get('sortino_ratio', 0.0):.3f}")
    print(f"  Max drawdown: {metrics.get('max_drawdown', 0.0):.2f}%")
    print(f"  Calmar ratio: {metrics.get('calmar_ratio', 0.0):.3f}")
    print(f"  Win rate: {metrics.get('win_rate', 0.0):.1%}")
    tx_costs = metrics.get("transaction_costs", {})
    print(
        f"  Total transaction costs: ¥{tx_costs.get('total_cost', 0.0):,.0f} "
        f"({tx_costs.get('cost_pct_of_pv', 0.0):.2f}% of capital)"
    )
    print(f"  Avg daily cost: {tx_costs.get('avg_daily_cost_bps', 0.0):.2f} bps")

    trades_records = [trade_to_dict(trade) for trade in portfolio.get_trades()]
    history_records = portfolio.get_history()

    results = {
        "config": {
            "data_path": str(data_path),
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": initial_capital,
            "top_k": resolved_top_k,
            "target_top_k": optimization_config.target_top_k,
            "min_position_weight": optimization_config.min_weight,
            "turnover_limit": optimization_config.turnover_limit,
            "cost_penalty": optimization_config.cost_penalty,
            "candidate_multiplier": optimization_config.candidate_multiplier,
            "min_alpha": optimization_config.min_alpha,
            "horizon": horizon,
            "rebalance_frequency": rebalance_mode,
            "model_path": str(model_path) if model_path else None,
            "config_path": str(config_path) if config_path else None,
            "device": device,
            "panel_cache_salt": panel_cache_salt,
            "prediction_mode": "model" if inference_engine else "mock",
            "selection_k_ratio": effective_k_ratio,
            "selection_k_min": effective_k_min,
            "selection_sign": effective_sign,
        },
        "cost_model": cost_model_info,
        "summary": {
            "trading_days": len(daily_results),
            "total_trades": total_trades,
            "prediction_days_available": len(prediction_dates) if inference_engine else len(trading_dates),
            "rebalance_count": rebalance_count,
        },
        "performance": metrics,
        "daily_results_sample": history_records[:10],
        "trades_sample": trades_records[:10],
    }

    evaluation_payload: dict[str, object] = {}

    if cross_section_metrics:

        def _nanmean(key: str) -> float:
            values = [record.get(key) for record in cross_section_metrics]
            values = [v for v in values if v is not None and np.isfinite(v)]
            if not values:
                return float("nan")
            return float(np.mean(values))

        summary = {
            "count": len(cross_section_metrics),
            "rank_ic_mean": _nanmean("rank_ic"),
            "topk_overlap_mean": _nanmean("topk_overlap"),
            "p_at_k_pos_mean": _nanmean("p_at_k_pos"),
            "p_at_k_pos_rand_mean": _nanmean("p_at_k_pos_rand"),
            "delta_p_at_k_pos_mean": _nanmean("delta_p_at_k_pos"),
            "ndcg_at_k_mean": _nanmean("ndcg_at_k"),
            "ndcg_rand_mean": _nanmean("ndcg_rand"),
            "delta_ndcg_mean": _nanmean("delta_ndcg"),
            "wil_at_k_mean": _nanmean("wil_at_k"),
            "spread_mean": _nanmean("spread"),
            "k_over_n_mean": _nanmean("k_over_n"),
            "scored_count_mean": _nanmean("scored_count"),
            "cross_section_mean": _nanmean("cross_section_total"),
        }

        bootstrap_results: dict[str, dict[str, float]] = {}
        bootstrap_metrics = [
            "rank_ic",
            "delta_p_at_k_pos",
            "delta_ndcg",
            "topk_overlap",
            "spread",
        ]
        for metric in bootstrap_metrics:
            series = [
                float(record[metric])
                for record in cross_section_metrics
                if record.get(metric) is not None and np.isfinite(record[metric])
            ]
            if len(series) >= 5:
                block_len = max(2, int(np.sqrt(len(series))))
                bootstrap_results[metric] = block_bootstrap_ci(
                    series,
                    block_size=block_len,
                    n_bootstrap=500,
                )

        evaluation_payload["summary"] = summary
        evaluation_payload["per_day"] = cross_section_metrics
        if bootstrap_results:
            evaluation_payload["bootstrap"] = bootstrap_results

    returns_series = (
        np.array(
            [record["daily_return"] for record in history_records[1:]],
            dtype=float,
        )
        / 100.0
    )
    dsr_result = deflated_sharpe_ratio(
        returns_series,
        n_trials=1,
    )
    if dsr_result is not None:
        evaluation_payload["risk"] = {
            "sharpe": dsr_result.sharpe,
            "dsr": dsr_result.dsr,
            "threshold": dsr_result.sr_threshold,
            "sharpe_std": dsr_result.sr_std,
            "n_obs": dsr_result.n_obs,
            "n_trials": dsr_result.n_trials,
        }

    if evaluation_payload:
        results["evaluation_metrics"] = evaluation_payload

    artifacts: dict[str, str] = {}

    if daily_metrics_path and str(daily_metrics_path).strip():
        daily_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        flattened_history = []
        for record in history_records:
            flat = {k: v for k, v in record.items() if k not in {"positions", "target_weights", "optimization"}}
            if "positions" in record:
                flat["positions_json"] = json.dumps(record["positions"], ensure_ascii=False)
            if "target_weights" in record:
                flat["target_weights_json"] = json.dumps(record["target_weights"], ensure_ascii=False)
            if "optimization" in record:
                flat["optimization_json"] = json.dumps(record["optimization"], ensure_ascii=False)
            flattened_history.append(flat)
        if flattened_history:
            pl.DataFrame(flattened_history).write_csv(daily_metrics_path)
            artifacts["daily_metrics_csv"] = str(daily_metrics_path)

    if trades_path:
        trades_path.parent.mkdir(parents=True, exist_ok=True)
        if trades_records:
            pl.DataFrame(trades_records).write_csv(trades_path)
            artifacts["trades_csv"] = str(trades_path)

    if artifacts:
        results["artifacts"] = artifacts

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Backtest] Results saved to: {output_path}")

    print("\n✅ Backtest completed")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="APEX-Ranker Phase 3 backtest driver")
    parser.add_argument(
        "--data",
        default="output/ml_dataset_latest_full.parquet",
        help="Path to parquet dataset",
    )
    parser.add_argument(
        "--start-date",
        default="2025-09-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        default="2025-09-30",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10_000_000,
        help="Initial capital in JPY",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of stocks to hold",
    )
    parser.add_argument(
        "--target-top-k",
        type=int,
        default=35,
        help="Target holdings count after optimisation (default: 35)",
    )
    parser.add_argument(
        "--min-position-weight",
        type=float,
        default=0.02,
        help="Minimum allocation weight per position (default: 0.02)",
    )
    parser.add_argument(
        "--turnover-limit",
        type=float,
        default=0.35,
        help="Maximum turnover fraction allowed per rebalance (default: 0.35)",
    )
    parser.add_argument(
        "--cost-penalty",
        type=float,
        default=1.0,
        help="Penalty multiplier applied to estimated round-trip transaction costs",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=float,
        default=2.0,
        help="Multiplier controlling candidate pool size vs. target top-k (default: 2.0)",
    )
    parser.add_argument(
        "--min-alpha",
        type=float,
        default=0.1,
        help="Minimum adjustment factor when enforcing turnover constraints (default: 0.1)",
    )
    parser.add_argument(
        "--selection-k-ratio",
        type=float,
        default=None,
        help="Percentile gating ratio (0–1). Defaults to target_top_k divided by candidate count.",
    )
    parser.add_argument(
        "--selection-k-min",
        type=int,
        default=None,
        help="Minimum number of securities to keep after gating (default: target_top_k).",
    )
    parser.add_argument(
        "--selection-sign",
        type=int,
        default=None,
        help="Signal orientation for gating (1 keeps highest scores, -1 keeps lowest). Overrides config when set.",
    )
    parser.add_argument(
        "--use-enhanced-inference",
        action="store_true",
        help="Enable enhanced inference (A.3 Hysteresis + A.4 Risk Neutralization)",
    )
    parser.add_argument(
        "--ei-hysteresis-entry-k",
        type=int,
        default=35,
        help="Hysteresis entry threshold (default: 35)",
    )
    parser.add_argument(
        "--ei-hysteresis-exit-k",
        type=int,
        default=60,
        help="Hysteresis exit threshold (default: 60)",
    )
    parser.add_argument(
        "--ei-neutralize-risk",
        action="store_true",
        help="Enable risk neutralization (A.4)",
    )
    parser.add_argument(
        "--ei-risk-factors",
        type=str,
        default=None,
        help="Comma-separated risk factors (default: Sector33Code,volatility_60d)",
    )
    parser.add_argument(
        "--ei-neutralize-gamma",
        type=float,
        default=0.3,
        help="Partial neutralization factor (default: 0.3, range: [0.2, 0.5]). 0.0=no neutralization, 1.0=full neutralization",
    )
    parser.add_argument(
        "--ei-ridge-alpha",
        type=float,
        default=10.0,
        help="Ridge regression regularization (default: 10.0, higher=more conservative)",
    )
    parser.add_argument(
        "--infer-add-csz",
        action="store_true",
        help="Enable cross-sectional Z-score generation at inference (89→178 dims)",
    )
    parser.add_argument(
        "--infer-csz-eps",
        type=float,
        default=1e-6,
        help="Epsilon for CS-Z std protection (default: 1e-6)",
    )
    parser.add_argument(
        "--infer-csz-clip",
        type=float,
        default=5.0,
        help="Clip CS-Z outliers to ±N sigma (default: 5.0)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to trained model checkpoint (.pt). If omitted, uses mock predictions.",
    )
    parser.add_argument(
        "--config",
        default="apex-ranker/configs/v0_pruned.yaml",
        help="Model config YAML path (required when using --model).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=20,
        choices=[1, 5, 10, 20],
        help="Prediction horizon in days",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--rebalance-freq",
        default="weekly",
        choices=["daily", "weekly", "monthly"],
        help="Rebalance frequency (default: weekly)",
    )
    parser.add_argument(
        "--use-mock-predictions",
        action="store_true",
        help="Force mock predictions even if a model path is supplied.",
    )
    parser.add_argument(
        "--panel-cache-dir",
        default="cache/panel",
        help="Directory for persisted panel caches (default: cache/panel)",
    )
    parser.add_argument(
        "--daily-csv",
        default=None,
        help="Optional path to write daily metrics CSV",
    )
    parser.add_argument(
        "--trades-csv",
        default=None,
        help="Optional path to write trade log CSV",
    )
    parser.add_argument(
        "--features-mode",
        default="drop-missing",
        choices=["strict", "fill-zero", "drop-missing"],
        help="Feature validation mode: strict (error on missing), fill-zero (fill missing with 0.0), drop-missing (continue with available features)",
    )
    parser.add_argument(
        "--output",
        default="results/backtest_phase3.json",
        help="Output JSON summary path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    model_path = Path(args.model) if args.model else None
    config_path = Path(args.config) if args.config else None
    output_path = Path(args.output) if args.output else None
    daily_metrics_path = Path(args.daily_csv) if args.daily_csv else None
    trades_path = Path(args.trades_csv) if args.trades_csv else None
    panel_cache_dir = Path(args.panel_cache_dir).expanduser() if args.panel_cache_dir else None

    run_backtest_smoke_test(
        data_path=data_path,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        top_k=args.top_k,
        output_path=output_path,
        model_path=model_path,
        config_path=config_path,
        horizon=args.horizon,
        device=args.device,
        use_mock=args.use_mock_predictions,
        daily_metrics_path=daily_metrics_path,
        trades_path=trades_path,
        rebalance_freq=args.rebalance_freq,
        optimization_target_top_k=args.target_top_k,
        min_position_weight=args.min_position_weight,
        turnover_limit=args.turnover_limit,
        cost_penalty=args.cost_penalty,
        candidate_multiplier=args.candidate_multiplier,
        min_alpha=args.min_alpha,
        panel_cache_dir=panel_cache_dir,
        selection_k_ratio=args.selection_k_ratio,
        selection_k_min=args.selection_k_min,
        selection_sign=args.selection_sign,
        use_enhanced_inference=args.use_enhanced_inference,
        ei_hysteresis_entry_k=args.ei_hysteresis_entry_k,
        ei_hysteresis_exit_k=args.ei_hysteresis_exit_k,
        ei_neutralize_risk=args.ei_neutralize_risk,
        ei_risk_factors=(args.ei_risk_factors.split(",") if args.ei_risk_factors else None),
        ei_neutralize_gamma=args.ei_neutralize_gamma,
        ei_ridge_alpha=args.ei_ridge_alpha,
        infer_add_csz=args.infer_add_csz,
        infer_csz_eps=args.infer_csz_eps,
        infer_csz_clip=args.infer_csz_clip,
        features_mode=args.features_mode,
    )


if __name__ == "__main__":
    main()
