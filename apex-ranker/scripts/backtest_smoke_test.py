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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping

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
    should_rebalance,
)
from apex_ranker.backtest.inference import (
    BacktestInferenceEngine,
    compute_weight_turnover,
    date_to_int,
    ensure_date,
    int_to_date,
    resolve_device,
)
from apex_ranker.data import (
    FeatureSelector,
    add_cross_sectional_zscores,
    build_panel_cache,
    load_panel_cache,
    panel_cache_key,
    save_panel_cache,
)
from apex_ranker.data.loader import load_backtest_frame
from apex_ranker.models import APEXRankerV0
from apex_ranker.utils import load_config

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

    return {
        row["Code"]: float(row["returns_5d"])
        for row in predictions.iter_rows(named=True)
    }


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
) -> dict:
    """
    Execute backtest using either mock predictions or model inference.

    Returns:
        Backtest summary dictionary (also written to disk when requested).
    """
    print("\n" + "=" * 80)
    print("Phase 3: Backtest Driver")
    print("=" * 80)

    rebalance_mode = normalise_frequency(rebalance_freq)

    config: dict | None = None
    feature_cols: list[str] | None = None
    lookback = 0

    if config_path is not None and config_path.exists():
        config = load_config(str(config_path))
        feature_cols = get_feature_columns(config)
        lookback = config["data"]["lookback"]
        print(f"[Backtest] Loaded config: {config_path}")
    elif model_path is not None and not use_mock:
        raise FileNotFoundError(
            "Model inference requested but config file was not provided or found."
        )

    frame = load_backtest_frame(
        data_path=data_path,
        start_date=start_date,
        end_date=end_date,
        feature_cols=feature_cols or [],
        lookback=lookback,
    )

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
    cache_directory = (
        Path(panel_cache_dir).expanduser()
        if panel_cache_dir is not None
        else Path("cache/panel")
    )

    if model_path is not None and not use_mock:
        inference_engine = BacktestInferenceEngine(
            model_path=model_path,
            config=config,
            frame=frame,
            feature_cols=feature_cols or [],
            device=device,
            dataset_path=data_path,
            panel_cache_dir=cache_directory,
        )
        prediction_dates = inference_engine.available_dates()
        print(
            f"[Backtest] Inference ready on {len(prediction_dates)} dates "
            f"(device={inference_engine.device})"
        )
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

    candidate_request = max(
        resolved_top_k,
        int(resolved_top_k * max(candidate_multiplier, 1.0)),
        optimization_config.target_top_k * 2,
    )

    daily_results: list[dict] = []
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
                    print(
                        f"[Backtest] {current_date}: insufficient lookback, skipping "
                        "rebalance attempt"
                    )
                else:
                    rankings = inference_engine.predict(
                        target_date=current_date,
                        horizon=horizon,
                        top_k=candidate_request,
                    )

                    if rankings.is_empty():
                        print(
                            f"[Backtest] {current_date}: model produced no candidates"
                        )
                    else:
                        available_codes = set(price_map.keys())
                        filtered = rankings.filter(
                            pl.col("Code").is_in(list(available_codes))
                        ).sort("Rank")
                        if filtered.is_empty():
                            print(
                                f"[Backtest] {current_date}: "
                                "no overlap between predictions and price data"
                            )
                        else:
                            pool_limit = optimization_config.candidate_count(
                                filtered.height
                            )
                            if pool_limit <= 0:
                                print(
                                    f"[Backtest] {current_date}: candidate pool exhausted"
                                )
                            else:
                                filtered = filtered.head(pool_limit)
                                predictions = {
                                    row["Code"]: float(row["Score"])
                                    for row in filtered.iter_rows(named=True)
                                }
                                prediction_source = "model"
            else:
                predictions = generate_mock_predictions(current_frame, candidate_request)
                prediction_source = "mock"

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
                    fallback_codes = list(predictions.keys())[
                        : optimization_config.target_top_k
                    ]
                    if fallback_codes:
                        weight = 1.0 / len(fallback_codes)
                        opt_weights = {code: weight for code in fallback_codes}
                        opt_result.selected_codes = list(opt_weights.keys())
                        fallback_turnover = compute_weight_turnover(
                            portfolio.weights, opt_weights
                        )
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
        state["prediction_source"] = (
            last_prediction_source if last_prediction_source else prediction_source
        )
        state["selection_count"] = len(portfolio.positions)
        state["optimized_top_k"] = (
            len(target_weights) if target_weights else len(portfolio.positions)
        )
        if target_weights:
            state["target_weights"] = {
                code: float(weight) for code, weight in target_weights.items()
            }
        if optimization_summary is not None:
            state["optimization"] = optimization_summary
        if portfolio.positions:
            state["selected_codes"] = ",".join(sorted(portfolio.positions.keys()))
        else:
            state["selected_codes"] = ""
        state["avg_prediction_score"] = (
            float(np.mean(list(active_predictions.values())))
            if active_predictions
            else None
        )
        state["num_trades"] = len(trades)
        state["rebalanced"] = did_rebalance
        state["last_rebalance_date"] = (
            str(last_rebalance_date) if last_rebalance_date else None
        )
        daily_results.append(state)

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
            "prediction_mode": "model" if inference_engine else "mock",
        },
        "cost_model": cost_model_info,
        "summary": {
            "trading_days": len(daily_results),
            "total_trades": total_trades,
            "prediction_days_available": len(prediction_dates)
            if inference_engine
            else len(trading_dates),
            "rebalance_count": rebalance_count,
        },
        "performance": metrics,
        "daily_results_sample": history_records[:10],
        "trades_sample": trades_records[:10],
    }

    artifacts: dict[str, str] = {}

    if daily_metrics_path and str(daily_metrics_path).strip():
        daily_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        flattened_history = []
        for record in history_records:
            flat = {
                k: v
                for k, v in record.items()
                if k not in {"positions", "target_weights", "optimization"}
            }
            if "positions" in record:
                flat["positions_json"] = json.dumps(
                    record["positions"], ensure_ascii=False
                )
            if "target_weights" in record:
                flat["target_weights_json"] = json.dumps(
                    record["target_weights"], ensure_ascii=False
                )
            if "optimization" in record:
                flat["optimization_json"] = json.dumps(
                    record["optimization"], ensure_ascii=False
                )
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
    )


if __name__ == "__main__":
    main()
