#!/usr/bin/env python3
"""
APEX-Ranker Phase 4.3.2: Regime-Adaptive Backtest Driver

Extends backtest_smoke_test.py with real-time regime detection and dynamic exposure control.

Key Features:
- Real-time regime calculation from portfolio history
- Dynamic position sizing based on market conditions
- Crisis protection (20% exposure in extreme volatility)
- Bull market preservation (100% exposure in strong trends)
- Comprehensive regime logging and diagnostics

Usage:
    python apex-ranker/scripts/backtest_regime_adaptive.py \
        --start-date 2021-11-01 --end-date 2022-03-31 \
        --model models/apex_ranker_v0_enhanced.pt \
        --config apex-ranker/configs/v0_base.yaml \
        --enable-regime-detection \
        --output results/regime_adaptive_crisis_test.json
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

# Add apex-ranker to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import base backtest functionality
import json
from datetime import date as Date
from datetime import datetime

import numpy as np
import polars as pl
from apex_ranker.backtest import (
    CostCalculator,
    OptimizationConfig,
    Portfolio,
    generate_target_weights,
    normalise_frequency,
    should_rebalance,
)
from apex_ranker.backtest.inference import (
    BacktestInferenceEngine,
    compute_weight_turnover,
)
from apex_ranker.data.loader import load_backtest_frame
from apex_ranker.utils import load_config
from backtest_smoke_test import (
    build_daily_lookup,
    generate_mock_predictions,
    get_feature_columns,
    trade_to_dict,
)

# Import regime detection
from realtime_regime import RealtimeRegimeCalculator
from regime_detection import DefensiveRiskManager


def run_regime_adaptive_backtest(
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
    rebalance_freq: str = "monthly",
    enable_regime_detection: bool = True,
    regime_lookback: int = 20,
    daily_metrics_path: Path | None = None,
    optimization_target_top_k: int | None = None,
    min_position_weight: float = 0.02,
    turnover_limit: float = 0.35,
    cost_penalty: float = 1.0,
    candidate_multiplier: float = 2.0,
    min_alpha: float = 0.1,
    panel_cache_dir: Path | None = None,
) -> dict:
    """
    Execute regime-adaptive backtest with real-time market regime detection.

    Args:
        data_path: Path to parquet dataset
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        initial_capital: Initial capital in JPY
        top_k: Number of stocks to hold
        output_path: Output JSON path
        model_path: Path to trained model (.pt)
        config_path: Model config YAML path
        horizon: Prediction horizon in days
        device: Device for inference ('auto', 'cuda', 'cpu')
        use_mock: Force mock predictions
        rebalance_freq: Rebalance frequency ('daily', 'weekly', 'monthly')
        enable_regime_detection: Enable regime-adaptive exposure control
        regime_lookback: Lookback window for regime calculation (days)
        daily_metrics_path: Optional path for daily CSV output

    Returns:
        Backtest summary dictionary with regime diagnostics
    """
    print("\n" + "=" * 80)
    print("Phase 4.3.2: Regime-Adaptive Backtest Driver")
    print("=" * 80)
    print(f"Regime Detection: {'ENABLED' if enable_regime_detection else 'DISABLED'}")
    print(f"Regime Lookback: {regime_lookback} days")
    print("=" * 80)

    rebalance_mode = normalise_frequency(rebalance_freq)

    # Load configuration and features
    config: dict | None = None
    feature_cols: list[str] | None = None
    lookback = 0

    if config_path is not None and config_path.exists():
        config = load_config(str(config_path))
        feature_cols = get_feature_columns(config)
        lookback = config["data"]["lookback"]
        print(f"[Regime Backtest] Loaded config: {config_path}")
    elif model_path is not None and not use_mock:
        raise FileNotFoundError(
            "Model inference requested but config file not provided."
        )

    # Load dataset
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
        raise ValueError("Not enough trading days in specified window.")

    # Initialize inference engine
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
            f"[Regime Backtest] Inference ready on {len(prediction_dates)} dates "
            f"(device={inference_engine.device})"
        )
    else:
        print("[Regime Backtest] Using mock predictions (returns_5d proxy)")

    # Initialize regime components
    regime_calculator = (
        RealtimeRegimeCalculator(lookback_days=regime_lookback)
        if enable_regime_detection
        else None
    )
    risk_manager = DefensiveRiskManager() if enable_regime_detection else None

    # Initialize portfolio
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

    print(f"[Regime Backtest] Initial capital: ¥{initial_capital:,.0f}")
    print(f"[Regime Backtest] Candidate Top-K: {resolved_top_k}")
    print(
        "[Regime Backtest] Optimised holdings target: "
        f"{optimization_config.target_top_k} "
        f"(min_weight={optimization_config.min_weight:.4f}, "
        f"turnover_limit={optimization_config.turnover_limit:.2f}, "
        f"cost_penalty={optimization_config.cost_penalty:.2f})"
    )
    print(f"[Regime Backtest] Horizon: {horizon}d")
    print(f"[Regime Backtest] Rebalance frequency: {rebalance_mode}")
    print(f"[Regime Backtest] Panel cache directory: {cache_directory}")

    candidate_request = max(
        resolved_top_k,
        int(resolved_top_k * max(candidate_multiplier, 1.0)),
        optimization_config.target_top_k * 2,
    )

    # Tracking variables
    daily_results: list[dict] = []
    regime_diagnostics: list[dict] = []
    rebalance_count = 0
    last_rebalance_date: Date | None = None
    last_predictions: dict[str, float] | None = None
    last_prediction_source: str | None = None

    # Main backtest loop
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

        # Regime detection and exposure calculation
        regime_exposure = 1.0  # Default: full exposure
        regime_info = None
        target_weights: dict[str, float] = {}
        optimization_summary: dict[str, object] | None = None

        if should_rebalance(current_date, last_rebalance_date, rebalance_mode):
            # Calculate regime signals
            if enable_regime_detection and regime_calculator and risk_manager:
                if len(daily_results) >= regime_calculator.min_observations:
                    try:
                        # Calculate regime from portfolio history
                        signals = regime_calculator.calculate_regime(
                            portfolio_history=daily_results,
                            current_date=current_date,
                        )

                        # Calculate current drawdown
                        if daily_results:
                            latest_pv = daily_results[-1]["portfolio_value"]
                            peak_pv = max(r["portfolio_value"] for r in daily_results)
                            current_dd = (
                                (latest_pv - peak_pv) / peak_pv if peak_pv > 0 else 0.0
                            )
                        else:
                            current_dd = 0.0

                        # Calculate recommended exposure
                        regime_exposure = risk_manager.calculate_exposure(
                            signals, current_dd
                        )

                        regime_info = {
                            "date": str(current_date),
                            "regime": signals.regime.value,
                            "confidence": signals.confidence,
                            "exposure": regime_exposure,
                            "realized_vol": signals.realized_vol,
                            "momentum_20d": signals.momentum_20d,
                            "max_dd_20d": signals.max_dd_20d,
                            "current_drawdown": current_dd,
                        }

                        regime_diagnostics.append(regime_info)

                        print(
                            f"[Regime] {current_date}: {signals.regime.value.upper():>8s} "
                            f"(conf={signals.confidence:.2f}) → Exposure {regime_exposure:.0%}, "
                            f"Vol={signals.realized_vol:.1%}, Mom={signals.momentum_20d:+.1%}"
                        )

                    except Exception as e:
                        print(
                            f"[Regime] {current_date}: Regime calculation failed: {e}"
                        )
                        regime_exposure = 1.0
                else:
                    print(
                        f"[Regime] {current_date}: Insufficient history ({len(daily_results)} days)"
                    )
                    regime_exposure = 1.0

            # Generate predictions (same as base backtest)
            if inference_engine is not None and not use_mock:
                if current_date not in prediction_dates:
                    print(
                        f"[Regime Backtest] {current_date}: insufficient lookback, skipping rebalance"
                    )
                else:
                    rankings = inference_engine.predict(
                        target_date=current_date,
                        horizon=horizon,
                        top_k=candidate_request,
                    )

                    if rankings.is_empty():
                        print(
                            f"[Regime Backtest] {current_date}: model produced no candidates"
                        )
                    else:
                        available_codes = set(price_map.keys())
                        filtered = rankings.filter(
                            pl.col("Code").is_in(list(available_codes))
                        ).sort("Rank")
                        if filtered.is_empty():
                            print(
                                f"[Regime Backtest] {current_date}: "
                                "no overlap between predictions and price data"
                            )
                        else:
                            pool_limit = optimization_config.candidate_count(
                                filtered.height
                            )
                            if pool_limit <= 0:
                                print(
                                    f"[Regime Backtest] {current_date}: candidate pool exhausted"
                                )
                            else:
                                filtered = filtered.head(pool_limit)
                                predictions = {
                                    row["Code"]: float(row["Score"])
                                    for row in filtered.iter_rows(named=True)
                                }
                                prediction_source = "model"
            else:
                predictions = generate_mock_predictions(
                    current_frame, candidate_request
                )
                prediction_source = "mock"

            # Apply regime-adaptive position sizing
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
                        opt_weights = dict.fromkeys(fallback_codes, weight)
                        opt_result.selected_codes = list(opt_weights.keys())
                        fallback_turnover = compute_weight_turnover(
                            portfolio.weights, opt_weights
                        )
                        opt_result.unconstrained_turnover = fallback_turnover
                        opt_result.constrained_turnover = fallback_turnover
                        opt_result.applied_alpha = 1.0
                        opt_result.notes.append("fallback_equal_weights")

                if opt_weights:
                    scaled_weights = {
                        code: weight * regime_exposure
                        for code, weight in opt_weights.items()
                    }
                    target_weights = scaled_weights
                    actual_turnover = compute_weight_turnover(
                        portfolio.weights, target_weights
                    )
                    opt_result.constrained_turnover = actual_turnover
                    optimization_summary = opt_result.to_dict()
                    optimization_summary["regime_exposure"] = regime_exposure

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

        # Update portfolio with next day prices
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

        # Log portfolio state
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

        # Add regime info to state
        if regime_info:
            state["regime"] = regime_info["regime"]
            state["regime_exposure"] = regime_info["exposure"]
            state["regime_confidence"] = regime_info["confidence"]
        else:
            state["regime"] = "n/a"
            state["regime_exposure"] = 1.0
            state["regime_confidence"] = None

        daily_results.append(state)

        if idx % 5 == 0:
            regime_str = (
                f"[{state['regime'].upper()}:{state['regime_exposure']:.0%}]"
                if enable_regime_detection
                else ""
            )
            print(
                f"[Regime Backtest] {next_date}: "
                f"PV=¥{state['portfolio_value']:,.0f}, "
                f"Return={state['daily_return']:.2f}%, "
                f"Turnover={daily_turnover:.2%} "
                f"{regime_str}"
            )

    # Calculate final metrics
    metrics = portfolio.calculate_metrics()
    total_trades = len(portfolio.get_trades())

    cost_cfg = cost_calculator.config
    cost_model_info = {
        "base_spread_bps": cost_cfg.base_spread_bps,
        "market_impact_factor": cost_cfg.market_impact_factor,
        "max_slippage_bps": cost_cfg.max_slippage_bps,
        "commission_tiers": cost_cfg.commission_tiers,
    }

    # Print results
    print("\n" + "=" * 80)
    print("Regime-Adaptive Backtest Results")
    print("=" * 80)
    print(f"  Prediction mode: {'Model' if inference_engine else 'Mock'}")
    print(f"  Regime detection: {'ENABLED' if enable_regime_detection else 'DISABLED'}")
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

    if enable_regime_detection and regime_diagnostics:
        print("\n" + "-" * 80)
        print("Regime Statistics")
        print("-" * 80)
        regimes = [r["regime"] for r in regime_diagnostics]
        exposures = [r["exposure"] for r in regime_diagnostics]
        print(f"  Regime detections: {len(regime_diagnostics)}")
        print(f"  Average exposure: {np.mean(exposures):.1%}")
        print(f"  Min exposure: {np.min(exposures):.1%}")
        print(f"  Max exposure: {np.max(exposures):.1%}")
        for regime_name in ["crisis", "bear", "bull", "sideways"]:
            count = regimes.count(regime_name)
            if count > 0:
                pct = 100 * count / len(regimes)
                avg_exp = np.mean(
                    [
                        r["exposure"]
                        for r in regime_diagnostics
                        if r["regime"] == regime_name
                    ]
                )
                print(
                    f"  {regime_name.capitalize():>8s}: {count:3d} days ({pct:4.1f}%), Avg Exposure={avg_exp:.0%}"
                )

    tx_costs = metrics.get("transaction_costs", {})
    print(
        f"\n  Total transaction costs: ¥{tx_costs.get('total_cost', 0.0):,.0f} "
        f"({tx_costs.get('cost_pct_of_pv', 0.0):.2f}% of capital)"
    )
    print(f"  Avg daily cost: {tx_costs.get('avg_daily_cost_bps', 0.0):.2f} bps")

    # Prepare output
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
            "regime_detection_enabled": enable_regime_detection,
            "regime_lookback": regime_lookback,
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
        "regime_diagnostics": regime_diagnostics if enable_regime_detection else [],
        "daily_results_sample": history_records[:10],
        "trades_sample": trades_records[:10],
    }

    # Write daily metrics CSV if requested
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
            results["daily_metrics_csv"] = str(daily_metrics_path)

    # Write JSON output
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Regime Backtest] Results saved to: {output_path}")

    print("\n✅ Regime-Adaptive Backtest completed")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="APEX-Ranker Phase 4.3.2 Regime-Adaptive Backtest Driver"
    )
    parser.add_argument(
        "--data",
        default="output/ml_dataset_latest_full.parquet",
        help="Path to parquet dataset",
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        required=True,
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
        help="Path to trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--config",
        default="apex-ranker/configs/v0_base.yaml",
        help="Model config YAML path (required when using --model)",
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
        default="monthly",
        choices=["daily", "weekly", "monthly"],
        help="Rebalance frequency",
    )
    parser.add_argument(
        "--use-mock-predictions",
        action="store_true",
        help="Force mock predictions",
    )
    parser.add_argument(
        "--enable-regime-detection",
        action="store_true",
        help="Enable regime-adaptive exposure control",
    )
    parser.add_argument(
        "--regime-lookback",
        type=int,
        default=20,
        help="Lookback window for regime calculation (days)",
    )
    parser.add_argument(
        "--daily-csv",
        default=None,
        help="Optional path to write daily metrics CSV",
    )
    parser.add_argument(
        "--panel-cache-dir",
        default="cache/panel",
        help="Directory for persisted panel caches (default: cache/panel)",
    )
    parser.add_argument(
        "--output",
        default="results/regime_adaptive_backtest.json",
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
    panel_cache_dir = (
        Path(args.panel_cache_dir).expanduser() if args.panel_cache_dir else None
    )

    run_regime_adaptive_backtest(
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
        rebalance_freq=args.rebalance_freq,
        enable_regime_detection=args.enable_regime_detection,
        regime_lookback=args.regime_lookback,
        daily_metrics_path=daily_metrics_path,
        panel_cache_dir=panel_cache_dir,
        optimization_target_top_k=args.target_top_k,
        min_position_weight=args.min_position_weight,
        turnover_limit=args.turnover_limit,
        cost_penalty=args.cost_penalty,
        candidate_multiplier=args.candidate_multiplier,
        min_alpha=args.min_alpha,
    )


if __name__ == "__main__":
    main()
