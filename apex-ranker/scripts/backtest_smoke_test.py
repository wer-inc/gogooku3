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
from dataclasses import asdict
from datetime import date as Date
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch
from apex_ranker.backtest import (
    CostCalculator,
    Portfolio,
    Trade,
    normalise_frequency,
    should_rebalance,
)
from apex_ranker.data import (
    FeatureSelector,
    add_cross_sectional_zscores,
    build_panel_cache,
)
from apex_ranker.models import APEXRankerV0
from apex_ranker.utils import load_config

DATE_EPOCH = Date(1970, 1, 1)


def ensure_date(value: Date | datetime | np.datetime64 | str) -> Date:
    """Convert various date-like objects to ``datetime.date``."""
    if isinstance(value, Date):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, np.datetime64):
        days = int(value.astype("datetime64[D]").astype("int64"))
        return DATE_EPOCH + timedelta(days=days)
    if isinstance(value, str):
        return datetime.strptime(value, "%Y-%m-%d").date()
    raise TypeError(f"Unsupported date type: {type(value)}")


def date_to_int(value: Date | datetime | np.datetime64 | str) -> int:
    """Convert a date-like value to integer days since epoch."""
    normalized = ensure_date(value)
    return (normalized - DATE_EPOCH).days


def int_to_date(value: int) -> Date:
    """Convert integer days since epoch back to ``datetime.date``."""
    return DATE_EPOCH + timedelta(days=int(value))


def resolve_device(device_str: str) -> torch.device:
    """Resolve device string to ``torch.device``."""
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_str)


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


def load_model_checkpoint(
    model_path: Path,
    config: dict,
    device: torch.device,
    n_features: int,
) -> APEXRankerV0:
    """Instantiate ``APEXRankerV0`` and load weights."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model_cfg = config["model"]
    horizons = config["train"]["horizons"]

    model = APEXRankerV0(
        in_features=n_features,
        horizons=horizons,
        d_model=model_cfg["d_model"],
        depth=model_cfg["depth"],
        patch_len=model_cfg["patch_len"],
        stride=model_cfg["stride"],
        n_heads=model_cfg["n_heads"],
        dropout=model_cfg.get("dropout", 0.1),
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


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


def load_dataset_for_backtest(
    data_path: Path,
    start_date: str | None,
    end_date: str | None,
    feature_cols: list[str] | None,
    lookback: int,
) -> pl.DataFrame:
    """Load parquet dataset with required columns for backtesting."""
    print(f"[Backtest] Loading dataset: {data_path}")

    required_cols: set[str] = {
        "Date",
        "Code",
        "Close",
        "Volume",
        "TurnoverValue",
        "returns_1d",
        "returns_5d",
        "returns_10d",
        "returns_20d",
    }
    if feature_cols:
        required_cols.update(feature_cols)

    frame = pl.read_parquet(data_path, columns=list(required_cols))
    frame = frame.with_columns(
        pl.col("Date").cast(pl.Date),
        pl.col("Code").cast(pl.Utf8),
    )

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else None
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else None
    buffer_start = (
        start_dt - timedelta(days=lookback * 2)
        if start_dt is not None and lookback > 0
        else start_dt
    )

    if buffer_start is not None:
        frame = frame.filter(pl.col("Date") >= buffer_start)
    if end_dt is not None:
        frame = frame.filter(pl.col("Date") <= end_dt)

    frame = frame.sort(["Date", "Code"])

    print(f"[Backtest] Loaded {len(frame):,} rows")
    print(
        "[Backtest] Date span:",
        frame["Date"].min(),
        "→",
        frame["Date"].max(),
    )
    print(f"[Backtest] Unique stocks: {frame['Code'].n_unique()}")

    return frame


class BacktestInferenceEngine:
    """Wrap Phase 2 inference utilities for backtest usage."""

    def __init__(
        self,
        model_path: Path,
        config: dict,
        frame: pl.DataFrame,
        feature_cols: list[str],
        *,
        device: str = "auto",
    ):
        self.config = config
        self.device = resolve_device(device)
        self.date_col = config["data"]["date_column"]
        self.code_col = config["data"]["code_column"]
        self.lookback = config["data"]["lookback"]
        self.feature_cols = list(feature_cols)

        self.model = load_model_checkpoint(
            model_path=model_path,
            config=config,
            device=self.device,
            n_features=len(self.feature_cols),
        )

        clip_sigma = config.get("normalization", {}).get("clip_sigma", 5.0)
        feature_frame = frame.select([self.date_col, self.code_col] + self.feature_cols)
        feature_frame = add_cross_sectional_zscores(
            feature_frame,
            columns=self.feature_cols,
            date_col=self.date_col,
            clip_sigma=clip_sigma,
        )

        self.z_features = [f"{col}_cs_z" for col in self.feature_cols]
        self.cache = build_panel_cache(
            feature_frame,
            feature_cols=self.z_features,
            target_cols=[],
            mask_cols=[],
            date_col=self.date_col,
            code_col=self.code_col,
            lookback=self.lookback,
            min_stocks_per_day=0,
        )
        self.horizons = set(config["train"]["horizons"])

    def available_dates(self) -> set[Date]:
        """Return set of dates for which inference can be generated."""
        return {int_to_date(date_int) for date_int in self.cache.date_to_codes.keys()}

    def _tensor_for_date(
        self,
        target_date: Date,
    ) -> tuple[torch.Tensor | None, list[str]]:
        date_int = date_to_int(target_date)
        codes = self.cache.date_to_codes.get(date_int)
        if not codes:
            return None, []

        feature_windows: list[np.ndarray] = []
        valid_codes: list[str] = []

        for code in codes:
            payload = self.cache.codes.get(code)
            if payload is None:
                continue
            dates = payload["dates"]
            idx = np.searchsorted(dates, date_int)
            if idx == len(dates) or dates[idx] != date_int:
                continue
            start = idx - self.lookback + 1
            if start < 0:
                continue
            window = payload["features"][start : idx + 1]
            if window.shape[0] != self.lookback:
                continue
            feature_windows.append(window)
            valid_codes.append(code)

        if not feature_windows:
            return None, []

        features = np.stack(feature_windows, axis=0).astype(np.float32, copy=False)
        tensor = torch.from_numpy(features)
        return tensor, valid_codes

    def predict(
        self,
        target_date: Date,
        horizon: int,
        top_k: int,
    ) -> pl.DataFrame:
        """Generate ranked predictions for a specific date."""
        if horizon not in self.horizons:
            raise ValueError(
                f"Horizon {horizon}d not available. "
                f"Configured horizons: {sorted(self.horizons)}"
            )

        tensor, codes = self._tensor_for_date(target_date)
        if tensor is None or not codes:
            return pl.DataFrame(
                {"Date": [], "Rank": [], "Code": [], "Score": [], "Horizon": []}
            )

        tensor = tensor.to(self.device)
        with torch.no_grad():
            output = self.model(tensor)

        if horizon not in output:
            available = list(output.keys())
            raise ValueError(
                f"Horizon {horizon}d missing from model output. "
                f"Available horizons: {available}"
            )

        scores = output[horizon].detach().cpu().numpy()
        ranked_idx = np.argsort(scores)[::-1]
        if top_k is not None:
            ranked_idx = ranked_idx[:top_k]

        ranked_codes = [codes[i] for i in ranked_idx]
        ranked_scores = [float(scores[i]) for i in ranked_idx]
        ranks = list(range(1, len(ranked_codes) + 1))

        return pl.DataFrame(
            {
                "Date": [str(target_date)] * len(ranks),
                "Rank": ranks,
                "Code": ranked_codes,
                "Score": ranked_scores,
                "Horizon": [f"{horizon}d"] * len(ranks),
            }
        )


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

    frame = load_dataset_for_backtest(
        data_path=data_path,
        start_date=start_date,
        end_date=end_date,
        feature_cols=feature_cols,
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

    if model_path is not None and not use_mock:
        inference_engine = BacktestInferenceEngine(
            model_path=model_path,
            config=config,
            frame=frame,
            feature_cols=feature_cols or [],
            device=device,
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

    print(f"[Backtest] Initial capital: ¥{initial_capital:,.0f}")
    print(f"[Backtest] Top-K allocation: {top_k}")
    print(f"[Backtest] Horizon: {horizon}d")

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
                        top_k=top_k * 3,
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
                            filtered = filtered.head(top_k)
                            predictions = {
                                row["Code"]: row["Score"]
                                for row in filtered.iter_rows(named=True)
                            }
                            prediction_source = "model"
            else:
                predictions = generate_mock_predictions(current_frame, top_k)
                prediction_source = "mock"

            if predictions:
                num_positions = len(predictions)
                target_weights = dict.fromkeys(predictions.keys(), 1.0 / num_positions)

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
            "top_k": top_k,
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
        flattened_history = [
            {k: v for k, v in record.items() if k != "positions"}
            for record in history_records
        ]
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
    )


if __name__ == "__main__":
    main()
