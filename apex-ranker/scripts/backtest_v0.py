#!/usr/bin/env python3
"""
APEX-Ranker v0 Backtest Script

Simple backtest framework to validate ranking model performance with portfolio metrics.

Usage:
    # Backtest enhanced model
    python apex-ranker/scripts/backtest_v0.py \
        --model models/apex_ranker_v0_enhanced.pt \
        --config apex-ranker/configs/v0_base.yaml \
        --output results/backtest_enhanced.json

    # Compare baseline vs enhanced
    python apex-ranker/scripts/backtest_v0.py \
        --model models/apex_ranker_v0_early_stopping.pt \
        --config apex-ranker/configs/v0_base.yaml \
        --output results/backtest_baseline.json
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
import yaml
from tqdm import tqdm

# Add apex-ranker to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from apex_ranker.data import (
    FeatureSelector,
    add_cross_sectional_zscores,
    build_panel_cache,
    DayPanelDataset,
)
from apex_ranker.models import APEXRankerV0

warnings.filterwarnings('ignore')


class SimplePortfolioBacktest:
    """
    Simple long-only equal-weight portfolio backtest.

    Strategy:
    - Each day, rank stocks by model predictions
    - Hold top-K stocks with equal weight
    - Rebalance daily
    - Track returns, Sharpe, turnover, drawdown
    """

    def __init__(
        self,
        top_k: int = 50,
        horizon: int = 20,  # Use 20d predictions
        initial_capital: float = 100_000_000,  # 100M JPY
    ):
        self.top_k = top_k
        self.horizon = horizon
        self.initial_capital = initial_capital

        # Tracking
        self.history = {
            'dates': [],
            'portfolio_value': [],
            'positions': [],
            'returns': [],
            'turnover': [],
        }

    def run(
        self,
        predictions: Dict[int, np.ndarray],  # date_int -> [stock_scores]
        actuals: Dict[int, np.ndarray],      # date_int -> [stock_returns]
        codes: Dict[int, List[str]],          # date_int -> [stock_codes]
        *,
        collect_details: bool = False,
    ) -> Tuple[Dict, List[Dict[str, object]]]:
        """
        Run backtest simulation.

        Args:
            predictions: Date -> model predictions for each stock
            actuals: Date -> actual forward returns for each stock
            codes: Date -> stock codes for each position
            collect_details: Whether to collect per-day top-K detail rows

        Returns:
            Tuple of backtest results dictionary and optional detail rows
        """
        dates = sorted(predictions.keys())
        portfolio_value = self.initial_capital
        current_positions = set()  # Set of stock codes
        detail_rows: List[Dict[str, object]] = []

        for step_idx, date in enumerate(tqdm(dates, desc="Backtesting")):
            pred_scores = predictions[date]
            actual_returns = actuals[date]
            stock_codes = codes[date]

            if not (
                len(pred_scores) == len(actual_returns) == len(stock_codes)
            ):
                raise ValueError(
                    f"Length mismatch on date {date}: "
                    f"pred={len(pred_scores)}, actual={len(actual_returns)}, codes={len(stock_codes)}"
                )

            if step_idx == 0:
                print(
                    "[DEBUG] First day stats – "
                    f"pred_mean={np.mean(pred_scores):.6f}, pred_std={np.std(pred_scores):.6f}, "
                    f"actual_mean={np.mean(actual_returns):.6f}, actual_std={np.std(actual_returns):.6f}"
                )
            if np.std(actual_returns) < 1e-8:
                print(
                    f"[WARN] Actual returns nearly constant on date {date}; "
                    "check target mapping."
                )

            # Rank stocks by predicted score (higher = better)
            ranked_indices = np.argsort(-pred_scores)[: self.top_k]
            new_positions = set(stock_codes[i] for i in ranked_indices)

            # Compute turnover (fraction of portfolio changed)
            if current_positions:
                n_changed = len(current_positions.symmetric_difference(new_positions))
                turnover = n_changed / self.top_k
            else:
                turnover = 1.0  # First day

            # Compute portfolio return (equal weight top-K)
            portfolio_return = np.mean(actual_returns[ranked_indices])

            # Update portfolio value
            portfolio_value *= (1 + portfolio_return)

            # Record
            self.history['dates'].append(date)
            self.history['portfolio_value'].append(portfolio_value)
            self.history['positions'].append(list(new_positions))
            self.history['returns'].append(float(portfolio_return))
            self.history['turnover'].append(float(turnover))

            if collect_details:
                weight = 1.0 / self.top_k if self.top_k > 0 else 0.0
                for rank, idx in enumerate(ranked_indices, start=1):
                    detail_rows.append(
                        {
                            "date_int": int(date),
                            "rank": int(rank),
                            "code": stock_codes[idx],
                            "pred_score": float(pred_scores[idx]),
                            "actual_return": float(actual_returns[idx]),
                            "weight": float(weight),
                        }
                    )

            # Update current positions
            current_positions = new_positions

        # Compute summary statistics
        returns = np.array(self.history['returns'], dtype=np.float64)

        results = {
            'total_return': (portfolio_value / self.initial_capital - 1) * 100,  # %
            'annualized_return': self._annualized_return(returns),
            'sharpe_ratio': self._sharpe_ratio(returns),
            'max_drawdown': self._max_drawdown(self.history['portfolio_value']),
            'avg_turnover': np.mean(self.history['turnover']) if len(self.history['turnover']) else 0.0,
            'n_days': len(dates),
            'final_value': portfolio_value,
            'history': self.history,
        }
        return results, detail_rows

    @staticmethod
    def _annualized_return(returns: np.ndarray, periods_per_year: int = 252) -> float:
        """Compute annualized return from daily returns."""
        cumulative = np.prod(1 + returns)
        n_periods = len(returns)
        annualized = (cumulative ** (periods_per_year / n_periods) - 1) * 100
        return annualized

    @staticmethod
    def _sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
        """Compute Sharpe ratio (assuming 0 risk-free rate)."""
        if len(returns) == 0 or np.std(returns) < 1e-10:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year)

    @staticmethod
    def _max_drawdown(portfolio_values: List[float]) -> float:
        """Compute maximum drawdown in percentage."""
        values = np.array(portfolio_values)
        peaks = np.maximum.accumulate(values)
        drawdowns = (values - peaks) / peaks * 100
        return np.min(drawdowns)


def load_model_and_config(model_path: str, config_path: str) -> Tuple[APEXRankerV0, dict]:
    """Load APEX-Ranker model and configuration."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')

    # Extract model parameters from checkpoint or config
    model_state = checkpoint.get('model_state_dict', checkpoint)

    # Infer model architecture from checkpoint
    # Use attention in_proj_weight to get d_model: [3*d_model, d_model]
    attn_in_proj_key = 'encoder.blocks.0.attn.in_proj_weight'
    if attn_in_proj_key in model_state:
        # in_proj_weight shape is [3*d_model, d_model] for Q, K, V
        d_model_from_ckpt = model_state[attn_in_proj_key].shape[1]
    else:
        # Fallback to config
        d_model_from_ckpt = config['model'].get('d_model', 256)

    # Get patch_len from patch_embed.conv
    patch_conv_key = 'encoder.patch_embed.conv.weight'
    if patch_conv_key in model_state:
        patch_len_from_ckpt = model_state[patch_conv_key].shape[2]
    else:
        patch_len_from_ckpt = config['model'].get('patch_len', 16)

    # Infer n_heads - it must divide d_model evenly
    # Common values: 4, 8, 16
    # Try to infer from config first, then find a divisor
    n_heads_config = config['model'].get('n_heads', 8)
    if d_model_from_ckpt % n_heads_config == 0:
        n_heads_from_ckpt = n_heads_config
    else:
        # Find a suitable n_heads that divides d_model
        for candidate in [16, 8, 4, 2, 1]:
            if d_model_from_ckpt % candidate == 0:
                n_heads_from_ckpt = candidate
                break
        else:
            n_heads_from_ckpt = 1  # Fallback

    # Infer number of input features from patch_embed.proj: [d_model, d_model]
    # The actual in_features is inferred from lookback * n_features
    # We'll use a simpler approach: count from the checkpoint metadata if available
    # For now, we need to know in_features - check encoder input projection

    # Since PatchTST uses lookback dimension, we need to infer from the data
    # The safest approach is to use the checkpoint's saved architecture
    # Look for a key that contains input size information

    # Alternative: Look at the projection layer
    proj_key = 'encoder.patch_embed.proj.weight'
    if proj_key in model_state:
        # proj: [d_model, in_features_per_patch]
        # in_features_per_patch might equal d_model if there's a transformation
        in_features_per_patch = model_state[proj_key].shape[1]
    else:
        in_features_per_patch = d_model_from_ckpt

    # For PatchTST, in_features is the number of variates (features)
    # We need to derive this from the actual dataset, not the checkpoint
    # The checkpoint only tells us d_model, not the original feature count

    # Load the dataset to get the actual feature count
    data_cfg = config['data']

    # Use FeatureSelector to get feature count
    feature_selector = FeatureSelector(data_cfg['feature_groups_config'])
    groups = list(data_cfg.get('feature_groups', []))
    use_plus30 = data_cfg.get('use_plus30', True)
    if use_plus30:
        groups = groups + ["plus30"]

    selection = feature_selector.select(
        groups=groups,
        optional_groups=data_cfg.get('optional_groups', []),
        metadata_path=data_cfg.get('metadata_path'),
    )

    in_features = len(selection.features)

    # Create model with architecture from checkpoint
    horizons = config['train']['horizons']

    # Infer depth from checkpoint if possible
    depth_from_ckpt = None
    block_prefix = "encoder.blocks."
    for key in model_state.keys():
        if key.startswith(block_prefix):
            try:
                block_idx = int(key.split(".")[2])
                if depth_from_ckpt is None or block_idx + 1 > depth_from_ckpt:
                    depth_from_ckpt = block_idx + 1
            except (IndexError, ValueError):
                continue

    model = APEXRankerV0(
        in_features=in_features,
        horizons=horizons,
        d_model=d_model_from_ckpt,
        depth=depth_from_ckpt or config['model'].get('depth', 4),
        patch_len=patch_len_from_ckpt,
        stride=config['model'].get('stride', 8),
        n_heads=n_heads_from_ckpt,
        dropout=config['model'].get('dropout', 0.2),
    )

    # Load weights
    model.load_state_dict(model_state)
    model.eval()

    return model, config


def prepare_validation_data(config: dict) -> Tuple[pl.DataFrame, List[str], List[str], List[str]]:
    """Load and prepare validation dataset."""
    data_cfg = config['data']

    data_path = data_cfg['parquet_path']
    df = pl.read_parquet(data_path)
    print(f"[INFO] Loaded dataset: {len(df):,} rows")

    feature_selector = FeatureSelector(data_cfg['feature_groups_config'])
    groups = list(data_cfg.get('feature_groups', []))
    optional_groups = list(data_cfg.get('optional_groups', []))

    if data_cfg.get('use_plus30', True):
        groups = groups + ["plus30"]

    selection = feature_selector.select(
        groups=groups,
        optional_groups=optional_groups,
        metadata_path=data_cfg.get('metadata_path'),
    )

    print(f"[INFO] Selected {len(selection.features)} features, {len(selection.masks)} masks")

    # Apply cross-sectional z-score normalization to mirror training pipeline
    df = add_cross_sectional_zscores(
        df,
        columns=selection.features,
        date_col=data_cfg['date_column'],
        clip_sigma=config.get('normalization', {}).get('clip_sigma', 5.0),
    )
    feature_cols = [f"{col}_cs_z" for col in selection.features]
    mask_cols = selection.masks

    target_map = data_cfg['target_columns']
    horizons = config['train']['horizons']

    def resolve_target(h: int) -> str:
        if isinstance(target_map, dict):
            if str(h) in target_map:
                return target_map[str(h)]
            if h in target_map:
                return target_map[h]
        raise KeyError(f"target column for horizon {h} is not defined in config")

    target_cols = [resolve_target(int(h)) for h in horizons]

    return df, feature_cols, target_cols, mask_cols


def generate_predictions(
    model: APEXRankerV0,
    dataset: DayPanelDataset,
    horizon: int,
    date_filter: Optional[List[int]] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, List[str]]]:
    """
    Generate predictions for validation dataset.

    Args:
        model: APEX-Ranker model
        dataset: DayPanelDataset
        horizon: Prediction horizon (1, 5, 10, or 20)
        date_filter: Optional list of date_ints to generate predictions for.
                    If None, generates predictions for all dates.
        device: Device to run on

    Returns:
        predictions: date_int -> predicted scores
        actuals: date_int -> actual returns
        codes: date_int -> stock codes
    """
    model = model.to(device)
    model.eval()

    predictions = {}
    actuals = {}
    codes_dict = {}

    horizon_idx = model.horizons.index(horizon)

    # Convert date_filter to set for O(1) lookup
    date_filter_set = set(date_filter) if date_filter is not None else None

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Generating predictions"):
            item = dataset[i]
            if item is None:
                continue

            date_int = item['date_int']
            if date_filter_set is not None and date_int not in date_filter_set:
                continue

            X = item['X'].to(device)  # [stocks, lookback, features]
            y = item['y']
            stock_codes = item['codes']

            # Predict
            outputs = model(X)  # {horizon: [stocks]}
            pred_scores = outputs[horizon].detach().cpu().numpy()  # [stocks]

            # Actual returns
            actual_returns = y[:, horizon_idx].detach().cpu().numpy()

            predictions[date_int] = pred_scores
            actuals[date_int] = actual_returns
            codes_dict[date_int] = stock_codes

    return predictions, actuals, codes_dict


def main():
    parser = argparse.ArgumentParser(description="APEX-Ranker v0 Backtest")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path for results")
    parser.add_argument("--top-k", type=int, default=50, help="Number of stocks to hold (default: 50)")
    parser.add_argument("--horizon", type=int, default=20, help="Forecast horizon to use (default: 20)")
    parser.add_argument("--val-days", type=int, default=None, help="Override validation days (default: from config)")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--save-history", type=str, default=None, help="Optional path to save daily top-K selections (parquet)")

    args = parser.parse_args()

    print("="*80)
    print("APEX-Ranker v0 Backtest")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    print(f"Top-K: {args.top_k}")
    print(f"Horizon: {args.horizon}d")
    print()

    # Load model and config
    print("[1/5] Loading model and configuration...")
    model, config = load_model_and_config(args.model, args.config)
    print(f"      Model: {model.__class__.__name__}")
    print(f"      Horizons: {model.horizons}")
    print()

    # Prepare data
    print("[2/5] Loading validation dataset...")
    df, feature_cols, target_cols, mask_cols = prepare_validation_data(config)

    # Split validation data (last N days + lookback for panel cache)
    val_days = args.val_days or config['train'].get('val_days', 120)
    lookback = config['data']['lookback']
    date_col = config['data']['date_column']
    date_series = (
        df.select(pl.col(date_col).unique().sort())
        .to_series()
    )
    date_values = date_series.to_list()
    date_ints = np.asarray(date_series.to_numpy(), dtype='datetime64[D]').astype('int64')

    if val_days > len(date_values):
        val_days = len(date_values)

    val_start_idx = max(0, len(date_values) - val_days)
    data_start_idx = max(0, val_start_idx - lookback)

    val_start_date = date_values[val_start_idx]
    data_start_date = date_values[data_start_idx]

    val_df = df.filter(pl.col(date_col) >= data_start_date)
    print(f"      Validation period: {val_start_date} onwards ({val_days} days)")
    print(f"      Data loaded: {data_start_date} onwards (includes {lookback}d lookback)")
    print(f"      Validation samples: {len(val_df):,}")
    print()

    # Build panel cache
    print("[3/5] Building panel cache...")
    cache = build_panel_cache(
        val_df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        mask_cols=mask_cols,
        date_col=config['data']['date_column'],
        code_col=config['data']['code_column'],
        lookback=config['data']['lookback'],
        min_stocks_per_day=config['data']['min_stocks_per_day'],
    )

    # Filter cache to only validation period (not lookback period)
    # cache.date_ints contains all dates with valid panels
    valid_dates = sorted(cache.date_ints)
    val_start_int = int(date_ints[val_start_idx])
    val_dates_only = [d for d in valid_dates if d >= val_start_int]
    if not val_dates_only:
        take_n = min(len(valid_dates), val_days)
        val_dates_only = valid_dates[-take_n:]

    dataset = DayPanelDataset(
        cache,
        feature_cols=feature_cols,
        mask_cols=mask_cols,
        target_cols=target_cols,
    )
    print(f"      Panel days (all): {len(dataset)}")
    print(f"      Validation days: {len(val_dates_only)}")
    print()

    # Generate predictions
    print("[4/5] Generating predictions...")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    predictions, actuals, codes = generate_predictions(
        model, dataset, args.horizon, date_filter=val_dates_only, device=device
    )
    print(f"      Predictions generated for {len(predictions)} days")
    print()

    # Run backtest
    print("[5/5] Running backtest...")
    backtester = SimplePortfolioBacktest(
        top_k=args.top_k,
        horizon=args.horizon,
    )

    collect_details = bool(args.save_history)
    results, detail_rows = backtester.run(
        predictions,
        actuals,
        codes,
        collect_details=collect_details,
    )

    # Print results
    print()
    print("="*80)
    print("BACKTEST RESULTS")
    print("="*80)
    print(f"Total Return:        {results['total_return']:>10.2f}%")
    print(f"Annualized Return:   {results['annualized_return']:>10.2f}%")
    print(f"Sharpe Ratio:        {results['sharpe_ratio']:>10.2f}")
    print(f"Max Drawdown:        {results['max_drawdown']:>10.2f}%")
    print(f"Avg Daily Turnover:  {results['avg_turnover']:>10.2%}")
    print(f"Trading Days:        {results['n_days']:>10,}")
    print(f"Final Value:         {results['final_value']:>10,.0f} JPY")
    print("="*80)
    print()

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove history for JSON serialization (too large)
    def _to_builtin(value: object):
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        return value

    results_clean = {
        k: _to_builtin(v) for k, v in results.items() if k != 'history'
    }
    results_clean['model_path'] = args.model
    results_clean['config_path'] = args.config
    results_clean['top_k'] = args.top_k
    results_clean['horizon'] = args.horizon

    with open(output_path, 'w') as f:
        json.dump(results_clean, f, indent=2)

    print(f"[INFO] Results saved to: {output_path}")

    if args.save_history:
        if detail_rows:
            history_path = Path(args.save_history)
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_df = pl.DataFrame(detail_rows)
            history_df.write_parquet(history_path)
            print(f"[INFO] Saved top-{args.top_k} history to: {history_path}")
        else:
            print("[WARN] No detail rows were collected; history file not written.")
    print()


if __name__ == "__main__":
    main()
