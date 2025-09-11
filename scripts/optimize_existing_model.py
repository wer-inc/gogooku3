#!/usr/bin/env python3
"""
既存モデルの予測を最適化して一気に性能を上げる
"""

import torch
import numpy as np
import polars as pl
from pathlib import Path
import logging
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def optimize_predictions(predictions: np.ndarray, 
                        targets: np.ndarray,
                        invert_sign: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    予測を最適化してSharpe比を改善
    """
    # 1. 符号反転
    if invert_sign:
        predictions = -predictions
        logger.info("Applied sign inversion")
    
    # 2. 予測のクリッピング（外れ値を除去）
    pred_std = np.std(predictions)
    predictions = np.clip(predictions, -3*pred_std, 3*pred_std)
    
    # 3. 予測の正規化
    pred_mean = np.mean(predictions)
    pred_std = np.std(predictions) + 1e-8
    predictions_norm = (predictions - pred_mean) / pred_std
    
    # 4. ポートフォリオ最適化
    # 上位/下位のパーセンタイルのみ取引
    upper_threshold = np.percentile(predictions_norm, 80)
    lower_threshold = np.percentile(predictions_norm, 20)
    
    positions = np.zeros_like(predictions_norm)
    positions[predictions_norm > upper_threshold] = 1
    positions[predictions_norm < lower_threshold] = -1
    
    # 5. リスクパリティ調整
    # ポジションサイズを調整
    n_long = np.sum(positions > 0)
    n_short = np.sum(positions < 0)
    
    if n_long > 0:
        positions[positions > 0] = 0.5 / n_long
    if n_short > 0:
        positions[positions < 0] = -0.5 / n_short
    
    # 6. 結果の計算
    portfolio_returns = positions * targets
    
    # Sharpe比
    sharpe = np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-8)
    
    # その他のメトリクス
    metrics = {
        "sharpe": sharpe,
        "mean_return": np.mean(portfolio_returns),
        "volatility": np.std(portfolio_returns),
        "n_long": n_long,
        "n_short": n_short,
        "hit_rate": np.mean((positions * targets) > 0),
        "max_drawdown": calculate_max_drawdown(portfolio_returns)
    }
    
    return positions, metrics


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """最大ドローダウンを計算"""
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return np.min(drawdown)


def load_latest_checkpoint():
    """最新のチェックポイントを読み込み"""
    checkpoint_dir = Path("models/checkpoints")
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    
    if not checkpoints:
        logger.error("No checkpoints found")
        return None
        
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading checkpoint: {latest}")
    
    return torch.load(latest, map_location='cpu')


def main():
    """メイン実行"""
    logger.info("=== 既存モデルの最適化 ===")
    
    # ダミーデータで例を示す
    # 実際はモデルの予測を使用
    n_samples = 1000
    predictions = np.random.randn(n_samples) * 0.1
    targets = np.random.randn(n_samples) * 0.01  # 実際のリターンスケール
    
    # 最適化前
    raw_sharpe = np.mean(targets) / (np.std(targets) + 1e-8)
    logger.info(f"Raw Sharpe: {raw_sharpe:.4f}")
    
    # 最適化後
    positions, metrics = optimize_predictions(predictions, targets, invert_sign=True)
    
    logger.info("\n=== 最適化結果 ===")
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.4f}")
    
    # 改善率
    improvement = (metrics["sharpe"] - raw_sharpe) / abs(raw_sharpe) * 100
    logger.info(f"\nSharpe改善率: {improvement:.1f}%")
    
    # ポジション分布
    logger.info(f"\nポジション分布:")
    logger.info(f"ロング: {metrics['n_long']} ({metrics['n_long']/n_samples*100:.1f}%)")
    logger.info(f"ショート: {metrics['n_short']} ({metrics['n_short']/n_samples*100:.1f}%)")
    logger.info(f"ニュートラル: {n_samples - metrics['n_long'] - metrics['n_short']} ({(n_samples - metrics['n_long'] - metrics['n_short'])/n_samples*100:.1f}%)")


if __name__ == "__main__":
    main()
