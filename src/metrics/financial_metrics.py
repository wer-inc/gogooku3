"""
Financial Evaluation Metrics for Stock Prediction
日次クロスセクション評価による金融特化メトリクス
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from scipy import stats
from scipy.stats import spearmanr
import warnings

logger = logging.getLogger(__name__)


class FinancialMetrics:
    """
    金融時系列予測用の日次評価メトリクス
    
    バッチ vs 日次評価の正しい切り分け:
    - 学習損失: バッチ単位（勾配計算用）
    - 評価指標: 日次クロスセクション単位（実運用と整合）
    
    主要メトリクス:
    - IC (Information Coefficient): ピアソン相関
    - RankIC: スピアマン順位相関（外れ値にロバスト）
    - Decile Analysis: 予測値による十分位ポートフォリオ分析
    """

    def __init__(
        self,
        date_column: str = 'date',
        code_column: str = 'code',
        min_stocks_per_day: int = 20,
        decile_count: int = 10,
        return_columns: Optional[List[str]] = None
    ):
        """
        Args:
            date_column: 日付カラム名
            code_column: 銘柄コードカラム名  
            min_stocks_per_day: 日次計算の最小銘柄数
            decile_count: Decile分析の分位数（通常10）
            return_columns: リターンカラム名リスト
        """
        self.date_column = date_column
        self.code_column = code_column
        self.min_stocks_per_day = min_stocks_per_day
        self.decile_count = decile_count
        self.return_columns = return_columns or ['return_1d', 'return_5d', 'return_20d']

    def compute_information_coefficient(
        self,
        predictions: Union[np.ndarray, torch.Tensor, pd.Series],
        targets: Union[np.ndarray, torch.Tensor, pd.Series],
        dates: Union[np.ndarray, pd.Series],
        return_daily: bool = False
    ) -> Union[float, Dict[str, Any]]:
        """
        Information Coefficient (IC) を日次で計算
        
        IC = Pearson correlation between prediction and actual return
        
        Args:
            predictions: 予測値
            targets: 実際のリターン
            dates: 日付
            return_daily: True=日次結果も返す
            
        Returns:
            平均IC（return_daily=Trueなら詳細結果も含む）
        """
        # データ型統一
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        if isinstance(dates, torch.Tensor):
            dates = dates.detach().cpu().numpy()
            
        predictions = np.asarray(predictions).flatten()
        targets = np.asarray(targets).flatten()
        dates = pd.to_datetime(dates)
        
        if len(predictions) != len(targets) or len(predictions) != len(dates):
            raise ValueError("Predictions, targets, and dates must have same length")

        # 日次IC計算
        daily_ics = []
        daily_results = []
        unique_dates = sorted(pd.Series(dates).unique())
        
        valid_days = 0
        
        for date in unique_dates:
            mask = pd.Series(dates) == date
            day_pred = predictions[mask]
            day_target = targets[mask]
            
            # 有効データ数チェック
            if len(day_pred) < self.min_stocks_per_day:
                continue
                
            # NaN除去
            valid_mask = ~(np.isnan(day_pred) | np.isnan(day_target))
            if valid_mask.sum() < self.min_stocks_per_day:
                continue
                
            day_pred_clean = day_pred[valid_mask]
            day_target_clean = day_target[valid_mask]
            
            # ピアソン相関計算
            try:
                ic, p_value = stats.pearsonr(day_pred_clean, day_target_clean)
                if np.isnan(ic):
                    continue
                    
                daily_ics.append(ic)
                valid_days += 1
                
                if return_daily:
                    daily_results.append({
                        'date': date,
                        'ic': ic,
                        'p_value': p_value,
                        'n_stocks': len(day_pred_clean)
                    })
                    
            except Exception as e:
                logger.warning(f"IC calculation failed for {date.date()}: {e}")
                continue

        if not daily_ics:
            logger.warning("No valid days for IC calculation")
            return 0.0 if not return_daily else {'mean_ic': 0.0, 'daily_results': []}

        mean_ic = np.mean(daily_ics)
        
        if not return_daily:
            return float(mean_ic)
            
        return {
            'mean_ic': float(mean_ic),
            'std_ic': float(np.std(daily_ics)),
            'ic_ir': float(mean_ic / (np.std(daily_ics) + 1e-8)),  # IC Information Ratio
            'valid_days': valid_days,
            'total_days': len(unique_dates),
            'daily_results': daily_results
        }

    def compute_rank_ic(
        self,
        predictions: Union[np.ndarray, torch.Tensor, pd.Series],
        targets: Union[np.ndarray, torch.Tensor, pd.Series], 
        dates: Union[np.ndarray, pd.Series],
        return_daily: bool = False
    ) -> Union[float, Dict[str, Any]]:
        """
        Rank IC (Spearman correlation) を日次で計算
        
        RankIC = Spearman rank correlation (外れ値にロバスト)
        
        Args:
            predictions: 予測値
            targets: 実際のリターン
            dates: 日付
            return_daily: True=日次結果も返す
            
        Returns:
            平均RankIC（return_daily=Trueなら詳細結果も含む）
        """
        # データ型統一
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        if isinstance(dates, torch.Tensor):
            dates = dates.detach().cpu().numpy()
            
        predictions = np.asarray(predictions).flatten()
        targets = np.asarray(targets).flatten()
        dates = pd.to_datetime(dates)

        # 日次RankIC計算
        daily_rank_ics = []
        daily_results = []
        unique_dates = sorted(pd.Series(dates).unique())
        
        valid_days = 0
        
        for date in unique_dates:
            mask = pd.Series(dates) == date
            day_pred = predictions[mask]
            day_target = targets[mask]
            
            if len(day_pred) < self.min_stocks_per_day:
                continue
                
            # NaN除去
            valid_mask = ~(np.isnan(day_pred) | np.isnan(day_target))
            if valid_mask.sum() < self.min_stocks_per_day:
                continue
                
            day_pred_clean = day_pred[valid_mask]
            day_target_clean = day_target[valid_mask]
            
            # スピアマン順位相関計算
            try:
                rank_ic, p_value = spearmanr(day_pred_clean, day_target_clean)
                if np.isnan(rank_ic):
                    continue
                    
                daily_rank_ics.append(rank_ic)
                valid_days += 1
                
                if return_daily:
                    daily_results.append({
                        'date': date,
                        'rank_ic': rank_ic,
                        'p_value': p_value,
                        'n_stocks': len(day_pred_clean)
                    })
                    
            except Exception as e:
                logger.warning(f"RankIC calculation failed for {date.date()}: {e}")
                continue

        if not daily_rank_ics:
            logger.warning("No valid days for RankIC calculation")
            return 0.0 if not return_daily else {'mean_rank_ic': 0.0, 'daily_results': []}

        mean_rank_ic = np.mean(daily_rank_ics)
        
        if not return_daily:
            return float(mean_rank_ic)
            
        return {
            'mean_rank_ic': float(mean_rank_ic),
            'std_rank_ic': float(np.std(daily_rank_ics)),
            'rank_ic_ir': float(mean_rank_ic / (np.std(daily_rank_ics) + 1e-8)),
            'valid_days': valid_days,
            'total_days': len(unique_dates),
            'daily_results': daily_results
        }

    def compute_decile_analysis(
        self,
        predictions: Union[np.ndarray, torch.Tensor, pd.Series],
        returns: Union[np.ndarray, torch.Tensor, pd.Series],
        dates: Union[np.ndarray, pd.Series],
        return_daily: bool = False
    ) -> Dict[str, Any]:
        """
        Decile分析を日次で計算
        
        予測値でポートフォリオを10分位に分割し、各分位の平均リターンを計算
        Long-Short spread (top decile - bottom decile) が主要指標
        
        Args:
            predictions: 予測値
            returns: 実際のリターン
            dates: 日付
            return_daily: True=日次結果も返す
            
        Returns:
            Decile分析結果
        """
        # データ型統一
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(returns, torch.Tensor):
            returns = returns.detach().cpu().numpy()
        if isinstance(dates, torch.Tensor):
            dates = dates.detach().cpu().numpy()
            
        predictions = np.asarray(predictions).flatten()
        returns = np.asarray(returns).flatten()
        dates = pd.to_datetime(dates)

        daily_decile_returns = []
        daily_results = []
        unique_dates = sorted(pd.Series(dates).unique())
        
        valid_days = 0
        
        for date in unique_dates:
            mask = pd.Series(dates) == date
            day_pred = predictions[mask]
            day_ret = returns[mask]
            
            if len(day_pred) < self.min_stocks_per_day:
                continue
                
            # NaN除去
            valid_mask = ~(np.isnan(day_pred) | np.isnan(day_ret))
            if valid_mask.sum() < self.min_stocks_per_day:
                continue
                
            day_pred_clean = day_pred[valid_mask]
            day_ret_clean = day_ret[valid_mask]
            
            try:
                # 予測値で分位点を計算（重複値考慮）
                decile_bounds = np.percentile(
                    day_pred_clean, 
                    np.linspace(0, 100, self.decile_count + 1)
                )
                
                decile_returns = []
                decile_counts = []
                
                for i in range(self.decile_count):
                    if i == 0:
                        mask_decile = day_pred_clean <= decile_bounds[i + 1]
                    elif i == self.decile_count - 1:
                        mask_decile = day_pred_clean >= decile_bounds[i]
                    else:
                        mask_decile = (day_pred_clean > decile_bounds[i]) & (day_pred_clean <= decile_bounds[i + 1])
                    
                    if mask_decile.sum() > 0:
                        decile_ret = day_ret_clean[mask_decile].mean()
                        decile_returns.append(decile_ret)
                        decile_counts.append(mask_decile.sum())
                    else:
                        decile_returns.append(np.nan)
                        decile_counts.append(0)
                
                # 有効なdecileが十分にあるかチェック
                valid_deciles = ~np.isnan(decile_returns)
                if valid_deciles.sum() < self.decile_count // 2:
                    continue
                
                daily_decile_returns.append(decile_returns)
                valid_days += 1
                
                if return_daily:
                    # Long-Short spread計算
                    top_decile = decile_returns[-1] if not np.isnan(decile_returns[-1]) else 0.0
                    bottom_decile = decile_returns[0] if not np.isnan(decile_returns[0]) else 0.0
                    long_short = top_decile - bottom_decile
                    
                    daily_results.append({
                        'date': date,
                        'decile_returns': decile_returns,
                        'decile_counts': decile_counts,
                        'long_short_spread': long_short,
                        'n_stocks': len(day_pred_clean)
                    })
                    
            except Exception as e:
                logger.warning(f"Decile analysis failed for {date.date()}: {e}")
                continue

        if not daily_decile_returns:
            logger.warning("No valid days for decile analysis")
            return {
                'mean_decile_returns': [0.0] * self.decile_count,
                'long_short_spread': 0.0,
                'daily_results': [] if return_daily else None
            }

        # 日次結果を平均
        mean_decile_returns = np.nanmean(daily_decile_returns, axis=0).tolist()
        
        # Long-Short spread
        top_returns = [returns[-1] for returns in daily_decile_returns if not np.isnan(returns[-1])]
        bottom_returns = [returns[0] for returns in daily_decile_returns if not np.isnan(returns[0])]
        
        long_short_spread = 0.0
        if top_returns and bottom_returns:
            long_short_spread = np.mean(top_returns) - np.mean(bottom_returns)

        result = {
            'mean_decile_returns': mean_decile_returns,
            'long_short_spread': float(long_short_spread),
            'valid_days': valid_days,
            'total_days': len(unique_dates),
            'decile_count': self.decile_count
        }
        
        if return_daily:
            result['daily_results'] = daily_results
            
        return result

    def compute_all_metrics(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],  
        dates: Union[np.ndarray, pd.Series],
        return_daily: bool = False,
        prefix: str = ""
    ) -> Dict[str, Any]:
        """
        すべての金融メトリクスを一括計算
        
        Args:
            predictions: 予測値
            targets: 実際のリターン
            dates: 日付
            return_daily: True=日次結果も返す
            prefix: メトリクス名のプレフィックス
            
        Returns:
            すべてのメトリクス結果
        """
        results = {}
        p = f"{prefix}_" if prefix else ""
        
        # IC計算
        ic_result = self.compute_information_coefficient(
            predictions, targets, dates, return_daily=return_daily
        )
        if isinstance(ic_result, dict):
            results.update({f"{p}ic_{k}": v for k, v in ic_result.items()})
        else:
            results[f"{p}ic"] = ic_result

        # RankIC計算
        rank_ic_result = self.compute_rank_ic(
            predictions, targets, dates, return_daily=return_daily
        )
        if isinstance(rank_ic_result, dict):
            results.update({f"{p}rank_ic_{k}": v for k, v in rank_ic_result.items()})
        else:
            results[f"{p}rank_ic"] = rank_ic_result

        # Decile分析
        decile_result = self.compute_decile_analysis(
            predictions, targets, dates, return_daily=return_daily
        )
        results.update({f"{p}decile_{k}": v for k, v in decile_result.items()})

        return results

    def compute_multi_horizon_metrics(
        self,
        predictions_dict: Dict[str, Union[np.ndarray, torch.Tensor]],
        targets_dict: Dict[str, Union[np.ndarray, torch.Tensor]],
        dates: Union[np.ndarray, pd.Series],
        return_daily: bool = False
    ) -> Dict[str, Any]:
        """
        マルチホライズン予測のメトリクスを計算
        
        Args:
            predictions_dict: {horizon: predictions}
            targets_dict: {horizon: targets}
            dates: 日付
            return_daily: True=日次結果も返す
            
        Returns:
            ホライズン別メトリクス結果
        """
        all_results = {}
        
        for horizon in predictions_dict:
            if horizon not in targets_dict:
                logger.warning(f"No targets for horizon {horizon}")
                continue
                
            horizon_results = self.compute_all_metrics(
                predictions_dict[horizon],
                targets_dict[horizon], 
                dates,
                return_daily=return_daily,
                prefix=f"h{horizon}"
            )
            
            all_results.update(horizon_results)
            
        return all_results


def compute_sharpe_ratio(returns: Union[np.ndarray, torch.Tensor], risk_free_rate: float = 0.0) -> float:
    """
    Sharpe比を計算（安定化版）
    
    Args:
        returns: リターン配列
        risk_free_rate: リスクフリーレート（年率）
        
    Returns:
        Sharpe比
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.detach().cpu().numpy()
    
    returns = np.asarray(returns)
    
    if len(returns) == 0:
        return 0.0
        
    excess_returns = returns - risk_free_rate / 252  # 日次換算
    
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns)
    
    if std_return < 1e-8:
        return 0.0
        
    return float(mean_return / std_return * np.sqrt(252))  # 年率化


def compute_maximum_drawdown(returns: Union[np.ndarray, torch.Tensor]) -> float:
    """
    最大ドローダウンを計算
    
    Args:
        returns: リターン配列
        
    Returns:
        最大ドローダウン（負の値）
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.detach().cpu().numpy()
    
    returns = np.asarray(returns)
    
    if len(returns) == 0:
        return 0.0
        
    # 累積リターン
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    
    # ドローダウン
    drawdown = (cumulative - running_max) / running_max
    
    return float(np.min(drawdown))


def compute_hit_rate(predictions: Union[np.ndarray, torch.Tensor], targets: Union[np.ndarray, torch.Tensor]) -> float:
    """
    ヒットレート（方向性の一致率）を計算
    
    Args:
        predictions: 予測値
        targets: 実際の値
        
    Returns:
        ヒットレート（0-1）
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    
    # 符号の一致を計算
    pred_sign = np.sign(predictions)
    target_sign = np.sign(targets)
    
    matches = (pred_sign == target_sign)
    
    return float(np.mean(matches))