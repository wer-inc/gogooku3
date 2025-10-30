#!/usr/bin/env python3
"""
強化されたポートフォリオ最適化
爆上げのための後処理機能を追加
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def daily_zscore_normalize(df: pd.DataFrame, pred_col: str, date_col: str) -> pd.DataFrame:
    """日次クロスセクショナルZスコア正規化"""
    logger.info("Applying daily Z-score normalization...")

    def zscore(x):
        if len(x) < 2:
            return x
        return (x - x.mean()) / (x.std() + 1e-8)

    df[f"{pred_col}_zscore"] = df.groupby(date_col)[pred_col].transform(zscore)
    return df


def neutralize_factors(df: pd.DataFrame, pred_col: str, date_col: str,
                      neutralize_sector: bool = False, neutralize_size: bool = False) -> pd.DataFrame:
    """セクター・サイズ中立化"""
    logger.info(f"Neutralizing factors - Sector: {neutralize_sector}, Size: {neutralize_size}")

    # セクター列の準備
    sector_cols = [col for col in df.columns if col.startswith('sector33_')]

    # サイズプロキシ（対数時価総額など）
    size_proxy = None
    for col in ['log_mktcap', 'mktcap', 'market_cap']:
        if col in df.columns:
            if col == 'mktcap' or col == 'market_cap':
                df['log_mktcap'] = np.log(df[col] + 1)
                size_proxy = 'log_mktcap'
            else:
                size_proxy = col
            break

    residuals = []
    for date, group in df.groupby(date_col):
        if len(group) < 10:  # サンプル数が少ない場合はスキップ
            residuals.append(group[pred_col].values)
            continue

        # 回帰用の特徴量準備
        X_factors = []

        if neutralize_sector and sector_cols:
            X_factors.append(group[sector_cols].values)

        if neutralize_size and size_proxy:
            size_vals = group[size_proxy].values.reshape(-1, 1)
            # 外れ値処理
            size_vals = np.clip(size_vals, np.percentile(size_vals, 1), np.percentile(size_vals, 99))
            X_factors.append(size_vals)

        if X_factors:
            X = np.hstack(X_factors)
            y = group[pred_col].values

            # 線形回帰で残差取得
            try:
                reg = LinearRegression()
                reg.fit(X, y)
                resid = y - reg.predict(X)
                residuals.append(resid)
            except:
                logger.warning(f"Failed to neutralize for date {date}")
                residuals.append(y)
        else:
            residuals.append(group[pred_col].values)

    # 結果を格納
    df[f"{pred_col}_neutral"] = np.concatenate(residuals)
    return df


def apply_volatility_targeting(portfolio_returns: pd.Series, target_vol: float = 0.1) -> pd.Series:
    """ボラティリティターゲティング（年率）"""
    logger.info(f"Applying volatility targeting: {target_vol:.1%} annualized")

    # ローリングボラティリティ（20日）
    rolling_vol = portfolio_returns.rolling(20, min_periods=10).std() * np.sqrt(252)

    # スケーリング係数
    scaling = target_vol / (rolling_vol + 1e-8)
    scaling = scaling.clip(0.1, 3.0)  # 極端な値を制限

    # 調整後リターン
    adjusted_returns = portfolio_returns * scaling.shift(1).fillna(1.0)

    return adjusted_returns


def enhanced_portfolio_optimization(
    df: pd.DataFrame,
    date_col: str,
    code_col: str,
    pred_col: str,
    ret_col: str,
    long_frac: float = 0.15,
    short_frac: float = 0.15,
    mode: str = "ls",
    invert_sign: bool = False,
    cost_bps: float = 5.0,
    zscore: bool = False,
    neutralize_sector: bool = False,
    neutralize_size: bool = False,
    vol_target: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """強化されたポートフォリオ最適化"""

    # 元の予測値を保存
    df[f"{pred_col}_original"] = df[pred_col].copy()

    # 1. Zスコア正規化
    if zscore:
        df = daily_zscore_normalize(df, pred_col, date_col)
        pred_col = f"{pred_col}_zscore"

    # 2. セクター・サイズ中立化
    if neutralize_sector or neutralize_size:
        df = neutralize_factors(df, pred_col, date_col, neutralize_sector, neutralize_size)
        pred_col = f"{pred_col}_neutral"

    # 3. 符号反転
    if invert_sign:
        df[pred_col] = -df[pred_col]

    # 4. ポートフォリオ構築
    daily_returns = []

    for date, group in df.groupby(date_col):
        if len(group) < 10:
            continue

        # パーセンタイルでカットオフ
        long_cutoff = group[pred_col].quantile(1 - long_frac)
        short_cutoff = group[pred_col].quantile(short_frac)

        # ポジション決定
        long_mask = group[pred_col] >= long_cutoff
        short_mask = group[pred_col] <= short_cutoff

        # 等ウェイト
        n_long = long_mask.sum()
        n_short = short_mask.sum()

        if n_long == 0 or (mode == "ls" and n_short == 0):
            continue

        # リターン計算
        long_ret = group.loc[long_mask, ret_col].mean() if n_long > 0 else 0
        short_ret = group.loc[short_mask, ret_col].mean() if n_short > 0 else 0

        # ポートフォリオリターン
        if mode == "ls":
            port_ret = 0.5 * long_ret - 0.5 * short_ret
        elif mode == "lo":
            port_ret = long_ret
        else:  # so
            port_ret = -short_ret

        # コスト控除（片道）
        turnover = 1.0  # 仮定：日次100%回転
        cost = turnover * cost_bps / 10000
        port_ret -= cost

        daily_returns.append({
            date_col: date,
            'portfolio_return': port_ret,
            'long_return': long_ret,
            'short_return': short_ret,
            'n_long': n_long,
            'n_short': n_short
        })

    # 結果をDataFrame化
    portfolio_df = pd.DataFrame(daily_returns)
    portfolio_df[date_col] = pd.to_datetime(portfolio_df[date_col])
    portfolio_df = portfolio_df.sort_values(date_col)

    # 5. ボラティリティターゲティング
    if vol_target is not None:
        portfolio_df['portfolio_return_adjusted'] = apply_volatility_targeting(
            portfolio_df['portfolio_return'], vol_target
        )
        ret_col_final = 'portfolio_return_adjusted'
    else:
        ret_col_final = 'portfolio_return'

    # 統計計算
    returns = portfolio_df[ret_col_final]
    sharpe = returns.mean() / returns.std() * np.sqrt(252)

    stats = {
        'sharpe_ratio': sharpe,
        'annual_return': returns.mean() * 252,
        'annual_volatility': returns.std() * np.sqrt(252),
        'max_drawdown': (returns.cumsum() - returns.cumsum().cummax()).min(),
        'win_rate': (returns > 0).mean(),
        'avg_daily_return': returns.mean(),
        'n_days': len(returns),
        'enhancements': {
            'zscore': zscore,
            'neutralize_sector': neutralize_sector,
            'neutralize_size': neutralize_size,
            'vol_target': vol_target,
            'invert_sign': invert_sign
        }
    }

    return portfolio_df, stats


def main():
    parser = argparse.ArgumentParser(description="Enhanced Portfolio Optimization")
    parser.add_argument("--input", type=str, help="Input parquet file with predictions")
    parser.add_argument("--pred-col", type=str, default="predicted_return", help="Prediction column")
    parser.add_argument("--ret-col", type=str, help="Return column (auto-detect if not specified)")
    parser.add_argument("--date-col", type=str, default="date", help="Date column")
    parser.add_argument("--code-col", type=str, default="Code", help="Stock code column")
    parser.add_argument("--mode", type=str, default="ls", choices=["ls", "lo", "so"], help="Portfolio mode")
    parser.add_argument("--long-frac", type=float, default=0.15, help="Long fraction")
    parser.add_argument("--short-frac", type=float, default=0.15, help="Short fraction")
    parser.add_argument("--cost-bps", type=float, default=5.0, help="Transaction cost in bps")
    parser.add_argument("--invert-sign", action="store_true", help="Invert prediction sign")

    # 強化機能
    parser.add_argument("--zscore", action="store_true", help="Apply daily Z-score normalization")
    parser.add_argument("--neutralize-sector", action="store_true", help="Neutralize sector exposure")
    parser.add_argument("--neutralize-size", action="store_true", help="Neutralize size exposure")
    parser.add_argument("--vol-target", type=float, help="Volatility target (annualized)")

    parser.add_argument("--output", type=str, help="Output directory for results")

    args = parser.parse_args()

    # データ読み込み
    if args.input:
        logger.info(f"Loading predictions from {args.input}")
        df = pd.read_parquet(args.input)

        # リターン列の自動検出
        if not args.ret_col:
            candidates = ["actual_return", "returns_1d", "ret_1d", "target", "label"]
            for col in candidates:
                if col in df.columns:
                    args.ret_col = col
                    logger.info(f"Auto-detected return column: {col}")
                    break

        if not args.ret_col:
            raise ValueError("Could not auto-detect return column. Please specify --ret-col")
    else:
        logger.info("No input specified, running demo mode")
        # デモデータ生成
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='B')
        n_stocks = 500

        data = []
        for date in dates:
            for i in range(n_stocks):
                pred = np.random.randn() * 0.02
                ret = pred * 0.5 + np.random.randn() * 0.01  # 相関を持たせる
                data.append({
                    args.date_col: date,
                    args.code_col: f"STOCK_{i:04d}",
                    args.pred_col: pred,
                    'actual_return': ret,
                    f'sector33_{i % 33}': 1,  # セクターダミー
                    'log_mktcap': np.random.uniform(10, 15)  # サイズダミー
                })

        df = pd.DataFrame(data)
        args.ret_col = 'actual_return'

    # 最適化実行
    portfolio_df, stats = enhanced_portfolio_optimization(
        df=df,
        date_col=args.date_col,
        code_col=args.code_col,
        pred_col=args.pred_col,
        ret_col=args.ret_col,
        long_frac=args.long_frac,
        short_frac=args.short_frac,
        mode=args.mode,
        invert_sign=args.invert_sign,
        cost_bps=args.cost_bps,
        zscore=args.zscore,
        neutralize_sector=args.neutralize_sector,
        neutralize_size=args.neutralize_size,
        vol_target=args.vol_target,
    )

    # 結果表示
    logger.info("=" * 60)
    logger.info("PORTFOLIO OPTIMIZATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Sharpe Ratio: {stats['sharpe_ratio']:.3f}")
    logger.info(f"Annual Return: {stats['annual_return']:.2%}")
    logger.info(f"Annual Volatility: {stats['annual_volatility']:.2%}")
    logger.info(f"Max Drawdown: {stats['max_drawdown']:.2%}")
    logger.info(f"Win Rate: {stats['win_rate']:.2%}")
    logger.info(f"Number of Days: {stats['n_days']}")
    logger.info("-" * 60)
    logger.info("Enhancements Applied:")
    for key, value in stats['enhancements'].items():
        if value:
            logger.info(f"  - {key}: {value}")

    # 結果保存
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ポートフォリオリターン
        portfolio_df.to_csv(output_dir / "portfolio_returns.csv", index=False)

        # 統計情報
        with open(output_dir / "portfolio_stats.json", "w") as f:
            json.dump(stats, f, indent=2, default=str)

        logger.info(f"\nResults saved to {output_dir}")

    return stats['sharpe_ratio']


if __name__ == "__main__":
    main()
