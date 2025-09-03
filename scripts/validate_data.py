#!/usr/bin/env python3
"""
Parquetデータの検証スクリプト
設計書v3.2に基づく大規模データ検証
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from omegaconf import DictConfig, OmegaConf
import hydra
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def validate_data(config: DictConfig):
    """データの検証"""

    # Parquetファイルのリスト取得
    data_dir = Path(config.data.source.data_dir)
    pattern = config.data.source.file_pattern
    parquet_files = sorted(data_dir.glob(pattern))

    logger.info(f"Found {len(parquet_files)} parquet files in {data_dir}")

    # 最初のファイルを読み込んでスキーマを確認
    if parquet_files:
        df = pd.read_parquet(parquet_files[0], engine='pyarrow')
        logger.info(f"Sample file: {parquet_files[0].name}")
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # 基本統計
    logger.info("\\n=== Basic Statistics ===")
    logger.info(f"Number of unique stocks: {df['code'].nunique()}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Market codes: {df['market_code_name'].value_counts().to_dict()}")

    # 欠損値チェック
    logger.info("\\n=== Missing Values ===")
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'count': missing_counts[missing_counts > 0],
        'percentage': missing_pct[missing_counts > 0]
    })
    if not missing_df.empty:
        logger.info(f"\\n{missing_df}")
    else:
        logger.info("No missing values found")

    # 異常値チェック
    logger.info("\\n=== Outlier Check ===")

    # リターンの異常値
    return_cols = [col for col in df.columns if col.startswith('return_')]
    for col in return_cols[:5]:  # 最初の5つだけチェック
        outliers = df[col].abs() > 0.5  # 50%以上の変動
        if outliers.any():
            logger.info(f"{col}: {outliers.sum()} outliers (>{50}% change)")

    # 価格の整合性チェック
    price_issues = (
        (df['adjustment_high'] < df['adjustment_low']) |
        (df['adjustment_close'] > df['adjustment_high']) |
        (df['adjustment_close'] < df['adjustment_low'])
    )
    if price_issues.any():
        logger.warning(f"Found {price_issues.sum()} rows with price inconsistencies")

    # データ分布の可視化
    logger.info("\\n=== Creating visualization plots ===")

    # 出力ディレクトリ
    output_dir = Path("output/data_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. リターン分布
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    return_cols_to_plot = ['return_1d', 'return_5d', 'return_20d', 'rsi14']
    for i, col in enumerate(return_cols_to_plot):
        if col in df.columns:
            ax = axes[i // 2, i % 2]
            df[col].hist(bins=50, ax=ax)
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(output_dir / 'distributions.png')
    plt.close()

    # 2. 時系列サンプル
    sample_codes = df['code'].unique()[:5]
    fig, axes = plt.subplots(len(sample_codes), 1, figsize=(12, 4*len(sample_codes)))

    for i, code in enumerate(sample_codes):
        stock_data = df[df['code'] == code].sort_values('date')
        ax = axes[i] if len(sample_codes) > 1 else axes
        ax.plot(pd.to_datetime(stock_data['date']), stock_data['adjustment_close'])
        ax.set_title(f'Stock {code} - {stock_data.iloc[0]["company_name"]}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Adjusted Close Price')

    plt.tight_layout()
    plt.savefig(output_dir / 'sample_time_series.png')
    plt.close()

    # 3. 相関マトリックス
    feature_cols = ['return_1d', 'return_5d', 'return_20d', 'rsi14', 'atr14',
                   'adx14', 'macd', 'adjustment_volume']
    available_cols = [col for col in feature_cols if col in df.columns]
    if len(available_cols) > 1:
        corr_matrix = df[available_cols].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_matrix.png')
        plt.close()

    # データ品質レポート
    report = {
        'total_files': len(parquet_files),
        'sample_rows': len(df),
        'total_columns': len(df.columns),
        'unique_stocks': df['code'].nunique(),
        'date_range': f"{df['date'].min()} to {df['date'].max()}",
        'missing_values': missing_counts[missing_counts > 0].to_dict(),
        'price_inconsistencies': int(price_issues.sum()),
        'return_outliers': {col: int((df[col].abs() > 0.5).sum()) for col in return_cols[:5] if col in df.columns},
        'total_size_gb': config.data.large_scale.total_size_gb
    }

    # レポート保存
    report_path = output_dir / 'data_quality_report.yaml'
    with open(report_path, 'w') as f:
        OmegaConf.save(report, f)

    logger.info(f"\\nValidation complete. Results saved to {output_dir}")
    logger.info(f"Data quality report: {report_path}")

    return report


if __name__ == "__main__":
    validate_data()
