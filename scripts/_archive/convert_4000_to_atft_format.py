#!/usr/bin/env python3
# DEPRECATED: Moved to scripts/_archive on 2025-09-04
"""
4000銘柄データをATFT形式に変換するスクリプト
632銘柄データと同じ形式に変換
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_4000_stocks_to_atft_format():
    """4000銘柄データをATFT形式に変換"""

    logger.info("🚀 4000銘柄データをATFT形式に変換開始")

    # 4000銘柄データ読み込み
    data_4000_path = Path("output/ml_dataset_4000_stocks_extended.parquet")
    if not data_4000_path.exists():
        raise FileNotFoundError(f"4000銘柄データが見つかりません: {data_4000_path}")

    df_4000 = pd.read_parquet(data_4000_path)
    logger.info(f"📊 4000銘柄データ読み込み: {df_4000.shape[0]:,}行 × {df_4000.shape[1]}列")

    # 632銘柄データ読み込み（テンプレートとして使用）
    data_632_path = Path("output/ml_dataset_632_stocks.parquet")
    if not data_632_path.exists():
        raise FileNotFoundError(f"632銘柄データが見つかりません: {data_632_path}")

    df_632 = pd.read_parquet(data_632_path)
    logger.info(f"📋 632銘柄データ読み込み: {df_632.shape[0]:,}行 × {df_632.shape[1]}列")

    # 必要なカラムのマッピング
    column_mapping = {
        'Code': 'Code',
        'date': 'Date',
        'Close': 'Close',
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Volume': 'Volume',
        'returns_1d': 'returns_1d',
        'returns_5d': 'returns_5d',
        'returns_10d': 'returns_10d',
        'returns_20d': 'returns_20d',
        'ema_5': 'ema_5',
        'ema_10': 'ema_10',
        'ema_20': 'ema_20',
        'rsi_14': 'rsi_14'
    }

    # 4000銘柄データに632銘柄データのカラム構造を適用
    df_converted = df_4000.copy()

    # カラム名を統一
    df_converted = df_converted.rename(columns=column_mapping)

    # 632銘柄データに存在するが4000銘柄データにないカラムを追加
    missing_columns = []
    for col in df_632.columns:
        if col not in df_converted.columns:
            if col in ['row_idx']:
                # row_idxは自動生成
                df_converted[col] = range(len(df_converted))
            elif 'target' in col.lower():
                # ターゲット関連カラム
                if 'target' not in df_converted.columns:
                    df_converted[col] = df_converted.get('target', 0)
                else:
                    df_converted[col] = df_converted['target']
            else:
                # その他の欠損カラムは0で埋める
                df_converted[col] = 0
                missing_columns.append(col)

    logger.info(f"✅ 欠損カラムを追加: {len(missing_columns)}個")
    if missing_columns:
        logger.info(f"   追加されたカラム: {missing_columns[:10]}{'...' if len(missing_columns) > 10 else ''}")

    # データ型の統一
    for col in df_converted.columns:
        if col in df_632.columns:
            # 632銘柄データと同じデータ型に合わせる
            df_converted[col] = df_converted[col].astype(df_632[col].dtype)

    # 欠損値処理
    df_converted = df_converted.fillna(method='ffill').fillna(method='bfill').fillna(0)

    # 最終確認
    logger.info(f"📊 変換後データ: {df_converted.shape[0]:,}行 × {df_converted.shape[1]}列")
    logger.info(f"🏷️  銘柄数: {df_converted['Code'].nunique()}")

    # 変換結果を保存
    output_path = Path("output/ml_dataset_4000_atft_format.parquet")
    df_converted.to_parquet(output_path, index=False)
    logger.info(f"💾 変換データを保存: {output_path}")

    # メタデータも保存
    metadata = {
        'original_shape': df_4000.shape,
        'converted_shape': df_converted.shape,
        'num_stocks': df_converted['Code'].nunique(),
        'columns_added': len(missing_columns),
        'missing_values_after_conversion': df_converted.isnull().sum().sum()
    }

    metadata_path = Path("output/ml_dataset_4000_atft_format_metadata.json")
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"📋 メタデータを保存: {metadata_path}")
    logger.info("✅ 4000銘柄データ→ATFT形式変換完了")

    return str(output_path)


def main():
    """メイン関数"""
    try:
        output_path = convert_4000_stocks_to_atft_format()
        print("\n🎉 変換成功！")
        print(f"📁 変換後データ: {output_path}")
        print("\n次に以下のコマンドで4000銘柄トレーニングを実行できます:")
        print(f"cp {output_path} output/ml_dataset_production.parquet")
        print("python main.py complete-atft")
    except Exception as e:
        logger.error(f"❌ 変換失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
