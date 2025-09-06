#!/usr/bin/env python3
# DEPRECATED: Moved to scripts/_archive on 2025-09-04
"""
前期データセット作成スクリプト

営業日ベースで指定期間のデータを取得し、MLデータセットを作成
"""

import asyncio
import argparse
from datetime import datetime, timedelta
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jquants_pipeline.pipeline import JQuantsPipelineV4
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description='前期データセットを作成',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 1ヶ月分のデータ
  python create_historical_dataset.py --period 1m
  
  # 3ヶ月分のデータ  
  python create_historical_dataset.py --period 3m
  
  # カスタム期間
  python create_historical_dataset.py --start-date 2024-10-01 --end-date 2024-12-31
  
  # バッチサイズ指定（大量データ時のメモリ対策）
  python create_historical_dataset.py --period 1y --batch-days 30
        """
    )
    
    parser.add_argument(
        '--period',
        choices=['1w', '1m', '3m', '6m', '1y', '2y'],
        help='データ取得期間 (1w=1週間, 1m=1ヶ月, 3m=3ヶ月, 6m=6ヶ月, 1y=1年, 2y=2年)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='開始日 (YYYY-MM-DD形式)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='終了日 (YYYY-MM-DD形式、省略時は今日)'
    )
    
    parser.add_argument(
        '--batch-days',
        type=int,
        default=20,
        help='バッチサイズ（営業日数）。大量データ時は小さく設定 (default: 20)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/historical',
        help='出力ディレクトリ (default: output/historical)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='実行計画のみ表示（実際には取得しない）'
    )
    
    return parser.parse_args()


def calculate_date_range(period: str = None, start_date: str = None, end_date: str = None):
    """期間指定から日付範囲を計算"""
    
    if end_date:
        end = datetime.strptime(end_date, '%Y-%m-%d')
    else:
        end = datetime.now()
    
    if start_date:
        start = datetime.strptime(start_date, '%Y-%m-%d')
    elif period:
        period_days = {
            '1w': 7,
            '1m': 30,
            '3m': 90,
            '6m': 180,
            '1y': 365,
            '2y': 730
        }
        start = end - timedelta(days=period_days[period])
    else:
        raise ValueError("--period または --start-date を指定してください")
    
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')


async def create_dataset_batch(pipeline, start_date: str, end_date: str, batch_num: int):
    """単一バッチのデータセット作成"""
    logger.info(f"バッチ {batch_num}: {start_date} ~ {end_date}")
    
    try:
        df, metadata = await pipeline.run(
            use_jquants=True,
            start_date=start_date,
            end_date=end_date
        )
        
        if df is not None and metadata is not None:
            logger.info(f"  ✅ 取得成功: {len(df)}レコード, {metadata['stocks']}銘柄")
            return df, metadata
        else:
            logger.warning(f"  ⚠️ データ取得失敗")
            return None, None
            
    except Exception as e:
        logger.error(f"  ❌ エラー: {e}")
        return None, None


async def main():
    """メイン処理"""
    args = parse_arguments()
    
    # 日付範囲を計算
    start_date, end_date = calculate_date_range(args.period, args.start_date, args.end_date)
    
    # 実行計画を表示
    logger.info("=" * 60)
    logger.info("前期データセット作成計画")
    logger.info("=" * 60)
    logger.info(f"期間: {start_date} ~ {end_date}")
    
    # 営業日数を推定（土日を除く）
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    total_days = (end_dt - start_dt).days
    business_days = int(total_days * 0.714)  # 約5/7が営業日
    
    logger.info(f"推定営業日数: 約{business_days}日")
    logger.info(f"推定データ量: 約{business_days * 3800:,}レコード")
    
    # バッチ分割
    num_batches = (business_days + args.batch_days - 1) // args.batch_days
    logger.info(f"バッチサイズ: {args.batch_days}営業日")
    logger.info(f"バッチ数: {num_batches}")
    
    # API制限の確認
    estimated_api_calls = num_batches * 14  # 各バッチ約14回のAPIコール
    logger.info(f"推定APIコール数: 約{estimated_api_calls}回")
    
    if args.dry_run:
        logger.info("\n[DRY RUN] 実際のデータ取得は行いません")
        return
    
    # 実行確認
    logger.info("\n続行しますか？ (y/n): ")
    response = input().strip().lower()
    if response != 'y':
        logger.info("キャンセルしました")
        return
    
    # パイプライン初期化
    logger.info("\nパイプライン初期化中...")
    pipeline = JQuantsPipelineV4()
    
    # バッチ処理
    all_dataframes = []
    all_metadata = []
    
    current_date = start_dt
    batch_num = 1
    
    while current_date < end_dt:
        # バッチの終了日を計算（営業日ベース）
        batch_end = current_date + timedelta(days=int(args.batch_days * 1.4))  # 週末を考慮
        if batch_end > end_dt:
            batch_end = end_dt
        
        # バッチ実行
        batch_start_str = current_date.strftime('%Y-%m-%d')
        batch_end_str = batch_end.strftime('%Y-%m-%d')
        
        df, metadata = await create_dataset_batch(
            pipeline, 
            batch_start_str, 
            batch_end_str, 
            batch_num
        )
        
        if df is not None:
            all_dataframes.append(df)
            all_metadata.append(metadata)
        
        # 次のバッチへ
        current_date = batch_end + timedelta(days=1)
        batch_num += 1
        
        # API制限対策（少し待機）
        if batch_num <= num_batches:
            await asyncio.sleep(2)
    
    # 結果の集約
    if all_dataframes:
        import polars as pl
        
        logger.info("\n" + "=" * 60)
        logger.info("データセット集約中...")
        
        # 全データフレームを結合
        combined_df = pl.concat(all_dataframes)
        
        # 重複除去（同じ日付・銘柄の重複を除去）
        if 'Date' in combined_df.columns and 'Code' in combined_df.columns:
            combined_df = combined_df.unique(['Date', 'Code'])
        
        # 統計情報
        logger.info(f"総レコード数: {len(combined_df):,}")
        logger.info(f"総銘柄数: {combined_df['Code'].n_unique():,}")
        logger.info(f"期間: {combined_df['Date'].min()} ~ {combined_df['Date'].max()}")
        
        # 保存
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"{output_dir}/dataset_{start_date}_{end_date}_{timestamp}.parquet"
        
        combined_df.write_parquet(output_path)
        logger.info(f"\n✅ データセット保存完了: {output_path}")
        
        # メタデータも保存
        import json
        meta_path = output_path.replace('.parquet', '_metadata.json')
        final_metadata = {
            'created_at': datetime.now().isoformat(),
            'date_range': {
                'start': start_date,
                'end': end_date
            },
            'shape': {
                'rows': len(combined_df),
                'cols': len(combined_df.columns)
            },
            'stocks': combined_df['Code'].n_unique(),
            'batches': len(all_dataframes),
            'batch_size': args.batch_days
        }
        
        with open(meta_path, 'w') as f:
            json.dump(final_metadata, f, indent=2)
        
        logger.info(f"✅ メタデータ保存完了: {meta_path}")
        
    else:
        logger.error("❌ データ取得に失敗しました")


if __name__ == "__main__":
    asyncio.run(main())
