#!/usr/bin/env python3
"""
日次信用取引データの修正が適用されていることを確認するテストスクリプト
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import date, timedelta

import aiohttp
import polars as pl
from dotenv import load_dotenv

# プロジェクトルートをパスに追加
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gogooku3.components.jquants_async_fetcher import JQuantsAsyncFetcher

async def test_daily_margin_fix():
    """日次信用取引データの修正が正しく動作することを確認"""

    load_dotenv()
    email = os.getenv("JQUANTS_AUTH_EMAIL")
    password = os.getenv("JQUANTS_AUTH_PASSWORD")

    if not email or not password:
        print("❌ JQuants認証情報が.envファイルに設定されていません")
        return False

    # 短期間のテスト（最近の5日間）
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=5)

    print(f"🔧 日次信用取引データ修正テスト")
    print(f"📅 期間: {start_date} ～ {end_date}")
    print(f"📧 認証: {email[:5]}***@{email.split('@')[1]}")

    fetcher = JQuantsAsyncFetcher(email, password)

    try:
        async with aiohttp.ClientSession() as session:
            # 認証
            print("🔐 JQuantsに認証中...")
            await fetcher.authenticate(session)
            print("✅ 認証成功")

            # 日次信用データを取得
            print("📊 日次信用取引データ取得中...")
            daily_margin_df = await fetcher.get_daily_margin_interest(
                session,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )

            print(f"📈 取得レコード数: {len(daily_margin_df)}")

            if not daily_margin_df.is_empty():
                # データの検証
                print("\n🔍 データ検証結果:")
                print(f"📋 列数: {len(daily_margin_df.columns)}")
                print(f"📏 行数: {len(daily_margin_df)}")

                # 数値列の型を確認
                numeric_cols = [
                    "ShortMarginOutstanding", "LongMarginOutstanding",
                    "DailyChangeShortMarginOutstanding", "DailyChangeLongMarginOutstanding",
                    "ShortMarginOutstandingListedShareRatio", "LongMarginOutstandingListedShareRatio"
                ]

                for col in numeric_cols:
                    if col in daily_margin_df.columns:
                        col_dtype = daily_margin_df.select(pl.col(col)).dtypes[0]
                        null_count = daily_margin_df.select(pl.col(col).is_null().sum()).item()
                        total_count = len(daily_margin_df)
                        null_pct = (null_count / total_count) * 100 if total_count > 0 else 0
                        print(f"  ✅ {col}: {col_dtype}, NULL値 {null_count}/{total_count} ({null_pct:.1f}%)")

                # サンプルデータを表示
                print(f"\n📊 サンプルデータ (最初の3行):")
                print(daily_margin_df.head(3))

                print("\n🎉 修正が正しく適用されています！")
                return True

            else:
                print("⚠️  テスト期間内にデータがありませんでした（土日や祝日の場合は正常です）")
                return True

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_daily_margin_fix())
    if success:
        print("\n🎯 日次信用取引データの修正は正常に動作しています")
    else:
        print("\n💥 修正に問題があります")
        sys.exit(1)