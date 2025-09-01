#!/usr/bin/env python3
"""
拡張財務機能テストスクリプト
ユーザーの設計に基づくEvent-Centered DatasetとDaily Panel拡張をテスト
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# パス設定
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from scripts.data.direct_api_dataset_builder import DirectAPIDatasetBuilder

def main():
    """メイン実行関数"""
    print("🚀 拡張財務機能テスト開始")
    print("=" * 60)

    # 環境変数読み込み
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print("✅ .envファイル読み込み成功")
    else:
        print("⚠️ .envファイルが見つからないため、環境変数を手動設定してください")

    # ビルダー初期化
    builder = DirectAPIDatasetBuilder()

    # 認証テスト
    print("\\n🔐 J-Quants API認証テスト...")
    if builder.authenticate():
        print("✅ J-Quants API認証成功")

        # 拡張財務機能テスト
        print("\\n🧪 拡張財務機能テスト実行...")
        builder.test_enhanced_financial_features()

        print("\\n" + "=" * 60)
        print("🎉 テスト完了!")
        print("生成されたファイル:")
        print("- event_features_test.parquet (イベント中心データセット)")
        print("- enhanced_daily_test.parquet (拡張日次パネル)")
        print("=" * 60)

    else:
        print("❌ J-Quants API認証失敗")
        print("APIキーの設定を確認してください")

if __name__ == "__main__":
    main()
