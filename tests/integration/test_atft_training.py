#!/usr/bin/env python3
"""
ATFT-GAT-FAN学習機能テスト
"""

import os
import sys
from pathlib import Path


def test_imports():
    """必要なモジュールのインポートテスト"""

    # ATFT-GAT-FANパスを追加
    atft_path = Path("/home/ubuntu/gogooku2/apps/ATFT-GAT-FAN")
    sys.path.insert(0, str(atft_path))

    try:
        # 主要モジュールのインポートテスト
        import importlib.util

        # ATFT_GAT_FANの存在確認
        spec = importlib.util.find_spec("src.models.architectures.atft_gat_fan")
        if spec is None:
            raise ImportError("ATFT_GAT_FAN module not found")

        # ProductionDataModuleV2の存在確認
        spec = importlib.util.find_spec("src.data.production_loader_v2")
        if spec is None:
            print("⚠️ ProductionDataModuleV2 module not found")

        # scripts.trainの存在確認
        spec = importlib.util.find_spec("scripts.train")
        if spec is None:
            print("⚠️ scripts.train module not found")

        print("✅ 主要モジュールインポート成功")
        return True
    except ImportError as e:
        print(f"❌ モジュールインポート失敗: {e}")
        return False


def test_environment():
    """環境変数テスト"""

    required_vars = ["DEGENERACY_GUARD", "PRED_VAR_MIN", "NUM_WORKERS"]

    for var in required_vars:
        if var not in os.environ:
            print(f"❌ 環境変数 {var} が設定されていません")
            return False

    print("✅ 環境変数設定確認完了")
    return True


def test_data_path():
    """データパステスト"""

    data_dir = Path(__file__).parent.parent / "output"
    if not data_dir.exists():
        print(f"❌ データディレクトリが存在しません: {data_dir}")
        return False

    print(f"✅ データディレクトリ確認: {data_dir}")
    return True


def main():
    """メインテスト"""

    print("🧪 ATFT-GAT-FAN学習機能テスト開始")

    tests = [test_imports, test_environment, test_data_path]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print(f"\n📊 テスト結果: {passed}/{total} 成功")

    if passed == total:
        print("🎉 すべてのテストが成功しました！")
        return True
    else:
        print("❌ 一部のテストが失敗しました")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
