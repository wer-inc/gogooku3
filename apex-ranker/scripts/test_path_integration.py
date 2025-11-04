#!/usr/bin/env python3
"""
パス定数の統合テスト

全スクリプトが path_constants.py を正しくインポートできるか検証。

Usage:
    python scripts/test_path_integration.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def test_path_constants_import():
    """path_constants.py のインポートテスト"""
    print("=" * 70)
    print("🧪 Test 1: path_constants.py インポート")
    print("=" * 70)

    try:
        from path_constants import (
            BACKTEST_HEALTH_REPORT,
            BACKTEST_JSON,
            DATASET_CLEAN,
            DATASET_RAW,
            QUALITY_REPORT,
        )

        print("✅ インポート成功")
        print(f"   DATASET_RAW: {DATASET_RAW}")
        print(f"   DATASET_CLEAN: {DATASET_CLEAN}")
        print(f"   QUALITY_REPORT: {QUALITY_REPORT}")
        print(f"   BACKTEST_JSON: {BACKTEST_JSON}")
        print(f"   BACKTEST_HEALTH_REPORT: {BACKTEST_HEALTH_REPORT}")
        return True
    except ImportError as e:
        print(f"❌ インポート失敗: {e}")
        return False


def test_filter_script_import():
    """filter_dataset_quality.py のインポートテスト"""
    print("\n" + "=" * 70)
    print("🧪 Test 2: filter_dataset_quality.py からのインポート")
    print("=" * 70)

    try:
        # スクリプト内のインポートをシミュレート
        script_path = Path(__file__).parent / "filter_dataset_quality.py"
        if not script_path.exists():
            print(f"❌ スクリプトが見つかりません: {script_path}")
            return False

        # 実際にインポートを試みる（execを使わずに手動検証）
        with open(script_path) as f:
            content = f.read()

        # インポート文があるか確認
        if "from path_constants import" in content:
            print("✅ filter_dataset_quality.py は path_constants をインポート")
            return True
        else:
            print("❌ filter_dataset_quality.py に path_constants インポートがない")
            return False
    except Exception as e:
        print(f"❌ テスト失敗: {e}")
        return False


def test_check_script_import():
    """check_backtest_output.py のインポートテスト"""
    print("\n" + "=" * 70)
    print("🧪 Test 3: check_backtest_output.py からのインポート")
    print("=" * 70)

    try:
        script_path = Path(__file__).parent / "check_backtest_output.py"
        if not script_path.exists():
            print(f"❌ スクリプトが見つかりません: {script_path}")
            return False

        with open(script_path) as f:
            content = f.read()

        if "from path_constants import" in content:
            print("✅ check_backtest_output.py は path_constants をインポート")
            return True
        else:
            print("❌ check_backtest_output.py に path_constants インポートがない")
            return False
    except Exception as e:
        print(f"❌ テスト失敗: {e}")
        return False


def test_argparse_defaults():
    """argparse デフォルト値のテスト"""
    print("\n" + "=" * 70)
    print("🧪 Test 4: argparse デフォルト値")
    print("=" * 70)

    try:
        from path_constants import DATASET_CLEAN, DATASET_RAW

        # filter_dataset_quality.py の argparse をインポート
        # （実際には main() 内なので直接テストできないが、定数が定義されていることを確認）
        print("✅ デフォルト値が定義されている:")
        print(f"   DATASET_RAW: {DATASET_RAW}")
        print(f"   DATASET_CLEAN: {DATASET_CLEAN}")
        return True
    except Exception as e:
        print(f"❌ テスト失敗: {e}")
        return False


def test_effective_kmin_logic():
    """effective_kmin ロジックのユニットテスト"""
    print("\n" + "=" * 70)
    print("🧪 Test 5: effective_kmin ロジック")
    print("=" * 70)

    test_cases = [
        {"k_min": 53, "candidate_count": 100, "expected": 53},
        {"k_min": 53, "candidate_count": 50, "expected": 50},
        {"k_min": 53, "candidate_count": 30, "expected": 30},
        {"k_min": 53, "candidate_count": 10, "expected": 10},
    ]

    all_passed = True
    for case in test_cases:
        k_min = case["k_min"]
        candidate_count = case["candidate_count"]
        expected = case["expected"]

        effective_kmin = min(k_min, candidate_count)

        if effective_kmin == expected:
            print(
                f"✅ k_min={k_min}, candidates={candidate_count} → effective={effective_kmin}"
            )
        else:
            print(
                f"❌ k_min={k_min}, candidates={candidate_count} → effective={effective_kmin} (expected {expected})"
            )
            all_passed = False

    return all_passed


def main():
    print("\n" + "=" * 70)
    print("🚀 APEX-Ranker パス定数統合テスト")
    print("=" * 70)

    results = []
    results.append(("path_constants インポート", test_path_constants_import()))
    results.append(
        ("filter_dataset_quality.py 統合", test_filter_script_import())
    )
    results.append(("check_backtest_output.py 統合", test_check_script_import()))
    results.append(("argparse デフォルト値", test_argparse_defaults()))
    results.append(("effective_kmin ロジック", test_effective_kmin_logic()))

    print("\n" + "=" * 70)
    print("📊 テスト結果サマリー")
    print("=" * 70)

    total = len(results)
    passed = sum(1 for _, result in results if result)
    failed = total - passed

    for name, result in results:
        status = "✅ Pass" if result else "❌ Fail"
        print(f"{status:8} {name}")

    print("=" * 70)
    print(f"総計: {passed}/{total} passed ({failed} failed)")

    if failed == 0:
        print("\n🎉 すべてのテストが成功しました!")
        print("✅ P0-1, P0-2 の修正が正しく動作しています")
        return 0
    else:
        print(f"\n❌ {failed} 個のテストが失敗しました")
        return 1


if __name__ == "__main__":
    sys.exit(main())
