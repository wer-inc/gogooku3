#!/usr/bin/env python3
"""
バックテスト結果の健全性チェック

チェック項目:
- 全リバランス日で selected_count >= effective_k_min
- fallback_rate < threshold
- 日次リターン |portfolio_return| <= abs_ret_day_max

Usage:
    # デフォルトパス使用（推奨）
    python scripts/check_backtest_output.py

    # カスタムパス指定
    python scripts/check_backtest_output.py \
      --input output/backtest/backtest_result.json \
      --k-min 53 \
      --fallback-threshold 0.20 \
      --abs-ret-day-max 0.15 \
      --report output/reports/backtest_health_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# パス定数をインポート（デフォルト値として使用）
try:
    from path_constants import BACKTEST_HEALTH_REPORT, BACKTEST_JSON
except ImportError:
    # フォールバック: path_constants.py が見つからない場合
    BACKTEST_JSON = "output/backtest/backtest_result.json"
    BACKTEST_HEALTH_REPORT = "output/reports/backtest_health_report.json"


def detect_rebalance_key(data: dict) -> str | None:
    """リバランスデータのキーを自動検出"""
    candidates = ["rebalances", "timeline", "days", "events", "portfolio_history"]
    for key in candidates:
        if key in data:
            return key
    return None


def extract_selected_count(entry: dict) -> int | None:
    """selected_count を抽出（柔軟な検出）"""
    # 直接のキー
    if "selected_count" in entry:
        val = entry["selected_count"]
        if isinstance(val, int):
            return val

    # selected/positions の配列長
    for key in ["selected", "positions", "holdings"]:
        if key in entry:
            val = entry[key]
            if isinstance(val, list):
                return len(val)

    return None


def extract_fallback_used(entry: dict) -> bool:
    """fallback_used を抽出"""
    if "fallback_used" in entry:
        return bool(entry["fallback_used"])
    if "is_fallback" in entry:
        return bool(entry["is_fallback"])
    return False  # デフォルトは False


def extract_portfolio_return(entry: dict) -> float | None:
    """ポートフォリオリターンを抽出"""
    candidates = ["portfolio_return", "daily_return", "ret", "return"]
    for key in candidates:
        if key in entry:
            val = entry[key]
            if isinstance(val, (int, float)):
                return float(val)
    return None


def extract_candidate_count(entry: dict) -> int | None:
    """候補銘柄数を抽出（effective_kmin計算用）"""
    candidates = ["candidate_count", "num_candidates", "universe_size", "n_candidates"]
    for key in candidates:
        if key in entry:
            val = entry[key]
            if isinstance(val, int):
                return val

    # selected + dropped から推定
    if "dropped" in entry and "selected_count" in entry:
        dropped = entry["dropped"]
        selected = entry["selected_count"]
        if isinstance(dropped, int) and isinstance(selected, int):
            return selected + dropped

    return None


def main():
    parser = argparse.ArgumentParser(description="バックテスト結果の健全性チェック")
    parser.add_argument(
        "--input",
        default=BACKTEST_JSON,
        help=f"バックテストJSON（デフォルト: {BACKTEST_JSON}）",
    )
    parser.add_argument("--k-min", type=int, default=53, help="最低選定数")
    parser.add_argument(
        "--fallback-threshold", type=float, default=0.20, help="Fallback率閾値"
    )
    parser.add_argument(
        "--abs-ret-day-max", type=float, default=0.15, help="日次リターン絶対値最大"
    )
    parser.add_argument(
        "--report",
        default=BACKTEST_HEALTH_REPORT,
        help=f"健全性レポート出力先（デフォルト: {BACKTEST_HEALTH_REPORT}）",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("🔍 バックテスト結果の健全性チェック")
    print("=" * 70)
    print(f"入力: {args.input}")
    print(f"最低選定数: {args.k_min}")
    print(f"Fallback閾値: {args.fallback_threshold * 100:.0f}%")
    print(f"日次リターン最大: {args.abs_ret_day_max * 100:.0f}%")
    print("=" * 70)

    # JSON読み込み
    with open(args.input) as f:
        data = json.load(f)

    # リバランスデータの検出
    rebalance_key = detect_rebalance_key(data)
    if not rebalance_key:
        print("❌ リバランスデータが見つかりません")
        print(f"   検出されたキー: {list(data.keys())}")
        sys.exit(1)

    print(f"✅ リバランスデータ検出: '{rebalance_key}'")

    rebalances = data[rebalance_key]
    if not isinstance(rebalances, list):
        print(f"❌ '{rebalance_key}' はリスト形式ではありません")
        sys.exit(1)

    print(f"📊 リバランス回数: {len(rebalances)}")

    # 統計収集
    selected_counts = []
    candidate_counts = []
    fallback_count = 0
    portfolio_returns = []

    for idx, entry in enumerate(rebalances):
        # selected_count
        count = extract_selected_count(entry)
        if count is not None:
            selected_counts.append(count)
        else:
            print(f"⚠️  リバランス#{idx}: selected_count を検出できません")

        # candidate_count
        cand_count = extract_candidate_count(entry)
        if cand_count is not None:
            candidate_counts.append(cand_count)

        # fallback_used
        if extract_fallback_used(entry):
            fallback_count += 1

        # portfolio_return
        ret = extract_portfolio_return(entry)
        if ret is not None:
            portfolio_returns.append(ret)

    print("\n📈 統計:")
    print(f"  - selected_count 取得: {len(selected_counts)}/{len(rebalances)}")
    print(f"  - candidate_count 取得: {len(candidate_counts)}/{len(rebalances)}")
    print(f"  - fallback使用: {fallback_count}/{len(rebalances)}")
    print(f"  - portfolio_return 取得: {len(portfolio_returns)}/{len(rebalances)}")

    # 健全性チェック
    issues = []

    # Check 1: selected_count >= effective_k_min
    # effective_k_min = min(k_min, candidate_count) で現実的な供給条件を判定
    if selected_counts:
        violations = []
        violation_details = []

        for i, sc in enumerate(selected_counts):
            # candidate_count がある場合は effective_kmin を計算
            if i < len(candidate_counts) and candidate_counts[i] is not None:
                effective_kmin = min(args.k_min, candidate_counts[i])
                if sc < effective_kmin:
                    violations.append(sc)
                    violation_details.append(
                        f"    #{i}: selected={sc}, effective_kmin={effective_kmin}, candidates={candidate_counts[i]}"
                    )
            else:
                # candidate_count がない場合は従来通り k_min で判定
                if sc < args.k_min:
                    violations.append(sc)
                    violation_details.append(
                        f"    #{i}: selected={sc}, k_min={args.k_min} (candidate_count不明)"
                    )

        if violations:
            issues.append(
                f"❌ selected_count < effective_k_min が{len(violations)}回発生"
            )
            print("\n❌ selected_count 不足:")
            print(f"   - 違反回数: {len(violations)}/{len(selected_counts)}")
            print(f"   - 最小値: {min(selected_counts)}")
            print(f"   - 平均値: {sum(selected_counts) / len(selected_counts):.1f}")
            if violation_details[:5]:  # 最大5件表示
                print("   - 違反例（最大5件）:")
                for detail in violation_details[:5]:
                    print(detail)
        else:
            print("\n✅ selected_count: 全て >= effective_k_min")
            print(f"   - 最小値: {min(selected_counts)}")
            print(f"   - 平均値: {sum(selected_counts) / len(selected_counts):.1f}")
            if candidate_counts:
                print(f"   - candidate平均: {sum(candidate_counts) / len(candidate_counts):.1f}")
    else:
        print("\n⚠️  selected_count データなし（チェックスキップ）")

    # Check 2: fallback_rate < threshold
    if rebalances:
        fallback_rate = fallback_count / len(rebalances)
        if fallback_rate > args.fallback_threshold:
            issues.append(
                f"❌ fallback率が高い: {fallback_rate * 100:.1f}% (> {args.fallback_threshold * 100:.0f}%)"
            )
            print("\n❌ Fallback率:")
            print(f"   - {fallback_rate * 100:.1f}% ({fallback_count}/{len(rebalances)})")
        else:
            print(f"\n✅ Fallback率: {fallback_rate * 100:.1f}% (< {args.fallback_threshold * 100:.0f}%)")

    # Check 3: |portfolio_return| <= abs_ret_day_max
    if portfolio_returns:
        extreme_returns = [r for r in portfolio_returns if abs(r) > args.abs_ret_day_max]
        if extreme_returns:
            issues.append(
                f"❌ |portfolio_return| > {args.abs_ret_day_max * 100:.0f}% が{len(extreme_returns)}回発生"
            )
            print("\n❌ 極端リターン:")
            print(f"   - 違反回数: {len(extreme_returns)}/{len(portfolio_returns)}")
            print(f"   - 最大値: {max(portfolio_returns) * 100:.2f}%")
            print(f"   - 最小値: {min(portfolio_returns) * 100:.2f}%")
        else:
            print(f"\n✅ 日次リターン: 全て |r| <= {args.abs_ret_day_max * 100:.0f}%")
            if portfolio_returns:
                print(f"   - 最大値: {max(portfolio_returns) * 100:.2f}%")
                print(f"   - 最小値: {min(portfolio_returns) * 100:.2f}%")
    else:
        print("\n⚠️  portfolio_return データなし（チェックスキップ）")

    # レポート保存
    report = {
        "input": args.input,
        "rebalance_key": rebalance_key,
        "rebalance_count": len(rebalances),
        "checks": {
            "k_min": args.k_min,
            "fallback_threshold": args.fallback_threshold,
            "abs_ret_day_max": args.abs_ret_day_max,
        },
        "statistics": {
            "selected_counts": {
                "count": len(selected_counts),
                "min": min(selected_counts) if selected_counts else None,
                "max": max(selected_counts) if selected_counts else None,
                "mean": (
                    sum(selected_counts) / len(selected_counts)
                    if selected_counts
                    else None
                ),
            },
            "fallback": {
                "count": fallback_count,
                "total": len(rebalances),
                "rate": fallback_count / max(len(rebalances), 1),
            },
            "portfolio_returns": {
                "count": len(portfolio_returns),
                "min": min(portfolio_returns) if portfolio_returns else None,
                "max": max(portfolio_returns) if portfolio_returns else None,
                "mean": (
                    sum(portfolio_returns) / len(portfolio_returns)
                    if portfolio_returns
                    else None
                ),
            },
        },
        "health_check": {"passed": len(issues) == 0, "issues": issues},
    }

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n📝 健全性レポート: {args.report}")

    # 健全性チェック失敗時は終了
    if issues:
        print("\n❌ 健全性チェック失敗:")
        for issue in issues:
            print(f"  {issue}")
        print("=" * 70)
        sys.exit(1)

    print("\n✅ 健全性チェック成功")
    print("=" * 70)


if __name__ == "__main__":
    main()
