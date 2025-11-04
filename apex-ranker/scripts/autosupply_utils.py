#!/usr/bin/env python3
"""
銘柄供給自動調整ユーティリティ

バックテストで銘柄供給不足を防ぐための動的k調整機能

Usage:
    from scripts.autosupply_utils import autosupply_k_ratio, ensure_k_min

    # リバランス日ごとに
    ratio = autosupply_k_ratio(
        candidate_count=len(candidates),
        target_top_k=35,
        alpha=1.5,
        floor=0.15
    )
    k_min = ensure_k_min(current_k_min=0, hard_floor=53)
    k_pick = max(int(ratio * max(len(candidates), 1)), k_min)

    # 上位 k_pick 銘柄を選定
    selected = candidates[:k_pick]
"""

from __future__ import annotations


def autosupply_k_ratio(
    candidate_count: int,
    target_top_k: int = 35,
    alpha: float = 1.5,
    floor: float = 0.15,
) -> float:
    """
    銘柄供給率を動的に調整

    Args:
        candidate_count: 候補銘柄数
        target_top_k: 目標保有銘柄数（デフォルト: 35）
        alpha: 供給倍率（デフォルト: 1.5 = 35 × 1.5 = 53最低供給）
        floor: 最低供給率（デフォルト: 0.15 = 15%）

    Returns:
        float: 供給率（0.15～1.0）

    Example:
        >>> autosupply_k_ratio(candidate_count=100, target_top_k=35, alpha=1.5)
        0.53  # 100銘柄のうち53銘柄（53%）を選定

        >>> autosupply_k_ratio(candidate_count=30, target_top_k=35, alpha=1.5)
        1.0  # 候補不足時は全銘柄（100%）を選定
    """
    if candidate_count <= 0:
        return 0.0

    # 目標供給数 = target_top_k × alpha
    target_supply = target_top_k * alpha

    # 供給率 = target_supply / candidate_count
    ratio = target_supply / candidate_count

    # floor（最低15%）～ 1.0（全銘柄）の範囲に制限
    ratio = max(floor, min(ratio, 1.0))

    return ratio


def ensure_k_min(current_k_min: int = 0, hard_floor: int = 53) -> int:
    """
    最低選定数を保証

    Args:
        current_k_min: 現在の最低選定数
        hard_floor: 絶対最低値（デフォルト: 53 = 35 × 1.5）

    Returns:
        int: 保証された最低選定数

    Example:
        >>> ensure_k_min(current_k_min=30, hard_floor=53)
        53

        >>> ensure_k_min(current_k_min=100, hard_floor=53)
        100
    """
    return max(current_k_min, hard_floor)


def calculate_dynamic_k(
    candidate_count: int,
    target_top_k: int = 35,
    alpha: float = 1.5,
    floor_ratio: float = 0.15,
    hard_floor: int = 53,
) -> int:
    """
    動的k値を計算（ワンステップ版）

    Args:
        candidate_count: 候補銘柄数
        target_top_k: 目標保有銘柄数
        alpha: 供給倍率
        floor_ratio: 最低供給率
        hard_floor: 絶対最低値

    Returns:
        int: 選定すべき銘柄数

    Example:
        >>> calculate_dynamic_k(candidate_count=100, target_top_k=35)
        53  # 100 × 0.53 = 53銘柄

        >>> calculate_dynamic_k(candidate_count=30, target_top_k=35)
        53  # 候補不足だが hard_floor=53 を保証
    """
    ratio = autosupply_k_ratio(
        candidate_count=candidate_count,
        target_top_k=target_top_k,
        alpha=alpha,
        floor=floor_ratio,
    )

    k_from_ratio = int(ratio * max(candidate_count, 1))
    k_min = ensure_k_min(current_k_min=k_from_ratio, hard_floor=hard_floor)

    return k_min


if __name__ == "__main__":
    # テスト実行
    print("=" * 70)
    print("🔧 autosupply_utils テスト")
    print("=" * 70)

    test_cases = [
        {"candidate_count": 100, "desc": "十分な候補（100銘柄）"},
        {"candidate_count": 50, "desc": "中程度の候補（50銘柄）"},
        {"candidate_count": 30, "desc": "不足気味（30銘柄）"},
        {"candidate_count": 10, "desc": "極度の不足（10銘柄）"},
    ]

    for case in test_cases:
        count = case["candidate_count"]
        desc = case["desc"]

        ratio = autosupply_k_ratio(count, target_top_k=35, alpha=1.5)
        k_min = ensure_k_min(hard_floor=53)
        k_pick = calculate_dynamic_k(count, target_top_k=35)

        print(f"\n{desc}:")
        print(f"  - 候補数: {count}")
        print(f"  - 供給率: {ratio * 100:.1f}%")
        print(f"  - 最低保証: {k_min}")
        print(f"  - 選定数: {k_pick}")
        print(f"  - 結果: {min(k_pick, count)}銘柄選定")

    print("\n" + "=" * 70)
    print("✅ テスト完了")
