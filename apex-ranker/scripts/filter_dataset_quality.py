#!/usr/bin/env python3
"""
データ品質ゲート & クリーニング

品質基準:
- 価格フィルタ（min_price >= 100円）
- リターン異常値除外（|ret_1d| <= max_ret_1d）
- ADV60（平均売買代金60日、当日除外）>= min_adv
- 株価フリーズ検出（同値連続5日以上の削減確認）

Usage:
    # デフォルトパス使用（推奨）
    python scripts/filter_dataset_quality.py

    # カスタムパス指定
    python scripts/filter_dataset_quality.py \
      --input output/ml_dataset_latest_full.parquet \
      --output output/ml_dataset_clean.parquet \
      --min-price 100 \
      --max-ret-1d 0.15 \
      --min-adv 50000000 \
      --report output/reports/quality_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

# パス定数をインポート（デフォルト値として使用）
try:
    from path_constants import DATASET_CLEAN, DATASET_RAW, QUALITY_REPORT
except ImportError:
    # フォールバック: path_constants.py が見つからない場合
    DATASET_RAW = "output/ml_dataset_latest_full.parquet"
    DATASET_CLEAN = "output/ml_dataset_clean.parquet"
    QUALITY_REPORT = "output/reports/quality_report.json"

# 列名候補（自動検出用）
CAND_CODE = ["code", "Code", "ticker", "symbol"]
CAND_DATE = ["timestamp", "trading_date", "date", "Date"]
CAND_PRICE = ["adj_close", "AdjClose", "close", "Close"]
CAND_VOLUME = ["volume", "Volume"]
CAND_TURNOVER = ["turnover", "TurnoverValue", "turnover_value"]


def detect_column(df: pl.DataFrame, candidates: list[str], name: str) -> str:
    """列名を自動検出"""
    for cand in candidates:
        if cand in df.columns:
            return cand
    raise ValueError(f"Could not find {name} column in {candidates}")


def compute_adv60_trailing(
    df: pl.DataFrame,
    code_col: str,
    date_col: str,
    turnover_col: str | None,
    volume_col: str | None,
    price_col: str,
) -> pl.DataFrame:
    """
    ADV60（平均売買代金60日、当日除外）を計算

    ルックアヘッド防止のため、rolling().shift(1)を使用
    turnover列がない場合は volume * price から算出
    """
    # turnover列の準備
    if turnover_col and turnover_col in df.columns:
        df = df.with_columns(pl.col(turnover_col).alias("_turnover"))
    elif volume_col and volume_col in df.columns:
        # volume * price で turnover を算出
        df = df.with_columns(
            (pl.col(volume_col) * pl.col(price_col)).alias("_turnover")
        )
    else:
        raise ValueError("Neither turnover nor volume column found")

    # 整列・重複排除
    df = df.sort([code_col, date_col]).unique([code_col, date_col], keep="first")

    # ADV60（当日除外）
    df = df.with_columns(
        pl.col("_turnover")
        .rolling_mean(window_size=60)
        .shift(1)  # 当日除外（ルックアヘッド防止）
        .over(code_col)
        .alias("adv60_trailing")
    )

    return df.drop("_turnover")


def compute_ret_1d(df: pl.DataFrame, code_col: str, price_col: str) -> pl.DataFrame:
    """ret_1d（1日リターン）を計算"""
    if "ret_1d" in df.columns:
        return df

    df = df.with_columns(
        pl.col(price_col).pct_change().over(code_col).alias("ret_1d")
    )

    return df


def detect_price_freezes(
    df: pl.DataFrame,
    code_col: str,
    date_col: str,
    price_col: str,
    min_freeze_days: int = 5,
) -> dict:
    """
    株価フリーズ（同値連続5日以上）を検出

    Returns:
        dict: {
            "total_freeze_sequences": int,  # フリーズシーケンス数
            "total_freeze_days": int,       # フリーズ合計日数
            "affected_codes": int,          # 影響銘柄数
        }
    """
    # 同値連続日数を計算
    df_freeze = df.sort([code_col, date_col]).with_columns(
        [
            # 前日と同じ価格かどうか
            (pl.col(price_col) == pl.col(price_col).shift(1))
            .over(code_col)
            .fill_null(False)
            .alias("_is_same"),
        ]
    )

    # 連続同値のグループID
    df_freeze = df_freeze.with_columns(
        [
            (~pl.col("_is_same")).cum_sum().over(code_col).alias("_freeze_group"),
        ]
    )

    # グループごとの連続日数
    freeze_stats = (
        df_freeze.group_by([code_col, "_freeze_group"])
        .agg(
            [
                pl.count().alias("freeze_days"),
                pl.col("_is_same").first().alias("is_freeze"),
            ]
        )
        .filter(pl.col("is_freeze") & (pl.col("freeze_days") >= min_freeze_days))
    )

    total_sequences = freeze_stats.height
    total_days = freeze_stats["freeze_days"].sum() or 0
    affected_codes = freeze_stats[code_col].n_unique()

    return {
        "total_freeze_sequences": total_sequences,
        "total_freeze_days": int(total_days),
        "affected_codes": affected_codes,
    }


def main():
    parser = argparse.ArgumentParser(description="データ品質ゲート & クリーニング")
    parser.add_argument(
        "--input",
        default=DATASET_RAW,
        help=f"入力parquetファイル（デフォルト: {DATASET_RAW}）",
    )
    parser.add_argument(
        "--output",
        default=DATASET_CLEAN,
        help=f"出力parquetファイル（デフォルト: {DATASET_CLEAN}）",
    )
    parser.add_argument(
        "--min-price", type=float, default=100.0, help="最低価格（円）"
    )
    parser.add_argument(
        "--max-ret-1d", type=float, default=0.15, help="最大1日リターン（絶対値）"
    )
    parser.add_argument(
        "--min-adv", type=float, default=50_000_000, help="最低ADV60（円）"
    )
    parser.add_argument(
        "--freeze-reduction-ratio-max",
        type=float,
        default=0.5,
        help="フリーズ削減率の最大値（post/pre）",
    )
    parser.add_argument(
        "--freeze-abs-max",
        type=int,
        default=100,
        help="フリーズ合計日数の最大値（post）",
    )
    parser.add_argument(
        "--report",
        default=QUALITY_REPORT,
        help=f"品質レポート出力先（デフォルト: {QUALITY_REPORT}）",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("📊 データ品質ゲート & クリーニング")
    print("=" * 70)
    print(f"入力: {args.input}")
    print(f"出力: {args.output}")
    print(f"最低価格: {args.min_price}円")
    print(f"最大リターン: {args.max_ret_1d * 100:.1f}%")
    print(f"最低ADV60: {args.min_adv / 1e8:.1f}億円")
    print("=" * 70)

    # データ読み込み
    df = pl.read_parquet(args.input)
    print(f"✅ データ読み込み: {df.height:,}行 × {df.width}列")

    # 列名検出
    code_col = detect_column(df, CAND_CODE, "code")
    date_col = detect_column(df, CAND_DATE, "date")
    price_col = detect_column(df, CAND_PRICE, "price")

    # volume/turnover（どちらか必須）
    volume_col = None
    turnover_col = None
    for cand in CAND_VOLUME:
        if cand in df.columns:
            volume_col = cand
            break
    for cand in CAND_TURNOVER:
        if cand in df.columns:
            turnover_col = cand
            break

    print("🔍 検出された列:")
    print(f"  - Code: {code_col}")
    print(f"  - Date: {date_col}")
    print(f"  - Price: {price_col}")
    print(f"  - Volume: {volume_col or '(not found)'}")
    print(f"  - Turnover: {turnover_col or '(not found)'}")

    # Pre-filter統計
    print("\n📈 Pre-filter統計:")
    pre_rows = df.height
    pre_codes = df[code_col].n_unique()
    print(f"  - 総行数: {pre_rows:,}")
    print(f"  - 銘柄数: {pre_codes:,}")

    # フリーズ検出（Pre）
    pre_freeze = detect_price_freezes(df, code_col, date_col, price_col)
    print(f"  - フリーズシーケンス: {pre_freeze['total_freeze_sequences']:,}")
    print(f"  - フリーズ日数: {pre_freeze['total_freeze_days']:,}")
    print(f"  - 影響銘柄数: {pre_freeze['affected_codes']:,}")

    # ret_1d計算
    df = compute_ret_1d(df, code_col, price_col)

    # 異常リターン統計（Pre）
    ret_extreme_10_pre = (
        df.filter(pl.col("ret_1d").abs() > 0.10).height / max(df.height, 1)
    )
    ret_extreme_15_pre = (
        df.filter(pl.col("ret_1d").abs() > 0.15).height / max(df.height, 1)
    )
    print(f"  - |ret_1d| > 10%: {ret_extreme_10_pre * 100:.2f}%")
    print(f"  - |ret_1d| > 15%: {ret_extreme_15_pre * 100:.2f}%")

    # ADV60計算
    df = compute_adv60_trailing(df, code_col, date_col, turnover_col, volume_col, price_col)

    # フィルタ適用
    print("\n🔧 フィルタ適用中...")
    df_clean = df.filter(
        (pl.col(price_col) >= args.min_price)
        & (pl.col("ret_1d").abs() <= args.max_ret_1d)
        & (pl.col("adv60_trailing") >= args.min_adv)
    )

    # Post-filter統計
    print("\n📊 Post-filter統計:")
    post_rows = df_clean.height
    post_codes = df_clean[code_col].n_unique()
    print(f"  - 総行数: {post_rows:,} ({(post_rows / max(pre_rows, 1)) * 100:.1f}%)")
    print(f"  - 銘柄数: {post_codes:,} ({(post_codes / max(pre_codes, 1)) * 100:.1f}%)")

    # フリーズ検出（Post）
    post_freeze = detect_price_freezes(df_clean, code_col, date_col, price_col)
    print(f"  - フリーズシーケンス: {post_freeze['total_freeze_sequences']:,}")
    print(f"  - フリーズ日数: {post_freeze['total_freeze_days']:,}")
    print(f"  - 影響銘柄数: {post_freeze['affected_codes']:,}")

    # 異常リターン統計（Post）
    ret_extreme_10_post = (
        df_clean.filter(pl.col("ret_1d").abs() > 0.10).height / max(df_clean.height, 1)
    )
    ret_extreme_15_post = (
        df_clean.filter(pl.col("ret_1d").abs() > 0.15).height / max(df_clean.height, 1)
    )
    print(f"  - |ret_1d| > 10%: {ret_extreme_10_post * 100:.2f}%")
    print(f"  - |ret_1d| > 15%: {ret_extreme_15_post * 100:.2f}%")

    # 品質ゲート判定
    print("\n🚦 品質ゲート判定:")
    issues = []

    # Check 1: ret_1d extreme values
    if ret_extreme_10_post > 0.005:  # 0.5%
        issues.append(
            f"❌ |ret_1d| > 10% の割合が高い: {ret_extreme_10_post * 100:.2f}% (> 0.5%)"
        )
    else:
        print(f"  ✅ |ret_1d| > 10%: {ret_extreme_10_post * 100:.2f}% (< 0.5%)")

    if ret_extreme_15_post > 1e-6:  # ≈ 0%
        issues.append(
            f"❌ |ret_1d| > 15% が存在: {ret_extreme_15_post * 100:.4f}% (> 0%)"
        )
    else:
        print(f"  ✅ |ret_1d| > 15%: {ret_extreme_15_post * 100:.4f}% (≈ 0%)")

    # Check 2: price violations
    price_violations = df_clean.filter(pl.col(price_col) < args.min_price).height
    if price_violations > 0:
        issues.append(f"❌ 価格 < {args.min_price}円が{price_violations}件存在")
    else:
        print(f"  ✅ 価格フィルタ: 全て >= {args.min_price}円")

    # Check 3: freeze reduction
    freeze_ratio = (
        post_freeze["total_freeze_days"] / max(pre_freeze["total_freeze_days"], 1)
    )
    if freeze_ratio > args.freeze_reduction_ratio_max:
        issues.append(
            f"❌ フリーズ削減不足: {freeze_ratio * 100:.1f}% (> {args.freeze_reduction_ratio_max * 100:.0f}%)"
        )
    else:
        print(
            f"  ✅ フリーズ削減: {freeze_ratio * 100:.1f}% (< {args.freeze_reduction_ratio_max * 100:.0f}%)"
        )

    if post_freeze["total_freeze_days"] > args.freeze_abs_max:
        issues.append(
            f"❌ フリーズ日数が多い: {post_freeze['total_freeze_days']}日 (> {args.freeze_abs_max}日)"
        )
    else:
        print(
            f"  ✅ フリーズ日数: {post_freeze['total_freeze_days']}日 (< {args.freeze_abs_max}日)"
        )

    # レポート保存
    report = {
        "input": args.input,
        "output": args.output,
        "filters": {
            "min_price": args.min_price,
            "max_ret_1d": args.max_ret_1d,
            "min_adv": args.min_adv,
        },
        "pre_filter": {
            "rows": pre_rows,
            "codes": pre_codes,
            "ret_extreme_10_pct": ret_extreme_10_pre,
            "ret_extreme_15_pct": ret_extreme_15_pre,
            "freeze_sequences": pre_freeze["total_freeze_sequences"],
            "freeze_days": pre_freeze["total_freeze_days"],
            "affected_codes": pre_freeze["affected_codes"],
        },
        "post_filter": {
            "rows": post_rows,
            "codes": post_codes,
            "ret_extreme_10_pct": ret_extreme_10_post,
            "ret_extreme_15_pct": ret_extreme_15_post,
            "freeze_sequences": post_freeze["total_freeze_sequences"],
            "freeze_days": post_freeze["total_freeze_days"],
            "affected_codes": post_freeze["affected_codes"],
        },
        "quality_gate": {"passed": len(issues) == 0, "issues": issues},
    }

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n📝 品質レポート: {args.report}")

    # 品質ゲート失敗時は終了
    if issues:
        print("\n❌ 品質ゲート失敗:")
        for issue in issues:
            print(f"  {issue}")
        sys.exit(1)

    # データ保存
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_clean.write_parquet(args.output)
    print(f"\n✅ クリーンデータ保存: {args.output}")
    print(f"   {post_rows:,}行 × {df_clean.width}列")
    print("=" * 70)


if __name__ == "__main__":
    main()
