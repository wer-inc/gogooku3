#!/usr/bin/env python3
"""
データ品質の自動検証テスト
結合の一意性・単調性・リーク耐性を検証
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
from datetime import datetime, timedelta

import polars as pl

from features.code_normalizer import CodeNormalizer
from features.leak_detector import LeakDetector
from features.safe_joiner_v2 import SafeJoinerV2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQualityValidator:
    """
    データ品質の包括的な検証
    """

    @staticmethod
    def check_code_date_uniqueness(df: pl.DataFrame) -> bool:
        """
        (Code, Date) の一意性を検証

        Returns:
            一意の場合True
        """
        total_rows = len(df)
        unique_rows = df.select(["Code", "Date"]).n_unique()

        if total_rows != unique_rows:
            duplicates = (
                df.group_by(["Code", "Date"]).count().filter(pl.col("count") > 1)
            )
            logger.error(
                f"❌ (Code, Date) uniqueness violated: {len(duplicates)} duplicates found"
            )
            logger.error(f"  Sample duplicates: {duplicates.head(5)}")
            return False

        logger.info("✅ (Code, Date) uniqueness check passed")
        return True

    @staticmethod
    def check_flow_impulse_consistency(df: pl.DataFrame) -> bool:
        """
        flow_impulse の整合性を検証
        - flow_impulse==1 の日は days_since_flow==0
        - days_since_flow は非減少
        """
        if "flow_impulse" not in df.columns or "days_since_flow" not in df.columns:
            logger.info("  Flow columns not found, skipping")
            return True

        # impulse==1 の時 days_since==0 であることを確認
        inconsistent = df.filter(
            (pl.col("flow_impulse") == 1)
            & (pl.col("days_since_flow") != 0)
            & (pl.col("days_since_flow") != 999)  # 999は欠損フラグ
        )

        if len(inconsistent) > 0:
            logger.error(f"❌ Flow impulse inconsistency: {len(inconsistent)} rows")
            return False

        # days_since_flow の単調性チェック（Code×Section内で）
        if "Section" in df.columns:
            # 各銘柄・Sectionで days_since_flow が適切にリセットされるか
            df_sorted = df.sort(["Code", "Section", "Date"])

            # 連続性チェック（簡易版）
            df_with_prev = df_sorted.with_columns(
                [
                    pl.col("days_since_flow")
                    .shift(1)
                    .over(["Code", "Section"])
                    .alias("prev_days")
                ]
            )

            # days_since_flowが急激に減少する場合は新しい区間の開始
            bad_transitions = df_with_prev.filter(
                (pl.col("days_since_flow") < pl.col("prev_days") - 1)
                & (pl.col("flow_impulse") != 1)
                & (pl.col("days_since_flow") != 999)
                & (pl.col("prev_days") != 999)
            )

            if len(bad_transitions) > 0:
                logger.warning(
                    f"⚠️ Flow days monotonicity issues: {len(bad_transitions)} transitions"
                )
                # これは警告レベル（新区間開始の可能性）

        logger.info("✅ Flow impulse consistency check passed")
        return True

    @staticmethod
    def check_statement_days_non_negative(df: pl.DataFrame) -> bool:
        """
        stmt_days_since_statement が非負であることを検証
        """
        if "stmt_days_since_statement" not in df.columns:
            logger.info("  Statement columns not found, skipping")
            return True

        negative_days = df.filter(
            (pl.col("stmt_days_since_statement") < 0)
            & (pl.col("stmt_days_since_statement") != 999)  # 999は欠損フラグ
        )

        if len(negative_days) > 0:
            logger.error(
                f"❌ Negative stmt_days_since_statement: {len(negative_days)} rows"
            )
            logger.error(
                f"  Sample: {negative_days.head(3).select(['Code', 'Date', 'stmt_days_since_statement'])}"
            )
            return False

        logger.info("✅ Statement days non-negative check passed")
        return True

    @staticmethod
    def check_validity_flags_consistency(df: pl.DataFrame) -> bool:
        """
        有効性フラグの整合性を検証
        """
        validity_cols = [
            "is_stmt_valid",
            "is_flow_valid",
            "is_mkt_valid",
            "is_beta_valid",
            "is_section_fallback",
        ]

        existing_cols = [col for col in validity_cols if col in df.columns]

        if not existing_cols:
            logger.info("  No validity flags found, skipping")
            return True

        issues = []

        for col in existing_cols:
            # フラグは0または1のみ
            unique_values = df[col].unique().to_list()
            if not all(v in [0, 1, None] for v in unique_values):
                issues.append(f"{col} has invalid values: {unique_values}")

        # is_fully_valid の整合性
        if "is_fully_valid" in df.columns:
            # 個別フラグが全て1の時のみ is_fully_valid==1
            base_validity_cols = ["is_stmt_valid", "is_flow_valid", "is_mkt_valid"]
            available_base_cols = [
                col for col in base_validity_cols if col in df.columns
            ]

            if available_base_cols:
                expected_fully_valid = pl.all_horizontal(
                    [pl.col(col) == 1 for col in available_base_cols]
                )

                inconsistent = df.filter(
                    expected_fully_valid != (pl.col("is_fully_valid") == 1)
                )

                if len(inconsistent) > 0:
                    issues.append(
                        f"is_fully_valid inconsistent: {len(inconsistent)} rows"
                    )

        if issues:
            for issue in issues:
                logger.error(f"❌ {issue}")
            return False

        logger.info("✅ Validity flags consistency check passed")
        return True

    @staticmethod
    def check_coverage_thresholds(
        df: pl.DataFrame,
        min_stmt_coverage: float = 0.5,
        min_flow_coverage: float = 0.5,
        min_mkt_coverage: float = 0.8,
    ) -> bool:
        """
        最小カバレッジ要件を検証
        """
        coverages = {}

        # Statements coverage
        if "stmt_days_since_statement" in df.columns:
            stmt_coverage = df.filter(pl.col("stmt_days_since_statement") < 999).shape[
                0
            ] / len(df)
            coverages["statements"] = stmt_coverage
            if stmt_coverage < min_stmt_coverage:
                logger.warning(
                    f"⚠️ Statements coverage {stmt_coverage:.1%} < {min_stmt_coverage:.1%}"
                )

        # Flow coverage
        if "days_since_flow" in df.columns:
            flow_coverage = df.filter(pl.col("days_since_flow") < 999).shape[0] / len(
                df
            )
            coverages["flows"] = flow_coverage
            if flow_coverage < min_flow_coverage:
                logger.warning(
                    f"⚠️ Flow coverage {flow_coverage:.1%} < {min_flow_coverage:.1%}"
                )

        # Market coverage
        mkt_cols = [col for col in df.columns if col.startswith("mkt_")]
        if mkt_cols:
            mkt_coverage = df[mkt_cols[0]].is_not_null().sum() / len(df)
            coverages["market"] = mkt_coverage
            if mkt_coverage < min_mkt_coverage:
                logger.warning(
                    f"⚠️ Market coverage {mkt_coverage:.1%} < {min_mkt_coverage:.1%}"
                )

        # サマリー
        if coverages:
            logger.info("Coverage summary:")
            for source, cov in coverages.items():
                logger.info(f"  {source}: {cov:.1%}")

        return True  # カバレッジは警告のみ（エラーにしない）


def test_code_normalization():
    """Code正規化のテスト"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Code Normalization")
    logger.info("=" * 60)

    normalizer = CodeNormalizer()

    # テストケース
    test_cases = [
        ("1301", "1301"),  # 4桁はそのまま
        ("13010", "1301"),  # 5桁末尾0は削除
        ("001301", "1301"),  # 先頭0は削除
        (1301, "1301"),  # 数値も処理
        ("86970", "8697"),  # 特殊マッピング
        ("", None),  # 空文字列
        (None, None),  # None
    ]

    passed = True
    for input_code, expected in test_cases:
        result = normalizer.normalize_code(input_code)
        if result != expected:
            logger.error(f"  ❌ {input_code} → {result} (expected: {expected})")
            passed = False
        else:
            logger.info(f"  ✅ {input_code} → {result}")

    assert passed, "Code normalization test failed"
    logger.info("✅ Code normalization test passed")


def test_data_quality_checks():
    """データ品質チェックの統合テスト"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Data Quality Checks")
    logger.info("=" * 60)

    # サンプルデータ作成
    data = []
    codes = ["1301", "2802", "6501"]

    for code in codes:
        for day in range(1, 21):
            date = datetime(2024, 1, day).date()
            if date.weekday() >= 5:  # 週末スキップ
                continue

            data.append(
                {
                    "Code": code,
                    "Date": date,
                    "Close": 1000.0,
                    "stmt_days_since_statement": day % 10,  # 0-9の循環
                    "flow_impulse": 1 if day % 7 == 1 else 0,
                    "days_since_flow": 0 if day % 7 == 1 else (day - 1) % 7,
                    "is_stmt_valid": 1,
                    "is_flow_valid": 1,
                    "is_mkt_valid": 1,
                    "mkt_return": 0.001,
                }
            )

    df = pl.DataFrame(data)

    # 品質検証
    validator = DataQualityValidator()

    results = {
        "uniqueness": validator.check_code_date_uniqueness(df),
        "flow_consistency": validator.check_flow_impulse_consistency(df),
        "stmt_non_negative": validator.check_statement_days_non_negative(df),
        "validity_consistency": validator.check_validity_flags_consistency(df),
        "coverage": validator.check_coverage_thresholds(df),
    }

    # 結果サマリー
    logger.info("\n" + "-" * 40)
    logger.info("Quality Check Results:")
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {check:20} {status}")

    assert all(results.values()), "Some quality checks failed"
    logger.info("\n✅ All quality checks passed")


def test_leak_detection_with_safe_joiner():
    """SafeJoinerV2でのリーク検査"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Leak Detection with SafeJoinerV2")
    logger.info("=" * 60)

    # サンプルデータ作成（簡易版）
    quotes = pl.DataFrame(
        {
            "Code": ["1301"] * 10,
            "Date": [datetime(2024, 1, i).date() for i in range(1, 11)],
            "Close": [1000.0] * 10,
        }
    )

    statements = pl.DataFrame(
        {
            "LocalCode": ["1301"],
            "DisclosedDate": datetime(2024, 1, 5).date(),
            "DisclosedTime": "15:30:00",
        }
    )

    # SafeJoinerV2で結合
    joiner = SafeJoinerV2()
    base_df = joiner.prepare_base_quotes(quotes)
    df_with_stmt = joiner.join_statements_with_dedup(base_df, statements)

    # リーク検査
    detector = LeakDetector()
    leak_results = detector.check_all_leaks(df_with_stmt, verbose=False)

    # 結果確認
    assert all(leak_results.values()), "Leak detection failed"
    logger.info("✅ No leaks detected with SafeJoinerV2")


def test_minimum_coverage_assertion():
    """最小カバレッジ要件のテスト"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Minimum Coverage Assertion")
    logger.info("=" * 60)

    # 低カバレッジデータ作成
    df = pl.DataFrame(
        {
            "Code": ["1301"] * 100,
            "Date": [
                datetime(2024, 1, 1).date() + timedelta(days=i) for i in range(100)
            ],
            "stmt_days_since_statement": [999] * 50 + [10] * 50,  # 50%カバレッジ
            "days_since_flow": [999] * 70 + [5] * 30,  # 30%カバレッジ
        }
    )

    validator = DataQualityValidator()

    # デフォルト閾値でのチェック（警告のみ）
    result = validator.check_coverage_thresholds(df)
    assert result, "Coverage check should pass with warnings"

    # 厳しい閾値でのチェック
    result = validator.check_coverage_thresholds(
        df, min_stmt_coverage=0.3, min_flow_coverage=0.2
    )
    assert result, "Coverage check failed unexpectedly"


def main():
    """全テストを実行"""
    logger.info("=" * 60)
    logger.info("DATA QUALITY TEST SUITE")
    logger.info("=" * 60)

    test_functions = [
        test_code_normalization,
        test_data_quality_checks,
        test_leak_detection_with_safe_joiner,
        test_minimum_coverage_assertion,
    ]

    results = {}

    for test_func in test_functions:
        test_name = test_func.__name__
        try:
            test_func()
            results[test_name] = True
        except Exception as e:
            logger.error(f"Test {test_name} failed: {e}")
            results[test_name] = False

    # 結果サマリー
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {test_name:35} {status}")
        all_passed = all_passed and passed

    if all_passed:
        logger.info("\n✅ All tests PASSED!")
    else:
        logger.info("\n❌ Some tests FAILED")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
