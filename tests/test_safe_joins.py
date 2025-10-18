#!/usr/bin/env python3
"""
安全な結合処理のテストスイート
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
from datetime import datetime, timedelta

import polars as pl

from features.calendar_utils import TradingCalendarUtil
from features.leak_detector import LeakDetector
from features.safe_joiner import SafeJoiner
from features.section_mapper import SectionMapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_calendar():
    """サンプルの営業日カレンダーを作成"""
    dates = []
    holiday_divisions = []

    # 2024年1月のカレンダー
    for day in range(1, 32):
        date = datetime(2024, 1, day).date()

        # 土日は非営業日
        if date.weekday() >= 5:
            holiday_div = 0
        # 1月1-3日は正月休み
        elif day <= 3:
            holiday_div = 0
        else:
            holiday_div = 1  # 営業日

        dates.append(date)
        holiday_divisions.append(holiday_div)

    return pl.DataFrame({"Date": dates, "HolidayDivision": holiday_divisions})


def create_sample_quotes():
    """サンプルのdaily_quotesデータを作成"""
    data = []
    codes = ["1301", "2802", "6501", "7203", "8306"]

    for code in codes:
        for day in range(4, 32):  # 1月4日から
            date = datetime(2024, 1, day).date()
            if date.weekday() >= 5:  # 週末スキップ
                continue

            data.append(
                {
                    "Code": code,
                    "Date": date,
                    "Open": 1000.0,
                    "High": 1010.0,
                    "Low": 990.0,
                    "Close": 1005.0,
                    "Volume": 1000000.0,
                }
            )

    return pl.DataFrame(data)


def create_sample_statements():
    """サンプルの財務諸表データを作成"""
    data = []
    codes = ["1301", "2802", "6501", "7203", "8306"]

    for code in codes:
        # 各銘柄に1つの決算開示
        data.append(
            {
                "LocalCode": code,
                "DisclosedDate": datetime(2024, 1, 10).date(),  # 1月10日開示
                "DisclosedTime": "15:30:00",  # 15:30開示（場後）
                "Revenue": 1000000000,
                "NetIncome": 100000000,
            }
        )

    return pl.DataFrame(data)


def create_sample_trades():
    """サンプルのtrades_specデータを作成"""
    data = []
    sections = ["TSEPrime", "TSEStandard", "TSEGrowth"]

    # 週次データ（金曜日公表）
    fridays = [
        datetime(2024, 1, 5).date(),
        datetime(2024, 1, 12).date(),
        datetime(2024, 1, 19).date(),
        datetime(2024, 1, 26).date(),
    ]

    for friday in fridays:
        for section in sections:
            data.append(
                {
                    "PublishedDate": friday,
                    "Section": section,
                    "ForeignersBalance": 100000000,
                    "ForeignersTotal": 500000000,
                    "IndividualsBalance": -50000000,
                    "IndividualsTotal": 300000000,
                }
            )

    return pl.DataFrame(data)


def create_sample_topix():
    """サンプルのTOPIXデータを作成"""
    data = []

    for day in range(4, 32):
        date = datetime(2024, 1, day).date()
        if date.weekday() >= 5:
            continue

        data.append(
            {
                "Date": date,
                "Close": 2500.0 + day,
                "mkt_return": 0.001,
                "mkt_volatility": 0.02,
                "mkt_trend": 0.5,
            }
        )

    return pl.DataFrame(data)


def create_sample_listed_info():
    """サンプルのlisted_infoデータを作成"""
    data = []

    # 銘柄とMarketCodeのマッピング
    mappings = [
        ("1301", "0101"),  # TSEPrime
        ("2802", "0101"),  # TSEPrime
        ("6501", "0102"),  # TSEStandard
        ("7203", "0101"),  # TSEPrime
        ("8306", "0103"),  # TSEGrowth
    ]

    for code, market_code in mappings:
        data.append(
            {
                "Code": code,
                "MarketCode": market_code,
                "Date": datetime(2024, 1, 1).date(),
            }
        )

    return pl.DataFrame(data)


def test_calendar_utils():
    """営業日カレンダーユーティリティのテスト"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Calendar Utils")
    logger.info("=" * 60)

    calendar_df = create_sample_calendar()
    util = TradingCalendarUtil(calendar_df)

    # 営業日判定テスト
    test_date = datetime(2024, 1, 10).date()
    is_bd = util.is_business_day(test_date)
    logger.info(f"Is {test_date} a business day? {is_bd}")

    # 次営業日テスト
    friday = datetime(2024, 1, 5).date()
    next_bd = util.next_business_day(friday)
    logger.info(f"Next business day after {friday}: {next_bd}")

    # 営業日カレンダー作成
    cal = util.create_business_day_calendar("2024-01-01", "2024-01-31")
    logger.info(f"Business days in January 2024: {len(cal)} days")

    return True


def test_section_mapper():
    """Sectionマッピングのテスト"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Section Mapper")
    logger.info("=" * 60)

    listed_info_df = create_sample_listed_info()
    mapper = SectionMapper()

    # マッピングテーブル作成
    mapping_df = mapper.create_section_mapping(listed_info_df)
    logger.info(f"Created mapping for {len(mapping_df)} stocks")

    # 日次データへの付与
    quotes_df = create_sample_quotes()
    quotes_with_section = mapper.attach_section_to_daily(quotes_df, mapping_df)

    # カバレッジ検証
    stats = mapper.validate_section_coverage(quotes_with_section)
    logger.info(f"Section coverage: {stats['section_coverage']:.1%}")

    for section, info in stats["sections"].items():
        logger.info(f"  {section}: {info['count']} rows, {info['unique_codes']} stocks")

    return stats["section_coverage"] > 0.9


def test_safe_joiner():
    """安全な結合処理のテスト"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Safe Joiner")
    logger.info("=" * 60)

    # データ準備
    quotes_df = create_sample_quotes()
    statements_df = create_sample_statements()
    trades_df = create_sample_trades()
    topix_df = create_sample_topix()
    listed_info_df = create_sample_listed_info()
    calendar_df = create_sample_calendar()

    # ユーティリティ準備
    calendar_util = TradingCalendarUtil(calendar_df)
    section_mapper_obj = SectionMapper()

    # SafeJoiner初期化
    joiner = SafeJoiner(calendar_util, section_mapper_obj)

    # 基盤データ準備
    base_df = joiner.prepare_base_quotes(quotes_df)
    logger.info(f"Base data: {base_df.shape}")

    # Sectionマッピング
    mapping_df = section_mapper_obj.create_section_mapping(listed_info_df)
    base_df = section_mapper_obj.attach_section_to_daily(base_df, mapping_df)

    # 1. Statements結合
    logger.info("\nJoining statements...")
    df_with_stmt = joiner.join_statements_asof(
        base_df, statements_df, use_time_cutoff=True
    )

    # 2. Trades結合
    logger.info("\nJoining trades_spec...")
    df_with_flow = joiner.join_trades_spec_interval(df_with_stmt, trades_df)

    # 3. TOPIX結合
    logger.info("\nJoining TOPIX...")
    df_complete = joiner.join_topix_same_day(df_with_flow, topix_df)

    # 結果サマリー
    logger.info("\n" + "-" * 40)
    logger.info("Join Summary:")
    summary = joiner.get_join_summary()

    for source, stats in summary.items():
        logger.info(f"  {source}:")
        for key, value in stats.items():
            if isinstance(value, float):
                logger.info(f"    {key}: {value:.1%}")
            else:
                logger.info(f"    {key}: {value}")

    return df_complete


def test_leak_detector(df: pl.DataFrame):
    """リーク検査のテスト"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Leak Detector")
    logger.info("=" * 60)

    detector = LeakDetector()

    # 全リーク検査
    results = detector.check_all_leaks(df)

    # レポート生成
    report = detector.generate_leak_report(df)
    print("\n" + report)

    return all(results.values())


def test_coverage_improvement():
    """カバレッジ改善の検証"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Coverage Improvement")
    logger.info("=" * 60)

    # 簡易結合（現在の方法）のシミュレーション
    quotes_df = create_sample_quotes()
    trades_df = create_sample_trades()

    # 現在: 完全一致結合
    trades_simple = trades_df.with_columns(
        [(pl.col("PublishedDate") + timedelta(days=1)).alias("effective_date")]
    )

    current = quotes_df.join(
        trades_simple.select(["effective_date", "ForeignersBalance"]),
        left_on="Date",
        right_on="effective_date",
        how="left",
    )

    current_coverage = current["ForeignersBalance"].is_not_null().sum() / len(current)

    # 改善版: 区間結合（SafeJoinerを使用）
    calendar_df = create_sample_calendar()
    calendar_util = TradingCalendarUtil(calendar_df)
    joiner = SafeJoiner(calendar_util)

    base_df = joiner.prepare_base_quotes(quotes_df)
    improved = joiner.join_trades_spec_interval(base_df, trades_df)

    improved_coverage = improved["flow_foreign_net_ratio"].is_not_null().sum() / len(
        improved
    )

    # 比較結果
    logger.info(f"Current method coverage: {current_coverage:.1%}")
    logger.info(f"Improved method coverage: {improved_coverage:.1%}")

    if current_coverage > 0:
        improvement = (improved_coverage - current_coverage) / current_coverage * 100
    else:
        improvement = float("inf") if improved_coverage > 0 else 0

    logger.info(f"Improvement: {improvement:.1f}%")

    return improved_coverage > current_coverage * 2  # 2倍以上の改善


def main():
    """全テストを実行"""
    logger.info("=" * 60)
    logger.info("SAFE JOINS TEST SUITE")
    logger.info("=" * 60)

    test_results = {}

    # 各テスト実行
    test_results["calendar"] = test_calendar_utils()
    test_results["mapper"] = test_section_mapper()

    # Safe Joiner テスト
    df_complete = test_safe_joiner()
    test_results["joiner"] = df_complete is not None

    # リーク検査
    if df_complete is not None:
        test_results["leak"] = test_leak_detector(df_complete)

    # カバレッジ改善テスト
    test_results["coverage"] = test_coverage_improvement()

    # 結果サマリー
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)

    all_passed = True
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {test_name:15} {status}")
        all_passed = all_passed and passed

    if all_passed:
        logger.info("\n✅ All tests PASSED!")
    else:
        logger.info("\n❌ Some tests FAILED")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
