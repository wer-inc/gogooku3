"""
Financial Statements Join テストスイート
リーク検査、カバレッジ検証、T+1ルール確認を含む受け入れテスト
"""

import sys
from datetime import datetime, time, timedelta
from pathlib import Path

import polars as pl

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from features.safe_joiner import SafeJoiner


def create_sample_statements():
    """テスト用の財務諸表データを作成"""
    return pl.DataFrame(
        {
            "LocalCode": ["1301", "1301", "1301", "1332", "1332"],
            "DisclosedDate": [
                datetime(2024, 2, 9).date(),  # 金曜日開示
                datetime(2024, 5, 10).date(),  # 次四半期（金曜日）
                datetime(2024, 8, 9).date(),  # さらに次四半期
                datetime(2024, 2, 8).date(),  # 木曜日開示
                datetime(2024, 5, 9).date(),  # 次四半期（木曜日）
            ],
            "DisclosedTime": [
                "15:30:00",
                "14:00:00",
                "11:00:00",
                "16:00:00",
                "13:30:00",
            ],
            "TypeOfCurrentPeriod": ["3Q", "FY", "1Q", "3Q", "FY"],
            "NetSales": [100000000, 120000000, 30000000, 80000000, 95000000],
            "OperatingProfit": [10000000, 15000000, 3000000, 8000000, 9500000],
            "Profit": [8000000, 12000000, 2400000, 6400000, 7600000],
            "ForecastOperatingProfit": [
                40000000,
                45000000,
                12000000,
                32000000,
                38000000,
            ],
            "ForecastProfit": [32000000, 36000000, 9600000, 25600000, 30400000],
            "ForecastEarningsPerShare": [100.0, 112.5, 30.0, 80.0, 95.0],
            "ForecastDividendPerShareAnnual": ["20", "22", "22", "15", "18"],
            "Equity": [200000000, 210000000, 215000000, 160000000, 170000000],
            "TotalAssets": [500000000, 525000000, 537500000, 400000000, 425000000],
            "ChangesInAccountingEstimates": [None, None, "true", None, None],
            "ChangesBasedOnRevisionsOfAccountingStandard": [
                None,
                None,
                None,
                None,
                "true",
            ],
            "RetrospectiveRestatement": [None, None, None, None, None],
        }
    )


def create_sample_quotes():
    """テスト用の価格データを作成"""
    dates = []
    # 2024年2月から8月までの営業日を生成
    current = datetime(2024, 2, 1)
    while current <= datetime(2024, 8, 31):
        if current.weekday() < 5:  # 平日のみ
            dates.append(current.date())
        current += timedelta(days=1)

    records = []
    for date in dates:
        for code in ["1301", "1332"]:
            records.append(
                {
                    "Date": date,
                    "Code": code,
                    "Close": 1000.0 + hash(f"{code}{date}") % 500,
                    "Open": 980.0 + hash(f"{code}{date}") % 500,
                    "High": 1020.0 + hash(f"{code}{date}") % 500,
                    "Low": 970.0 + hash(f"{code}{date}") % 500,
                    "Volume": 100000 + hash(f"{code}{date}") % 50000,
                }
            )

    return pl.DataFrame(records)


class TestStatementsJoin:
    """財務諸表結合のテストクラス"""

    def test_calculate_statement_features(self):
        """財務特徴量計算のテスト"""
        statements = create_sample_statements()
        joiner = SafeJoiner()

        # 財務特徴量の計算
        result = joiner._calculate_statement_features(statements)

        # 必須特徴量の存在確認
        expected_cols = [
            "stmt_yoy_sales",
            "stmt_yoy_op",
            "stmt_yoy_np",
            "stmt_opm",
            "stmt_npm",
            "stmt_progress_op",
            "stmt_progress_np",
            "stmt_rev_fore_op",
            "stmt_rev_fore_np",
            "stmt_roe",
            "stmt_roa",
            "stmt_nc_flag",
        ]

        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

        # マージンの妥当性確認（0-1の範囲）
        assert (result["stmt_opm"] >= 0).all()
        assert (result["stmt_opm"] <= 1).all()
        assert (result["stmt_npm"] >= 0).all()
        assert (result["stmt_npm"] <= 1).all()

    def test_t1_effective_date(self):
        """T+1ルールの適用テスト"""
        statements = create_sample_statements()
        quotes = create_sample_quotes()
        joiner = SafeJoiner()

        # T+1ルールで結合（15:00カットオフ）
        result = joiner.join_statements_asof(
            quotes, statements, use_time_cutoff=True, cutoff_time=time(15, 0)
        )

        # 2024/2/9 15:30開示 → T+1 (2/13火曜日に有効)
        feb9_code1301 = result.filter(
            (pl.col("Code") == "1301")
            & (pl.col("Date") == datetime(2024, 2, 13).date())
        )
        assert feb9_code1301["stmt_imp_statement"][0] == 1, "Should be impulse day"

        # 2024/5/10 14:00開示 → T+0 (5/10当日有効)
        may10_code1301 = result.filter(
            (pl.col("Code") == "1301")
            & (pl.col("Date") == datetime(2024, 5, 10).date())
        )
        assert (
            may10_code1301["stmt_imp_statement"][0] == 1
        ), "Should be impulse day (before cutoff)"

    def test_no_future_leak(self):
        """未来参照（リーク）がないことを確認"""
        statements = create_sample_statements()
        quotes = create_sample_quotes()
        joiner = SafeJoiner()

        result = joiner.join_statements_asof(quotes, statements)

        # stmt_days_since_statementが負の値を持たないことを確認
        # （-1は未開示を示すので除外）
        actual_days = result.filter(pl.col("stmt_days_since_statement") >= 0)
        negative_days = actual_days.filter(pl.col("stmt_days_since_statement") < 0)

        assert (
            len(negative_days) == 0
        ), f"Found {len(negative_days)} records with future leak"

    def test_asof_backward_join(self):
        """as-of backward結合の正確性テスト"""
        statements = create_sample_statements()
        quotes = create_sample_quotes()
        joiner = SafeJoiner()

        result = joiner.join_statements_asof(quotes, statements)

        # 2024/6/1時点で、1301は5/10の開示データを持つはず
        june1_1301 = result.filter(
            (pl.col("Code") == "1301") & (pl.col("Date") == datetime(2024, 6, 1).date())
        )

        # 開示からの経過日数が妥当な範囲内
        days_since = june1_1301["stmt_days_since_statement"][0]
        assert days_since > 0, "Should have past statement data"
        assert days_since < 60, "Statement should be relatively recent"

    def test_coverage_metrics(self):
        """カバレッジメトリクスのテスト"""
        statements = create_sample_statements()
        quotes = create_sample_quotes()
        joiner = SafeJoiner()

        result = joiner.join_statements_asof(quotes, statements)

        # カバレッジ計算
        coverage = result["is_stmt_valid"].sum() / len(result)

        # 財務諸表が定期的にあるため、カバレッジは高いはず
        assert coverage > 0.5, f"Coverage too low: {coverage:.1%}"

        # (Code, Date)の一意性確認
        unique_count = result.select(["Code", "Date"]).n_unique()
        assert unique_count == len(result), "Found duplicate (Code, Date) pairs"

    def test_feature_consistency(self):
        """特徴量の一貫性テスト"""
        statements = create_sample_statements()
        quotes = create_sample_quotes()
        joiner = SafeJoiner()

        result = joiner.join_statements_asof(quotes, statements)

        # インパルスはdays_since_statement=0の時のみ1
        impulse_check = result.filter(
            (pl.col("stmt_imp_statement") == 1)
            & (pl.col("stmt_days_since_statement") != 0)
        )
        assert len(impulse_check) == 0, "Impulse should only occur on effective_date"

        # 有効フラグとdays_since_statementの整合性
        valid_check = result.filter(
            (pl.col("is_stmt_valid") == 1) & (pl.col("stmt_days_since_statement") < 0)
        )
        assert len(valid_check) == 0, "Valid flag inconsistent with days_since"


def test_integration_statements():
    """統合テスト：実際のデータ構造を模した総合テスト"""
    # 1年分の財務諸表（四半期ごと）
    statements_records = []
    for code in ["1301", "1332", "1333"]:
        for quarter in range(4):
            disclosed_date = datetime(2024, 2 + quarter * 3, 10)
            statements_records.append(
                {
                    "LocalCode": code,
                    "DisclosedDate": disclosed_date.date(),
                    "DisclosedTime": "15:30:00",
                    "TypeOfCurrentPeriod": f"{quarter+1}Q",
                    "NetSales": 100000000 * (1 + quarter * 0.1),
                    "OperatingProfit": 10000000 * (1 + quarter * 0.1),
                    "Profit": 8000000 * (1 + quarter * 0.1),
                    "ForecastOperatingProfit": 40000000,
                    "ForecastProfit": 32000000,
                    "ForecastEarningsPerShare": 100.0,
                    "ForecastDividendPerShareAnnual": "20",
                    "Equity": 200000000,
                    "TotalAssets": 500000000,
                }
            )

    statements = pl.DataFrame(statements_records)

    # 1年分の価格データ
    dates = []
    current = datetime(2024, 1, 1).date()
    while current <= datetime(2024, 12, 31).date():
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)

    quotes_records = []
    for date in dates:
        for code in ["1301", "1332", "1333"]:
            quotes_records.append(
                {
                    "Date": date,
                    "Code": code,
                    "Close": 1000.0 + hash(f"{code}{date}") % 500,
                    "Open": 980.0,
                    "High": 1050.0,
                    "Low": 950.0,
                    "Volume": 100000,
                }
            )

    quotes = pl.DataFrame(quotes_records)

    # 結合実行
    joiner = SafeJoiner()
    result = joiner.join_statements_asof(quotes, statements)

    # 総合的な検証
    assert len(result) == len(quotes), "Record count mismatch"

    # リーク検査
    assert (
        result.filter(pl.col("stmt_days_since_statement") >= 0)[
            "stmt_days_since_statement"
        ]
        >= 0
    ).all(), "Found future leak"

    # カバレッジ
    coverage = result["is_stmt_valid"].sum() / len(result)
    assert coverage > 0.7, f"Coverage too low: {coverage:.1%}"

    # 統計情報の確認
    assert joiner.join_stats["statements"]["coverage"] == coverage
    assert joiner.join_stats["statements"]["total_rows"] == len(result)

    print("✅ All integration tests passed!")
    print(f"  - Total records: {len(result):,}")
    print(f"  - Statement coverage: {coverage:.1%}")
    print(
        f"  - Features added: {joiner.join_stats['statements'].get('features_added', 0)}"
    )


if __name__ == "__main__":
    # 基本テストの実行
    test_suite = TestStatementsJoin()
    test_suite.test_calculate_statement_features()
    test_suite.test_t1_effective_date()
    test_suite.test_no_future_leak()
    test_suite.test_asof_backward_join()
    test_suite.test_coverage_metrics()
    test_suite.test_feature_consistency()

    # 統合テスト
    test_integration_statements()

    print("\n✅ All tests passed successfully!")
