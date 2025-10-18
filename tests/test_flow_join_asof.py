"""
Trade-Spec Flow Join テストスイート
リーク検査とカバレッジ検証を含む受け入れテスト
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from features.flow_joiner import (
    add_flow_features,
    attach_flow_to_quotes,
    build_flow_intervals,
    expand_flow_daily,
)


def create_sample_trades_spec():
    """テスト用のtrades_specデータを作成"""
    return pl.DataFrame(
        {
            "PublishedDate": [
                datetime(2024, 1, 5),  # 金曜日公表
                datetime(2024, 1, 12),  # 次週金曜日
                datetime(2024, 1, 19),  # さらに次週
            ],
            "Section": ["TSEPrime", "TSEPrime", "TSEPrime"],
            "ForeignersBalance": [1000000, -500000, 2000000],
            "ForeignersTotal": [10000000, 8000000, 12000000],
            "IndividualsBalance": [-800000, 600000, -1500000],
            "IndividualsTotal": [5000000, 4000000, 6000000],
            "TotalTotal": [20000000, 18000000, 25000000],
            "ProprietaryBalance": [100000, -50000, 200000],
            "BrokerageBalance": [-50000, 30000, -100000],
            "TrustBanksBalance": [200000, -100000, 300000],
            "InvestmentTrustsBalance": [150000, -80000, 250000],
        }
    )


def create_sample_quotes():
    """テスト用の価格データを作成"""
    dates = []
    codes = ["1301", "1332", "1333"]

    # 2024年1月の営業日を生成
    current = datetime(2024, 1, 4)
    while current <= datetime(2024, 1, 31):
        if current.weekday() < 5:  # 平日のみ
            dates.append(current)
        current += timedelta(days=1)

    records = []
    for date in dates:
        for code in codes:
            records.append(
                {
                    "Date": date,
                    "Code": code,
                    "Section": "TSEPrime",
                    "Close": 1000.0 + hash(code) % 1000,
                }
            )

    return pl.DataFrame(records)


def next_bd_simple(d):
    """翌営業日を返す（週末スキップ）"""
    next_day = d + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    return next_day


class TestFlowJoiner:
    """フロー結合のテストクラス"""

    def test_build_flow_intervals(self):
        """有効区間テーブルの構築テスト"""
        trades_spec = create_sample_trades_spec()
        flow_intervals = build_flow_intervals(trades_spec, next_bd_simple)

        # 区間数の確認
        assert len(flow_intervals) == 3

        # effective_startがPublishedDateの翌営業日であることを確認
        first_start = flow_intervals[0]["effective_start"].item()
        # 2024/1/5(金)の翌営業日は2024/1/8(月)
        expected_start = datetime(2024, 1, 8).date()
        assert (
            first_start == expected_start
        ), f"Expected {expected_start}, got {first_start}"

        # effective_endが次のeffective_startの前日であることを確認
        # （3つのレコードがあるので、2番目のeffective_startを確認）
        if len(flow_intervals) > 1:
            second_start = flow_intervals[1]["effective_start"].item()
            first_end = flow_intervals[0]["effective_end"].item()
            # effective_endは次のeffective_startの前日
            expected_end = second_start - timedelta(days=1)
            assert (
                first_end == expected_end
            ), f"Expected {expected_end}, got {first_end}"

    def test_add_flow_features(self):
        """フロー特徴量生成のテスト"""
        trades_spec = create_sample_trades_spec()
        flow_intervals = build_flow_intervals(trades_spec, next_bd_simple)
        flow_feat = add_flow_features(flow_intervals)

        # 必須特徴量の存在確認
        expected_cols = [
            "foreigners_net_ratio",
            "individuals_net_ratio",
            "foreign_share_activity",
            "breadth_pos",
            "smart_money_idx",
            "flow_shock_flag",
        ]

        for col in expected_cols:
            assert col in flow_feat.columns, f"Missing column: {col}"

        # 値の妥当性確認
        assert flow_feat["foreigners_net_ratio"].is_not_null().all()
        assert (flow_feat["breadth_pos"] >= 0).all()
        assert (flow_feat["breadth_pos"] <= 1).all()

    def test_expand_flow_daily(self):
        """日次展開のテスト"""
        trades_spec = create_sample_trades_spec()
        flow_intervals = build_flow_intervals(trades_spec, next_bd_simple)
        flow_feat = add_flow_features(flow_intervals)

        # 営業日リストの生成
        business_days = []
        current = datetime(2024, 1, 4).date()
        while current <= datetime(2024, 1, 31).date():
            if current.weekday() < 5:
                business_days.append(current)
            current += timedelta(days=1)

        flow_daily = expand_flow_daily(flow_feat, business_days)

        # 各区間が正しく展開されているか確認
        assert len(flow_daily) > 0

        # flow_impulseがdays_since_flow=0の時のみ1であることを確認
        impulse_check = flow_daily.filter(
            (pl.col("flow_impulse") == 1) & (pl.col("days_since_flow") != 0)
        )
        assert (
            len(impulse_check) == 0
        ), "flow_impulse should be 1 only when days_since_flow=0"

    def test_attach_flow_to_quotes(self):
        """価格データへの結合テスト"""
        trades_spec = create_sample_trades_spec()
        quotes = create_sample_quotes()

        # フロー特徴量の準備
        flow_intervals = build_flow_intervals(trades_spec, next_bd_simple)
        flow_feat = add_flow_features(flow_intervals)

        business_days = quotes["Date"].unique().sort()
        flow_daily = expand_flow_daily(flow_feat, business_days)

        # 結合実行
        result = attach_flow_to_quotes(quotes, flow_daily, "Section")

        # 結果の検証
        assert len(result) == len(quotes)
        assert "smart_money_idx" in result.columns
        assert "days_since_flow" in result.columns
        assert "is_flow_valid" in result.columns

    def test_no_future_leak(self):
        """未来参照（リーク）がないことを確認"""
        trades_spec = create_sample_trades_spec()
        quotes = create_sample_quotes()

        # フロー特徴量の準備と結合
        flow_intervals = build_flow_intervals(trades_spec, next_bd_simple)
        flow_feat = add_flow_features(flow_intervals)
        business_days = quotes["Date"].unique().sort()
        flow_daily = expand_flow_daily(flow_feat, business_days)
        result = attach_flow_to_quotes(quotes, flow_daily, "Section")

        # days_since_flowが負の値を持たないことを確認
        negative_days = result.filter(
            pl.col("days_since_flow") < -1
        )  # -1は未経験を示す
        assert (
            len(negative_days) == 0
        ), f"Found {len(negative_days)} records with future leak"

    def test_coverage_metrics(self):
        """カバレッジメトリクスのテスト"""
        trades_spec = create_sample_trades_spec()
        quotes = create_sample_quotes()

        # フロー特徴量の準備と結合
        flow_intervals = build_flow_intervals(trades_spec, next_bd_simple)
        flow_feat = add_flow_features(flow_intervals)
        business_days = quotes["Date"].unique().sort()
        flow_daily = expand_flow_daily(flow_feat, business_days)
        result = attach_flow_to_quotes(quotes, flow_daily, "Section")

        # カバレッジ計算
        coverage = result["is_flow_valid"].sum() / len(result)

        # 期待されるカバレッジ（trades_specが3週分あるため、大部分がカバーされるはず）
        assert coverage > 0.5, f"Coverage too low: {coverage:.1%}"

        # (Code, Date)の一意性確認
        unique_count = result.select(["Code", "Date"]).n_unique()
        assert unique_count == len(result), "Found duplicate (Code, Date) pairs"


def test_integration_with_real_structure():
    """実際のデータ構造を模したintegrationテスト"""
    # より現実的なデータ構造でテスト
    trades_spec = pl.DataFrame(
        {
            "PublishedDate": [
                datetime(2024, 1, 5) + timedelta(weeks=i) for i in range(52)
            ],
            "Section": ["TSEPrime"] * 52,
            "ForeignersBalance": [1000000 * (1 + i % 3 - 1) for i in range(52)],
            "ForeignersTotal": [10000000] * 52,
            "IndividualsBalance": [-800000 * (1 + i % 3 - 1) for i in range(52)],
            "IndividualsTotal": [5000000] * 52,
            "TotalTotal": [20000000] * 52,
            "ProprietaryBalance": [100000] * 52,
            "BrokerageBalance": [-50000] * 52,
            "TrustBanksBalance": [200000] * 52,
            "InvestmentTrustsBalance": [150000] * 52,
        }
    )

    # 1年分の価格データ
    dates = []
    current = datetime(2024, 1, 1).date()
    while current <= datetime(2024, 12, 31).date():
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)

    quotes_records = []
    for date in dates:
        for code in range(1301, 1401):  # 100銘柄
            quotes_records.append(
                {
                    "Date": date,
                    "Code": str(code),
                    "Section": "TSEPrime" if code < 1350 else "TSEStandard",
                    "Close": 1000.0 + code,
                }
            )

    quotes = pl.DataFrame(quotes_records)

    # フロー結合実行
    flow_intervals = build_flow_intervals(trades_spec, next_bd_simple)
    flow_feat = add_flow_features(flow_intervals)
    flow_daily = expand_flow_daily(flow_feat, dates)
    result = attach_flow_to_quotes(quotes, flow_daily, "Section")

    # 総合的な検証
    assert len(result) == len(quotes), "Record count mismatch"

    # TSEPrimeのカバレッジが高いことを確認
    prime_coverage = (
        result.filter(pl.col("Section") == "TSEPrime")["is_flow_valid"].sum()
        / result.filter(pl.col("Section") == "TSEPrime").height
    )
    assert prime_coverage > 0.9, f"TSEPrime coverage too low: {prime_coverage:.1%}"

    # リーク検査
    assert (result["days_since_flow"] >= -1).all(), "Found future leak"

    print("✅ All integration tests passed!")
    print(f"  - Total records: {len(result):,}")
    print(f"  - Flow coverage: {result['is_flow_valid'].sum() / len(result):.1%}")
    print(f"  - TSEPrime coverage: {prime_coverage:.1%}")


if __name__ == "__main__":
    # 基本テストの実行
    test_suite = TestFlowJoiner()
    test_suite.test_build_flow_intervals()
    test_suite.test_add_flow_features()
    test_suite.test_expand_flow_daily()
    test_suite.test_attach_flow_to_quotes()
    test_suite.test_no_future_leak()
    test_suite.test_coverage_metrics()

    # 統合テスト
    test_integration_with_real_structure()

    print("\n✅ All tests passed successfully!")
