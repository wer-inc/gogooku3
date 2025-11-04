"""
Leak Detector - データリーケージ検査
時系列データの未来参照を検出
"""

import logging
from datetime import datetime

import polars as pl

logger = logging.getLogger(__name__)


class LeakDetector:
    """
    時系列データリーケージの検出と検証
    """

    def __init__(self):
        self.leak_results = {}
        self.validation_passed = True

    def check_all_leaks(
        self,
        df: pl.DataFrame,
        date_col: str = "Date",
        verbose: bool = True
    ) -> dict[str, bool]:
        """
        全てのリーク検査を実行

        Args:
            df: 検査対象のデータフレーム
            date_col: 日付列名
            verbose: 詳細ログを出力するか

        Returns:
            検査結果の辞書
        """
        logger.info("=" * 60)
        logger.info("Running comprehensive leak detection...")
        logger.info("=" * 60)

        results = {}

        # 1. Statement leaks
        if any(col.startswith("stmt_") for col in df.columns):
            results["statements"] = self.check_statement_leaks(df, date_col, verbose)

        # 2. Flow leaks
        if any(col.startswith("flow_") for col in df.columns):
            results["flows"] = self.check_flow_leaks(df, date_col, verbose)

        # 3. TOPIX leaks
        if any(col.startswith("mkt_") for col in df.columns):
            results["topix"] = self.check_topix_leaks(df, date_col, verbose)

        # 4. Target leaks
        if any("target" in col or "ret" in col for col in df.columns):
            results["targets"] = self.check_target_leaks(df, date_col, verbose)

        # 5. General feature leaks
        results["general"] = self.check_general_leaks(df, date_col, verbose)

        # サマリー
        all_passed = all(results.values())
        self.validation_passed = all_passed

        if all_passed:
            logger.info("✅ All leak checks PASSED")
        else:
            failed_checks = [k for k, v in results.items() if not v]
            logger.error(f"❌ Leak checks FAILED: {failed_checks}")

        logger.info("=" * 60)

        return results

    def check_statement_leaks(
        self,
        df: pl.DataFrame,
        date_col: str = "Date",
        verbose: bool = True
    ) -> bool:
        """
        財務諸表データのリークチェック

        Returns:
            リークがない場合True
        """
        logger.info("Checking statement leaks...")

        # stmt_days_since_statement が負の値を持つかチェック
        if "stmt_days_since_statement" in df.columns:
            negative_days = df.filter(
                pl.col("stmt_days_since_statement") < 0
            )

            if len(negative_days) > 0:
                logger.error(f"  ❌ Found {len(negative_days)} rows with negative days_since_statement")
                if verbose:
                    sample = negative_days.head(5).select([
                        "Code", date_col, "stmt_days_since_statement"
                    ])
                    logger.error(f"  Sample leaks:\n{sample}")
                return False
            else:
                logger.info("  ✅ No statement leaks detected")

        # stmt_imp_statement の整合性チェック
        if "stmt_imp_statement" in df.columns and "stmt_days_since_statement" in df.columns:
            inconsistent = df.filter(
                (pl.col("stmt_imp_statement") == 1) &
                (pl.col("stmt_days_since_statement") != 0)
            )

            if len(inconsistent) > 0:
                logger.warning(f"  ⚠️ Found {len(inconsistent)} rows with inconsistent impulse flags")
                return False

        return True

    def check_flow_leaks(
        self,
        df: pl.DataFrame,
        date_col: str = "Date",
        verbose: bool = True
    ) -> bool:
        """
        フローデータのリークチェック

        Returns:
            リークがない場合True
        """
        logger.info("Checking flow leaks...")

        # days_since_flow が負の値を持つかチェック
        if "days_since_flow" in df.columns:
            negative_days = df.filter(
                (pl.col("days_since_flow") < 0) &
                (pl.col("days_since_flow") != 999)  # 999は欠損値フラグ
            )

            if len(negative_days) > 0:
                logger.error(f"  ❌ Found {len(negative_days)} rows with negative days_since_flow")
                if verbose:
                    sample = negative_days.head(5).select([
                        "Code", date_col, "days_since_flow"
                    ])
                    logger.error(f"  Sample leaks:\n{sample}")
                return False
            else:
                logger.info("  ✅ No flow leaks detected")

        # flow_impulse の整合性チェック
        if "flow_impulse" in df.columns and "days_since_flow" in df.columns:
            inconsistent = df.filter(
                (pl.col("flow_impulse") == 1) &
                (pl.col("days_since_flow") != 0) &
                (pl.col("days_since_flow") != 999)
            )

            if len(inconsistent) > 0:
                logger.warning(f"  ⚠️ Found {len(inconsistent)} rows with inconsistent flow impulse")
                return False

        return True

    def check_topix_leaks(
        self,
        df: pl.DataFrame,
        date_col: str = "Date",
        verbose: bool = True
    ) -> bool:
        """
        TOPIXデータのリークチェック

        Returns:
            リークがない場合True
        """
        logger.info("Checking TOPIX leaks...")

        # TOPIXは同日結合なので、基本的にリークはないはず
        # ただし、先物データや夜間取引データが混入していないかチェック

        mkt_cols = [col for col in df.columns if col.startswith("mkt_")]

        if not mkt_cols:
            logger.info("  No market features to check")
            return True

        # 全てのmkt_特徴量が同一日付で同じ値を持つかチェック
        for mkt_col in mkt_cols[:3]:  # サンプルチェック
            grouped = df.group_by(date_col).agg([
                pl.col(mkt_col).n_unique().alias("unique_values")
            ])

            multi_value_dates = grouped.filter(
                pl.col("unique_values") > 1
            )

            if len(multi_value_dates) > 0:
                logger.warning(f"  ⚠️ {mkt_col} has multiple values on same date")
                if verbose:
                    logger.warning(f"    Affected dates: {len(multi_value_dates)}")

        logger.info("  ✅ TOPIX data consistency check passed")
        return True

    def check_target_leaks(
        self,
        df: pl.DataFrame,
        date_col: str = "Date",
        verbose: bool = True
    ) -> bool:
        """
        ターゲット変数のリークチェック

        Returns:
            リークがない場合True
        """
        logger.info("Checking target leaks...")

        # 将来リターンが特徴量に含まれていないかチェック
        future_cols = [
            col for col in df.columns
            if any(keyword in col.lower() for keyword in ["future", "forward", "target"])
            and not col.startswith("target_")  # target_は正しいターゲット列
        ]

        if future_cols:
            logger.error(f"  ❌ Found potential future information in features: {future_cols}")
            return False

        # リターン計算の整合性チェック
        return_cols = [col for col in df.columns if "return" in col.lower() or "ret" in col.lower()]

        for ret_col in return_cols:
            if ret_col.startswith("target_"):
                continue  # ターゲット列はスキップ

            # 特徴量に未来のリターンが含まれていないか
            if any(future in ret_col for future in ["_forward", "_future", "_ahead"]):
                logger.error(f"  ❌ Feature {ret_col} appears to contain future information")
                return False

        logger.info("  ✅ No target leaks detected")
        return True

    def check_general_leaks(
        self,
        df: pl.DataFrame,
        date_col: str = "Date",
        verbose: bool = True
    ) -> bool:
        """
        一般的なリークパターンのチェック

        Returns:
            リークがない場合True
        """
        logger.info("Checking general leak patterns...")

        issues = []

        # 1. Peek-ahead bias: _lead, _shift(-) などのパターン
        lead_cols = [
            col for col in df.columns
            if "_lead" in col or "_ahead" in col or "_next" in col
        ]

        if lead_cols:
            issues.append(f"Found lead columns: {lead_cols}")

        # 2. 完全に同じ値を持つ列（定数列）
        for col in df.columns:
            if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                if df[col].n_unique() == 1:
                    logger.warning(f"  ⚠️ Column {col} has constant value")

        # 3. 日付列の整合性
        if date_col in df.columns:
            # 未来の日付がないかチェック
            max_date = df[date_col].max()
            today = datetime.now().date()

            if max_date > today:
                issues.append(f"Data contains future dates (max: {max_date})")

        if issues:
            for issue in issues:
                logger.error(f"  ❌ {issue}")
            return False

        logger.info("  ✅ No general leaks detected")
        return True

    def check_train_test_overlap(
        self,
        train_df: pl.DataFrame,
        test_df: pl.DataFrame,
        date_col: str = "Date",
        embargo_days: int = 0
    ) -> bool:
        """
        訓練データとテストデータの時間的重複をチェック

        Args:
            train_df: 訓練データ
            test_df: テストデータ
            date_col: 日付列名
            embargo_days: エンバーゴ期間（日数）

        Returns:
            重複がない場合True
        """
        logger.info("Checking train/test temporal overlap...")

        train_max_date = train_df[date_col].max()
        test_min_date = test_df[date_col].min()

        # エンバーゴを考慮した判定
        from datetime import timedelta
        required_gap = timedelta(days=embargo_days)
        actual_gap = test_min_date - train_max_date

        if actual_gap < required_gap:
            logger.error("  ❌ Train/test overlap detected!")
            logger.error(f"    Train ends: {train_max_date}")
            logger.error(f"    Test starts: {test_min_date}")
            logger.error(f"    Gap: {actual_gap.days} days (required: {embargo_days})")
            return False

        logger.info(f"  ✅ No overlap (gap: {actual_gap.days} days)")
        return True

    def generate_leak_report(
        self,
        df: pl.DataFrame,
        output_path: str | None = None
    ) -> str:
        """
        詳細なリークレポートを生成

        Args:
            df: 検査対象データ
            output_path: レポートの保存先（省略時は文字列で返す）

        Returns:
            レポート文字列
        """
        report = []
        report.append("=" * 60)
        report.append("DATA LEAK DETECTION REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Data shape: {df.shape}")
        report.append("")

        # 各種チェック結果
        results = self.check_all_leaks(df, verbose=False)

        report.append("Check Results:")
        for check_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            report.append(f"  {check_name:15} {status}")

        report.append("")
        report.append("Feature Statistics:")

        # 特徴量グループ別の統計
        feature_groups = {
            "Statements": [c for c in df.columns if c.startswith("stmt_")],
            "Flows": [c for c in df.columns if c.startswith("flow_")],
            "Market": [c for c in df.columns if c.startswith("mkt_")],
            "Cross": [c for c in df.columns if c.startswith("cross_")],
            "Technical": [c for c in df.columns if any(t in c for t in ["ma_", "ema_", "rsi_", "macd_"])]
        }

        for group_name, cols in feature_groups.items():
            if cols:
                null_ratio = sum(df[col].is_null().sum() for col in cols) / (len(cols) * len(df))
                report.append(f"  {group_name:15} {len(cols):3} features, {(1-null_ratio):.1%} coverage")

        report.append("")
        report.append("Recommendations:")

        if not all(results.values()):
            report.append("  ⚠️ Critical issues detected. Please review:")
            if not results.get("statements", True):
                report.append("    - Check statement effective_date calculation")
            if not results.get("flows", True):
                report.append("    - Review flow data interval join logic")
            if not results.get("targets", True):
                report.append("    - Ensure no future information in features")
        else:
            report.append("  ✅ No critical issues detected")

        report.append("=" * 60)

        report_text = "\n".join(report)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report_text)
            logger.info(f"Leak report saved to {output_path}")

        return report_text
