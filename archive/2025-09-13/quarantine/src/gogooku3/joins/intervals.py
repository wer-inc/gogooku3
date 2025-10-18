"""
時系列データの高度な結合ユーティリティ
interval join, as-of join, business day handling
"""

from datetime import datetime, timedelta

import jpholiday
import pandas as pd
import polars as pl


class BusinessCalendar:
    """JPX営業日カレンダー管理"""

    @staticmethod
    def get_business_days(start_date: str, end_date: str) -> pl.DataFrame:
        """
        JPX営業日リストを取得

        Args:
            start_date: 開始日 (YYYY-MM-DD)
            end_date: 終了日 (YYYY-MM-DD)

        Returns:
            営業日のDataFrame（Date列とbidx列）
        """
        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq="D")

        # Filter out weekends and Japanese holidays
        business_days = []
        for date in dates:
            if date.weekday() < 5 and not jpholiday.is_holiday(date):
                # Additional JPX-specific holidays (year-end, etc.)
                if not BusinessCalendar._is_jpx_holiday(date):
                    business_days.append(date.date())

        # Create DataFrame with business day index
        df = pl.DataFrame({"Date": business_days}).with_row_count("bidx")

        return df

    @staticmethod
    def _is_jpx_holiday(date: pd.Timestamp) -> bool:
        """JPX特有の休場日チェック"""
        # Year-end holidays (Dec 31 - Jan 3)
        if date.month == 12 and date.day == 31:
            return True
        if date.month == 1 and date.day <= 3:
            return True
        return False

    @staticmethod
    def next_business_day(
        date: datetime.date, calendar_df: pl.DataFrame
    ) -> datetime.date:
        """次営業日を取得"""
        future_days = calendar_df.filter(pl.col("Date") > date)
        if len(future_days) > 0:
            return future_days["Date"][0]
        else:
            # Fallback: add 1 day (should not happen with proper calendar)
            return date + timedelta(days=1)

    @staticmethod
    def add_business_days(df: pl.DataFrame, calendar_df: pl.DataFrame) -> pl.DataFrame:
        """DataFrameに営業日インデックスを追加"""
        return df.join(calendar_df, on="Date", how="left")


class IntervalJoiner:
    """区間結合ユーティリティ"""

    @staticmethod
    def interval_join_by_asof(
        left: pl.DataFrame,
        right: pl.DataFrame,
        key: str,
        date_col: str = "Date",
        start: str = "valid_from",
        end: str = "valid_to",
        suffix: str = "_right",
    ) -> pl.DataFrame:
        """
        区間結合（as-of + end filter方式）

        Args:
            left: 左側DataFrame (key + date_col)
            right: 右側DataFrame (key + start + end + features)
            key: 結合キー（例: "Code", "Section"）
            date_col: 日付列名
            start: 開始日列名
            end: 終了日列名
            suffix: 右側列のサフィックス

        Returns:
            結合後のDataFrame
        """
        # Sort for as-of join
        left_sorted = left.sort([key, date_col])
        right_sorted = right.sort([key, start])

        # As-of join (backward strategy)
        joined = left_sorted.join_asof(
            right_sorted,
            left_on=date_col,
            right_on=start,
            by=key,
            strategy="backward",
            suffix=suffix,
        )

        # Filter by end date
        if end in joined.columns or f"{end}{suffix}" in joined.columns:
            end_col = f"{end}{suffix}" if f"{end}{suffix}" in joined.columns else end
            joined = joined.filter(pl.col(date_col) <= pl.col(end_col))

        return joined

    @staticmethod
    def validate_intervals(
        df: pl.DataFrame, key: str, start: str = "valid_from", end: str = "valid_to"
    ) -> dict[str, any]:
        """
        区間の妥当性検証

        Returns:
            検証結果の辞書
        """
        results = {
            "has_gaps": False,
            "has_overlaps": False,
            "invalid_ranges": 0,
            "details": [],
        }

        # Check invalid ranges (start > end)
        invalid = df.filter(pl.col(start) > pl.col(end))
        results["invalid_ranges"] = len(invalid)

        # Check for gaps and overlaps per key
        for key_val in df[key].unique():
            key_df = df.filter(pl.col(key) == key_val).sort(start)

            if len(key_df) <= 1:
                continue

            for i in range(len(key_df) - 1):
                curr_end = key_df[end][i]
                next_start = key_df[start][i + 1]

                # Check gap
                if curr_end < next_start - timedelta(days=1):
                    results["has_gaps"] = True
                    results["details"].append(
                        {
                            "type": "gap",
                            "key": key_val,
                            "between": (curr_end, next_start),
                        }
                    )

                # Check overlap
                if curr_end >= next_start:
                    results["has_overlaps"] = True
                    results["details"].append(
                        {
                            "type": "overlap",
                            "key": key_val,
                            "between": (curr_end, next_start),
                        }
                    )

        return results


class AsOfJoiner:
    """As-of結合ユーティリティ（財務データ用）"""

    @staticmethod
    def asof_join_by_code(
        quotes: pl.DataFrame,
        statements: pl.DataFrame,
        date_col: str = "Date",
        effective_col: str = "effective_date",
        code_col: str = "Code",
    ) -> pl.DataFrame:
        """
        財務データのas-of結合

        Args:
            quotes: 価格データ
            statements: 財務データ（T+1処理済み）
            date_col: 価格データの日付列
            effective_col: 財務データの有効日列
            code_col: 銘柄コード列

        Returns:
            結合後のDataFrame
        """
        # Sort for as-of join
        quotes_sorted = quotes.sort([code_col, date_col])
        statements_sorted = statements.sort([code_col, effective_col])

        # As-of join (backward strategy for point-in-time data)
        joined = quotes_sorted.join_asof(
            statements_sorted,
            left_on=date_col,
            right_on=effective_col,
            by=code_col,
            strategy="backward",
            suffix="_stmt",
        )

        return joined

    @staticmethod
    def add_tplus1_effective_date(
        df: pl.DataFrame,
        disclosed_date_col: str = "DisclosedDate",
        disclosed_time_col: str = "DisclosedTime",
        calendar_df: pl.DataFrame = None,
        cutoff_time: str = "15:00:00",
    ) -> pl.DataFrame:
        """
        T+1ルールで有効日を計算

        Args:
            df: 財務データ
            disclosed_date_col: 開示日列
            disclosed_time_col: 開示時刻列
            calendar_df: 営業日カレンダー
            cutoff_time: カットオフ時刻（デフォルト15:00）

        Returns:
            effective_date列が追加されたDataFrame
        """
        # Parse time and check if after cutoff
        df = df.with_columns(
            [
                pl.col(disclosed_time_col)
                .str.strptime(pl.Time, "%H:%M:%S")
                .alias("_time_parsed")
            ]
        )

        cutoff = datetime.strptime(cutoff_time, "%H:%M:%S").time()

        # Apply T+1 rule
        df = df.with_columns(
            [
                pl.when(pl.col("_time_parsed") < cutoff)
                .then(pl.col(disclosed_date_col))
                .otherwise(pl.col(disclosed_date_col) + pl.duration(days=1))
                .alias("effective_date_raw")
            ]
        )

        # Adjust to next business day if calendar provided
        if calendar_df is not None:
            # This would require row-wise operation or join with calendar
            # Simplified version: just use the raw date
            df = df.with_columns([pl.col("effective_date_raw").alias("effective_date")])
        else:
            df = df.with_columns([pl.col("effective_date_raw").alias("effective_date")])

        # Clean up temporary columns
        df = df.drop(["_time_parsed", "effective_date_raw"])

        return df


class FlowDataJoiner:
    """フローデータ結合ユーティリティ"""

    @staticmethod
    def expand_flow_intervals(
        flow_df: pl.DataFrame,
        calendar_df: pl.DataFrame,
        published_date_col: str = "PublishedDate",
        section_col: str = "Section",
    ) -> pl.DataFrame:
        """
        フローデータの区間を展開（T+1ルール適用）

        Args:
            flow_df: フローデータ
            calendar_df: 営業日カレンダー
            published_date_col: 公表日列
            section_col: セクション列

        Returns:
            effective_start, effective_end列が追加されたDataFrame
        """
        # Sort by section and published date
        flow_sorted = flow_df.sort([section_col, published_date_col])

        # Calculate effective start (T+1)
        flow_sorted = flow_sorted.with_columns(
            [
                # Simplified: add 1 day for T+1
                (pl.col(published_date_col) + pl.duration(days=1)).alias(
                    "effective_start"
                )
            ]
        )

        # Calculate effective end (next record's start - 1 day)
        flow_sorted = flow_sorted.with_columns(
            [pl.col("effective_start").shift(-1).over(section_col).alias("next_start")]
        )

        flow_sorted = flow_sorted.with_columns(
            [
                pl.when(pl.col("next_start").is_not_null())
                .then(pl.col("next_start") - pl.duration(days=1))
                .otherwise(pl.lit(datetime(2099, 12, 31).date()))  # Far future date
                .alias("effective_end")
            ]
        )

        # Drop temporary column
        flow_sorted = flow_sorted.drop(["next_start"])

        return flow_sorted

    @staticmethod
    def calculate_days_since_flow(
        df: pl.DataFrame,
        calendar_df: pl.DataFrame,
        date_col: str = "Date",
        start_col: str = "effective_start",
    ) -> pl.DataFrame:
        """
        フロー開始からの営業日数を計算

        Args:
            df: 結合済みデータ
            calendar_df: 営業日カレンダー（bidx列付き）
            date_col: 現在日付列
            start_col: フロー開始日列

        Returns:
            days_since_flow列が追加されたDataFrame
        """
        # Join with calendar to get bidx for both dates
        df = df.join(
            calendar_df.select(["Date", "bidx"]), on=date_col, how="left"
        ).rename({"bidx": "bidx_current"})

        df = df.join(
            calendar_df.select(["Date", "bidx"]),
            left_on=start_col,
            right_on="Date",
            how="left",
            suffix="_start",
        ).rename({"bidx": "bidx_start"})

        # Calculate business days difference
        df = df.with_columns(
            [(pl.col("bidx_current") - pl.col("bidx_start")).alias("days_since_flow")]
        )

        # Clean up temporary columns
        df = df.drop(["bidx_current", "bidx_start", "Date_start"])

        return df


class MarketDataJoiner:
    """市場データ結合ユーティリティ"""

    @staticmethod
    def join_market_features(
        quotes: pl.DataFrame, market: pl.DataFrame, date_col: str = "Date"
    ) -> pl.DataFrame:
        """
        市場特徴量を日付で結合

        Args:
            quotes: 個別銘柄データ
            market: 市場データ
            date_col: 日付列名

        Returns:
            結合後のDataFrame
        """
        # Simple date join
        return quotes.join(market, on=date_col, how="left", suffix="_mkt")

    @staticmethod
    def calculate_cross_features(
        df: pl.DataFrame, beta_window: int = 60, ridge_lambda: float = 1e-8
    ) -> pl.DataFrame:
        """
        クロスセクション特徴量を計算

        Args:
            df: 個別銘柄と市場データが結合済みのDataFrame
            beta_window: ベータ計算期間
            ridge_lambda: リッジ回帰の正則化パラメータ

        Returns:
            クロス特徴量が追加されたDataFrame
        """
        # Calculate rolling beta (simplified - actual implementation would use ridge regression)
        df = df.with_columns(
            [
                # Covariance and variance over rolling window
                pl.corr("px_returns_1d", "mkt_ret_1d")
                .over(pl.col("Code").list.slice(-beta_window, beta_window))
                .alias("cross_corr_60d"),
                # Beta approximation (correlation * std_ratio)
                (
                    pl.corr("px_returns_1d", "mkt_ret_1d").over(
                        pl.col("Code").list.slice(-beta_window, beta_window)
                    )
                    * (
                        pl.col("px_returns_1d").std()
                        / (pl.col("mkt_ret_1d").std() + ridge_lambda)
                    )
                )
                .over(pl.col("Code").list.slice(-beta_window, beta_window))
                .alias("cross_beta_60d"),
            ]
        )

        # Alpha calculations
        df = df.with_columns(
            [
                (
                    pl.col("px_returns_1d")
                    - pl.col("cross_beta_60d") * pl.col("mkt_ret_1d")
                ).alias("cross_alpha_1d"),
                (
                    pl.col("px_returns_5d")
                    - pl.col("cross_beta_60d") * pl.col("mkt_ret_5d")
                ).alias("cross_alpha_5d"),
            ]
        )

        # Relative strength
        df = df.with_columns(
            [
                (pl.col("px_returns_5d") / (pl.col("mkt_ret_5d") + 1e-10)).alias(
                    "cross_rel_strength_5d"
                )
            ]
        )

        # Trend alignment
        df = df.with_columns(
            [
                (pl.col("px_ma_gap_5_20").sign() == pl.col("mkt_gap_5_20").sign())
                .cast(pl.Int8)
                .alias("cross_trend_align_flag")
            ]
        )

        # Regime conditional alpha
        df = df.with_columns(
            [
                (pl.col("cross_alpha_1d") * pl.col("mkt_bull_200")).alias(
                    "cross_alpha_vs_regime"
                )
            ]
        )

        # Idiosyncratic volatility ratio
        df = df.with_columns(
            [
                (pl.col("px_volatility_20d") / (pl.col("mkt_vol_20d") + 1e-10)).alias(
                    "cross_idio_vol_ratio"
                )
            ]
        )

        # Beta stability (inverse of rolling std of beta)
        df = df.with_columns(
            [
                (
                    1.0
                    / (pl.col("cross_beta_60d").rolling_std(20).over("Code") + 1e-10)
                ).alias("cross_beta_stability_60d")
            ]
        )

        return df
