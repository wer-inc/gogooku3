"""
営業日カレンダーユーティリティ
Trading Calendar APIを活用した営業日計算
"""

import polars as pl
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class TradingCalendarUtil:
    """
    取引所営業日の計算ユーティリティ
    """
    
    def __init__(self, calendar_df: Optional[pl.DataFrame] = None):
        """
        Args:
            calendar_df: Trading Calendar APIから取得したDataFrame
                        (Date, HolidayDivision列を含む)
        """
        self.calendar_df = calendar_df
        self._business_days = None
        
        if calendar_df is not None:
            self._process_calendar()
    
    def _process_calendar(self):
        """カレンダーデータを処理して営業日リストを作成"""
        if self.calendar_df is None:
            return
        
        # HolidayDivision: 1=営業日, 0=非営業日, 2=半日, 3=祝日取引
        # 営業日として扱うのは 1, 2, 3
        business_days_df = self.calendar_df.filter(
            pl.col("HolidayDivision").is_in([1, 2, 3])
        )
        
        # Date列を日付型に変換
        if "Date" in business_days_df.columns:
            if business_days_df["Date"].dtype == pl.Utf8:
                business_days_df = business_days_df.with_columns(
                    pl.col("Date").str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
                )
        
        # 営業日リストを作成
        self._business_days = set(business_days_df["Date"].to_list())
        logger.info(f"Loaded {len(self._business_days)} business days")
    
    def is_business_day(self, date: pl.Date) -> bool:
        """指定日が営業日かどうか判定"""
        if self._business_days is None:
            # カレンダーがない場合は平日を営業日とする簡易版
            if isinstance(date, str):
                date = datetime.strptime(date, "%Y-%m-%d").date()
            return date.weekday() < 5
        
        return date in self._business_days
    
    def next_business_day(self, date: pl.Date, days: int = 1) -> pl.Date:
        """
        指定日から次のN営業日後を取得
        
        Args:
            date: 基準日
            days: 何営業日後か（デフォルト1）
        
        Returns:
            N営業日後の日付
        """
        if isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d").date()
        
        count = 0
        current = date
        
        while count < days:
            current = current + timedelta(days=1)
            if self.is_business_day(current):
                count += 1
        
        return current
    
    def prev_business_day(self, date: pl.Date, days: int = 1) -> pl.Date:
        """
        指定日から前のN営業日前を取得
        
        Args:
            date: 基準日
            days: 何営業日前か（デフォルト1）
        
        Returns:
            N営業日前の日付
        """
        if isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d").date()
        
        count = 0
        current = date
        
        while count < days:
            current = current - timedelta(days=1)
            if self.is_business_day(current):
                count += 1
        
        return current
    
    def business_days_between(self, start_date: pl.Date, end_date: pl.Date) -> int:
        """
        2つの日付間の営業日数を計算
        
        Args:
            start_date: 開始日
            end_date: 終了日
        
        Returns:
            営業日数
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        if self._business_days:
            # カレンダーがある場合
            count = sum(
                1 for d in self._business_days
                if start_date <= d <= end_date
            )
        else:
            # 簡易版：平日をカウント
            count = 0
            current = start_date
            while current <= end_date:
                if current.weekday() < 5:
                    count += 1
                current += timedelta(days=1)
        
        return count
    
    def create_business_day_calendar(
        self, 
        start_date: str, 
        end_date: str
    ) -> pl.DataFrame:
        """
        指定期間の営業日カレンダーを作成
        
        Args:
            start_date: 開始日（YYYY-MM-DD）
            end_date: 終了日（YYYY-MM-DD）
        
        Returns:
            営業日のDataFrame
        """
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        dates = []
        current = start
        
        while current <= end:
            if self.is_business_day(current):
                dates.append(current)
            current += timedelta(days=1)
        
        return pl.DataFrame({
            "Date": dates
        })


def create_next_bd_expr(calendar_df: pl.DataFrame) -> pl.Expr:
    """
    Polars式として使える next_business_day 関数を作成
    
    Args:
        calendar_df: Trading Calendar データ
    
    Returns:
        次営業日を計算するPolars式
    
    使用例:
        df.with_columns(next_bd_expr(calendar_df).alias("next_bd"))
    """
    # 営業日リストを作成
    business_days = calendar_df.filter(
        pl.col("HolidayDivision").is_in([1, 2, 3])
    )["Date"].to_list()
    
    # 各日付に対して次の営業日をマッピング
    next_bd_map = {}
    for i, date in enumerate(business_days[:-1]):
        next_bd_map[date] = business_days[i + 1]
    
    # 最後の営業日の次は仮に+1日とする
    if business_days:
        last_date = business_days[-1]
        if isinstance(last_date, str):
            last_date = datetime.strptime(last_date, "%Y-%m-%d").date()
        next_bd_map[business_days[-1]] = last_date + timedelta(days=1)
    
    # Polars式として返す
    return pl.col("Date").map_dict(next_bd_map, default=pl.col("Date") + pl.duration(days=1))


def validate_business_day_coverage(
    data_df: pl.DataFrame,
    calendar_df: pl.DataFrame,
    date_col: str = "Date"
) -> Dict[str, float]:
    """
    データの営業日カバレッジを検証
    
    Args:
        data_df: 検証対象のデータ
        calendar_df: Trading Calendar
        date_col: 日付列名
    
    Returns:
        カバレッジ統計
    """
    # 営業日を取得
    business_days = set(
        calendar_df.filter(
            pl.col("HolidayDivision").is_in([1, 2, 3])
        )["Date"].to_list()
    )
    
    # データの日付を取得
    data_dates = set(data_df[date_col].unique().to_list())
    
    # 統計計算
    covered_days = data_dates & business_days
    missing_days = business_days - data_dates
    extra_days = data_dates - business_days
    
    coverage = len(covered_days) / len(business_days) if business_days else 0
    
    return {
        "coverage_ratio": coverage,
        "covered_days": len(covered_days),
        "total_business_days": len(business_days),
        "missing_days": len(missing_days),
        "extra_days": len(extra_days)  # 非営業日のデータ
    }