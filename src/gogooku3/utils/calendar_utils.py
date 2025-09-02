"""
JPX Calendar Utilities
東京証券取引所の営業日管理
"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Optional, Set, Tuple
import pytz


class JPXCalendar:
    """JPX営業日カレンダー管理"""
    
    # JPX営業時間
    MARKET_OPEN = (9, 0)   # 9:00
    MARKET_CLOSE = (15, 0)  # 15:00
    LUNCH_START = (11, 30)  # 11:30
    LUNCH_END = (12, 30)    # 12:30
    
    # 半日取引日（大納会・大発会）
    HALF_DAY_DATES = {
        date(2023, 12, 29),  # 2023年大納会
        date(2024, 1, 4),    # 2024年大発会
        date(2024, 12, 30),  # 2024年大納会
        date(2025, 1, 6),    # 2025年大発会
    }
    
    # 日本の祝日（簡略版、実際は祝日APIや専用ライブラリを使用推奨）
    HOLIDAYS_2023_2025 = {
        # 2023年
        date(2023, 1, 1),   # 元日
        date(2023, 1, 2),   # 振替休日
        date(2023, 1, 9),   # 成人の日
        date(2023, 2, 11),  # 建国記念の日
        date(2023, 2, 23),  # 天皇誕生日
        date(2023, 3, 21),  # 春分の日
        date(2023, 4, 29),  # 昭和の日
        date(2023, 5, 3),   # 憲法記念日
        date(2023, 5, 4),   # みどりの日
        date(2023, 5, 5),   # こどもの日
        date(2023, 7, 17),  # 海の日
        date(2023, 8, 11),  # 山の日
        date(2023, 9, 18),  # 敬老の日
        date(2023, 9, 23),  # 秋分の日
        date(2023, 10, 9),  # スポーツの日
        date(2023, 11, 3),  # 文化の日
        date(2023, 11, 23), # 勤労感謝の日
        
        # 2024年
        date(2024, 1, 1),   # 元日
        date(2024, 1, 8),   # 成人の日
        date(2024, 2, 11),  # 建国記念の日
        date(2024, 2, 12),  # 振替休日
        date(2024, 2, 23),  # 天皇誕生日
        date(2024, 3, 20),  # 春分の日
        date(2024, 4, 29),  # 昭和の日
        date(2024, 5, 3),   # 憲法記念日
        date(2024, 5, 4),   # みどりの日
        date(2024, 5, 5),   # こどもの日
        date(2024, 5, 6),   # 振替休日
        date(2024, 7, 15),  # 海の日
        date(2024, 8, 11),  # 山の日
        date(2024, 8, 12),  # 振替休日
        date(2024, 9, 16),  # 敬老の日
        date(2024, 9, 22),  # 秋分の日
        date(2024, 9, 23),  # 振替休日
        date(2024, 10, 14), # スポーツの日
        date(2024, 11, 3),  # 文化の日
        date(2024, 11, 4),  # 振替休日
        date(2024, 11, 23), # 勤労感謝の日
        
        # 2025年（予定）
        date(2025, 1, 1),   # 元日
        date(2025, 1, 13),  # 成人の日
        date(2025, 2, 11),  # 建国記念の日
        date(2025, 2, 23),  # 天皇誕生日
        date(2025, 2, 24),  # 振替休日
        date(2025, 3, 20),  # 春分の日
        date(2025, 4, 29),  # 昭和の日
        date(2025, 5, 3),   # 憲法記念日
        date(2025, 5, 4),   # みどりの日
        date(2025, 5, 5),   # こどもの日
        date(2025, 5, 6),   # 振替休日
        date(2025, 7, 21),  # 海の日
        date(2025, 8, 11),  # 山の日
        date(2025, 9, 15),  # 敬老の日
        date(2025, 9, 23),  # 秋分の日
        date(2025, 10, 13), # スポーツの日
        date(2025, 11, 3),  # 文化の日
        date(2025, 11, 23), # 勤労感謝の日
        date(2025, 11, 24), # 振替休日
    }
    
    def __init__(self):
        """Initialize JPX Calendar"""
        self.jst = pytz.timezone('Asia/Tokyo')
        self._business_days_cache = {}
    
    def is_business_day(self, check_date: date) -> bool:
        """
        指定日がJPX営業日かどうかを判定
        
        Args:
            check_date: チェックする日付
            
        Returns:
            営業日の場合True
        """
        # 週末チェック
        if check_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # 祝日チェック
        if check_date in self.HOLIDAYS_2023_2025:
            return False
        
        # 年末年始休業（12/31-1/3は基本的に休場）
        if check_date.month == 12 and check_date.day == 31:
            return False
        if check_date.month == 1 and check_date.day <= 3:
            return False
        
        return True
    
    def is_half_day(self, check_date: date) -> bool:
        """
        半日取引日かどうかを判定
        
        Args:
            check_date: チェックする日付
            
        Returns:
            半日取引日の場合True
        """
        return check_date in self.HALF_DAY_DATES
    
    def get_business_days(
        self, 
        start_date: date, 
        end_date: date,
        include_half_days: bool = True
    ) -> List[date]:
        """
        指定期間の営業日リストを取得
        
        Args:
            start_date: 開始日
            end_date: 終了日
            include_half_days: 半日取引日を含めるか
            
        Returns:
            営業日のリスト
        """
        cache_key = (start_date, end_date, include_half_days)
        if cache_key in self._business_days_cache:
            return self._business_days_cache[cache_key]
        
        business_days = []
        current_date = start_date
        
        while current_date <= end_date:
            if self.is_business_day(current_date):
                if include_half_days or not self.is_half_day(current_date):
                    business_days.append(current_date)
            current_date += timedelta(days=1)
        
        self._business_days_cache[cache_key] = business_days
        return business_days
    
    def add_business_days(
        self, 
        start_date: date, 
        n_days: int,
        skip_half_days: bool = False
    ) -> date:
        """
        営業日ベースで日数を加算
        
        Args:
            start_date: 開始日
            n_days: 加算する営業日数（負の値も可）
            skip_half_days: 半日取引日をスキップするか
            
        Returns:
            n営業日後の日付
        """
        if n_days == 0:
            return start_date
        
        direction = 1 if n_days > 0 else -1
        days_to_add = abs(n_days)
        current_date = start_date
        days_added = 0
        
        while days_added < days_to_add:
            current_date += timedelta(days=direction)
            if self.is_business_day(current_date):
                if not skip_half_days or not self.is_half_day(current_date):
                    days_added += 1
        
        return current_date
    
    def count_business_days(
        self,
        start_date: date,
        end_date: date,
        include_half_days: bool = True
    ) -> int:
        """
        期間内の営業日数をカウント
        
        Args:
            start_date: 開始日
            end_date: 終了日
            include_half_days: 半日取引日を含めるか
            
        Returns:
            営業日数
        """
        if end_date < start_date:
            return 0
        
        return len(self.get_business_days(start_date, end_date, include_half_days))
    
    def get_previous_business_day(
        self,
        check_date: date,
        skip_half_days: bool = False
    ) -> date:
        """
        直前の営業日を取得
        
        Args:
            check_date: 基準日
            skip_half_days: 半日取引日をスキップするか
            
        Returns:
            直前の営業日
        """
        return self.add_business_days(check_date, -1, skip_half_days)
    
    def get_next_business_day(
        self,
        check_date: date,
        skip_half_days: bool = False
    ) -> date:
        """
        翌営業日を取得
        
        Args:
            check_date: 基準日
            skip_half_days: 半日取引日をスキップするか
            
        Returns:
            翌営業日
        """
        return self.add_business_days(check_date, 1, skip_half_days)
    
    def is_market_hours(
        self,
        check_datetime: datetime,
        include_lunch: bool = False
    ) -> bool:
        """
        市場取引時間内かどうかを判定
        
        Args:
            check_datetime: チェックする日時
            include_lunch: 昼休みも取引時間に含めるか
            
        Returns:
            取引時間内の場合True
        """
        # Convert to JST if needed
        if check_datetime.tzinfo is None:
            check_datetime = self.jst.localize(check_datetime)
        else:
            check_datetime = check_datetime.astimezone(self.jst)
        
        # Check if business day
        if not self.is_business_day(check_datetime.date()):
            return False
        
        # Check time
        time_tuple = (check_datetime.hour, check_datetime.minute)
        
        # Half day check (morning session only)
        if self.is_half_day(check_datetime.date()):
            return self.MARKET_OPEN <= time_tuple <= self.LUNCH_START
        
        # Normal trading day
        if self.MARKET_OPEN <= time_tuple < self.LUNCH_START:
            return True
        elif self.LUNCH_END <= time_tuple <= self.MARKET_CLOSE:
            return True
        elif include_lunch and self.LUNCH_START <= time_tuple < self.LUNCH_END:
            return True
        
        return False
    
    def get_trading_sessions(
        self,
        check_date: date
    ) -> List[Tuple[datetime, datetime]]:
        """
        指定日の取引セッション時間を取得
        
        Args:
            check_date: チェックする日付
            
        Returns:
            [(セッション開始時刻, セッション終了時刻), ...]のリスト
        """
        if not self.is_business_day(check_date):
            return []
        
        sessions = []
        
        # Morning session
        morning_start = self.jst.localize(
            datetime.combine(check_date, 
                           datetime.min.time().replace(hour=self.MARKET_OPEN[0], 
                                                      minute=self.MARKET_OPEN[1]))
        )
        morning_end = self.jst.localize(
            datetime.combine(check_date,
                           datetime.min.time().replace(hour=self.LUNCH_START[0],
                                                      minute=self.LUNCH_START[1]))
        )
        sessions.append((morning_start, morning_end))
        
        # Afternoon session (not on half days)
        if not self.is_half_day(check_date):
            afternoon_start = self.jst.localize(
                datetime.combine(check_date,
                               datetime.min.time().replace(hour=self.LUNCH_END[0],
                                                          minute=self.LUNCH_END[1]))
            )
            afternoon_end = self.jst.localize(
                datetime.combine(check_date,
                               datetime.min.time().replace(hour=self.MARKET_CLOSE[0],
                                                          minute=self.MARKET_CLOSE[1]))
            )
            sessions.append((afternoon_start, afternoon_end))
        
        return sessions
    
    @staticmethod
    def add_business_day_features(df: pl.DataFrame, date_col: str = "Date") -> pl.DataFrame:
        """
        DataFrameに営業日関連の特徴量を追加
        
        Features added:
        - is_monday: 月曜日フラグ
        - is_friday: 金曜日フラグ
        - is_month_start: 月初営業日フラグ
        - is_month_end: 月末営業日フラグ
        - is_quarter_end: 四半期末フラグ
        - days_in_month: 月内営業日数
        - day_of_month: 月内営業日番号
        - is_half_day: 半日取引日フラグ
        """
        calendar = JPXCalendar()
        
        # Convert to date if datetime
        dates = df[date_col].to_list()
        if dates and isinstance(dates[0], datetime):
            dates = [d.date() if isinstance(d, datetime) else d for d in dates]
        
        # Calculate features
        is_monday = []
        is_friday = []
        is_month_start = []
        is_month_end = []
        is_quarter_end = []
        is_half_day = []
        days_in_month = []
        day_of_month = []
        
        for d in dates:
            if isinstance(d, (datetime, date)):
                # Day of week
                is_monday.append(1 if d.weekday() == 0 else 0)
                is_friday.append(1 if d.weekday() == 4 else 0)
                
                # Half day
                is_half_day.append(1 if calendar.is_half_day(d) else 0)
                
                # Month start/end
                month_start = date(d.year, d.month, 1)
                if d.month == 12:
                    month_end = date(d.year + 1, 1, 1) - timedelta(days=1)
                else:
                    month_end = date(d.year, d.month + 1, 1) - timedelta(days=1)
                
                month_bdays = calendar.get_business_days(month_start, month_end)
                
                is_month_start.append(1 if month_bdays and d == month_bdays[0] else 0)
                is_month_end.append(1 if month_bdays and d == month_bdays[-1] else 0)
                
                # Quarter end
                is_q_end = (d.month in [3, 6, 9, 12] and 
                           is_month_end[-1] == 1)
                is_quarter_end.append(1 if is_q_end else 0)
                
                # Days in month and day of month
                days_in_month.append(len(month_bdays))
                try:
                    day_of_month.append(month_bdays.index(d) + 1)
                except ValueError:
                    day_of_month.append(0)
            else:
                # Handle null values
                is_monday.append(None)
                is_friday.append(None)
                is_month_start.append(None)
                is_month_end.append(None)
                is_quarter_end.append(None)
                is_half_day.append(None)
                days_in_month.append(None)
                day_of_month.append(None)
        
        # Add features to dataframe
        return df.with_columns([
            pl.Series("cal_is_monday", is_monday, dtype=pl.Int8),
            pl.Series("cal_is_friday", is_friday, dtype=pl.Int8),
            pl.Series("cal_is_month_start", is_month_start, dtype=pl.Int8),
            pl.Series("cal_is_month_end", is_month_end, dtype=pl.Int8),
            pl.Series("cal_is_quarter_end", is_quarter_end, dtype=pl.Int8),
            pl.Series("cal_is_half_day", is_half_day, dtype=pl.Int8),
            pl.Series("cal_days_in_month", days_in_month, dtype=pl.Int32),
            pl.Series("cal_day_of_month", day_of_month, dtype=pl.Int32),
        ])