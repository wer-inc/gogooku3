"""
営業日カレンダー ユーティリティ

- Trading Calendar API または既存の株式日次グリッドから、
  翌営業日（T+1）を計算するための共通ヘルパーを提供します。
  パイプライン全体で同一の定義に基づく T+1 を実現することが目的です。

主なAPI:
- TradingCalendarUtil: 営業日判定, 前後営業日の計算
- create_next_bd_expr(calendar_df): カレンダーDFから pl.Expr を返す
- build_next_bday_expr_from_dates(business_days): 営業日リストから pl.Expr を返す
- build_next_bday_expr_from_quotes(quotes_df): quotesのDate列から pl.Expr を返す

既定の営業日定義（equity_mode=True）:
- HolidayDivision in {1,2} を営業日とする（1=営業日, 2=半日）
  祝日取引 (3) はデフォルトでは含めない。
  先物/オプション用途などで含めたい場合は、明示的に include_divisions を渡すこと。
"""

import logging
from datetime import datetime, timedelta

try:
    import jpholiday
except ImportError:  # pragma: no cover - optional dependency
    jpholiday = None

import polars as pl

logger = logging.getLogger(__name__)


class TradingCalendarUtil:
    """
    取引所営業日の計算ユーティリティ
    """

    def __init__(self, calendar_df: pl.DataFrame | None = None):
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
        # 既定は株式モード: 1,2 のみを営業日として扱う（必要に応じて3を含める設計に拡張）
        if "HolidayDivision" in self.calendar_df.columns:
            business_days_df = self.calendar_df.filter(
                pl.col("HolidayDivision").is_in([1, 2])
            )
        else:
            business_days_df = self.calendar_df

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
            # カレンダーがない場合でも日本の祝日を考慮した平日判定を行う
            if isinstance(date, str):
                date = datetime.strptime(date, "%Y-%m-%d").date()
            if isinstance(date, datetime):
                date = date.date()

            if date.weekday() >= 5:
                return False

            if jpholiday is not None and jpholiday.is_holiday(date):
                return False

            return True

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
    include = [1, 2]
    if "HolidayDivision" in calendar_df.columns:
        business_days = (
            calendar_df.filter(pl.col("HolidayDivision").is_in(include))["Date"].to_list()
        )
    else:
        business_days = calendar_df["Date"].to_list()

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


# =============== New public helpers ===============
def build_next_bday_expr_from_dates(dates: list) -> callable:
    """営業日リストから次営業日 pl.Expr→pl.Expr を返す関数を生成。

    Args:
        dates: 営業日（pl.Date あるいは date 互換）の昇順リスト

    Returns:
        Callable[[pl.Expr], pl.Expr]: 入力日付列に対応する翌営業日列を返す関数
    """
    if not dates:
        def _fallback(col: pl.Expr) -> pl.Expr:
            return col + pl.duration(days=1)
        return _fallback

    # 作業用に昇順ユニーク化
    uniq = []
    seen = set()
    for d in dates:
        if d in seen:
            continue
        uniq.append(d)
        seen.add(d)

    next_map: dict = {}
    for i in range(len(uniq) - 1):
        next_map[uniq[i]] = uniq[i + 1]

    # 最終日の次は +1 日（安全側のダミー）
    last = uniq[-1]
    if isinstance(last, str):
        from datetime import datetime, timedelta
        last_dt = datetime.strptime(last, "%Y-%m-%d").date()
        next_map[last] = last_dt + timedelta(days=1)
    else:
        from datetime import timedelta
        next_map[last] = last + timedelta(days=1)

    def _expr(col: pl.Expr) -> pl.Expr:
        return col.map_dict(next_map, default=col + pl.duration(days=1))

    return _expr


def build_next_bday_expr_from_quotes(quotes_df: pl.DataFrame) -> callable:
    """quotesのDate列から次営業日関数を生成（従来の実装を共通化）。"""
    if quotes_df.is_empty() or "Date" not in quotes_df.columns:
        def _fallback(col: pl.Expr) -> pl.Expr:
            return col + pl.duration(days=1)
        return _fallback

    dates = quotes_df.select("Date").unique().sort("Date")["Date"].to_list()
    return build_next_bday_expr_from_dates(dates)


def build_next_bday_expr_from_calendar_df(calendar_df: pl.DataFrame, *, include_divisions: list[int] | None = None) -> callable:
    """カレンダーDFから次営業日関数を生成。

    Args:
        calendar_df: Trading Calendar データ（Date, HolidayDivision を含む想定）
        include_divisions: 営業日に含める区分（既定: [1,2] 株式モード）
    """
    include = include_divisions or [1, 2]
    if calendar_df.is_empty():
        def _fallback(col: pl.Expr) -> pl.Expr:
            return col + pl.duration(days=1)
        return _fallback

    if "HolidayDivision" in calendar_df.columns:
        df = calendar_df.filter(pl.col("HolidayDivision").is_in(include))
    else:
        df = calendar_df
    dates = df.select("Date").unique().sort("Date")["Date"].to_list()
    return build_next_bday_expr_from_dates(dates)


def validate_business_day_coverage(
    data_df: pl.DataFrame,
    calendar_df: pl.DataFrame,
    date_col: str = "Date"
) -> dict[str, float]:
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
