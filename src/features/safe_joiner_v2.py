"""
Safe Joiner V2 - 改善版の安全な結合処理
実運用での問題を解決した堅牢な実装
"""

import logging
from datetime import datetime, time

import polars as pl

from .calendar_utils import TradingCalendarUtil
from .code_normalizer import CodeNormalizer
from .section_mapper import SectionMapper
from .validity_flags import ValidityFlagManager

logger = logging.getLogger(__name__)


class SafeJoinerV2:
    """
    改善版の安全な結合処理
    
    主な改善点:
    - Code正規化の徹底
    - Statements多重開示の解決
    - 半日取引対応
    - Trades_spec区間の完全性保証
    - Sectionフォールバック戦略
    - 有効性フラグの体系化
    """

    def __init__(
        self,
        calendar_util: TradingCalendarUtil | None = None,
        section_mapper: SectionMapper | None = None,
        code_normalizer: CodeNormalizer | None = None,
        validity_manager: ValidityFlagManager | None = None
    ):
        self.calendar_util = calendar_util or TradingCalendarUtil()
        self.section_mapper = section_mapper or SectionMapper()
        self.code_normalizer = code_normalizer or CodeNormalizer()
        self.validity_manager = validity_manager or ValidityFlagManager()
        self.join_stats = {}

    def prepare_base_quotes(self, quotes_df: pl.DataFrame) -> pl.DataFrame:
        """
        daily_quotesを基盤データとして準備（Code正規化込み）
        """
        logger.info("Preparing base quotes data with code normalization...")

        # Code正規化を最初に適用
        quotes = self.code_normalizer.normalize_dataframe(
            quotes_df,
            code_columns=["Code"],
            target_column="Code"
        )

        # 型変換
        quotes = quotes.with_columns([
            pl.col("Code").cast(pl.Utf8),
            pl.col("Date").cast(pl.Date),
        ])

        # 重複チェック（Code正規化後）
        before_dedup = len(quotes)
        quotes = quotes.unique(subset=["Code", "Date"], keep="last")
        after_dedup = len(quotes)

        if before_dedup != after_dedup:
            logger.warning(f"Removed {before_dedup - after_dedup} duplicate (Code, Date) pairs after normalization")

        # 検証
        validation = self.code_normalizer.validate_normalization(quotes)
        logger.info(f"Code normalization rate: {validation['normalization_rate']:.1%}")

        logger.info(f"Base quotes prepared: {len(quotes)} rows, {quotes['Code'].n_unique()} stocks")

        return quotes

    def join_statements_with_dedup(
        self,
        base_df: pl.DataFrame,
        statements_df: pl.DataFrame,
        use_time_cutoff: bool = True,
        calendar_df: pl.DataFrame | None = None
    ) -> pl.DataFrame:
        """
        財務諸表データをas-of結合（多重開示対策・半日取引対応）
        """
        logger.info("Joining statements with deduplication and half-day handling...")

        # Code正規化
        stm = self.code_normalizer.normalize_dataframe(
            statements_df,
            code_columns=["LocalCode", "Code"],
            target_column="Code"
        )

        stm = stm.with_columns([
            pl.col("Code").cast(pl.Utf8),
            pl.col("DisclosedDate").cast(pl.Date),
        ])

        # join用の正規化コード（4桁）
        stm = stm.with_columns(
            pl.col("Code")
            .map_elements(CodeNormalizer.normalize_code, return_dtype=pl.Utf8)
            .alias("_join_code")
        )

        # 開示時刻の処理（同日複数開示の解決）
        if "DisclosedTime" in stm.columns:
            # 開示タイムスタンプを作成
            stm = stm.with_columns([
                pl.datetime(
                    pl.col("DisclosedDate").dt.year(),
                    pl.col("DisclosedDate").dt.month(),
                    pl.col("DisclosedDate").dt.day(),
                    pl.col("DisclosedTime").str.slice(0, 2).cast(pl.Int32),
                    pl.col("DisclosedTime").str.slice(3, 2).cast(pl.Int32),
                    pl.col("DisclosedTime").str.slice(6, 2).cast(pl.Int32).fill_null(0)
                ).alias("disclosed_ts")
            ])
        else:
            # 時刻がない場合はデフォルト（15:30）を設定
            stm = stm.with_columns([
                pl.datetime(
                    pl.col("DisclosedDate").dt.year(),
                    pl.col("DisclosedDate").dt.month(),
                    pl.col("DisclosedDate").dt.day(),
                    15, 30, 0
                ).alias("disclosed_ts")
            ])

        # 同日・同銘柄の複数開示を最新時刻のみに絞る
        logger.info(f"  Before dedup: {len(stm)} statements")
        stm = (stm.sort(["Code", "DisclosedDate", "disclosed_ts"])
               .group_by(["Code", "DisclosedDate"])
               .last()
               .sort(["Code", "DisclosedDate"]))
        logger.info(f"  After dedup: {len(stm)} statements")

        # 半日取引の判定（カレンダーから取得）
        if calendar_df is not None and "HolidayDivision" in calendar_df.columns:
            # 半日取引日のリスト作成
            half_days = set(
                calendar_df.filter(pl.col("HolidayDivision") == 2)["Date"].to_list()
            )

            # 動的cutoff時刻の設定
            stm = stm.with_columns([
                pl.when(pl.col("DisclosedDate").is_in(half_days))
                .then(pl.lit(time(11, 30, 0)))  # 半日は11:30
                .otherwise(pl.lit(time(15, 0, 0)))  # 通常は15:00
                .alias("cutoff_time")
            ])
        else:
            # デフォルトは15:00
            stm = stm.with_columns(pl.lit(time(15, 0, 0)).alias("cutoff_time"))

        # effective_dateの計算
        if use_time_cutoff and "DisclosedTime" in stm.columns:
            # 時刻を考慮したeffective_date
            stm = stm.with_columns([
                pl.when(
                    pl.col("disclosed_ts").dt.time() < pl.col("cutoff_time")
                )
                .then(pl.col("DisclosedDate"))  # 当日有効
                .otherwise(
                    self._next_business_day_expr(pl.col("DisclosedDate"))  # T+1
                )
                .alias("effective_date")
            ])
        else:
            # 保守的にT+1ルール
            stm = stm.with_columns([
                self._next_business_day_expr(pl.col("DisclosedDate")).alias("effective_date")
            ])

        # 財務特徴量の計算（厳密FY×Q YoYを先に付与）
        try:
            from features.statements_yoy import add_strict_yoy
            stm = add_strict_yoy(stm)
        except Exception as e:
            logger.warning(f"Failed to add strict YoY: {e}")
        stm = self._calculate_statement_features(stm)

        # ソート
        stm = stm.sort(["Code", "effective_date"])
        base_df = base_df.with_columns(
            pl.col("Code")
            .cast(pl.Utf8)
            .map_elements(CodeNormalizer.normalize_code, return_dtype=pl.Utf8)
            .alias("_join_code")
        )

        base_df = base_df.sort(["_join_code", "Date"])

        # as-of結合
        stmt_cols = [col for col in stm.columns if col.startswith("stmt_")]

        # Fix column conflicts: exclude "Code" from right DataFrame since it's used in "by" parameter
        join_cols = ["_join_code", "effective_date"] + stmt_cols
        result = base_df.join_asof(
            stm.select(join_cols),
            left_on="Date",
            right_on="effective_date",
            by="_join_code",
            strategy="backward"
        )

        # Clean up potential column conflicts
        if "Code_right" in result.columns:
            result = result.drop("Code_right")
        if "Date_right" in result.columns:
            result = result.drop("Date_right")
        if "_join_code_right" in result.columns:
            result = result.drop("_join_code_right")

        # joinキーを削除
        if "_join_code" in result.columns:
            result = result.drop("_join_code")

        # インパルスと経過日数
        if "effective_date" in result.columns:
            result = result.with_columns([
                (pl.col("Date") == pl.col("effective_date"))
                    .cast(pl.Int8)
                    .alias("stmt_imp_statement"),
                (pl.col("Date") - pl.col("effective_date"))
                    .dt.total_days()
                    .fill_null(999)
                    .alias("stmt_days_since_statement")
            ])
            result = result.drop("effective_date")

        # 有効性フラグを追加
        result = self.validity_manager.add_statement_validity_flags(result)

        # 統計情報
        coverage = result.filter(pl.col("stmt_days_since_statement") < 999).shape[0] / len(result)
        self.join_stats["statements"] = {
            "coverage": coverage,
            "rows_with_data": result.filter(pl.col("stmt_days_since_statement") < 999).shape[0],
            "total_rows": len(result),
            "valid_rows": result["is_stmt_valid"].sum()
        }

        logger.info(f"  Statements coverage: {coverage:.1%}")
        logger.info(f"  Valid statements: {result['is_stmt_valid'].mean():.1%}")

        return result

    def join_trades_spec_with_gap_filling(
        self,
        base_df: pl.DataFrame,
        trades_df: pl.DataFrame,
        section_mapping_df: pl.DataFrame | None = None
    ) -> pl.DataFrame:
        """
        週次フローデータを区間結合（区間の完全性保証・フォールバック対応）
        """
        logger.info("Joining trades_spec with gap filling and fallback...")

        # Sectionを付与（フォールバック戦略込み）
        if "Section" not in base_df.columns:
            if section_mapping_df is not None:
                base_df = self.section_mapper.attach_section_to_daily(
                    base_df, section_mapping_df
                )

            # フォールバック処理
            base_df = self.validity_manager.add_section_fallback_flag(base_df)

        # trades_specの準備
        flow = trades_df.with_columns([
            pl.col("PublishedDate").cast(pl.Date),
            pl.col("Section").cast(pl.Utf8)
        ])

        # 同週複数公表の解決（最新のみ採用）
        flow = (flow.sort(["Section", "PublishedDate"])
                .group_by(["Section", pl.col("PublishedDate").dt.truncate("1w")])
                .last())

        # effective_start = 翌営業日
        flow = flow.with_columns([
            self._next_business_day_expr(pl.col("PublishedDate")).alias("effective_start")
        ])

        # Section内でソート
        flow = flow.sort(["Section", "effective_start"])

        # 次のeffective_startを取得（区間の終端計算用）
        flow = flow.with_columns([
            pl.col("effective_start").shift(-1).over("Section").alias("next_start")
        ])

        # effective_end = next_start - 1日（最後は当日まで延長）
        flow = flow.with_columns([
            pl.when(pl.col("next_start").is_not_null())
            .then(pl.col("next_start") - pl.duration(days=1))
            .otherwise(pl.lit(datetime.now().date()))  # 最新は当日まで有効
            .alias("effective_end")
        ])

        # フロー特徴量の計算
        flow = self._calculate_flow_features(flow)

        # 区間結合（asof + フィルタ方式）
        flow_cols = [col for col in flow.columns if col.startswith("flow_")]

        # Sectionごとに処理
        result_dfs = []

        for section in base_df["Section"].unique():
            section_df = base_df.filter(pl.col("Section") == section)
            section_flow = flow.filter(pl.col("Section") == section)

            if section_flow.is_empty() or section == "AllMarket":
                # フローデータがない場合はデフォルト値
                for col in flow_cols:
                    section_df = section_df.with_columns(pl.lit(0.0).alias(col))
                section_df = section_df.with_columns([
                    pl.lit(0).cast(pl.Int8).alias("flow_impulse"),
                    pl.lit(999).alias("days_since_flow")
                ])
            else:
                # asof結合
                section_df = section_df.sort("Date").join_asof(
                    section_flow.select(["effective_start", "effective_end"] + flow_cols),
                    left_on="Date",
                    right_on="effective_start",
                    strategy="backward"
                )

                # Clean up potential column conflicts
                if "Date_right" in section_df.columns:
                    section_df = section_df.drop("Date_right")
                if "effective_start_right" in section_df.columns:
                    section_df = section_df.drop("effective_start_right")

                # 区間内チェック
                for col in flow_cols:
                    section_df = section_df.with_columns([
                        pl.when(
                            (pl.col("Date") >= pl.col("effective_start")) &
                            (pl.col("Date") <= pl.col("effective_end"))
                        )
                        .then(pl.col(col))
                        .otherwise(None)
                        .alias(col)
                    ])

                # インパルスと経過日数
                section_df = section_df.with_columns([
                    pl.when(pl.col("Date") == pl.col("effective_start"))
                    .then(pl.lit(1))
                    .when(pl.col("effective_start").is_not_null())
                    .then(pl.lit(0))
                    .otherwise(pl.lit(0))
                    .cast(pl.Int8)
                    .alias("flow_impulse"),

                    pl.when(pl.col("effective_start").is_not_null())
                    .then((pl.col("Date") - pl.col("effective_start")).dt.total_days())
                    .otherwise(pl.lit(999))
                    .alias("days_since_flow")
                ])

                section_df = section_df.drop(["effective_start", "effective_end"])

            result_dfs.append(section_df)

        if result_dfs:
            result = pl.concat(result_dfs)
        else:
            result = base_df

        # 有効性フラグを追加
        result = self.validity_manager.add_flow_validity_flags(result)

        # 統計情報
        coverage = result.filter(pl.col("days_since_flow") < 999).shape[0] / len(result)
        self.join_stats["trades_spec"] = {
            "coverage": coverage,
            "rows_with_data": result.filter(pl.col("days_since_flow") < 999).shape[0],
            "total_rows": len(result),
            "valid_rows": result["is_flow_valid"].sum()
        }

        logger.info(f"  Trades_spec coverage: {coverage:.1%}")
        logger.info(f"  Valid flows: {result['is_flow_valid'].mean():.1%}")

        return result

    def join_topix_with_validation(
        self,
        base_df: pl.DataFrame,
        topix_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        TOPIXデータを同日結合（欠損時の安全処理）
        """
        logger.info("Joining TOPIX with validation...")

        # TOPIXデータの準備
        topix = topix_df.with_columns([
            pl.col("Date").cast(pl.Date)
        ])

        # mkt_* 特徴量の確認
        mkt_cols = [col for col in topix.columns if col.startswith("mkt_")]

        # 同日結合
        result = base_df.join(
            topix.select(["Date"] + mkt_cols),
            on="Date",
            how="left"
        )

        # 市場データ有効性フラグ
        result = self.validity_manager.add_market_validity_flags(result)

        # 銘柄×市場Cross特徴量（市場データが有効な場合のみ）
        if mkt_cols:
            result = self._calculate_cross_features_safe(result)
            result = self.validity_manager.add_beta_validity_flags(result)

        # 統計情報
        if mkt_cols:
            coverage = result[mkt_cols[0]].is_not_null().sum() / len(result)
            self.join_stats["topix"] = {
                "coverage": coverage,
                "mkt_features": len(mkt_cols),
                "cross_features": len([c for c in result.columns if c.startswith("cross_")]),
                "valid_rows": result["is_mkt_valid"].sum()
            }
            logger.info(f"  TOPIX coverage: {coverage:.1%}")
            logger.info(f"  Valid market data: {result['is_mkt_valid'].mean():.1%}")

        return result

    def _next_business_day_expr(self, date_col: pl.Expr) -> pl.Expr:
        """次営業日をTradingCalendarUtil経由で取得"""
        util = self.calendar_util

        def next_bd(value: datetime | None):
            if value is None:
                return None
            if isinstance(value, datetime):
                base = value.date()
            else:
                base = value
            return util.next_business_day(base)

        return date_col.map_elements(next_bd, return_dtype=pl.Date)

    def _pad_calendar(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return df

        df = df.with_columns(pl.col("Date").cast(pl.Date))
        date_min = df["Date"].min()
        date_max = df["Date"].max()

        calendar = pl.date_range(date_min, date_max, interval="1d", eager=True, closed="both").to_frame("Date")
        codes = df["Code"].unique().to_frame("Code")

        expanded = codes.join(calendar, how="cross")
        expanded = expanded.join(
            df.with_columns(pl.lit(0, dtype=pl.Int8).alias("is_calendar_pad")),
            on=["Code", "Date"],
            how="left"
        )

        expanded = expanded.with_columns([
            pl.col("is_calendar_pad").fill_null(1).cast(pl.Int8),
            pl.col("Date").map_elements(
                lambda d: self.calendar_util.is_business_day(d),
                return_dtype=pl.Boolean
            ).alias("_is_bd")
        ])

        expanded = expanded.with_columns(
            pl.col("_is_bd").cast(pl.Int8).alias("is_trading_day")
        ).drop("_is_bd")

        return expanded.sort(["Code", "Date"])

    def _calculate_statement_features(self, stm: pl.DataFrame) -> pl.DataFrame:
        """財務諸表特徴量を計算"""
        # 既に十分な stmt_* 列が存在する場合は再計算を省略
        existing_stmt_cols = {col for col in stm.columns if col.startswith("stmt_")}
        required_stmt_cols = {
            "stmt_yoy_sales",
            "stmt_yoy_op",
            "stmt_yoy_np",
            "stmt_opm",
            "stmt_npm",
            "stmt_progress_op",
            "stmt_progress_np",
            "stmt_rev_fore_op",
            "stmt_rev_fore_np",
            "stmt_rev_fore_eps",
            "stmt_rev_div_fore",
            "stmt_roe",
            "stmt_roa",
            "stmt_change_in_est",
            "stmt_nc_flag",
        }
        if required_stmt_cols.issubset(existing_stmt_cols):
            # 互換用の別名列のみ欠損している場合があるので補完しておく
            alias_exprs: list[pl.Expr] = []
            if "stmt_yoy_sales" in existing_stmt_cols and "stmt_revenue_growth" not in existing_stmt_cols:
                alias_exprs.append(pl.col("stmt_yoy_sales").alias("stmt_revenue_growth"))
            if "stmt_opm" in existing_stmt_cols and "stmt_profit_margin" not in existing_stmt_cols:
                alias_exprs.append(pl.col("stmt_opm").alias("stmt_profit_margin"))
            return stm.with_columns(alias_exprs) if alias_exprs else stm

        s = stm

        # Code列を正規化（4桁）
        if "Code" not in s.columns and "LocalCode" in s.columns:
            s = s.with_columns(
                pl.col("LocalCode").cast(pl.Utf8).alias("Code")
            )

        if "Code" in s.columns:
            s = s.with_columns(pl.col("Code").cast(pl.Utf8))

        # 欠損があっても安全に型変換
        s = s.with_columns([
            pl.col("DisclosedDate").cast(pl.Date, strict=False).alias("DisclosedDate"),
        ])

        # 会計年度の導出（FiscalYear が無ければ CurrentFiscalYearStartDate の年を使用）
        fiscal_year_expr = None
        if "FiscalYear" in s.columns:
            fiscal_year_expr = pl.col("FiscalYear").cast(pl.Int32, strict=False)
        elif "CurrentFiscalYearStartDate" in s.columns:
            fiscal_year_expr = (
                pl.col("CurrentFiscalYearStartDate")
                .cast(pl.Utf8, strict=False)
                .str.slice(0, 4)
                .cast(pl.Int32, strict=False)
            )
        else:
            fiscal_year_expr = pl.lit(None, dtype=pl.Int32)

        type_col = "TypeOfCurrentPeriod" if "TypeOfCurrentPeriod" in s.columns else None
        quarter_expr = (
            pl.when(pl.col(type_col).str.contains("1Q"))
            .then(1)
            .when(pl.col(type_col).str.contains("2Q"))
            .then(2)
            .when(pl.col(type_col).str.contains("3Q"))
            .then(3)
            .when(pl.col(type_col).str.contains("FY|4Q"))
            .then(4)
            .otherwise(0)
            if type_col
            else pl.lit(0)
        )

        s = s.sort(["Code", "DisclosedDate"]).with_columns([
            fiscal_year_expr.alias("fiscal_year"),
            quarter_expr.alias("quarter"),
        ])

        numeric_cols = [
            "NetSales",
            "OperatingProfit",
            "Profit",
            "ForecastOperatingProfit",
            "ForecastProfit",
            "ForecastEarningsPerShare",
            "ForecastDividendPerShareAnnual",
            "Equity",
            "TotalAssets",
        ]
        cast_exprs = [
            pl.col(col).cast(pl.Float64, strict=False).alias(col)
            for col in numeric_cols
            if col in s.columns
        ]
        if cast_exprs:
            s = s.with_columns(cast_exprs)

        # 前年同期値を自己結合で取得
        yoy_base = s.select([
            pl.col("Code"),
            pl.col("fiscal_year").alias("base_fy"),
            pl.col("quarter").alias("base_q"),
            pl.col("NetSales").alias("yoy_sales_base"),
            pl.col("OperatingProfit").alias("yoy_op_base"),
            pl.col("Profit").alias("yoy_np_base"),
        ])

        s = s.with_columns((pl.col("fiscal_year") - 1).alias("prev_fy"))
        s = s.join(
            yoy_base,
            left_on=["Code", "prev_fy", "quarter"],
            right_on=["Code", "base_fy", "base_q"],
            how="left",
        ).drop(["base_fy", "base_q"])

        # 直近開示値
        s = s.with_columns([
            pl.col("NetSales").shift(1).over("Code").alias("prev_sales"),
            pl.col("OperatingProfit").shift(1).over("Code").alias("prev_op"),
            pl.col("Profit").shift(1).over("Code").alias("prev_np"),
            pl.col("ForecastOperatingProfit").shift(1).over("Code").alias("prev_fore_op"),
            pl.col("ForecastProfit").shift(1).over("Code").alias("prev_fore_np"),
            pl.col("ForecastEarningsPerShare").shift(1).over("Code").alias("prev_fore_eps"),
            pl.col("ForecastDividendPerShareAnnual").shift(1).over("Code").alias("prev_fore_div"),
        ])

        def pct_change(current: pl.Expr, previous: pl.Expr) -> pl.Expr:
            return (current - previous) / (previous.abs() + 1e-12)

        s = s.with_columns([
            pct_change(pl.col("NetSales"), pl.col("yoy_sales_base")).alias("stmt_yoy_sales"),
            pct_change(pl.col("OperatingProfit"), pl.col("yoy_op_base")).alias("stmt_yoy_op"),
            pct_change(pl.col("Profit"), pl.col("yoy_np_base")).alias("stmt_yoy_np"),

            (pl.col("OperatingProfit") / (pl.col("NetSales") + 1e-12)).alias("stmt_opm"),
            (pl.col("Profit") / (pl.col("NetSales") + 1e-12)).alias("stmt_npm"),

            (pl.col("OperatingProfit") / (pl.col("ForecastOperatingProfit") + 1e-12)).alias("stmt_progress_op"),
            (pl.col("Profit") / (pl.col("ForecastProfit") + 1e-12)).alias("stmt_progress_np"),

            pct_change(pl.col("ForecastOperatingProfit"), pl.col("prev_fore_op")).alias("stmt_rev_fore_op"),
            pct_change(pl.col("ForecastProfit"), pl.col("prev_fore_np")).alias("stmt_rev_fore_np"),
            pct_change(pl.col("ForecastEarningsPerShare"), pl.col("prev_fore_eps")).alias("stmt_rev_fore_eps"),
            pct_change(
                pl.col("ForecastDividendPerShareAnnual").cast(pl.Float64, strict=False),
                pl.col("prev_fore_div").cast(pl.Float64, strict=False),
            ).alias("stmt_rev_div_fore"),

            (pl.col("Profit") / (pl.col("Equity") + 1e-12)).alias("stmt_roe"),
            (pl.col("Profit") / (pl.col("TotalAssets") + 1e-12)).alias("stmt_roa"),
        ])

        # 会計方針変更等のフラグ
        change_expr = pl.lit(False)
        if "ChangesInAccountingEstimates" in s.columns:
            change_expr = change_expr | (
                pl.col("ChangesInAccountingEstimates")
                .cast(pl.Utf8, strict=False)
                .str.to_lowercase()
                .is_in(["true", "1", "yes"])
                .fill_null(False)
            )
        s = s.with_columns(change_expr.alias("stmt_change_in_est"))

        nc_expr = pl.lit(False)
        if "ChangesBasedOnRevisionsOfAccountingStandard" in s.columns:
            nc_expr = nc_expr | (
                pl.col("ChangesBasedOnRevisionsOfAccountingStandard")
                .cast(pl.Utf8, strict=False)
                .str.to_lowercase()
                .is_in(["true", "1", "yes"])
                .fill_null(False)
            )
        if "RetrospectiveRestatement" in s.columns:
            nc_expr = nc_expr | (
                pl.col("RetrospectiveRestatement")
                .cast(pl.Utf8, strict=False)
                .str.to_lowercase()
                .is_in(["true", "1", "yes"])
                .fill_null(False)
            )
        s = s.with_columns(nc_expr.alias("stmt_nc_flag"))

        # 互換のための別名列を付与
        alias_exprs = []
        if "stmt_yoy_sales" in s.columns:
            alias_exprs.append(pl.col("stmt_yoy_sales").alias("stmt_revenue_growth"))
        if "stmt_opm" in s.columns:
            alias_exprs.append(pl.col("stmt_opm").alias("stmt_profit_margin"))
        if alias_exprs:
            s = s.with_columns(alias_exprs)

        drop_cols = [
            "prev_sales",
            "prev_op",
            "prev_np",
            "prev_fore_op",
            "prev_fore_np",
            "prev_fore_eps",
            "prev_fore_div",
            "yoy_sales_base",
            "yoy_op_base",
            "yoy_np_base",
            "prev_fy",
        ]
        s = s.drop([col for col in drop_cols if col in s.columns])

        return s

    def _calculate_flow_features(self, flow: pl.DataFrame) -> pl.DataFrame:
        """フロー特徴量を計算"""
        if "ForeignersBalance" in flow.columns:
            flow = flow.with_columns([
                (pl.col("ForeignersBalance") / (pl.col("ForeignersTotal") + 1e-12))
                    .alias("flow_foreign_net_ratio"),
                (pl.col("IndividualsBalance") / (pl.col("IndividualsTotal") + 1e-12))
                    .alias("flow_individual_net_ratio"),
                pl.lit(0.0).alias("flow_smart_money_idx")
            ])
        else:
            flow = flow.with_columns([
                pl.lit(0.0).alias("flow_foreign_net_ratio"),
                pl.lit(0.0).alias("flow_individual_net_ratio"),
                pl.lit(0.0).alias("flow_smart_money_idx")
            ])

        return flow

    def _calculate_cross_features_safe(self, df: pl.DataFrame) -> pl.DataFrame:
        """銘柄×市場Cross特徴量を安全に計算"""
        # is_mkt_valid == 1 の場合のみ計算
        return df.with_columns([
            pl.when(pl.col("is_mkt_valid") == 1)
            .then(pl.lit(1.0))  # 実際のベータ計算をここに
            .otherwise(None)
            .alias("cross_beta_60d"),

            pl.when(pl.col("is_mkt_valid") == 1)
            .then(pl.lit(0.0))  # 実際のアルファ計算をここに
            .otherwise(None)
            .alias("cross_alpha_1d"),

            pl.when(pl.col("is_mkt_valid") == 1)
            .then(pl.lit(0.0))  # 実際の相対力計算をここに
            .otherwise(None)
            .alias("cross_rel_strength")
        ])

    def get_join_summary(self) -> dict:
        """結合統計のサマリーを取得"""
        return self.join_stats
