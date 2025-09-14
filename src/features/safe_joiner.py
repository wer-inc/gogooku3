"""
Safe Joiner - 安全なas-of結合実装
daily_quotes を基盤として各データソースを時間整合性を保って結合
"""

import logging
from datetime import time

import polars as pl

from .calendar_utils import TradingCalendarUtil
from .section_mapper import SectionMapper

logger = logging.getLogger(__name__)


class SafeJoiner:
    """
    時間整合性を保証した安全な結合処理
    """

    def __init__(
        self,
        calendar_util: TradingCalendarUtil | None = None,
        section_mapper: SectionMapper | None = None
    ):
        """
        Args:
            calendar_util: 営業日カレンダーユーティリティ
            section_mapper: Sectionマッピングユーティリティ
        """
        self.calendar_util = calendar_util or TradingCalendarUtil()
        self.section_mapper = section_mapper or SectionMapper()
        self.join_stats = {}

    def prepare_base_quotes(self, quotes_df: pl.DataFrame) -> pl.DataFrame:
        """
        daily_quotes を基盤データとして準備
        
        Args:
            quotes_df: daily_quotes データ
        
        Returns:
            型変換・重複排除済みの基盤データ
        """
        logger.info("Preparing base quotes data...")

        # 型変換（Codeは必ず文字列で0埋め4桁）
        quotes = quotes_df.with_columns([
            pl.col("Code").cast(pl.Utf8).str.zfill(4),
            pl.col("Date").cast(pl.Date),
        ])

        # 重複チェック
        duplicate_count = quotes.group_by(["Code", "Date"]).count().filter(
            pl.col("count") > 1
        ).shape[0]

        if duplicate_count > 0:
            logger.warning(f"Found {duplicate_count} duplicate (Code, Date) pairs, removing...")
            quotes = quotes.unique(subset=["Code", "Date"], keep="last")

        # 必要な列の存在確認
        required_cols = ["Code", "Date", "Open", "High", "Low", "Close", "Volume"]
        missing_cols = set(required_cols) - set(quotes.columns)
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Missing columns in quotes data: {missing_cols}")

        logger.info(f"Base quotes prepared: {len(quotes)} rows, {quotes['Code'].n_unique()} stocks")

        return quotes

    def join_statements_asof(
        self,
        base_df: pl.DataFrame,
        statements_df: pl.DataFrame,
        use_time_cutoff: bool = True,
        cutoff_time: time = time(15, 0)
    ) -> pl.DataFrame:
        """
        財務諸表データをas-of結合（T+1ルール）
        
        Args:
            base_df: 基盤データ（daily_quotes）
            statements_df: 財務諸表データ
            use_time_cutoff: 15:00判定を使用するか（デフォルトTrue）
            cutoff_time: 当日有効とする締切時刻
        
        Returns:
            財務諸表特徴量が付与されたデータ
        """
        logger.info("Joining statements with as-of rule...")

        if statements_df.is_empty():
            logger.warning("Statements DataFrame is empty, skipping join")
            return base_df

        # base_dfのCode列も確実に文字列型で0埋め
        base_df = base_df.with_columns([
            pl.col("Code").cast(pl.Utf8).str.zfill(4)
        ])

        # statementsの準備（LocalCode/Code正規化）
        if "LocalCode" in statements_df.columns:
            stm = statements_df.with_columns([
                pl.col("LocalCode").cast(pl.Utf8).str.zfill(4).alias("Code")
            ])
        elif "Code" in statements_df.columns:
            # Codeが数値型の場合もあるため、まずUtf8にキャストして0埋め
            # 型に関わらず常にUtf8にキャストしてから処理
            stm = statements_df.with_columns([
                pl.col("Code").cast(pl.Utf8).str.zfill(4)
            ])
        else:
            logger.error("No Code/LocalCode column found in statements")
            return base_df

        # 日付型変換（文字列から日付型へ）
        if stm.schema.get("DisclosedDate") == pl.Utf8:
            stm = stm.with_columns([
                pl.col("DisclosedDate").str.strptime(pl.Date, format="%Y-%m-%d", strict=False).alias("disclosed_date")
            ])
        else:
            stm = stm.with_columns([
                pl.col("DisclosedDate").cast(pl.Date).alias("disclosed_date")
            ])

        # 開示時刻の処理とT+1有効日計算
        if "DisclosedTime" in stm.columns and use_time_cutoff:
            # タイムスタンプを作成
            stm = stm.with_columns([
                pl.datetime(
                    pl.col("disclosed_date").dt.year(),
                    pl.col("disclosed_date").dt.month(),
                    pl.col("disclosed_date").dt.day(),
                    pl.col("DisclosedTime").str.slice(0, 2).cast(pl.Int32, strict=False).fill_null(0),
                    pl.col("DisclosedTime").str.slice(3, 2).cast(pl.Int32, strict=False).fill_null(0),
                    0
                ).alias("disclosed_ts")
            ])

            # 時刻判定によるT+0/T+1
            stm = stm.with_columns([
                pl.when(pl.col("disclosed_ts").dt.time() < cutoff_time)
                .then(pl.col("disclosed_date"))  # 当日有効
                .otherwise(
                    self._next_business_day_expr(pl.col("disclosed_date"))  # T+1
                )
                .alias("effective_date")
            ])
        else:
            # 保守的にT+1ルール
            stm = stm.with_columns([
                self._next_business_day_expr(pl.col("disclosed_date")).alias("effective_date")
            ])

        # 同日・同銘柄の重複を除去（最新のみ保持）
        if "disclosed_ts" in stm.columns:
            stm = stm.sort(["Code", "disclosed_ts"]).group_by(["Code", "effective_date"]).tail(1)
        else:
            stm = stm.sort(["Code", "disclosed_date"]).group_by(["Code", "effective_date"]).tail(1)

        # 財務特徴量の計算
        stm = self._calculate_statement_features(stm)

        # 結合用の列を選択
        stmt_feature_cols = [
            "stmt_yoy_sales", "stmt_yoy_op", "stmt_yoy_np",
            "stmt_opm", "stmt_npm",
            "stmt_progress_op", "stmt_progress_np",
            "stmt_rev_fore_op", "stmt_rev_fore_np", "stmt_rev_fore_eps", "stmt_rev_div_fore",
            "stmt_roe", "stmt_roa",
            "stmt_change_in_est", "stmt_nc_flag"
        ]

        available_stmt_cols = [col for col in stmt_feature_cols if col in stm.columns]
        join_cols = ["Code", "effective_date"] + available_stmt_cols

        # ソートしてas-of結合
        stm_for_join = stm.select(join_cols).sort(["Code", "effective_date"])
        base_sorted = base_df.sort(["Code", "Date"])

        # as-of結合（backward: その日以前の最新の開示を使用）
        result = base_sorted.join_asof(
            stm_for_join,
            left_on="Date",
            right_on="effective_date",
            by="Code",
            strategy="backward"
        )

        # インパルスと経過日数の計算
        if "effective_date" in result.columns:
            result = result.with_columns([
                # 開示当日フラグ
                (pl.col("Date") == pl.col("effective_date"))
                    .cast(pl.Int8)
                    .alias("stmt_imp_statement"),
                # 開示からの経過日数
                (pl.col("Date") - pl.col("effective_date"))
                    .dt.total_days()
                    .alias("stmt_days_since_statement")
            ])

            # effective_dateは不要なので削除
            result = result.drop("effective_date")
        else:
            # 結合できなかった場合のデフォルト値
            result = result.with_columns([
                pl.lit(0).cast(pl.Int8).alias("stmt_imp_statement"),
                pl.lit(-1).alias("stmt_days_since_statement")
            ])

        # 欠損値処理と有効性フラグ
        result = result.with_columns([
            # 数値特徴量の欠損値を0で埋める
            *[pl.col(c).fill_null(0.0) for c in available_stmt_cols
              if result.schema.get(c, pl.Float64) in [pl.Float64, pl.Float32]],
            # フラグ系の欠損値を0で埋める
            pl.col("stmt_imp_statement").fill_null(0),
            pl.col("stmt_days_since_statement").fill_null(-1),
            # 財務データ有効性フラグ
            (pl.col("stmt_days_since_statement") >= 0).cast(pl.Int8).alias("is_stmt_valid")
        ])

        # カバレッジ統計
        valid_count = result.filter(pl.col("is_stmt_valid") == 1).height
        coverage = valid_count / len(result) if len(result) > 0 else 0

        self.join_stats["statements"] = {
            "coverage": coverage,
            "rows_with_data": valid_count,
            "total_rows": len(result),
            "features_added": len(available_stmt_cols)
        }

        logger.info(f"  Statements coverage: {coverage:.1%} ({valid_count:,}/{len(result):,} rows)")
        logger.info(f"  Added {len(available_stmt_cols)} statement features")

        return result

    def join_trades_spec_interval(
        self,
        base_df: pl.DataFrame,
        trades_df: pl.DataFrame,
        section_mapping_df: pl.DataFrame | None = None
    ) -> pl.DataFrame:
        """
        週次フローデータを区間結合
        
        Args:
            base_df: 基盤データ（Section付き）
            trades_df: trades_spec データ
            section_mapping_df: Section マッピング（省略時は base_df から推定）
        
        Returns:
            フロー特徴量が付与されたデータ
        """
        logger.info("Joining trades_spec with interval rule...")

        # Sectionを付与（まだない場合）
        if "Section" not in base_df.columns:
            if section_mapping_df is not None:
                base_df = self.section_mapper.attach_section_to_daily(
                    base_df, section_mapping_df
                )
            else:
                # 簡易マッピング（非推奨）
                logger.warning("Using simplified section mapping")
                base_df = base_df.with_columns([
                    pl.when(pl.col("Code") < "5000")
                    .then(pl.lit("TSEPrime"))
                    .when(pl.col("Code") < "7000")
                    .then(pl.lit("TSEStandard"))
                    .otherwise(pl.lit("TSEGrowth"))
                    .alias("Section")
                ])

        # trades_spec の準備
        flow = trades_df.with_columns([
            pl.col("PublishedDate").cast(pl.Date),
            pl.col("Section").cast(pl.Utf8)
        ])

        # effective_start = 翌営業日
        flow = flow.with_columns([
            self._next_business_day_expr(pl.col("PublishedDate")).alias("effective_start")
        ])

        # Section内でソート
        flow = flow.sort(["Section", "effective_start"])

        # 次のeffective_startを取得
        flow = flow.with_columns([
            pl.col("effective_start").shift(-1).over("Section").alias("next_start")
        ])

        # effective_end = next_start - 1日（最後は2999-12-31）
        flow = flow.with_columns([
            pl.when(pl.col("next_start").is_not_null())
            .then(pl.col("next_start") - pl.duration(days=1))
            .otherwise(pl.date(2999, 12, 31))
            .alias("effective_end")
        ])

        # フロー特徴量の計算
        flow = self._calculate_flow_features(flow)

        # 区間結合（asof + フィルタ方式）
        flow_cols = [col for col in flow.columns if col.startswith("flow_")]

        # まずasofで候補を取得
        result = base_df.sort(["Section", "Date"]).join_asof(
            flow.select(["Section", "effective_start", "effective_end"] + flow_cols),
            left_on="Date",
            right_on="effective_start",
            by="Section",
            strategy="backward"
        )

        # 区間内のみ残す
        result = result.with_columns([
            pl.when(
                (pl.col("Date") >= pl.col("effective_start")) &
                (pl.col("Date") <= pl.col("effective_end"))
            )
            .then(pl.col(col))
            .otherwise(None)
            .alias(col)
            for col in flow_cols
        ])

        # インパルスと経過日数
        result = result.with_columns([
            pl.when(pl.col("Date") == pl.col("effective_start"))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .cast(pl.Int8)
            .alias("flow_impulse"),

            pl.when(pl.col("effective_start").is_not_null())
            .then((pl.col("Date") - pl.col("effective_start")).dt.total_days())
            .otherwise(pl.lit(999))
            .alias("days_since_flow")
        ])

        # 不要な列を削除
        result = result.drop(["effective_start", "effective_end"])

        # 統計情報
        coverage = result.filter(pl.col("days_since_flow") < 999).shape[0] / len(result)
        self.join_stats["trades_spec"] = {
            "coverage": coverage,
            "rows_with_data": result.filter(pl.col("days_since_flow") < 999).shape[0],
            "total_rows": len(result)
        }

        logger.info(f"  Trades_spec coverage: {coverage:.1%}")

        return result

    def join_topix_same_day(
        self,
        base_df: pl.DataFrame,
        topix_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        TOPIXデータを同日結合
        
        Args:
            base_df: 基盤データ
            topix_df: TOPIXデータ（市場特徴量計算済み）
        
        Returns:
            市場特徴量が付与されたデータ
        """
        logger.info("Joining TOPIX with same-day rule...")

        # TOPIXデータの準備
        topix = topix_df.with_columns([
            pl.col("Date").cast(pl.Date)
        ])

        # mkt_* 特徴量の確認
        mkt_cols = [col for col in topix.columns if col.startswith("mkt_")]

        if not mkt_cols:
            logger.warning("No mkt_* features found in TOPIX data")

        # 同日結合
        result = base_df.join(
            topix.select(["Date"] + mkt_cols),
            on="Date",
            how="left",
            coalesce=True
        )

        # 銘柄×市場Cross特徴量の計算
        if mkt_cols:
            result = self._calculate_cross_features(result)

        # 統計情報
        if mkt_cols:
            coverage = result[mkt_cols[0]].is_not_null().sum() / len(result)
            self.join_stats["topix"] = {
                "coverage": coverage,
                "mkt_features": len(mkt_cols),
                "cross_features": len([c for c in result.columns if c.startswith("cross_")])
            }
            logger.info(f"  TOPIX coverage: {coverage:.1%}")

        return result

    def _next_business_day_expr(self, date_col: pl.Expr) -> pl.Expr:
        """次営業日を計算するPolars式（簡易版）"""
        # TODO: calendar_util を使った正確な実装
        # 簡易版: +1日（週末なら月曜まで）
        return (pl.when(date_col.dt.weekday() == 4)  # 金曜
                .then(date_col + pl.duration(days=3))    # 月曜
                .when(date_col.dt.weekday() == 5)            # 土曜
                .then(date_col + pl.duration(days=2))    # 月曜
                .otherwise(date_col + pl.duration(days=1)))   # 翌日

    def _calculate_statement_features(self, stm: pl.DataFrame) -> pl.DataFrame:
        """
        財務諸表特徴量を計算
        P0-3: YoY成長率をFY×Qベースで正確に計算
        """
        # コード列の正規化（LocalCode → Code）
        if "LocalCode" in stm.columns and "Code" not in stm.columns:
            stm = stm.with_columns([
                pl.col("LocalCode").cast(pl.Utf8).str.zfill(4).alias("Code")
            ])
        elif "Code" in stm.columns:
            stm = stm.with_columns([
                pl.col("Code").cast(pl.Utf8).str.zfill(4)
            ])

        # 前提: Code, DisclosedDate, 主要数値が存在
        by = ["Code"]

        # P0-3: FY×Q ベースの正確な処理
        s = stm.sort(["Code", "DisclosedDate"])

        # FiscalYear列の処理
        if "FiscalYear" in stm.columns:
            s = s.with_columns([
                pl.col("FiscalYear").cast(pl.Int32, strict=False).alias("fiscal_year")
            ])
        else:
            s = s.with_columns([
                pl.lit(None).cast(pl.Int32).alias("fiscal_year")
            ])

        # Quarter列の処理
        s = s.with_columns([
            (
                pl.when(pl.col("TypeOfCurrentPeriod").str.contains("1Q"))
                .then(1)
                .when(pl.col("TypeOfCurrentPeriod").str.contains("2Q"))
                .then(2)
                .when(pl.col("TypeOfCurrentPeriod").str.contains("3Q"))
                .then(3)
                .when(pl.col("TypeOfCurrentPeriod").str.contains("FY|4Q"))
                .then(4)
                .otherwise(0)
                if "TypeOfCurrentPeriod" in stm.columns
                else pl.lit(0)
            ).alias("quarter"),
        ])

        # 数値演算に用いる主要カラムを安全に数値型へキャスト
        # JQuants APIのレスポンスは環境により文字列になることがあるため、strict=FalseでFloat64へ
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
        existing_numeric = [c for c in numeric_cols if c in s.columns]
        if existing_numeric:
            s = s.with_columns([
                pl.col(c).cast(pl.Float64, strict=False).alias(c) for c in existing_numeric
            ])

        # P0-3: 前年同期を (FY-1, Q) で特定
        s = s.with_columns([
            (pl.col("fiscal_year") - 1).alias("prev_fy")
        ])

        # 前年同期データを自己結合で取得（この時点でNetSales等は数値型）
        yoy_base = s.select([
            pl.col("Code"),
            pl.col("fiscal_year").alias("base_fy"),
            pl.col("quarter").alias("base_q"),
            pl.col("NetSales").alias("yoy_sales_base"),
            pl.col("OperatingProfit").alias("yoy_op_base"),
            pl.col("Profit").alias("yoy_np_base"),
        ])

        # 前年同期と結合
        s = s.join(
            yoy_base,
            left_on=["Code", "prev_fy", "quarter"],
            right_on=["Code", "base_fy", "base_q"],
            how="left",
            coalesce=True
        ).drop(["base_fy", "base_q"])

        # 直近値（前回開示）の準備
        s = s.with_columns([
            pl.col("NetSales").shift(1).over(by).alias("prev_sales"),
            pl.col("OperatingProfit").shift(1).over(by).alias("prev_op"),
            pl.col("Profit").shift(1).over(by).alias("prev_np"),
            pl.col("ForecastOperatingProfit").shift(1).over(by).alias("prev_fore_op"),
            pl.col("ForecastProfit").shift(1).over(by).alias("prev_fore_np"),
            pl.col("ForecastEarningsPerShare").shift(1).over(by).alias("prev_fore_eps"),
            pl.col("ForecastDividendPerShareAnnual").shift(1).over(by).alias("prev_fore_div"),
        ])

        # パーセント変化率計算用ヘルパー
        def pct_change(numerator, denominator):
            return ((numerator - denominator) / (denominator.abs() + 1e-12))

        # 財務特徴量の計算
        s = s.with_columns([
            # YoY成長率
            pct_change(pl.col("NetSales"), pl.col("yoy_sales_base")).alias("stmt_yoy_sales"),
            pct_change(pl.col("OperatingProfit"), pl.col("yoy_op_base")).alias("stmt_yoy_op"),
            pct_change(pl.col("Profit"), pl.col("yoy_np_base")).alias("stmt_yoy_np"),

            # マージン
            (pl.col("OperatingProfit") / (pl.col("NetSales") + 1e-12)).alias("stmt_opm"),
            (pl.col("Profit") / (pl.col("NetSales") + 1e-12)).alias("stmt_npm"),

            # 通期ガイダンス進捗率
            (pl.col("OperatingProfit") / (pl.col("ForecastOperatingProfit") + 1e-12)).alias("stmt_progress_op"),
            (pl.col("Profit") / (pl.col("ForecastProfit") + 1e-12)).alias("stmt_progress_np"),

            # ガイダンス改定率（今回-前回）/|前回|
            pct_change(pl.col("ForecastOperatingProfit"), pl.col("prev_fore_op")).alias("stmt_rev_fore_op"),
            pct_change(pl.col("ForecastProfit"), pl.col("prev_fore_np")).alias("stmt_rev_fore_np"),
            pct_change(pl.col("ForecastEarningsPerShare"), pl.col("prev_fore_eps")).alias("stmt_rev_fore_eps"),
            pct_change(
                pl.col("ForecastDividendPerShareAnnual").cast(pl.Float64, strict=False),
                pl.col("prev_fore_div").cast(pl.Float64, strict=False)
            ).alias("stmt_rev_div_fore"),

            # ROE/ROA（簡易版）
            (pl.col("Profit") / (pl.col("Equity") + 1e-12)).alias("stmt_roe"),
            (pl.col("Profit") / (pl.col("TotalAssets") + 1e-12)).alias("stmt_roa"),

        ])

        # 品質フラグ（列が存在する場合のみ処理）
        if "ChangesInAccountingEstimates" in s.columns:
            s = s.with_columns([
                pl.when(pl.col("ChangesInAccountingEstimates").is_not_null())
                .then(
                    pl.col("ChangesInAccountingEstimates")
                    .cast(pl.Utf8, strict=False)
                    .str.to_lowercase()
                    .is_in(["true", "1", "yes"])
                )
                .otherwise(False)
                .alias("stmt_change_in_est")
            ])
        else:
            s = s.with_columns([
                pl.lit(False).alias("stmt_change_in_est")
            ])

        # 比較不能フラグ（会計方針変更等）
        nc_flag_expr = pl.lit(False)
        if "ChangesBasedOnRevisionsOfAccountingStandard" in s.columns:
            nc_flag_expr = nc_flag_expr | (
                pl.col("ChangesBasedOnRevisionsOfAccountingStandard")
                .cast(pl.Utf8, strict=False)
                .str.to_lowercase()
                .is_in(["true", "1", "yes"])
                .fill_null(False)
            )
        if "RetrospectiveRestatement" in s.columns:
            nc_flag_expr = nc_flag_expr | (
                pl.col("RetrospectiveRestatement")
                .cast(pl.Utf8, strict=False)
                .str.to_lowercase()
                .is_in(["true", "1", "yes"])
                .fill_null(False)
            )

        s = s.with_columns([
            nc_flag_expr.alias("stmt_nc_flag")
        ])

        # 中間列を削除
        drop_cols = ["prev_sales", "prev_op", "prev_np", "prev_fore_op", "prev_fore_np",
                     "prev_fore_eps", "prev_fore_div", "yoy_sales_base", "yoy_op_base", "yoy_np_base"]
        s = s.drop([col for col in drop_cols if col in s.columns])

        return s

    def _calculate_flow_features(self, flow: pl.DataFrame) -> pl.DataFrame:
        """フロー特徴量を計算（簡略版）"""
        # TODO: 実際のフロー特徴量計算
        # ここでは例として基本的な特徴量を追加
        if "ForeignersBalance" in flow.columns:
            flow = flow.with_columns([
                (pl.col("ForeignersBalance") / (pl.col("ForeignersTotal") + 1e-12))
                    .alias("flow_foreign_net_ratio"),
                (pl.col("IndividualsBalance") / (pl.col("IndividualsTotal") + 1e-12))
                    .alias("flow_individual_net_ratio")
            ])
        else:
            flow = flow.with_columns([
                pl.lit(0.0).alias("flow_foreign_net_ratio"),
                pl.lit(0.0).alias("flow_individual_net_ratio")
            ])

        return flow

    def _calculate_cross_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """銘柄×市場Cross特徴量を計算（簡略版）"""
        # TODO: beta, alpha等の実際の計算
        # ここでは例として簡単な特徴量を追加
        return df.with_columns([
            pl.lit(0.0).alias("cross_beta_60d"),
            pl.lit(0.0).alias("cross_alpha_1d"),
            pl.lit(0.0).alias("cross_rel_strength")
        ])

    def get_join_summary(self) -> dict:
        """結合統計のサマリーを取得"""
        return self.join_stats
