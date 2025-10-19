"""
Flow Features Generator V2 - 改善版の週次フロー特徴量生成

主な改善点:
1. 正確な銘柄→Sectionマッピング（listed_infoから取得）
2. 適切なas-of結合（週次データを日次に前方補完）
3. 複数Sectionに属する銘柄への対応
"""

import logging
from datetime import timedelta

import polars as pl

logger = logging.getLogger(__name__)


class FlowFeaturesGeneratorV2:
    """
    改善版: 週次投資主体別売買動向からフロー特徴量を生成
    """

    def __init__(self, z_score_window: int = 52):
        self.z_score_window = z_score_window
        self.epsilon = 1e-12

    def create_section_mapping_from_listed_info(
        self,
        listed_info_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        listed_infoから正確な銘柄→Sectionマッピングを作成
        
        Args:
            listed_info_df: listed_info APIから取得したデータ
                           (Code, Date, MarketCode, SectorCode等を含む)
        
        Returns:
            銘柄コードとSectionのマッピングテーブル
        """
        # MarketCode → Section のマッピング定義
        market_to_section = {
            "0101": "TSEPrime",      # 東証プライム
            "0102": "TSEStandard",    # 東証スタンダード
            "0103": "TSEGrowth",      # 東証グロース
            "0104": "TSEPrime",       # 東証1部/2部（旧）→プライムとして扱う
            "0106": "TSEPrime",       # 東証JASDAQ（旧）→プライムとして扱う
            "0107": "TSEPrime",       # 東証マザーズ（旧）→プライムとして扱う
            "0108": "TSEPrime",       # TOKYO PRO Market（プロ向け）
            # 名証、福証、札証なども必要に応じて追加
            "0301": "NSEPremier",     # 名証プレミア
            "0302": "NSEMain",        # 名証メイン
        }

        # MarketCodeをSectionに変換
        # Note: Polars 1.x uses replace() instead of map_dict()
        mapping_df = listed_info_df.select([
            "Code",
            "Date",
            "MarketCode"
        ]).with_columns(
            pl.col("MarketCode").replace(market_to_section, default="Other").alias("Section")
        )

        # 最新の情報を取得（銘柄が市場を移動する場合があるため）
        mapping_df = mapping_df.sort(["Code", "Date"]).group_by("Code").last()

        logger.info(f"Created section mapping for {len(mapping_df)} stocks")
        logger.info(f"Section distribution: {mapping_df['Section'].value_counts()}")

        return mapping_df.select(["Code", "Section"])

    def build_flow_event_table_with_forward_fill(
        self,
        trades_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        週次フローイベント表を作成し、日次データ用に前方補完
        
        Args:
            trades_df: trades_specデータ（週次）
        
        Returns:
            日次に展開されたフロー特徴量テーブル
        """
        # 元の週次フロー特徴量を生成
        weekly_flow = self._build_weekly_flow_features(trades_df)

        # 日次データに展開（前方補完）
        daily_flow = self._expand_weekly_to_daily(weekly_flow)

        return daily_flow

    def _build_weekly_flow_features(self, trades_df: pl.DataFrame) -> pl.DataFrame:
        """週次フロー特徴量を生成（元のロジックを再利用）"""

        if trades_df is None or trades_df.is_empty():
            return pl.DataFrame()

        df = trades_df.sort(["Section", "PublishedDate"])

        # Date列の型変換
        if "PublishedDate" in df.columns:
            if df["PublishedDate"].dtype == pl.Utf8:
                df = df.with_columns(
                    pl.col("PublishedDate").str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
                )

        # 基本比率計算
        df = df.with_columns([
            (pl.col("ForeignersBalance") / (pl.col("ForeignersTotal") + self.epsilon))
                .alias("flow_foreign_net_ratio"),
            (pl.col("IndividualsBalance") / (pl.col("IndividualsTotal") + self.epsilon))
                .alias("flow_individual_net_ratio"),
            (pl.col("ForeignersTotal") / (pl.col("TotalTotal") + self.epsilon))
                .alias("flow_foreign_share_activity"),
        ])

        # Section内ローリングZ-score
        df = df.with_columns([
            self._rolling_zscore("ForeignersBalance", "Section").alias("foreign_balance_z"),
            self._rolling_zscore("IndividualsBalance", "Section").alias("individual_balance_z"),
            self._rolling_zscore("TotalTotal", "Section").alias("flow_activity_z"),
        ])

        # スマートマネー指標
        df = df.with_columns([
            (pl.col("foreign_balance_z") - pl.col("individual_balance_z"))
                .alias("flow_smart_money_idx")
        ])

        # 買い越し主体の広がり
        balance_cols = [
            "ProprietaryBalance", "BrokerageBalance", "IndividualsBalance",
            "ForeignersBalance", "SecuritiesCosBalance", "InvestmentTrustsBalance"
        ]

        available_balance_cols = [col for col in balance_cols if col in df.columns]

        if available_balance_cols:
            for col in available_balance_cols:
                df = df.with_columns(
                    (pl.col(col) > 0).cast(pl.Int8).alias(f"{col}_positive")
                )

            positive_cols = [f"{col}_positive" for col in available_balance_cols]
            df = df.with_columns(
                (pl.sum_horizontal(positive_cols) / len(available_balance_cols))
                    .alias("flow_breadth_pos")
            )
            df = df.drop(positive_cols)
        else:
            df = df.with_columns(pl.lit(0.5).alias("flow_breadth_pos"))

        # 4週モメンタム
        df = df.with_columns([
            pl.col("flow_smart_money_idx")
                .rolling_mean(window_size=4)
                .over("Section")
                .alias("smart_money_sma4")
        ])

        df = df.with_columns([
            (pl.col("flow_smart_money_idx") - pl.col("smart_money_sma4"))
                .alias("flow_smart_money_mom4")
        ])

        # ショックフラグ
        df = df.with_columns([
            pl.col("flow_smart_money_idx")
                .rolling_std(window_size=self.z_score_window)
                .over("Section")
                .alias("smart_money_std")
        ])

        df = df.with_columns([
            (pl.col("flow_smart_money_idx").abs() >= 2 * pl.col("smart_money_std"))
                .cast(pl.Int8)
                .alias("flow_shock_flag")
        ])

        # 必要な列のみ選択
        keep_cols = [
            "PublishedDate", "Section",
            "flow_foreign_net_ratio", "flow_individual_net_ratio",
            "flow_smart_money_idx", "flow_activity_z",
            "flow_foreign_share_activity", "flow_breadth_pos",
            "flow_smart_money_mom4", "flow_shock_flag"
        ]

        return df.select([col for col in keep_cols if col in df.columns])

    def _expand_weekly_to_daily(self, weekly_flow: pl.DataFrame) -> pl.DataFrame:
        """
        週次データを日次に展開（前方補完）
        
        金曜日に公表されたデータを翌週の月〜金まで使用
        """
        if weekly_flow.is_empty():
            return weekly_flow

        # 各週次データに対して有効期間を設定
        # PublishedDate（金曜日）の翌営業日から次の金曜日まで有効
        weekly_flow = weekly_flow.with_columns([
            (pl.col("PublishedDate") + timedelta(days=3)).alias("effective_start"),  # 翌月曜日
            (pl.col("PublishedDate") + timedelta(days=10)).alias("effective_end")    # 翌々金曜日
        ])

        return weekly_flow

    def _rolling_zscore(self, col_name: str, group_col: str) -> pl.Expr:
        """Section内でローリングZ-scoreを計算"""
        mu = pl.col(col_name).rolling_mean(self.z_score_window).over(group_col)
        sd = pl.col(col_name).rolling_std(self.z_score_window).over(group_col) + self.epsilon
        return (pl.col(col_name) - mu) / sd

    def attach_flow_to_daily_v2(
        self,
        stock_df: pl.DataFrame,
        flow_event_df: pl.DataFrame,
        section_mapping_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        改善版: 日次パネルにフロー特徴量を正確にas-of結合
        
        Args:
            stock_df: 銘柄日次データ
            flow_event_df: 週次フローイベント表（展開済み）
            section_mapping_df: 銘柄→Sectionマッピング
        
        Returns:
            フロー特徴量を含む日次パネル
        """
        # 1. 銘柄にSectionを付与
        stock_df = stock_df.join(section_mapping_df, on="Code", how="left")

        # Sectionがない銘柄はデフォルト値を設定
        stock_df = stock_df.with_columns(
            pl.col("Section").fill_null("Other")
        )

        # 2. Date列の型を統一
        if "Date" in stock_df.columns:
            stock_df = stock_df.with_columns(
                pl.col("Date").cast(pl.Date)
            )

        # 3. 各Section×日付でフロー特徴量を結合
        # as-of結合: 各日付に対して、その日付以前の最新の週次データを使用
        result_dfs = []

        for section in stock_df["Section"].unique():
            if section is None or section == "Other":
                # フローデータがないSectionは0埋め
                section_stocks = stock_df.filter(pl.col("Section") == section)
                flow_cols = [
                    "flow_foreign_net_ratio", "flow_individual_net_ratio",
                    "flow_smart_money_idx", "flow_activity_z",
                    "flow_foreign_share_activity", "flow_breadth_pos",
                    "flow_smart_money_mom4", "flow_shock_flag"
                ]
                for col in flow_cols:
                    section_stocks = section_stocks.with_columns(
                        pl.lit(0.0).alias(col)
                    )
                result_dfs.append(section_stocks)
                continue

            section_stocks = stock_df.filter(pl.col("Section") == section)
            section_flows = flow_event_df.filter(pl.col("Section") == section)

            if section_flows.is_empty():
                # フローデータがない場合
                for col in flow_cols:
                    section_stocks = section_stocks.with_columns(
                        pl.lit(0.0).alias(col)
                    )
                result_dfs.append(section_stocks)
                continue

            # as-of結合の実装
            # 各日付に対して、effective_start <= Date <= effective_end のフローデータを使用
            merged = self._perform_asof_join(section_stocks, section_flows)
            result_dfs.append(merged)

        if result_dfs:
            df = pl.concat(result_dfs)
        else:
            df = stock_df

        # 4. インパルスと経過日数を計算
        if "PublishedDate" in df.columns:
            df = df.with_columns([
                # 公表日から3営業日以内なら1
                ((pl.col("Date") - pl.col("PublishedDate")).dt.total_days() <= 3)
                    .cast(pl.Int8)
                    .alias("flow_impulse"),

                # 経過日数（営業日ベース）
                (pl.col("Date") - pl.col("PublishedDate"))
                    .dt.total_days()
                    .alias("days_since_flow")
            ])
        else:
            df = df.with_columns([
                pl.lit(0).cast(pl.Int8).alias("flow_impulse"),
                pl.lit(0).alias("days_since_flow")
            ])

        logger.info(f"✅ Attached flow features to {len(df)} daily records")

        return df

    def _perform_asof_join(
        self,
        section_stocks: pl.DataFrame,
        section_flows: pl.DataFrame
    ) -> pl.DataFrame:
        """
        適切なas-of結合を実行
        """
        # Polars 0.20+ ではjoin_asofが利用可能
        try:
            # 両方のデータフレームを日付でソート
            section_stocks = section_stocks.sort("Date")
            section_flows = section_flows.sort("effective_start")

            # join_asofを使用（Polarsのバージョンによっては利用不可）
            merged = section_stocks.join_asof(
                section_flows,
                left_on="Date",
                right_on="effective_start",
                strategy="backward",  # 過去の最新データを使用
                tolerance="7d"  # 7日以内のデータのみ使用
            )

        except AttributeError:
            # join_asofが使えない場合の代替実装
            logger.info("Using alternative as-of join implementation")

            # 各日付に対して有効なフローデータを探す
            flow_cols = [
                "flow_foreign_net_ratio", "flow_individual_net_ratio",
                "flow_smart_money_idx", "flow_activity_z",
                "flow_foreign_share_activity", "flow_breadth_pos",
                "flow_smart_money_mom4", "flow_shock_flag"
            ]

            # 初期化
            for col in flow_cols:
                section_stocks = section_stocks.with_columns(
                    pl.lit(None).alias(col)
                )

            # 簡易的な実装: 範囲内のデータを結合
            for _, flow_row in section_flows.iter_rows(named=True):
                mask = (
                    (section_stocks["Date"] >= flow_row["effective_start"]) &
                    (section_stocks["Date"] <= flow_row["effective_end"])
                )

                for col in flow_cols:
                    if col in flow_row:
                        section_stocks = section_stocks.with_columns(
                            pl.when(mask)
                            .then(pl.lit(flow_row[col]))
                            .otherwise(pl.col(col))
                            .alias(col)
                        )

            # Noneを0で埋める
            for col in flow_cols:
                section_stocks = section_stocks.with_columns(
                    pl.col(col).fill_null(0.0)
                )

            merged = section_stocks

        return merged
