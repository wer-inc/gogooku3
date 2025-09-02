"""
Flow Features Generator - 週次投資主体別売買動向特徴量

このモジュールは、trades_spec（売買内訳）データから週次フローイベント特徴量を生成します。
外国人vs個人投資家の資金フローを中心に、短期1-3日予測に有効な需給コンテキストを提供します。

特徴量カテゴリ:
- コア特徴（flow_* prefix、8個）: ネット比率、スマートマネー指標、活動度等
- 付加特徴（2個）: インパルス、経過日数

Author: gogooku3 team
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class FlowFeaturesGenerator:
    """
    週次投資主体別売買動向からフロー特徴量を生成
    
    8個のコア特徴量 + 2個の付加特徴量を生成:
    - flow_foreign_net_ratio: 外国人ネット買い比率
    - flow_individual_net_ratio: 個人ネット買い比率
    - flow_smart_money_idx: スマートマネー指標（外国人vs個人のZ-score差）
    - flow_activity_z: 市場活動度Z-score
    - flow_foreign_share_activity: 外国人シェア
    - flow_breadth_pos: 買い越し主体の広がり
    - flow_smart_money_mom4: スマートマネー4週モメンタム
    - flow_shock_flag: 極端フローフラグ
    - flow_impulse: 公表日インパルス
    - days_since_flow: 公表からの経過日数
    """
    
    def __init__(self, z_score_window: int = 52):
        """
        Args:
            z_score_window: Z-score計算の窓サイズ（デフォルト52週=1年）
        """
        self.z_score_window = z_score_window
        self.epsilon = 1e-12  # ゼロ除算防止
    
    def build_flow_event_table(self, trades_df: pl.DataFrame) -> pl.DataFrame:
        """
        週次フローイベント表を作成（Section単位）
        
        Args:
            trades_df: trades_specデータ（PublishedDate, Section, 各主体のBalance等）
        
        Returns:
            フロー特徴量を含むイベント表
        """
        if trades_df is None or trades_df.is_empty():
            logger.warning("No trades_spec data provided")
            return pl.DataFrame()
        
        df = trades_df.sort(["Section", "PublishedDate"])
        
        # Date列の型を確認・変換
        if "PublishedDate" in df.columns:
            # 文字列の場合は日付型に変換
            if df["PublishedDate"].dtype == pl.Utf8:
                df = df.with_columns(
                    pl.col("PublishedDate").str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
                )
        
        # ========== 基本比率計算 ==========
        df = df.with_columns([
            # 外国人ネット買い比率
            (pl.col("ForeignersBalance") / (pl.col("ForeignersTotal") + self.epsilon)).alias("flow_foreign_net_ratio"),
            
            # 個人ネット買い比率
            (pl.col("IndividualsBalance") / (pl.col("IndividualsTotal") + self.epsilon)).alias("flow_individual_net_ratio"),
            
            # 外国人活動シェア
            (pl.col("ForeignersTotal") / (pl.col("TotalTotal") + self.epsilon)).alias("flow_foreign_share_activity"),
        ])
        
        # ========== Section内ローリング52週Z-score ==========
        # 各Sectionごとに計算
        df = df.with_columns([
            # 外国人バランスのZ-score
            self._rolling_zscore("ForeignersBalance", "Section").alias("foreign_balance_z"),
            
            # 個人バランスのZ-score
            self._rolling_zscore("IndividualsBalance", "Section").alias("individual_balance_z"),
            
            # 全体活動度のZ-score
            self._rolling_zscore("TotalTotal", "Section").alias("flow_activity_z"),
        ])
        
        # スマートマネー指標（外国人Z - 個人Z）
        df = df.with_columns([
            (pl.col("foreign_balance_z") - pl.col("individual_balance_z")).alias("flow_smart_money_idx")
        ])
        
        # ========== 買い越し主体の広がり ==========
        # 各主体のBalanceが正の数をカウント
        balance_cols = [
            "ProprietaryBalance", "BrokerageBalance", "IndividualsBalance",
            "ForeignersBalance", "SecuritiesCosBalance", "InvestmentTrustsBalance",
            "BusinessCosBalance", "InsuranceCosBalance", "TrustBanksBalance",
            "CityBKsRegionalBKsEtcBalance", "OtherFinancialInstitutionsBalance"
        ]
        
        # 利用可能な列のみ使用
        available_balance_cols = [col for col in balance_cols if col in df.columns]
        
        if available_balance_cols:
            # 各主体が買い越しかどうかのフラグを作成
            for col in available_balance_cols:
                df = df.with_columns(
                    (pl.col(col) > 0).cast(pl.Int8).alias(f"{col}_positive")
                )
            
            # 買い越し主体数をカウント
            positive_cols = [f"{col}_positive" for col in available_balance_cols]
            df = df.with_columns(
                pl.sum_horizontal(positive_cols).alias("positive_count")
            )
            
            # 買い越し比率
            df = df.with_columns(
                (pl.col("positive_count") / len(available_balance_cols)).alias("flow_breadth_pos")
            )
            
            # 中間変数を削除
            df = df.drop(positive_cols + ["positive_count"])
        else:
            df = df.with_columns(pl.lit(0.5).alias("flow_breadth_pos"))
        
        # ========== スマートマネー4週モメンタム ==========
        # 4週移動平均との差
        df = df.with_columns([
            pl.col("flow_smart_money_idx")
            .rolling_mean(window_size=4)
            .over("Section")
            .alias("smart_money_sma4")
        ])
        
        df = df.with_columns([
            (pl.col("flow_smart_money_idx") - pl.col("smart_money_sma4")).alias("flow_smart_money_mom4")
        ])
        
        # ========== ショックフラグ ==========
        # スマートマネー指標の絶対値が2σを超える場合
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
        
        # ========== 有効性フラグ ==========
        # 52週分のデータが揃っているかチェック
        df = df.with_columns([
            pl.col("PublishedDate")
            .cum_count()
            .over("Section")
            .alias("week_count")
        ])
        
        df = df.with_columns([
            (pl.col("week_count") >= self.z_score_window)
            .cast(pl.Int8)
            .alias("is_flow_valid")
        ])
        
        # ========== effective_date（T+1営業日）を計算 ==========
        # 簡易的に+1日（実際は営業日カレンダーを使うべき）
        df = df.with_columns([
            (pl.col("PublishedDate") + timedelta(days=1)).alias("effective_date")
        ])
        
        # 必要な列のみ選択
        keep_cols = [
            "PublishedDate", "effective_date", "Section",
            "flow_foreign_net_ratio", "flow_individual_net_ratio",
            "flow_smart_money_idx", "flow_activity_z",
            "flow_foreign_share_activity", "flow_breadth_pos",
            "flow_smart_money_mom4", "flow_shock_flag",
            "is_flow_valid"
        ]
        
        # 中間変数を削除
        drop_cols = ["foreign_balance_z", "individual_balance_z", "smart_money_sma4", 
                     "smart_money_std", "week_count"]
        df = df.drop([col for col in drop_cols if col in df.columns])
        
        result_df = df.select([col for col in keep_cols if col in df.columns])
        
        logger.info(f"✅ Generated flow event table: {result_df.shape}")
        
        return result_df
    
    def _rolling_zscore(self, col_name: str, group_col: str) -> pl.Expr:
        """
        Section内でローリングZ-scoreを計算
        
        Args:
            col_name: Z-score計算対象の列名
            group_col: グループ化列名（Section）
        
        Returns:
            Z-score計算式
        """
        mu = pl.col(col_name).rolling_mean(self.z_score_window).over(group_col)
        sd = pl.col(col_name).rolling_std(self.z_score_window).over(group_col) + self.epsilon
        return (pl.col(col_name) - mu) / sd
    
    def attach_flow_to_daily(
        self, 
        stock_df: pl.DataFrame,
        flow_event_df: pl.DataFrame,
        section_mapping: Optional[Dict[str, str]] = None
    ) -> pl.DataFrame:
        """
        日次パネルにフロー特徴量をas-of結合
        
        Args:
            stock_df: 銘柄日次データ（Code, Date等）
            flow_event_df: フローイベント表
            section_mapping: 銘柄→Sectionのマッピング（省略時は簡易マッピング）
        
        Returns:
            フロー特徴量を含む日次パネル
        """
        if flow_event_df is None or flow_event_df.is_empty():
            logger.warning("No flow event data to attach")
            return stock_df
        
        # Date列の型を統一
        if "Date" in stock_df.columns:
            stock_df = stock_df.with_columns(
                pl.col("Date").cast(pl.Date)
            )
        
        if "effective_date" in flow_event_df.columns:
            flow_event_df = flow_event_df.with_columns(
                pl.col("effective_date").cast(pl.Date)
            )
        
        # ========== Section マッピング ==========
        # 簡易版: コード範囲でSectionを推定（実際は銘柄マスタから取得すべき）
        if section_mapping is None:
            # デフォルトマッピング（例）
            stock_df = stock_df.with_columns([
                pl.when(pl.col("Code") < "5000")
                .then(pl.lit("TSEPrime"))
                .when(pl.col("Code") < "7000")
                .then(pl.lit("TSEStandard"))
                .otherwise(pl.lit("TSEGrowth"))
                .alias("Section")
            ])
        else:
            # 提供されたマッピングを使用
            mapping_df = pl.DataFrame({
                "Code": list(section_mapping.keys()),
                "Section": list(section_mapping.values())
            })
            stock_df = stock_df.join(mapping_df, on="Code", how="left")
        
        # ========== as-of結合 ==========
        # 各銘柄・日付に対して、最新のフローイベントを結合
        # polarsではjoin_asofが直接使えないため、通常のjoinで対応
        
        # フロー特徴量列
        flow_cols = [
            "flow_foreign_net_ratio", "flow_individual_net_ratio",
            "flow_smart_money_idx", "flow_activity_z",
            "flow_foreign_share_activity", "flow_breadth_pos",
            "flow_smart_money_mom4", "flow_shock_flag",
            "is_flow_valid", "effective_date"
        ]
        
        # 利用可能な列のみ選択
        available_flow_cols = [col for col in flow_cols if col in flow_event_df.columns]
        
        # Section別に結合
        result_dfs = []
        for section in stock_df["Section"].unique():
            if section is None:
                continue
                
            section_stocks = stock_df.filter(pl.col("Section") == section)
            section_flows = flow_event_df.filter(pl.col("Section") == section)
            
            if section_flows.is_empty():
                # フローデータがない場合は0埋め
                for col in available_flow_cols:
                    if col != "effective_date":
                        section_stocks = section_stocks.with_columns(
                            pl.lit(0).alias(col)
                        )
                result_dfs.append(section_stocks)
                continue
            
            # 各日付に対して最新のeffective_dateを見つける
            # これは簡易実装で、実際にはもっと効率的な方法がある
            for col in available_flow_cols:
                if col == "effective_date":
                    continue
                section_stocks = section_stocks.with_columns(
                    pl.lit(None).alias(col)
                )
            
            # 実際の結合処理（簡易版）
            # TODO: より効率的なas-of結合の実装
            # effective_dateは既にavailable_flow_colsに含まれているので除外
            flow_cols_for_join = [col for col in available_flow_cols if col != "effective_date"]
            merged = section_stocks.join(
                section_flows.select(["effective_date"] + flow_cols_for_join),
                left_on="Date",
                right_on="effective_date",
                how="left"
            )
            
            result_dfs.append(merged)
        
        if result_dfs:
            df = pl.concat(result_dfs)
        else:
            df = stock_df
        
        # ========== flow_impulse と days_since_flow ==========
        if "effective_date" in df.columns:
            df = df.with_columns([
                # インパルス: 公表日当日なら1
                (pl.col("Date") == pl.col("effective_date"))
                .cast(pl.Int8)
                .alias("flow_impulse"),
                
                # 経過日数
                (pl.col("Date") - pl.col("effective_date"))
                .dt.total_days()
                .fill_null(0)
                .alias("days_since_flow")
            ])
        else:
            df = df.with_columns([
                pl.lit(0).cast(pl.Int8).alias("flow_impulse"),
                pl.lit(0).alias("days_since_flow")
            ])
        
        # effective_dateは不要なので削除
        if "effective_date" in df.columns:
            df = df.drop("effective_date")
        
        logger.info(f"✅ Attached flow features to daily panel")
        
        return df