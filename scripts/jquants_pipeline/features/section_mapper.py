"""
Section マッピングユーティリティ
listed_info から正確な MarketCode → Section 変換
"""

import polars as pl
from typing import Dict, Optional, List
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)


class SectionMapper:
    """
    銘柄コード → Section（市場区分）のマッピング管理
    """
    
    # MarketCode → Section のマッピング定義
    MARKET_TO_SECTION = {
        "0101": "TSEPrime",      # 東証プライム
        "0102": "TSEStandard",    # 東証スタンダード
        "0103": "TSEGrowth",      # 東証グロース
        "0104": "TSE1st",         # 東証1部（旧）
        "0105": "TSE2nd",         # 東証2部（旧）
        "0106": "TSEMothers",     # 東証マザーズ（旧）
        "0107": "TSEJASDAQ",      # 東証JASDAQ（旧）
        "0108": "TSEPro",         # TOKYO PRO Market
        "0109": "TSEOther",       # その他
        "0301": "NSEPremier",     # 名証プレミア
        "0302": "NSEMain",        # 名証メイン
        "0303": "NSENext",        # 名証ネクスト
        "0304": "NSEOther",       # 名証その他
        "0501": "SSEMain",        # 札証本則
        "0502": "SSEAmbitious",   # 札証アンビシャス
        "0503": "SSEOther",       # 札証その他
        "0701": "FSEMain",        # 福証本則
        "0702": "FSEQBoard",      # 福証Q-Board
        "0703": "FSEOther",       # 福証その他
    }
    
    # 2022年4月の市場再編マッピング
    MARKET_TRANSITION = {
        "0104": "0101",  # 東証1部 → プライム（デフォルト）
        "0105": "0102",  # 東証2部 → スタンダード
        "0106": "0103",  # マザーズ → グロース
        "0107": "0102",  # JASDAQ → スタンダード（デフォルト）
    }
    
    def __init__(self, transition_date: str = "2022-04-04"):
        """
        Args:
            transition_date: 市場再編日（デフォルト：2022年4月4日）
        """
        self.transition_date = datetime.strptime(transition_date, "%Y-%m-%d").date()
        self._mapping_cache = {}
    
    def create_section_mapping(
        self,
        listed_info_df: pl.DataFrame,
        as_of_date: Optional[str] = None
    ) -> pl.DataFrame:
        """
        listed_info から Section マッピングテーブルを作成
        
        Args:
            listed_info_df: listed_info APIから取得したデータ
            as_of_date: 基準日（省略時は最新）
        
        Returns:
            Code, Section, valid_from, valid_to を含むマッピングテーブル
        """
        df = listed_info_df.select([
            pl.col("Code").cast(pl.Utf8),
            pl.col("MarketCode").cast(pl.Utf8),
            pl.col("Date").cast(pl.Date).alias("valid_from")
        ])
        
        # MarketCode が変更された履歴を考慮
        df = df.sort(["Code", "valid_from"])
        
        # 各コードの次の変更日を取得
        df = df.with_columns([
            pl.col("valid_from").shift(-1).over("Code").alias("next_change")
        ])
        
        # valid_to を設定（次の変更日の前日、最後は無限大相当）
        df = df.with_columns([
            pl.when(pl.col("next_change").is_not_null())
            .then(pl.col("next_change") - pl.duration(days=1))
            .otherwise(pl.date(2999, 12, 31))
            .alias("valid_to")
        ])
        
        # MarketCode → Section 変換
        df = df.with_columns([
            self._map_market_code_expr().alias("Section")
        ])
        
        # 市場再編の処理
        df = self._apply_market_transition(df)
        
        # 必要な列のみ選択
        result = df.select([
            "Code", "Section", "valid_from", "valid_to"
        ])
        
        # as_of_date が指定されている場合はその時点で有効なものだけ
        if as_of_date:
            as_of = datetime.strptime(as_of_date, "%Y-%m-%d").date()
            result = result.filter(
                (pl.col("valid_from") <= as_of) &
                (pl.col("valid_to") >= as_of)
            )
        
        logger.info(f"Created section mapping for {result['Code'].n_unique()} stocks")
        
        # Section分布を表示
        section_counts = result.group_by("Section").agg(
            pl.count().alias("count")
        ).sort("count", descending=True)
        
        for row in section_counts.iter_rows(named=True):
            logger.info(f"  {row['Section']}: {row['count']} stocks")
        
        return result
    
    def _map_market_code_expr(self) -> pl.Expr:
        """MarketCode を Section に変換する Polars 式"""
        return pl.col("MarketCode").map_dict(
            self.MARKET_TO_SECTION,
            default=pl.lit("Other")
        )
    
    def _apply_market_transition(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        2022年4月の市場再編を適用
        
        Args:
            df: マッピングデータ
        
        Returns:
            市場再編を考慮したマッピング
        """
        # 市場再編日以降のデータ
        post_transition = df.filter(
            pl.col("valid_from") >= self.transition_date
        )
        
        # 再編対象のMarketCodeを持つものを変換
        for old_code, new_code in self.MARKET_TRANSITION.items():
            post_transition = post_transition.with_columns([
                pl.when(pl.col("MarketCode") == old_code)
                .then(pl.lit(self.MARKET_TO_SECTION.get(new_code, "Other")))
                .otherwise(pl.col("Section"))
                .alias("Section")
            ])
        
        # 再編前のデータ
        pre_transition = df.filter(
            pl.col("valid_from") < self.transition_date
        )
        
        # 結合
        return pl.concat([pre_transition, post_transition])
    
    def attach_section_to_daily(
        self,
        daily_df: pl.DataFrame,
        mapping_df: pl.DataFrame,
        date_col: str = "Date",
        code_col: str = "Code"
    ) -> pl.DataFrame:
        """
        日次データに Section を付与（期間を考慮）
        
        Args:
            daily_df: 日次データ（Code, Date を含む）
            mapping_df: Section マッピングテーブル
            date_col: 日付列名
            code_col: 銘柄コード列名
        
        Returns:
            Section が付与されたデータ
        """
        # 日付の型を統一
        daily_df = daily_df.with_columns([
            pl.col(date_col).cast(pl.Date),
            pl.col(code_col).cast(pl.Utf8)
        ])
        
        mapping_df = mapping_df.with_columns([
            pl.col("valid_from").cast(pl.Date),
            pl.col("valid_to").cast(pl.Date),
            pl.col("Code").cast(pl.Utf8)
        ])
        
        # asof join で期間を考慮した結合
        # 各銘柄・日付に対して、その時点で有効なSectionを取得
        result = daily_df.sort([code_col, date_col]).join_asof(
            mapping_df.sort(["Code", "valid_from"]),
            left_on=date_col,
            right_on="valid_from",
            by_left=code_col,
            by_right="Code",
            strategy="backward"  # その日以前の最新のマッピングを使用
        )
        
        # valid_to でフィルタリング（期間外のものを除外）
        result = result.filter(
            (pl.col(date_col) <= pl.col("valid_to")) |
            pl.col("valid_to").is_null()
        )
        
        # 不要な列を削除
        result = result.drop(["valid_from", "valid_to"])
        
        # Sectionがnullの場合はOtherで埋める
        result = result.with_columns([
            pl.col("Section").fill_null("Other")
        ])
        
        logger.info(f"Attached sections to {len(result)} daily records")
        
        return result
    
    def get_section_for_code(
        self,
        code: str,
        as_of_date: str,
        mapping_df: pl.DataFrame
    ) -> str:
        """
        特定の銘柄・日付のSectionを取得
        
        Args:
            code: 銘柄コード
            as_of_date: 基準日
            mapping_df: マッピングテーブル
        
        Returns:
            Section名
        """
        as_of = datetime.strptime(as_of_date, "%Y-%m-%d").date()
        
        filtered = mapping_df.filter(
            (pl.col("Code") == code) &
            (pl.col("valid_from") <= as_of) &
            (pl.col("valid_to") >= as_of)
        )
        
        if filtered.is_empty():
            return "Other"
        
        return filtered["Section"][0]
    
    def validate_section_coverage(
        self,
        daily_df: pl.DataFrame,
        section_col: str = "Section"
    ) -> Dict[str, float]:
        """
        Section カバレッジを検証
        
        Args:
            daily_df: Section付きの日次データ
            section_col: Section列名
        
        Returns:
            カバレッジ統計
        """
        total_rows = len(daily_df)
        
        # Section別の統計
        section_stats = daily_df.group_by(section_col).agg([
            pl.count().alias("count"),
            pl.col("Code").n_unique().alias("unique_codes")
        ]).sort("count", descending=True)
        
        # Other以外のカバレッジ
        non_other_rows = daily_df.filter(
            pl.col(section_col) != "Other"
        ).shape[0]
        
        coverage = non_other_rows / total_rows if total_rows > 0 else 0
        
        stats = {
            "total_rows": total_rows,
            "section_coverage": coverage,
            "other_ratio": 1 - coverage,
            "sections": {}
        }
        
        for row in section_stats.iter_rows(named=True):
            stats["sections"][row[section_col]] = {
                "count": row["count"],
                "ratio": row["count"] / total_rows,
                "unique_codes": row["unique_codes"]
            }
        
        return stats