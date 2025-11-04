"""
Section マッピングユーティリティ
listed_info から正確な MarketCode → Section 変換
"""

import logging
from datetime import datetime

import polars as pl

logger = logging.getLogger(__name__)


class SectionMapper:
    """
    銘柄コード → Section（市場区分）のマッピング管理

    注意:
    - listed_info の MarketCode を trades_spec の Section に変換
    - JASDAQは listed_info では Standard/Growth に分かれているが、
      trades_spec では TSEJASDAQ として統合される場合がある
    """

    # MarketCode → Section のマッピング定義
    MARKET_TO_SECTION = {
        # 旧市場コード（2022年4月3日まで）
        "0101": "TSE1st",         # 東証一部
        "0102": "TSE2nd",         # 東証二部
        "0104": "TSEMothers",     # マザーズ
        "0105": "TSEPro",         # TOKYO PRO MARKET
        "0106": "JASDAQStandard", # JASDAQ スタンダード
        "0107": "JASDAQGrowth",   # JASDAQ グロース
        "0109": "Other",          # その他

        # 新市場コード（2022年4月4日以降）
        "0111": "TSEPrime",       # プライム
        "0112": "TSEStandard",    # スタンダード
        "0113": "TSEGrowth",      # グロース

        # 地方市場
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
    # 旧市場コード → 新市場コードへの変換（デフォルト）
    MARKET_TRANSITION = {
        "0101": "0111",  # 東証1部 → プライム（デフォルト）
        "0102": "0112",  # 東証2部 → スタンダード
        "0104": "0113",  # マザーズ → グロース
        "0106": "0112",  # JASDAQ スタンダード → スタンダード
        "0107": "0113",  # JASDAQ グロース → グロース
        # 注: 実際の移行先は銘柄により異なる場合があります
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
        as_of_date: str | None = None
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
            pl.len().alias("count")
        ).sort("count", descending=True)

        for row in section_counts.iter_rows(named=True):
            logger.info(f"  {row['Section']}: {row['count']} stocks")

        return result

    def _map_market_code_expr(self) -> pl.Expr:
        """MarketCode を Section に変換する Polars 式"""
        return pl.col("MarketCode").replace(
            self.MARKET_TO_SECTION,
            default="Other"
        )

    def _apply_market_transition(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        2022年4月の市場再編を適用

        市場再編のルール:
        - 2022年4月4日以降: 新市場コード（0111-0113）が使用される
        - 2022年4月3日まで: 旧市場コード（0101,0102,0104,0106,0107）が使用される
        - 新市場コードが既に正しくマッピングされているため、特別な変換は不要

        Args:
            df: マッピングデータ

        Returns:
            市場再編を考慮したマッピング
        """
        # 現在のマッピングで既に正しく処理されているため、そのまま返す
        # （MARKET_TO_SECTIONに旧・新両方のコードが定義済み）
        return df

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
    ) -> dict[str, float]:
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

    def get_section_for_trades_spec(self, section: str) -> str:
        """
        trades_spec用にSectionを調整

        JASDAQStandard/JASDAQGrowth を TSEJASDAQ に統合する場合などに使用

        Args:
            section: 元のSection名

        Returns:
            trades_spec用に調整されたSection名
        """
        # JASDAQの統合（必要に応じて）
        jasdaq_mapping = {
            "JASDAQStandard": "TSEJASDAQ",
            "JASDAQGrowth": "TSEJASDAQ",
        }

        return jasdaq_mapping.get(section, section)
