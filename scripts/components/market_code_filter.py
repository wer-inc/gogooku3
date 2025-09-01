#!/usr/bin/env python3
"""
Market Code Filter - 市場コードによる銘柄フィルタリング
8つの主要市場のみを取得対象とし、PRO Market等を除外
"""

import polars as pl
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MarketCodeFilter:
    """Market Codeによる銘柄フィルタリング"""
    
    # 取得対象の市場コード（8市場）
    TARGET_MARKET_CODES = [
        "0101",  # 東証一部
        "0102",  # 東証二部
        "0104",  # マザーズ
        "0106",  # JASDAQ スタンダード
        "0107",  # JASDAQ グロース
        "0111",  # プライム
        "0112",  # スタンダード
        "0113",  # グロース
    ]
    
    # 除外する市場コード
    EXCLUDE_MARKET_CODES = [
        "0105",  # TOKYO PRO MARKET
        "0109",  # その他
    ]
    
    # 市場名称マッピング
    MARKET_NAMES = {
        "0101": "東証一部",
        "0102": "東証二部",
        "0104": "マザーズ",
        "0105": "TOKYO PRO MARKET",
        "0106": "JASDAQ スタンダード",
        "0107": "JASDAQ グロース",
        "0109": "その他",
        "0111": "プライム",
        "0112": "スタンダード",
        "0113": "グロース"
    }
    
    @classmethod
    def filter_stocks(cls, df: pl.DataFrame) -> pl.DataFrame:
        """
        8つの主要市場の銘柄のみにフィルタリング
        
        Args:
            df: Listed InfoのDataFrame（MarketCode列を含む）
            
        Returns:
            フィルタリング後のDataFrame
        """
        if df.is_empty():
            return df
        
        # MarketCode列の存在確認
        if "MarketCode" not in df.columns:
            logger.warning("MarketCode列が存在しません")
            return df
        
        # フィルタリング前の統計
        original_count = len(df)
        
        # 8市場のみに絞る（ホワイトリスト方式）
        filtered_df = df.filter(
            pl.col("MarketCode").is_in(cls.TARGET_MARKET_CODES)
        )
        
        filtered_count = len(filtered_df)
        
        # 統計情報をログ出力
        logger.info(f"Market Codeフィルタリング: {original_count} → {filtered_count} 銘柄")
        
        # 市場別の詳細統計
        cls._log_market_statistics(filtered_df)
        
        # 除外された未知のコードを検出
        cls._detect_unknown_codes(df)
        
        return filtered_df
    
    @classmethod
    def _log_market_statistics(cls, df: pl.DataFrame):
        """市場別の統計をログ出力"""
        if df.is_empty() or "MarketCode" not in df.columns:
            return
        
        stats = df.group_by("MarketCode").agg(
            pl.count().alias("count")
        ).sort("MarketCode")
        
        logger.info("市場別銘柄数:")
        for row in stats.iter_rows(named=True):
            code = row["MarketCode"]
            count = row["count"]
            name = cls.MARKET_NAMES.get(code, f"未定義({code})")
            logger.info(f"  {code}: {name} - {count}銘柄")
    
    @classmethod
    def _detect_unknown_codes(cls, df: pl.DataFrame):
        """未知のMarket Codeを検出してログ出力"""
        if df.is_empty() or "MarketCode" not in df.columns:
            return
        
        all_codes = df["MarketCode"].unique().to_list()
        known_codes = set(cls.TARGET_MARKET_CODES) | set(cls.EXCLUDE_MARKET_CODES)
        unknown_codes = [code for code in all_codes if code not in known_codes]
        
        if unknown_codes:
            logger.warning(f"⚠️ 未知のMarket Code検出: {unknown_codes}")
            
            # 未知コードの銘柄数を集計
            for code in unknown_codes:
                count = len(df.filter(pl.col("MarketCode") == code))
                logger.warning(f"  {code}: {count}銘柄（未分類）")
    
    @classmethod
    def get_target_stocks_by_date(
        cls, 
        df: pl.DataFrame, 
        target_date: Optional[str] = None
    ) -> pl.DataFrame:
        """
        特定日における対象銘柄を取得
        
        Args:
            df: Listed InfoのDataFrame
            target_date: 対象日（YYYY-MM-DD形式）、Noneの場合は最新
            
        Returns:
            フィルタリング後のDataFrame
        """
        # Market Codeでフィルタリング
        filtered_df = cls.filter_stocks(df)
        
        if filtered_df.is_empty():
            return filtered_df
        
        # 日付指定がある場合、上場/廃止日でさらにフィルタリング
        if target_date and "Date" in filtered_df.columns:
            # 上場日が対象日以前
            filtered_df = filtered_df.filter(
                pl.col("Date") <= target_date
            )
            
            # 上場廃止日が対象日より後（またはNULL）
            if "DelistingDate" in filtered_df.columns:
                filtered_df = filtered_df.filter(
                    pl.col("DelistingDate").is_null() | 
                    (pl.col("DelistingDate") > target_date)
                )
            
            logger.info(f"{target_date}時点の対象銘柄数: {len(filtered_df)}")
        
        return filtered_df
    
    @classmethod
    def validate_market_coverage(cls, df: pl.DataFrame) -> Dict:
        """
        市場カバレッジの検証
        
        Returns:
            検証結果の辞書
        """
        if df.is_empty() or "MarketCode" not in df.columns:
            return {"valid": False, "error": "データが空またはMarketCode列なし"}
        
        all_codes = df["MarketCode"].unique().to_list()
        
        # 対象市場のカバレッジ
        covered_targets = [code for code in cls.TARGET_MARKET_CODES if code in all_codes]
        missing_targets = [code for code in cls.TARGET_MARKET_CODES if code not in all_codes]
        
        # 除外市場の確認
        found_excludes = [code for code in cls.EXCLUDE_MARKET_CODES if code in all_codes]
        
        # 未知のコード
        known_codes = set(cls.TARGET_MARKET_CODES) | set(cls.EXCLUDE_MARKET_CODES)
        unknown_codes = [code for code in all_codes if code not in known_codes]
        
        result = {
            "valid": len(missing_targets) == 0,
            "covered_markets": len(covered_targets),
            "missing_markets": missing_targets,
            "excluded_markets_found": found_excludes,
            "unknown_markets": unknown_codes,
            "total_stocks": len(df),
            "filtered_stocks": len(cls.filter_stocks(df))
        }
        
        # 結果をログ出力
        logger.info(f"市場カバレッジ検証:")
        logger.info(f"  対象市場: {result['covered_markets']}/8")
        if result['missing_markets']:
            logger.warning(f"  欠落市場: {result['missing_markets']}")
        if result['unknown_markets']:
            logger.warning(f"  未知の市場: {result['unknown_markets']}")
        logger.info(f"  総銘柄数: {result['total_stocks']} → {result['filtered_stocks']} (フィルタ後)")
        
        return result


def test_market_filter():
    """テスト関数"""
    # テストデータ作成
    test_data = {
        "Code": ["1234", "5678", "9012", "3456", "7890", "1111"],
        "MarketCode": ["0111", "0112", "0113", "0105", "0109", "9999"],
        "Name": ["テストA", "テストB", "テストC", "PRO銘柄", "その他銘柄", "未知銘柄"]
    }
    
    df = pl.DataFrame(test_data)
    
    print("元データ:")
    print(df)
    print()
    
    # フィルタリング実行
    filtered_df = MarketCodeFilter.filter_stocks(df)
    
    print("フィルタリング後:")
    print(filtered_df)
    print()
    
    # カバレッジ検証
    coverage = MarketCodeFilter.validate_market_coverage(df)
    print("カバレッジ検証結果:")
    for key, value in coverage.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_market_filter()