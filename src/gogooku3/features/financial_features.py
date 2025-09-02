"""
Financial Statement Features Module
財務諸表特徴量（仕様準拠: stmt_* prefix）
"""

import polars as pl
import numpy as np
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta, date


class FinancialFeatures:
    """財務諸表特徴量計算（仕様準拠）"""
    
    @staticmethod
    def calculate_yoy_strict(
        statements_df: pl.DataFrame,
        fiscal_year_col: str = "FiscalYear",
        quarter_col: str = "Quarter"
    ) -> pl.DataFrame:
        """
        厳密な前年同期比(YoY)計算 - FY×Qベースで正確に計算
        
        Args:
            statements_df: 財務諸表データ (FiscalYear, Quarter含む)
            fiscal_year_col: 会計年度列名
            quarter_col: 四半期列名
            
        Returns:
            YoY成長率を追加したDataFrame
        """
        # 前年のFYを計算
        df = statements_df.with_columns(
            (pl.col(fiscal_year_col) - 1).alias("prev_fy")
        )
        
        # 前年同期データを自己結合で取得
        yoy_base = df.select([
            pl.col("Code"),
            pl.col(fiscal_year_col).alias("base_fy"),
            pl.col(quarter_col).alias("base_q"),
            pl.col("Revenue").alias("yoy_revenue_base"),
            pl.col("OperatingIncome").alias("yoy_oi_base"),
            pl.col("NetIncome").alias("yoy_ni_base"),
        ])
        
        # 前年同期と結合
        df = df.join(
            yoy_base,
            left_on=["Code", "prev_fy", quarter_col],
            right_on=["Code", "base_fy", "base_q"],
            how="left"
        )
        
        # YoY成長率を計算
        df = df.with_columns([
            # Revenue YoY
            ((pl.col("Revenue") - pl.col("yoy_revenue_base")) / 
             (pl.col("yoy_revenue_base").abs() + 1e-12)).alias("stmt_yoy_sales"),
            
            # Operating Income YoY
            ((pl.col("OperatingIncome") - pl.col("yoy_oi_base")) / 
             (pl.col("yoy_oi_base").abs() + 1e-12)).alias("stmt_yoy_oi"),
            
            # Net Income YoY
            ((pl.col("NetIncome") - pl.col("yoy_ni_base")) / 
             (pl.col("yoy_ni_base").abs() + 1e-12)).alias("stmt_yoy_ni"),
        ])
        
        # 中間列を削除
        df = df.drop(["prev_fy", "yoy_revenue_base", "yoy_oi_base", "yoy_ni_base"])
        
        return df
    
    @staticmethod
    def generate_financial_features(
        df: pl.DataFrame,
        financial_df: Optional[pl.DataFrame] = None,
        use_as_of_join: bool = True
    ) -> pl.DataFrame:
        """
        財務諸表特徴量を生成（仕様準拠: stmt_* prefix）
        
        Features (all with stmt_ prefix):
        - stmt_yoy_sales: 売上高前年同期比
        - stmt_yoy_oi: 営業利益前年同期比
        - stmt_yoy_ni: 純利益前年同期比
        - stmt_roe: ROE（自己資本利益率）
        - stmt_roa: ROA（総資産利益率）
        - stmt_roic: ROIC（投下資本利益率）
        - stmt_profit_margin: 営業利益率
        - stmt_asset_turnover: 総資産回転率
        - stmt_equity_ratio: 自己資本比率
        - stmt_current_ratio: 流動比率
        - stmt_de_ratio: D/Eレシオ
        - stmt_per: PER（株価収益率）
        - stmt_pbr: PBR（株価純資産倍率）
        - stmt_ev_ebitda: EV/EBITDA
        - stmt_dividend_yield: 配当利回り
        
        Args:
            df: 株価データ
            financial_df: 財務データ（オプション）
            use_as_of_join: As-of joinを使用するか
        
        Returns:
            財務特徴量を追加したDataFrame
        """
        
        if financial_df is not None:
            # 実データがある場合の処理
            df = FinancialFeatures._merge_financial_data(
                df, financial_df, use_as_of_join
            )
        else:
            # シミュレーションデータ生成（テスト用）
            df = FinancialFeatures._generate_simulated_financial_data(df)
        
        # 財務比率の計算
        df = FinancialFeatures._calculate_financial_ratios(df)
        
        # バリュエーション指標の計算
        df = FinancialFeatures._calculate_valuation_metrics(df)
        
        # 財務スコアの計算
        df = FinancialFeatures._calculate_financial_scores(df)
        
        # 有効性フラグの追加
        df = FinancialFeatures._add_financial_validity_flags(df)
        
        return df
    
    @staticmethod
    def _merge_financial_data(
        df: pl.DataFrame,
        financial_df: pl.DataFrame,
        use_as_of_join: bool = True
    ) -> pl.DataFrame:
        """
        財務データを株価データにマージ（As-of join）
        
        Args:
            df: 株価データ
            financial_df: 財務データ
            use_as_of_join: As-of joinを使用するか
        
        Returns:
            マージ済みDataFrame
        """
        
        # Ensure financial data has required columns
        required_cols = [
            "Code", "ReportDate", "ReleaseDate",
            "Revenue", "OperatingIncome", "NetIncome",
            "TotalAssets", "Equity", "CurrentAssets", "CurrentLiabilities",
            "Debt", "EBITDA", "DividendPerShare", "SharesOutstanding"
        ]
        
        # Add missing columns with default values
        for col in required_cols:
            if col not in financial_df.columns:
                if col in ["Code", "ReportDate", "ReleaseDate"]:
                    continue  # Skip key columns
                financial_df = financial_df.with_columns(
                    pl.lit(None).alias(col)
                )
        
        if use_as_of_join:
            # Use release date for as-of join (prevent look-ahead bias)
            # Financial data becomes available on release date, not report date
            df = df.sort(["meta_code", "meta_date"])
            financial_df = financial_df.sort(["Code", "ReleaseDate"])
            
            # As-of join: use most recent financial data available at each date
            df = df.join_asof(
                financial_df.select([
                    pl.col("Code"),
                    pl.col("ReleaseDate").alias("meta_date"),
                    *[col for col in financial_df.columns 
                      if col not in ["Code", "ReportDate", "ReleaseDate"]]
                ]),
                on="meta_date",
                by="Code" if "Code" in financial_df.columns else None,
                strategy="backward"  # Use most recent available data
            )
        else:
            # Simple join (for testing)
            df = df.join(
                financial_df,
                left_on=["meta_code", "meta_date"],
                right_on=["Code", "ReleaseDate"],
                how="left"
            )
        
        return df
    
    @staticmethod
    def _generate_simulated_financial_data(df: pl.DataFrame) -> pl.DataFrame:
        """
        シミュレーション用財務データ生成（テスト用）
        
        Args:
            df: 株価データ
        
        Returns:
            シミュレーション財務データを追加したDataFrame
        """
        
        # Generate correlated financial metrics based on price/volume
        price_col = "Close" if "Close" in df.columns else "px_close"
        volume_col = "Volume" if "Volume" in df.columns else "px_volume"
        
        if price_col in df.columns:
            # Market cap proxy
            market_cap = pl.col(price_col) * pl.col(volume_col) * 100
            
            # Revenue (correlated with market cap)
            df = df.with_columns(
                (market_cap * 0.5 + np.random.randn() * 1000000).alias("Revenue")
            )
            
            # Operating Income (15% margin)
            df = df.with_columns(
                (pl.col("Revenue") * 0.15 + np.random.randn() * 100000).alias("OperatingIncome")
            )
            
            # Net Income (10% margin)
            df = df.with_columns(
                (pl.col("Revenue") * 0.10 + np.random.randn() * 50000).alias("NetIncome")
            )
            
            # Total Assets (2x revenue)
            df = df.with_columns(
                (pl.col("Revenue") * 2 + np.random.randn() * 1000000).alias("TotalAssets")
            )
            
            # Equity (40% of assets)
            df = df.with_columns(
                (pl.col("TotalAssets") * 0.4).alias("Equity")
            )
            
            # Current Assets (50% of total)
            df = df.with_columns(
                (pl.col("TotalAssets") * 0.5).alias("CurrentAssets")
            )
            
            # Current Liabilities (40% of current assets)
            df = df.with_columns(
                (pl.col("CurrentAssets") * 0.4).alias("CurrentLiabilities")
            )
            
            # Debt (30% of assets)
            df = df.with_columns(
                (pl.col("TotalAssets") * 0.3).alias("Debt")
            )
            
            # EBITDA (20% margin)
            df = df.with_columns(
                (pl.col("Revenue") * 0.20).alias("EBITDA")
            )
            
            # Dividend per share
            df = df.with_columns(
                (pl.col("NetIncome") / 100000000 * 0.3).alias("DividendPerShare")
            )
        else:
            # Add zero columns if no price data
            financial_cols = [
                "Revenue", "OperatingIncome", "NetIncome",
                "TotalAssets", "Equity", "CurrentAssets",
                "CurrentLiabilities", "Debt", "EBITDA", "DividendPerShare"
            ]
            for col in financial_cols:
                df = df.with_columns(pl.lit(0.0).alias(col))
        
        # Add YoY growth (simulated - random walk for testing)
        # In production, use calculate_yoy_strict() for exact FY×Q matching
        df = df.with_columns([
            pl.lit(0.05 + np.random.randn() * 0.1).alias("stmt_yoy_sales"),
            pl.lit(0.08 + np.random.randn() * 0.15).alias("stmt_yoy_oi"),
            pl.lit(0.10 + np.random.randn() * 0.20).alias("stmt_yoy_ni"),
        ])
        
        return df
    
    @staticmethod
    def _calculate_financial_ratios(df: pl.DataFrame) -> pl.DataFrame:
        """
        財務比率を計算
        
        Args:
            df: 財務データを含むDataFrame
        
        Returns:
            財務比率を追加したDataFrame
        """
        
        # ROE (Return on Equity)
        if "NetIncome" in df.columns and "Equity" in df.columns:
            df = df.with_columns(
                (pl.col("NetIncome") / (pl.col("Equity") + 1e-12))
                .alias("stmt_roe")
            )
        
        # ROA (Return on Assets)
        if "NetIncome" in df.columns and "TotalAssets" in df.columns:
            df = df.with_columns(
                (pl.col("NetIncome") / (pl.col("TotalAssets") + 1e-12))
                .alias("stmt_roa")
            )
        
        # ROIC (Return on Invested Capital)
        if all(col in df.columns for col in ["OperatingIncome", "Equity", "Debt"]):
            df = df.with_columns(
                (pl.col("OperatingIncome") * (1 - 0.3) /  # Assume 30% tax rate
                 (pl.col("Equity") + pl.col("Debt") + 1e-12))
                .alias("stmt_roic")
            )
        
        # Profit Margin
        if "OperatingIncome" in df.columns and "Revenue" in df.columns:
            df = df.with_columns(
                (pl.col("OperatingIncome") / (pl.col("Revenue") + 1e-12))
                .alias("stmt_profit_margin")
            )
        
        # Asset Turnover
        if "Revenue" in df.columns and "TotalAssets" in df.columns:
            df = df.with_columns(
                (pl.col("Revenue") / (pl.col("TotalAssets") + 1e-12))
                .alias("stmt_asset_turnover")
            )
        
        # Equity Ratio
        if "Equity" in df.columns and "TotalAssets" in df.columns:
            df = df.with_columns(
                (pl.col("Equity") / (pl.col("TotalAssets") + 1e-12))
                .alias("stmt_equity_ratio")
            )
        
        # Current Ratio
        if "CurrentAssets" in df.columns and "CurrentLiabilities" in df.columns:
            df = df.with_columns(
                (pl.col("CurrentAssets") / (pl.col("CurrentLiabilities") + 1e-12))
                .alias("stmt_current_ratio")
            )
        
        # D/E Ratio
        if "Debt" in df.columns and "Equity" in df.columns:
            df = df.with_columns(
                (pl.col("Debt") / (pl.col("Equity") + 1e-12))
                .alias("stmt_de_ratio")
            )
        
        return df
    
    @staticmethod
    def _calculate_valuation_metrics(df: pl.DataFrame) -> pl.DataFrame:
        """
        バリュエーション指標を計算
        
        Args:
            df: 財務・株価データを含むDataFrame
        
        Returns:
            バリュエーション指標を追加したDataFrame
        """
        
        price_col = "Close" if "Close" in df.columns else "px_close"
        shares_col = "SharesOutstanding" if "SharesOutstanding" in df.columns else "shares_outstanding"
        
        # Market Cap
        if price_col in df.columns and shares_col in df.columns:
            df = df.with_columns(
                (pl.col(price_col) * pl.col(shares_col)).alias("MarketCap")
            )
        elif price_col in df.columns and "Volume" in df.columns:
            # Approximate market cap if shares outstanding not available
            df = df.with_columns(
                (pl.col(price_col) * pl.col("Volume") * 100).alias("MarketCap")
            )
        
        # PER (Price Earnings Ratio)
        if "MarketCap" in df.columns and "NetIncome" in df.columns:
            df = df.with_columns(
                (pl.col("MarketCap") / (pl.col("NetIncome") * 4 + 1e-12))  # Annualized
                .alias("stmt_per")
            )
        
        # PBR (Price Book Ratio)
        if "MarketCap" in df.columns and "Equity" in df.columns:
            df = df.with_columns(
                (pl.col("MarketCap") / (pl.col("Equity") + 1e-12))
                .alias("stmt_pbr")
            )
        
        # EV/EBITDA
        if all(col in df.columns for col in ["MarketCap", "Debt", "EBITDA"]):
            df = df.with_columns(
                ((pl.col("MarketCap") + pl.col("Debt")) / 
                 (pl.col("EBITDA") * 4 + 1e-12))  # Annualized
                .alias("stmt_ev_ebitda")
            )
        
        # Dividend Yield
        if "DividendPerShare" in df.columns and price_col in df.columns:
            df = df.with_columns(
                (pl.col("DividendPerShare") * 4 / (pl.col(price_col) + 1e-12))  # Annualized
                .alias("stmt_dividend_yield")
            )
        
        # Earnings Yield (inverse of PER)
        if "stmt_per" in df.columns:
            df = df.with_columns(
                (1.0 / (pl.col("stmt_per") + 1e-12))
                .alias("stmt_earnings_yield")
            )
        
        return df
    
    @staticmethod
    def _calculate_financial_scores(df: pl.DataFrame) -> pl.DataFrame:
        """
        総合財務スコアを計算（Piotroski F-Score風）
        
        Args:
            df: 財務データを含むDataFrame
        
        Returns:
            財務スコアを追加したDataFrame
        """
        
        score = pl.lit(0)
        
        # Profitability signals
        if "stmt_roe" in df.columns:
            score = score + (pl.col("stmt_roe") > 0).cast(pl.Int8)
        
        if "stmt_roa" in df.columns:
            score = score + (pl.col("stmt_roa") > 0).cast(pl.Int8)
            # ROA improvement
            score = score + (pl.col("stmt_roa") > pl.col("stmt_roa").shift(252)).cast(pl.Int8)
        
        if "OperatingIncome" in df.columns:
            score = score + (pl.col("OperatingIncome") > 0).cast(pl.Int8)
        
        # Leverage/Liquidity signals
        if "stmt_current_ratio" in df.columns:
            score = score + (pl.col("stmt_current_ratio") > 1).cast(pl.Int8)
            # Current ratio improvement
            score = score + (pl.col("stmt_current_ratio") > pl.col("stmt_current_ratio").shift(252)).cast(pl.Int8)
        
        if "stmt_de_ratio" in df.columns:
            # Lower D/E is better
            score = score + (pl.col("stmt_de_ratio") < pl.col("stmt_de_ratio").shift(252)).cast(pl.Int8)
        
        # Operating efficiency signals
        if "stmt_asset_turnover" in df.columns:
            # Asset turnover improvement
            score = score + (pl.col("stmt_asset_turnover") > pl.col("stmt_asset_turnover").shift(252)).cast(pl.Int8)
        
        if "stmt_profit_margin" in df.columns:
            # Margin improvement
            score = score + (pl.col("stmt_profit_margin") > pl.col("stmt_profit_margin").shift(252)).cast(pl.Int8)
        
        df = df.with_columns(score.alias("stmt_f_score"))
        
        # Quality score (combination of profitability and stability)
        quality_score = pl.lit(0.0)
        
        if "stmt_roe" in df.columns:
            quality_score = quality_score + pl.col("stmt_roe").clip(0, 0.3) * 10
        
        if "stmt_profit_margin" in df.columns:
            quality_score = quality_score + pl.col("stmt_profit_margin").clip(0, 0.2) * 10
        
        if "stmt_current_ratio" in df.columns:
            quality_score = quality_score + (pl.col("stmt_current_ratio").clip(1, 3) - 1) * 2
        
        df = df.with_columns(quality_score.alias("stmt_quality_score"))
        
        return df
    
    @staticmethod
    def _add_financial_validity_flags(df: pl.DataFrame) -> pl.DataFrame:
        """
        財務データの有効性フラグを追加
        
        Args:
            df: 財務データを含むDataFrame
        
        Returns:
            有効性フラグを追加したDataFrame
        """
        
        # Key financial metrics validity
        key_metrics = [
            "stmt_roe", "stmt_roa", "stmt_per", "stmt_pbr",
            "stmt_yoy_sales", "stmt_profit_margin"
        ]
        
        for metric in key_metrics:
            if metric in df.columns:
                df = df.with_columns(
                    pl.col(metric).is_not_null().alias(f"is_{metric}_valid")
                )
        
        # Overall financial data validity
        if any(col in df.columns for col in key_metrics):
            validity_expr = pl.lit(True)
            for col in key_metrics[:4]:  # Check main metrics
                if col in df.columns:
                    validity_expr = validity_expr & pl.col(col).is_not_null()
            
            df = df.with_columns(
                validity_expr.alias("is_financial_data_valid")
            )
        
        return df
    
    @staticmethod
    def add_financial_momentum_features(
        df: pl.DataFrame,
        windows: List[int] = [20, 60, 252]
    ) -> pl.DataFrame:
        """
        財務モメンタム特徴量を追加
        
        Features:
        - stmt_roe_momentum_*: ROEの変化率
        - stmt_earnings_momentum_*: 利益成長モメンタム
        - stmt_revision_*: 業績修正の方向性
        """
        
        for window in windows:
            # ROE momentum
            if "stmt_roe" in df.columns:
                df = df.with_columns(
                    (pl.col("stmt_roe") - pl.col("stmt_roe").shift(window))
                    .alias(f"stmt_roe_momentum_{window}d")
                )
            
            # Earnings momentum
            if "NetIncome" in df.columns:
                df = df.with_columns(
                    ((pl.col("NetIncome") / (pl.col("NetIncome").shift(window) + 1e-12)) - 1)
                    .alias(f"stmt_earnings_momentum_{window}d")
                )
            
            # Sales momentum
            if "Revenue" in df.columns:
                df = df.with_columns(
                    ((pl.col("Revenue") / (pl.col("Revenue").shift(window) + 1e-12)) - 1)
                    .alias(f"stmt_sales_momentum_{window}d")
                )
        
        return df