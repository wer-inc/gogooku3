"""
データスキーマ定義とバリデーション
Polars/Pandera互換のスキーマ管理
"""

import polars as pl
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class FeaturePrefix(Enum):
    """特徴量接頭辞の定義"""
    PRICE = "px"
    MARKET = "mkt"
    CROSS = "cross"
    FLOW = "flow"
    FINANCIAL = "fin"
    META = "meta"
    TARGET = "y"


@dataclass
class ColumnSchema:
    """列スキーマ定義"""
    name: str
    dtype: pl.DataType
    nullable: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    unique: bool = False
    prefix: Optional[FeaturePrefix] = None


class DataSchemas:
    """データスキーマ集約クラス"""
    
    @staticmethod
    def get_daily_quotes_schema() -> Dict[str, pl.DataType]:
        """日次価格データのスキーマ"""
        return {
            "Code": pl.Utf8,
            "Date": pl.Date,
            "Open": pl.Float32,
            "High": pl.Float32,
            "Low": pl.Float32,
            "Close": pl.Float32,
            "Volume": pl.Int64,
            "TurnoverValue": pl.Float64,
            "adj_factor": pl.Float32,
            "SharesOutstanding": pl.Int64,
        }
    
    @staticmethod
    def get_market_features_schema() -> Dict[str, pl.DataType]:
        """市場特徴量のスキーマ"""
        schema = {
            "Date": pl.Date,
        }
        
        # Market returns
        for horizon in [1, 5, 10, 20]:
            schema[f"mkt_ret_{horizon}d"] = pl.Float32
        
        # Market EMAs
        for window in [5, 20, 60, 200]:
            schema[f"mkt_ema_{window}"] = pl.Float32
        
        # Market volatility and risk
        schema.update({
            "mkt_vol_20d": pl.Float32,
            "mkt_atr_14": pl.Float32,
            "mkt_natr_14": pl.Float32,
            "mkt_dd_from_peak": pl.Float32,
            "mkt_big_move_flag": pl.Int8,
        })
        
        # Market Bollinger Bands
        schema.update({
            "mkt_bb_upper": pl.Float32,
            "mkt_bb_lower": pl.Float32,
            "mkt_bb_middle": pl.Float32,
            "mkt_bb_bw": pl.Float32,
            "mkt_bb_pct_b": pl.Float32,
        })
        
        # Market Z-scores
        schema.update({
            "mkt_ret_1d_z": pl.Float32,
            "mkt_vol_20d_z": pl.Float32,
            "mkt_bb_bw_z": pl.Float32,
            "mkt_dd_from_peak_z": pl.Float32,
        })
        
        # Market regimes
        schema.update({
            "mkt_bull_200": pl.Int8,
            "mkt_trend_up": pl.Int8,
            "mkt_high_vol": pl.Int8,
            "mkt_squeeze": pl.Int8,
        })
        
        return schema
    
    @staticmethod
    def get_section_mapping_schema() -> Dict[str, pl.DataType]:
        """セクションマッピングのスキーマ"""
        return {
            "Code": pl.Utf8,
            "Section": pl.Categorical,
            "valid_from": pl.Date,
            "valid_to": pl.Date,
        }
    
    @staticmethod
    def get_flow_daily_schema() -> Dict[str, pl.DataType]:
        """フローデータのスキーマ"""
        return {
            "Section": pl.Categorical,
            "PublishedDate": pl.Date,
            "effective_start": pl.Date,
            "effective_end": pl.Date,
            # Flow ratios
            "flow_buy_ratio": pl.Float32,
            "flow_sell_ratio": pl.Float32,
            "flow_net_ratio": pl.Float32,
            # Breadth
            "flow_breadth_pos": pl.Float32,
            # Z-scores
            "flow_buy_z": pl.Float32,
            "flow_sell_z": pl.Float32,
            "flow_net_z": pl.Float32,
            # Smart money
            "flow_smart_buy": pl.Float32,
            "flow_smart_sell": pl.Float32,
            "flow_smart_net": pl.Float32,
            # Timing
            "flow_impulse": pl.Float32,
            "days_since_flow": pl.Int32,
        }
    
    @staticmethod
    def get_statements_schema() -> Dict[str, pl.DataType]:
        """財務諸表データのスキーマ"""
        return {
            "Code": pl.Utf8,
            "DisclosedDate": pl.Date,
            "DisclosedTime": pl.Utf8,
            "effective_date": pl.Date,
            # YoY growth
            "fin_revenue_yoy": pl.Float32,
            "fin_profit_yoy": pl.Float32,
            "fin_eps_yoy": pl.Float32,
            # Margins
            "fin_gross_margin": pl.Float32,
            "fin_operating_margin": pl.Float32,
            # Progress
            "fin_progress_revenue": pl.Float32,
            "fin_progress_profit": pl.Float32,
            # Guidance revisions
            "fin_guide_rev_revenue": pl.Float32,
            "fin_guide_rev_profit": pl.Float32,
            "fin_guide_rev_count": pl.Int8,
            "fin_guide_momentum": pl.Float32,
            # ROE/ROA
            "fin_roe": pl.Float32,
            "fin_roa": pl.Float32,
            # Quality
            "fin_accruals_ratio": pl.Float32,
            "fin_nc_flag": pl.Int8,
            # Timing
            "fin_surprise_score": pl.Float32,
            "fin_days_since_report": pl.Int32,
        }
    
    @staticmethod
    def get_ml_panel_schema() -> Dict[str, pl.DataType]:
        """最終MLパネルのスキーマ（145列）"""
        schema = {
            # Meta columns
            "meta_code": pl.Utf8,
            "meta_date": pl.Date,
            "meta_section": pl.Categorical,
        }
        
        # Price/Technical indicators (~80 columns)
        price_features = [
            "returns", "log_returns", "volatility", "park_vol", "gk_vol",
            "sma", "ema", "price_to_sma", "price_to_ema", "ma_gap",
            "high_low_ratio", "close_position", "close_to_high", "close_to_low",
            "volume_ma", "volume_ratio", "turnover_rate", "dollar_volume",
            "rsi", "rsi_delta", "macd", "macd_signal", "macd_histogram",
            "bb_upper", "bb_lower", "bb_middle", "bb_width", "bb_position",
            "atr", "natr", "adx", "stoch_k", "stoch_d"
        ]
        
        # Add with appropriate suffixes
        for feature in price_features:
            # Simplified - actual implementation would handle all variations
            schema[f"px_{feature}_20"] = pl.Float32
        
        # Market features (26 columns)
        market_schema = DataSchemas.get_market_features_schema()
        del market_schema["Date"]  # Remove duplicate
        schema.update(market_schema)
        
        # Cross features (8 columns)
        cross_features = [
            "cross_beta_60d", "cross_alpha_1d", "cross_alpha_5d",
            "cross_rel_strength_5d", "cross_trend_align_flag",
            "cross_alpha_vs_regime", "cross_idio_vol_ratio",
            "cross_beta_stability_60d"
        ]
        for feature in cross_features:
            dtype = pl.Int8 if "flag" in feature else pl.Float32
            schema[feature] = dtype
        
        # Flow features (12 columns)
        flow_features = [
            "flow_buy_ratio", "flow_sell_ratio", "flow_net_ratio",
            "flow_breadth_pos", "flow_buy_z", "flow_sell_z", "flow_net_z",
            "flow_smart_buy", "flow_smart_sell", "flow_smart_net",
            "flow_impulse", "days_since_flow"
        ]
        for feature in flow_features:
            dtype = pl.Int32 if "days" in feature else pl.Float32
            schema[feature] = dtype
        
        # Financial features (17 columns)
        fin_features = [
            "fin_revenue_yoy", "fin_profit_yoy", "fin_eps_yoy",
            "fin_gross_margin", "fin_operating_margin",
            "fin_progress_revenue", "fin_progress_profit",
            "fin_guide_rev_revenue", "fin_guide_rev_profit",
            "fin_guide_rev_count", "fin_guide_momentum",
            "fin_roe", "fin_roa", "fin_accruals_ratio", "fin_nc_flag",
            "fin_surprise_score", "fin_days_since_report"
        ]
        for feature in fin_features:
            if "count" in feature or "flag" in feature:
                dtype = pl.Int8
            elif "days" in feature:
                dtype = pl.Int32
            else:
                dtype = pl.Float32
            schema[feature] = dtype
        
        # Targets (7 columns)
        for horizon in [1, 5, 10, 20]:
            schema[f"y_{horizon}d"] = pl.Float32
            if horizon != 20:  # Skip y_20d_bin for space
                schema[f"y_{horizon}d_bin"] = pl.Int8
        
        return schema


class SchemaValidator:
    """スキーマ検証ユーティリティ"""
    
    @staticmethod
    def validate_schema(df: pl.DataFrame, schema: Dict[str, pl.DataType]) -> Dict[str, any]:
        """
        DataFrameのスキーマを検証
        
        Args:
            df: 検証対象のDataFrame
            schema: 期待されるスキーマ
        
        Returns:
            検証結果の辞書
        """
        results = {
            "valid": True,
            "missing_columns": [],
            "extra_columns": [],
            "type_mismatches": [],
            "null_columns": []
        }
        
        df_columns = set(df.columns)
        schema_columns = set(schema.keys())
        
        # Check missing columns
        results["missing_columns"] = list(schema_columns - df_columns)
        
        # Check extra columns
        results["extra_columns"] = list(df_columns - schema_columns)
        
        # Check type mismatches
        for col, expected_type in schema.items():
            if col in df.columns:
                actual_type = df[col].dtype
                if actual_type != expected_type:
                    results["type_mismatches"].append({
                        "column": col,
                        "expected": str(expected_type),
                        "actual": str(actual_type)
                    })
        
        # Check for null values
        for col in df.columns:
            null_count = df[col].null_count()
            if null_count > 0:
                results["null_columns"].append({
                    "column": col,
                    "null_count": null_count,
                    "null_pct": null_count / len(df) * 100
                })
        
        # Set overall validity
        if (results["missing_columns"] or 
            results["type_mismatches"] or 
            len(results["extra_columns"]) > 10):  # Allow some extra columns
            results["valid"] = False
        
        return results
    
    @staticmethod
    def validate_value_ranges(
        df: pl.DataFrame,
        validations: Dict[str, Dict[str, any]]
    ) -> Dict[str, any]:
        """
        値の範囲を検証
        
        Args:
            df: 検証対象のDataFrame
            validations: 列名と検証ルールの辞書
        
        Returns:
            検証結果の辞書
        """
        results = {
            "valid": True,
            "violations": []
        }
        
        for col, rules in validations.items():
            if col not in df.columns:
                continue
            
            # Min value check
            if "min" in rules:
                min_violations = df.filter(pl.col(col) < rules["min"])
                if len(min_violations) > 0:
                    results["violations"].append({
                        "column": col,
                        "rule": f"min={rules['min']}",
                        "count": len(min_violations)
                    })
            
            # Max value check
            if "max" in rules:
                max_violations = df.filter(pl.col(col) > rules["max"])
                if len(max_violations) > 0:
                    results["violations"].append({
                        "column": col,
                        "rule": f"max={rules['max']}",
                        "count": len(max_violations)
                    })
            
            # Unique check
            if "unique" in rules and rules["unique"]:
                duplicates = df.group_by(col).count().filter(pl.col("count") > 1)
                if len(duplicates) > 0:
                    results["violations"].append({
                        "column": col,
                        "rule": "unique",
                        "count": len(duplicates)
                    })
        
        if results["violations"]:
            results["valid"] = False
        
        return results
    
    @staticmethod
    def check_data_leakage(
        df: pl.DataFrame,
        date_col: str = "meta_date",
        target_cols: List[str] = None
    ) -> Dict[str, any]:
        """
        データリーケージの検査
        
        Args:
            df: 検証対象のDataFrame
            date_col: 日付列名
            target_cols: ターゲット列のリスト
        
        Returns:
            検査結果の辞書
        """
        if target_cols is None:
            target_cols = [col for col in df.columns if col.startswith("y_")]
        
        results = {
            "has_leakage": False,
            "suspicious_features": [],
            "future_data_detected": []
        }
        
        # Check for future information in features
        for col in df.columns:
            if col in target_cols or col == date_col:
                continue
            
            # Check correlation with future targets
            for target in target_cols:
                if target in df.columns:
                    corr = df[col].corr(df[target])
                    if abs(corr) > 0.95:  # Suspiciously high correlation
                        results["suspicious_features"].append({
                            "feature": col,
                            "target": target,
                            "correlation": corr
                        })
        
        if results["suspicious_features"] or results["future_data_detected"]:
            results["has_leakage"] = True
        
        return results