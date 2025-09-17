"""
Validity Flags - 有効性フラグ管理
学習時のフィルタリング基準となる各種有効性フラグを体系的に管理
"""

import logging

import polars as pl

logger = logging.getLogger(__name__)


class ValidityFlagManager:
    """
    データの有効性フラグを管理
    
    各種フラグ:
    - is_stmt_valid: 財務諸表データの有効性
    - is_flow_valid: フローデータの有効性
    - is_mkt_valid: 市場データの有効性
    - is_beta_valid: ベータ計算の有効性
    - is_section_fallback: Sectionマッピングのフォールバック使用
    """

    # 各フラグの最小要件定義
    REQUIREMENTS = {
        "stmt": {
            "min_days_since": 0,       # 負値でないこと
            "max_days_since": 365,      # 1年以内の開示
            "required_cols": ["stmt_yoy_sales", "stmt_opm"]
        },
        "flow": {
            "min_periods": 13,          # 四半期（13週）以上のデータ
            "max_days_since": 14,       # 2週間以内
            "required_cols": ["flow_foreign_net_ratio", "flow_smart_money_idx"]
        },
        "mkt": {
            "required_cols": ["mkt_return", "mkt_volatility"],
            "max_null_ratio": 0.1      # 10%以下の欠損
        },
        "beta": {
            "min_periods": 40,          # ベータ計算に必要な最小日数
            "required_cols": ["cross_beta_60d"],
            "valid_range": (-3.0, 3.0)  # 妥当なベータの範囲
        }
    }

    @staticmethod
    def add_statement_validity_flags(
        df: pl.DataFrame,
        stmt_prefix: str = "stmt_"
    ) -> pl.DataFrame:
        """
        財務諸表データの有効性フラグを追加
        
        Args:
            df: 対象DataFrame
            stmt_prefix: 財務諸表列のプレフィックス
        
        Returns:
            有効性フラグが追加されたDataFrame
        """
        result = df

        # 基本的な有効性チェック
        conditions = []

        # days_since_statementのチェック
        if f"{stmt_prefix}days_since_statement" in df.columns:
            conditions.append(
                (pl.col(f"{stmt_prefix}days_since_statement") >= 0) &
                (pl.col(f"{stmt_prefix}days_since_statement") <= 365)
            )

        # 必要な財務指標の存在チェック（互換列も許容）
        stmt_required = ValidityFlagManager.REQUIREMENTS["stmt"]["required_cols"]
        optional_alias = {
            "stmt_yoy_sales": f"{stmt_prefix}revenue_growth",
            "stmt_opm": f"{stmt_prefix}profit_margin",
        }
        for col in stmt_required:
            prefixed = col if col.startswith(stmt_prefix) else f"{stmt_prefix}{col.removeprefix(stmt_prefix)}"
            if prefixed in df.columns:
                conditions.append(pl.col(prefixed).is_not_null())
                continue
            alias_col = optional_alias.get(col)
            if alias_col and alias_col in df.columns:
                conditions.append(pl.col(alias_col).is_not_null())

        # YoY計算の前提条件
        yoy_candidates = [f"{stmt_prefix}yoy_sales", f"{stmt_prefix}revenue_growth"]
        for cand in yoy_candidates:
            if cand in df.columns:
                conditions.append(
                    pl.col(cand).is_not_null()
                    & (pl.col(cand) > -100)
                    & (pl.col(cand) < 1000)
                )
                break

        # 全条件を満たす場合に有効
        if conditions:
            is_valid = pl.all_horizontal(conditions)
        else:
            is_valid = pl.lit(0)

        result = result.with_columns(
            is_valid.cast(pl.Int8).alias("is_stmt_valid")
        )

        # 統計情報
        valid_count = result["is_stmt_valid"].sum()
        total_count = len(result)
        logger.info(f"Statement validity: {valid_count}/{total_count} ({valid_count/total_count:.1%})")

        return result

    @staticmethod
    def add_flow_validity_flags(
        df: pl.DataFrame,
        flow_prefix: str = "flow_"
    ) -> pl.DataFrame:
        """
        フローデータの有効性フラグを追加
        
        Args:
            df: 対象DataFrame
            flow_prefix: フロー列のプレフィックス
        
        Returns:
            有効性フラグが追加されたDataFrame
        """
        result = df
        conditions = []

        # days_since_flowのチェック
        if "days_since_flow" in df.columns:
            conditions.append(
                (pl.col("days_since_flow") >= 0) &
                (pl.col("days_since_flow") <= 14)  # 2週間以内
            )

        # Z-scoreの妥当性チェック
        if f"{flow_prefix}smart_money_idx" in df.columns:
            conditions.append(
                pl.col(f"{flow_prefix}smart_money_idx").is_not_null() &
                (pl.col(f"{flow_prefix}smart_money_idx").abs() < 5)  # |Z| < 5
            )

        # 必要な列の存在チェック
        for col in ValidityFlagManager.REQUIREMENTS["flow"]["required_cols"]:
            if col in df.columns:
                conditions.append(pl.col(col).is_not_null())

        # 既存のis_flow_validがある場合は考慮
        if "is_flow_valid_original" in df.columns:
            conditions.append(pl.col("is_flow_valid_original") == 1)

        if conditions:
            is_valid = pl.all_horizontal(conditions)
        else:
            is_valid = pl.lit(0)

        result = result.with_columns(
            is_valid.cast(pl.Int8).alias("is_flow_valid")
        )

        # 統計情報
        valid_count = result["is_flow_valid"].sum()
        total_count = len(result)
        logger.info(f"Flow validity: {valid_count}/{total_count} ({valid_count/total_count:.1%})")

        return result

    @staticmethod
    def add_market_validity_flags(
        df: pl.DataFrame,
        mkt_prefix: str = "mkt_"
    ) -> pl.DataFrame:
        """
        市場データの有効性フラグを追加
        
        Args:
            df: 対象DataFrame
            mkt_prefix: 市場データ列のプレフィックス
        
        Returns:
            有効性フラグが追加されたDataFrame
        """
        result = df
        conditions = []

        # 市場データの存在チェック
        mkt_cols = [col for col in df.columns if col.startswith(mkt_prefix)]

        if mkt_cols:
            # 少なくとも1つの市場データが存在
            conditions.append(
                pl.any_horizontal([pl.col(col).is_not_null() for col in mkt_cols[:3]])
            )

        # TOPIXリターンの妥当性チェック
        if f"{mkt_prefix}return" in df.columns:
            conditions.append(
                pl.col(f"{mkt_prefix}return").is_not_null() &
                (pl.col(f"{mkt_prefix}return").abs() < 0.1)  # 日次10%未満
            )

        if conditions:
            is_valid = pl.all_horizontal(conditions)
        else:
            is_valid = pl.lit(1)  # 市場データがない場合はデフォルトで有効

        result = result.with_columns(
            is_valid.cast(pl.Int8).alias("is_mkt_valid")
        )

        return result

    @staticmethod
    def add_beta_validity_flags(
        df: pl.DataFrame,
        beta_col: str = "cross_beta_60d"
    ) -> pl.DataFrame:
        """
        ベータ計算の有効性フラグを追加
        
        Args:
            df: 対象DataFrame
            beta_col: ベータ列名
        
        Returns:
            有効性フラグが追加されたDataFrame
        """
        result = df

        if beta_col in df.columns:
            # ベータの妥当な範囲チェック
            min_beta, max_beta = ValidityFlagManager.REQUIREMENTS["beta"]["valid_range"]

            is_valid = (
                pl.col(beta_col).is_not_null() &
                (pl.col(beta_col) >= min_beta) &
                (pl.col(beta_col) <= max_beta)
            )
        else:
            is_valid = pl.lit(0)

        result = result.with_columns(
            is_valid.cast(pl.Int8).alias("is_beta_valid")
        )

        return result

    @staticmethod
    def add_section_fallback_flag(
        df: pl.DataFrame,
        section_col: str = "Section"
    ) -> pl.DataFrame:
        """
        Sectionマッピングのフォールバック使用フラグを追加
        
        Args:
            df: 対象DataFrame
            section_col: Section列名
        
        Returns:
            フォールバックフラグが追加されたDataFrame
        """
        result = df

        if section_col in df.columns:
            # AllMarket, Other, Unknown などのフォールバック値を検出
            fallback_values = ["AllMarket", "Other", "Unknown", None]

            is_fallback = pl.col(section_col).is_in(fallback_values) | pl.col(section_col).is_null()
        else:
            is_fallback = pl.lit(1)  # Section列がない場合はフォールバック扱い

        result = result.with_columns(
            is_fallback.cast(pl.Int8).alias("is_section_fallback")
        )

        # フォールバックしたSectionを安全な値で埋める
        if section_col in result.columns:
            result = result.with_columns(
                pl.when(pl.col("is_section_fallback") == 1)
                .then(pl.lit("AllMarket"))
                .otherwise(pl.col(section_col))
                .alias(section_col)
            )

        # 統計情報
        fallback_count = result["is_section_fallback"].sum()
        total_count = len(result)
        logger.info(f"Section fallback: {fallback_count}/{total_count} ({fallback_count/total_count:.1%})")

        return result

    @staticmethod
    def add_all_validity_flags(df: pl.DataFrame) -> pl.DataFrame:
        """
        全ての有効性フラグを追加
        
        Args:
            df: 対象DataFrame
        
        Returns:
            全有効性フラグが追加されたDataFrame
        """
        result = df

        # 各種フラグを順次追加
        result = ValidityFlagManager.add_statement_validity_flags(result)
        result = ValidityFlagManager.add_flow_validity_flags(result)
        result = ValidityFlagManager.add_market_validity_flags(result)
        result = ValidityFlagManager.add_beta_validity_flags(result)
        result = ValidityFlagManager.add_section_fallback_flag(result)

        # 総合的な有効性フラグ（全てが有効な場合のみ1）
        validity_cols = [
            "is_stmt_valid",
            "is_flow_valid",
            "is_mkt_valid",
            "is_beta_valid"
        ]

        available_validity_cols = [col for col in validity_cols if col in result.columns]

        if available_validity_cols:
            result = result.with_columns(
                pl.all_horizontal(
                    [pl.col(col) == 1 for col in available_validity_cols]
                ).cast(pl.Int8).alias("is_fully_valid")
            )

        # サマリー統計
        logger.info("\nValidity Flags Summary:")
        for col in ["is_stmt_valid", "is_flow_valid", "is_mkt_valid", "is_beta_valid",
                    "is_section_fallback", "is_fully_valid"]:
            if col in result.columns:
                valid_ratio = result[col].sum() / len(result)
                logger.info(f"  {col}: {valid_ratio:.1%}")

        return result

    @staticmethod
    def filter_by_validity(
        df: pl.DataFrame,
        require_stmt: bool = True,
        require_flow: bool = True,
        require_mkt: bool = True,
        require_beta: bool = False,
        allow_section_fallback: bool = True
    ) -> pl.DataFrame:
        """
        有効性フラグに基づいてデータをフィルタリング
        
        Args:
            df: 対象DataFrame
            require_stmt: 財務諸表データの有効性を要求
            require_flow: フローデータの有効性を要求
            require_mkt: 市場データの有効性を要求
            require_beta: ベータの有効性を要求
            allow_section_fallback: Sectionフォールバックを許可
        
        Returns:
            フィルタリングされたDataFrame
        """
        conditions = []

        if require_stmt and "is_stmt_valid" in df.columns:
            conditions.append(pl.col("is_stmt_valid") == 1)

        if require_flow and "is_flow_valid" in df.columns:
            conditions.append(pl.col("is_flow_valid") == 1)

        if require_mkt and "is_mkt_valid" in df.columns:
            conditions.append(pl.col("is_mkt_valid") == 1)

        if require_beta and "is_beta_valid" in df.columns:
            conditions.append(pl.col("is_beta_valid") == 1)

        if not allow_section_fallback and "is_section_fallback" in df.columns:
            conditions.append(pl.col("is_section_fallback") == 0)

        if conditions:
            result = df.filter(pl.all_horizontal(conditions))
        else:
            result = df

        # フィルタリング統計
        before_count = len(df)
        after_count = len(result)
        filter_rate = after_count / before_count if before_count > 0 else 0

        logger.info(f"Filtered by validity: {before_count} → {after_count} ({filter_rate:.1%} retained)")

        return result
