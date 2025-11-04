"""
Code Normalizer - 銘柄コード正規化
J-Quants API間のCode/LocalCode表記ゆれを統一
"""

import logging
import re

import polars as pl

logger = logging.getLogger(__name__)


class CodeNormalizer:
    """
    銘柄コードの正規化処理

    J-Quants APIのエンドポイント間で異なる表記を統一:
    - Code vs LocalCode
    - 4桁 vs 5桁（末尾0）
    - 文字列 vs 数値
    """

    # 特殊な銘柄コードのマッピング（必要に応じて追加）
    SPECIAL_MAPPINGS = {
        "86970": "8697",  # 日本取引所グループ（5桁→4桁）
        "13010": "1301",  # 極洋（末尾0削除）
        "94346": "9434",  # NTTデータG等: 5桁LocalCode→4桁Code
        # 追加のマッピングをここに
    }

    # LocalCodeは4桁銘柄コード + 市場区分(1桁)で構成される
    LOCAL_CODE_PATTERN = re.compile(r"^\d{5}$")

    @staticmethod
    def normalize_code(code: str | int | None) -> str | None:
        """
        単一の銘柄コードを正規化

        Args:
            code: 銘柄コード（文字列または数値）

        Returns:
            正規化された4桁の銘柄コード文字列

        Examples:
            >>> normalize_code("1301")
            "1301"
            >>> normalize_code("13010")
            "1301"
            >>> normalize_code(1301)
            "1301"
            >>> normalize_code("001301")
            "1301"
        """
        if code is None or (isinstance(code, str) and code.strip() == ""):
            return None

        # 文字列に変換して前後の空白を削除
        code_str = str(code).strip()

        # 数字以外の文字を削除（ハイフンなど）
        code_str = re.sub(r'[^0-9]', '', code_str)

        if not code_str:
            return None

        # 特殊マッピングをチェック
        if code_str in CodeNormalizer.SPECIAL_MAPPINGS:
            code_str = CodeNormalizer.SPECIAL_MAPPINGS[code_str]

        # 5桁LocalCodeは末尾1桁が市場区分なので切り落とす
        if CodeNormalizer.LOCAL_CODE_PATTERN.match(code_str):
            base_code = code_str[:-1]
            # base_codeが空文字になるケース（00000等）は後続処理でNoneに落とす
            if base_code:
                code_str = base_code

        # 5桁で末尾が0の場合は4桁に変換
        if len(code_str) == 5 and code_str.endswith("0"):
            code_str = code_str[:-1]

        # 先頭の0を削除して4桁にパディング
        code_int = int(code_str)
        if code_int == 0:
            return None

        # 4桁にゼロパディング
        normalized = str(code_int).zfill(4)

        # 4桁を超える場合は警告
        if len(normalized) > 4:
            logger.warning(f"Code {code} normalized to {normalized} (>4 digits)")

        return normalized

    @staticmethod
    def normalize_dataframe(
        df: pl.DataFrame,
        code_columns: list[str] | None = None,
        target_column: str = "Code"
    ) -> pl.DataFrame:
        """
        DataFrameの銘柄コード列を正規化

        Args:
            df: 対象のDataFrame
            code_columns: 正規化する列名のリスト（Noneの場合は自動検出）
            target_column: 正規化後の列名

        Returns:
            正規化されたDataFrame
        """
        if code_columns is None:
            # Code, LocalCode, コード などの列を自動検出
            potential_columns = [
                col for col in df.columns
                if any(pattern in col.lower() for pattern in ["code", "コード", "銘柄"])
            ]
            code_columns = potential_columns if potential_columns else []

        if not code_columns:
            logger.warning("No code columns found in DataFrame")
            return df

        result = df

        for col in code_columns:
            if col not in df.columns:
                continue

            # 正規化を適用
            normalized_col = f"{col}_normalized" if col != target_column else target_column

            result = result.with_columns(
                pl.col(col)
                .map_elements(CodeNormalizer.normalize_code, return_dtype=pl.Utf8)
                .alias(normalized_col)
            )

            # 元の列と異なる場合は置き換え
            if col != target_column and normalized_col != target_column:
                result = result.rename({normalized_col: target_column})
                if col != target_column:
                    result = result.drop(col)

        # 重複列の削除（LocalCode → Code など）
        if "LocalCode" in result.columns and "Code" in result.columns:
            result = result.drop("LocalCode")

        logger.info(f"Normalized code columns: {code_columns} → {target_column}")

        return result

    @staticmethod
    def validate_normalization(
        df: pl.DataFrame,
        code_column: str = "Code"
    ) -> dict:
        """
        正規化の検証とレポート

        Args:
            df: 検証対象のDataFrame
            code_column: 銘柄コード列名

        Returns:
            検証結果の辞書
        """
        if code_column not in df.columns:
            return {"error": f"Column {code_column} not found"}

        codes = df[code_column]

        # 統計情報
        stats = {
            "total_rows": len(codes),
            "unique_codes": codes.n_unique(),
            "null_codes": codes.is_null().sum(),
            "issues": []
        }

        # 4桁でないコードをチェック
        non_4digit = codes.filter(
            (codes.is_not_null()) &
            (codes.str.len_chars() != 4)
        )

        if len(non_4digit) > 0:
            stats["issues"].append({
                "type": "non_4digit",
                "count": len(non_4digit),
                "samples": non_4digit.head(5).to_list()
            })

        # 数字以外を含むコードをチェック
        non_numeric = codes.filter(
            (codes.is_not_null()) &
            (~codes.str.contains(r'^[0-9]+$'))
        )

        if len(non_numeric) > 0:
            stats["issues"].append({
                "type": "non_numeric",
                "count": len(non_numeric),
                "samples": non_numeric.head(5).to_list()
            })

        # 成功率
        valid_codes = codes.filter(
            (codes.is_not_null()) &
            (codes.str.len_chars() == 4) &
            (codes.str.contains(r'^[0-9]+$'))
        )

        stats["normalization_rate"] = len(valid_codes) / len(codes) if len(codes) > 0 else 0

        return stats

    @staticmethod
    def merge_with_normalization(
        left_df: pl.DataFrame,
        right_df: pl.DataFrame,
        left_code_col: str = "Code",
        right_code_col: str = "Code",
        on_date: bool = True,
        how: str = "left"
    ) -> pl.DataFrame:
        """
        正規化を適用してから結合

        Args:
            left_df: 左側のDataFrame
            right_df: 右側のDataFrame
            left_code_col: 左側の銘柄コード列名
            right_code_col: 右側の銘柄コード列名
            on_date: Date列でも結合するか
            how: 結合方法

        Returns:
            結合されたDataFrame
        """
        # 両側のコードを正規化
        left_normalized = CodeNormalizer.normalize_dataframe(
            left_df,
            code_columns=[left_code_col],
            target_column="Code_normalized"
        )

        right_normalized = CodeNormalizer.normalize_dataframe(
            right_df,
            code_columns=[right_code_col],
            target_column="Code_normalized"
        )

        # 結合キーの設定
        if on_date and "Date" in left_normalized.columns and "Date" in right_normalized.columns:
            join_keys = ["Code_normalized", "Date"]
        else:
            join_keys = ["Code_normalized"]

        # 結合実行
        result = left_normalized.join(
            right_normalized,
            on=join_keys,
            how=how
        )

        # 正規化列を元の名前に戻す
        if "Code" not in result.columns:
            result = result.rename({"Code_normalized": "Code"})
        else:
            result = result.drop("Code_normalized")

        # 結合統計のログ
        left_count = len(left_df)
        result_count = len(result)
        match_rate = result_count / left_count if left_count > 0 else 0

        logger.info(f"Merge completed: {left_count} → {result_count} rows ({match_rate:.1%} match rate)")

        return result


def apply_code_normalization_to_all_sources(
    daily_quotes: pl.DataFrame | None = None,
    statements: pl.DataFrame | None = None,
    trades_spec: pl.DataFrame | None = None,
    listed_info: pl.DataFrame | None = None,
    topix: pl.DataFrame | None = None
) -> dict:
    """
    全データソースにCode正規化を適用

    Args:
        各データソースのDataFrame（省略可）

    Returns:
        正規化されたDataFrameの辞書
    """
    normalizer = CodeNormalizer()
    results = {}

    # Daily quotes
    if daily_quotes is not None:
        logger.info("Normalizing daily_quotes...")
        results["daily_quotes"] = normalizer.normalize_dataframe(
            daily_quotes,
            code_columns=["Code"]
        )
        stats = normalizer.validate_normalization(results["daily_quotes"])
        logger.info(f"  Daily quotes normalization rate: {stats['normalization_rate']:.1%}")

    # Statements
    if statements is not None:
        logger.info("Normalizing statements...")
        results["statements"] = normalizer.normalize_dataframe(
            statements,
            code_columns=["LocalCode", "Code"],
            target_column="Code"
        )
        stats = normalizer.validate_normalization(results["statements"])
        logger.info(f"  Statements normalization rate: {stats['normalization_rate']:.1%}")

    # Trades spec（Sectionレベルなので通常Codeは含まない）
    if trades_spec is not None:
        results["trades_spec"] = trades_spec

    # Listed info
    if listed_info is not None:
        logger.info("Normalizing listed_info...")
        results["listed_info"] = normalizer.normalize_dataframe(
            listed_info,
            code_columns=["Code", "LocalCode"]
        )
        stats = normalizer.validate_normalization(results["listed_info"])
        logger.info(f"  Listed info normalization rate: {stats['normalization_rate']:.1%}")

    # TOPIX（通常Codeは含まない）
    if topix is not None:
        results["topix"] = topix

    return results
