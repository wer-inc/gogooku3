"""マクロインデックス特徴量生成モジュール（P0: 最小構成）"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import polars as pl

from ..utils.rolling import roll_mean_safe, roll_std_safe

LOGGER = logging.getLogger(__name__)


def load_indices_allowlist(allowlist_path: Optional[Path] = None) -> dict:
    """
    インデックスコードのallowlistを読み込む。

    Args:
        allowlist_path: allowlist JSONファイルのパス（Noneの場合はデフォルトパス）

    Returns:
        allowlist辞書
    """
    if allowlist_path is None:
        # デフォルトパス: config/indices_allowlist.json
        config_dir = Path(__file__).parent.parent.parent / "config"
        allowlist_path = config_dir / "indices_allowlist.json"

    if not allowlist_path.exists():
        LOGGER.warning(f"Indices allowlist not found at {allowlist_path}, using defaults")
        # デフォルト値
        return {
            "p0_indices": {
                "indices": [
                    {"code": "0000", "name": "TOPIX", "category": "benchmark", "prefix": "topix"},
                    {"code": "0101", "name": "日経225", "category": "benchmark", "prefix": "nk225"},
                ]
            },
            "spreads": {"pairs": []},
        }

    with open(allowlist_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index_core_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    インデックスの中核特徴量を生成（P0）。

    生成する特徴量:
    - リターン: ret_prev_{1d,5d,20d}, ret_oc, ret_co
    - トレンド・ボラ: atr_14, natr_14, mom_63d, trend_gap_20_100, realized_vol_20

    Args:
        df: インデックスOHLCデータ（date, code, open, high, low, close列が必要）

    Returns:
        特徴量DataFrame（date, code, idx_* 列）
    """
    if df.is_empty():
        return df

    # 列名の正規化
    col_map = {
        "Date": "date",
        "Code": "code",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename({old: new})

    # 型の正規化
    if "date" in df.columns:
        df = df.with_columns(pl.col("date").cast(pl.Date, strict=False))
    if "code" in df.columns:
        df = df.with_columns(pl.col("code").cast(pl.Utf8, strict=False))

    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))

    # codeごとにソート
    if "code" in df.columns:
        df = df.sort(["code", "date"])
    else:
        df = df.sort("date")

    eps = 1e-9

    # 1. リターン特徴量（shift(1)でリーク防止）
    df = df.with_columns(
        [
            # ret_prev_{1d,5d,20d}
            ((pl.col("close") / (pl.col("close").shift(1) + eps)) - 1.0).alias("ret_prev_1d"),
            ((pl.col("close") / (pl.col("close").shift(5) + eps)) - 1.0).alias("ret_prev_5d"),
            ((pl.col("close") / (pl.col("close").shift(20) + eps)) - 1.0).alias("ret_prev_20d"),
            # ret_oc = open / close_prev - 1
            ((pl.col("open") / (pl.col("close").shift(1) + eps)) - 1.0).alias("ret_oc"),
            # ret_co = close / open - 1
            ((pl.col("close") / (pl.col("open") + eps)) - 1.0).alias("ret_co"),
        ]
    )

    # codeごとにグループ化して計算
    if "code" in df.columns:
        group_by_code = lambda expr: expr.over("code")
    else:
        group_by_code = lambda expr: expr

    # 2. ATR/NATR（14日）
    df = df.with_columns(
        [
            # True Range
            pl.max_horizontal(
                pl.col("high") - pl.col("low"),
                (pl.col("high") - pl.col("close").shift(1)).abs(),
                (pl.col("low") - pl.col("close").shift(1)).abs(),
            ).alias("_tr"),
        ]
    )
    df = df.with_columns(
        [
            roll_mean_safe(pl.col("_tr"), 14, min_periods=5, by="code" if "code" in df.columns else None).alias(
                "atr_14"
            ),
            roll_mean_safe(
                pl.col("_tr") / (pl.col("close") + eps), 14, min_periods=5, by="code" if "code" in df.columns else None
            ).alias("natr_14"),
        ]
    )

    # 3. モメンタム（63日）
    df = df.with_columns(
        [
            ((pl.col("close") / (pl.col("close").shift(63) + eps)) - 1.0).alias("mom_63d"),
        ]
    )
    if "code" in df.columns:
        df = df.with_columns(pl.col("mom_63d").over("code").alias("mom_63d"))

    # 4. トレンド（SMA20/SMA100乖離）
    df = df.with_columns(
        [
            roll_mean_safe(pl.col("close"), 20, min_periods=10, by="code" if "code" in df.columns else None).alias(
                "_sma20"
            ),
            roll_mean_safe(pl.col("close"), 100, min_periods=20, by="code" if "code" in df.columns else None).alias(
                "_sma100"
            ),
        ]
    )
    df = df.with_columns(
        [
            ((pl.col("_sma20") / (pl.col("_sma100") + eps)) - 1.0).alias("trend_gap_20_100"),
        ]
    )

    # 5. 実現ボラティリティ（20日）
    # 日次リターンの標準偏差 * sqrt(252)
    df = df.with_columns(
        [
            roll_std_safe(pl.col("ret_prev_1d"), 20, min_periods=10, by="code" if "code" in df.columns else None)
            .fill_null(0.0)
            .mul(252.0**0.5)
            .alias("realized_vol_20"),
        ]
    )

    # 一時列を削除
    cleanup_cols = ["_tr", "_sma20", "_sma100"]
    for col in cleanup_cols:
        if col in df.columns:
            df = df.drop(col)

    return df


def build_index_spreads(
    df: pl.DataFrame | pl.LazyFrame,
    allowlist: Optional[dict] = None,
) -> pl.DataFrame:
    """
    インデックス間のスプレッド特徴量を生成（P0）。

    Args:
        df: インデックス特徴量DataFrame（date, code, close列が必要）
        allowlist: allowlist辞書（spreads.pairsを使用）

    Returns:
        スプレッド特徴量DataFrame（date, idx_spread_* 列）
    """
    if not isinstance(df, pl.DataFrame):
        try:
            df = df.collect()
        except AttributeError:
            df = pl.DataFrame(df)

    if df.is_empty():
        return df

    if allowlist is None:
        allowlist = load_indices_allowlist()

    spread_pairs = allowlist.get("spreads", {}).get("pairs", [])
    if not spread_pairs:
        LOGGER.debug("No spread pairs defined in allowlist")
        return pl.DataFrame(schema={"date": pl.Date})

    # 列名の正規化
    if "Date" in df.columns and "date" not in df.columns:
        df = df.rename({"Date": "date"})
    if "Code" in df.columns and "code" not in df.columns:
        df = df.rename({"Code": "code"})
    if "Close" in df.columns and "close" not in df.columns:
        df = df.rename({"Close": "close"})

    if "date" not in df.columns or "code" not in df.columns or "close" not in df.columns:
        LOGGER.warning("Required columns (date, code, close) not found")
        return pl.DataFrame(schema={"date": pl.Date})

    # 各ペアのスプレッドを計算
    spread_features = []
    for pair in spread_pairs:
        code1 = pair["code1"]
        code2 = pair["code2"]
        name = pair["name"]

        # 両方のインデックスデータを取得
        idx1 = df.filter(pl.col("code") == code1).select(["date", pl.col("close").alias("close1")])
        idx2 = df.filter(pl.col("code") == code2).select(["date", pl.col("close").alias("close2")])

        if idx1.is_empty() or idx2.is_empty():
            LOGGER.debug(f"Skipping spread {name}: missing data for {code1} or {code2}")
            continue

        # 日付で結合
        spread_df = idx1.join(idx2, on="date", how="inner")
        if spread_df.is_empty():
            continue

        eps = 1e-9
        # リターンスプレッド（1日）
        spread_df = spread_df.with_columns(
            [
                ((pl.col("close1") / (pl.col("close1").shift(1) + eps)) - 1.0).alias("ret1"),
                ((pl.col("close2") / (pl.col("close2").shift(1) + eps)) - 1.0).alias("ret2"),
            ]
        )
        spread_df = spread_df.with_columns(
            [
                (pl.col("ret2") - pl.col("ret1")).alias(f"idx_spread_{name}_1d"),
            ]
        )

        # 選択列
        keep_cols = ["date", f"idx_spread_{name}_1d"]
        spread_df = spread_df.select(keep_cols)

        if spread_features:
            spread_features = spread_features.join(spread_df, on="date", how="outer")
        else:
            spread_features = spread_df

    if isinstance(spread_features, list) or spread_features.is_empty():
        return pl.DataFrame(schema={"date": pl.Date})

    return spread_features


def build_index_features(
    indices_df: pl.DataFrame,
    allowlist: Optional[dict] = None,
) -> pl.DataFrame:
    """
    インデックス特徴量を生成（P0: 中核＋スプレッド）。

    Args:
        indices_df: インデックスOHLCデータ
        allowlist: allowlist辞書

    Returns:
        特徴量DataFrame（date, idx_* 列）
    """
    if indices_df.is_empty():
        return pl.DataFrame(schema={"date": pl.Date})

    # 中核特徴量を生成
    core_features = build_index_core_features(indices_df)

    if core_features.is_empty():
        return pl.DataFrame(schema={"date": pl.Date})

    # スプレッド特徴量を生成
    spread_features = build_index_spreads(core_features, allowlist=allowlist)

    # codeごとに特徴量を展開（codeを列名に含める）
    # 各codeの特徴量をdateごとに結合
    result_features = None

    # codeごとに処理
    if "code" in core_features.columns:
        for code_val, group in core_features.group_by("code", maintain_order=True):
            if group.is_empty():
                continue
            code_str = code_val[0] if isinstance(code_val, tuple) else code_val

            # このcodeの特徴量を取得（code列を除く）
            code_features = group.drop("code")

            # 列名にcodeプレフィックスを追加（TOPIXは特別扱い）
            if code_str == "0000":
                prefix = "topix"
            else:
                # allowlistからprefixを取得
                if allowlist:
                    p0_indices = allowlist.get("p0_indices", {}).get("indices", [])
                    idx_config = next((idx for idx in p0_indices if idx["code"] == code_str), None)
                    if idx_config:
                        prefix = idx_config.get("prefix", f"idx_{code_str}")
                    else:
                        prefix = f"idx_{code_str}"
                else:
                    prefix = f"idx_{code_str}"

            # 列名をリネーム（date以外）
            rename_map = {col: f"{prefix}_{col}" if col != "date" else col for col in code_features.columns}
            code_features = code_features.rename(rename_map)

            if result_features is not None and not result_features.is_empty():
                result_features = result_features.join(code_features, on="date", how="outer")
            else:
                result_features = code_features
    else:
        # code列がない場合はそのまま使用
        result_features = core_features
        # 列名にidx_プレフィックスを追加
        rename_map = {col: f"idx_{col}" if col != "date" else col for col in result_features.columns}
        result_features = result_features.rename(rename_map)

    # スプレッド特徴量を結合
    if not spread_features.is_empty():
        if result_features is None or result_features.is_empty():
            result_features = spread_features
        else:
            result_features = result_features.join(spread_features, on="date", how="outer")
    elif result_features is None or result_features.is_empty():
        return pl.DataFrame(schema={"date": pl.Date})

    if "date_right" in result_features.columns:
        result_features = (
            result_features.with_columns(
                pl.when(pl.col("date").is_null()).then(pl.col("date_right")).otherwise(pl.col("date")).alias("date")
            ).drop("date_right")
        )

    return result_features
