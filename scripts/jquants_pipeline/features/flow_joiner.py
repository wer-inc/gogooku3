"""
Trade-Spec Flow Joiner
週次投資部門別売買データを日次価格データに結合するユーティリティ

設計方針：
- 結合キー: (Section, Date) - 銘柄単位ではなく市場区分単位
- 有効区間: effective_start = 公表翌営業日(T+1), effective_end = 次回effective_startの前日
- 区間→日次展開してからSection×Dateでleft join（取りこぼしゼロ）
- 特徴量: コア8列 + flow_impulse/days_since_flow
- フォールバック: Section不明はAllMarketに載せ、is_section_fallback=1を残す
- リーク検査: 負のdays_since_flowが0件、(Code,Date)一意などを自動テスト
"""

import polars as pl
from typing import Iterable, Optional, Callable
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def normalize_section_name(name: str) -> str:
    """市場再編を跨ぐ名称ゆらぎを正規化。"""
    if name is None:
        return None
    mapping = {
        "TSE1st": "TSEPrime",
        "TSE2nd": "TSEStandard",
        "Mothers": "TSEGrowth",
        "JASDAQ": "TSEStandard",
        "JASDAQ Standard": "TSEStandard",
        "JASDAQ Growth": "TSEGrowth",
        "Prime": "TSEPrime",
        "Prime Market": "TSEPrime",
        "TSE Prime": "TSEPrime",
        "東証プライム": "TSEPrime",
        "Standard": "TSEStandard",
        "Standard Market": "TSEStandard",
        "TSE Standard": "TSEStandard",
        "東証スタンダード": "TSEStandard",
        "Growth": "TSEGrowth",
        "Growth Market": "TSEGrowth",
        "TSE Growth": "TSEGrowth",
        "東証グロース": "TSEGrowth",
        "All": "AllMarket",
        "ALL": "AllMarket",
        "All Market": "AllMarket",
        "ALL MARKET": "AllMarket",
        "Other": "AllMarket",
    }
    return mapping.get(str(name), str(name))


def build_flow_intervals(
    trades_spec: pl.DataFrame, 
    next_bd: Callable[[datetime], datetime]
) -> pl.DataFrame:
    """
    trades_spec → 有効区間テーブル
    
    Args:
        trades_spec: 投資部門別売買データ（PublishedDate, Section必須）
        next_bd: 翌営業日を返す関数
        
    Returns:
        section, effective_start, effective_end を含むDataFrame
    """
    logger.info(f"Building flow intervals from {len(trades_spec)} trade-spec records")
    
    # PublishedDateとSectionを整形
    df = trades_spec.with_columns([
        pl.col("PublishedDate").cast(pl.Date),
        pl.col("Section").map_elements(normalize_section_name, return_dtype=pl.Utf8).alias("section"),
    ])
    
    # PublishedDateの翌営業日をeffective_startとする（T+1ルール）
    df = df.with_columns([
        pl.col("PublishedDate").map_elements(
            lambda d: next_bd(d.to_pydatetime() if hasattr(d, 'to_pydatetime') else d) if d else None, 
            return_dtype=pl.Date
        ).alias("effective_start")
    ])
    
    # Section×PublishedDateでソート
    df = df.sort(["section", "PublishedDate"])
    
    # 同section・同PublishedDateに複数があれば最後（遅い方）を採用
    df = df.group_by(["section", "PublishedDate"]).tail(1)
    
    # 次回startの前日をendに設定
    df = df.with_columns([
        pl.col("effective_start").shift(-1).over("section").alias("next_start")
    ]).with_columns([
        (pl.col("next_start").fill_null(pl.date(2999, 12, 31)) - pl.duration(days=1)).alias("effective_end")
    ]).drop("next_start")
    
    logger.info(f"Created {len(df)} flow intervals")
    return df


def add_flow_features(flow_intervals: pl.DataFrame) -> pl.DataFrame:
    """
    フロー特徴量の生成（コア8列）
    
    Args:
        flow_intervals: effective_start/endを含む区間データ
        
    Returns:
        フロー特徴量を追加したDataFrame
    """
    df = flow_intervals
    
    def net_ratio(prefix: str) -> pl.Expr:
        """Net/Total比率を計算"""
        return (
            pl.col(f"{prefix}Balance") / (pl.col(f"{prefix}Total") + 1e-12)
        ).alias(f"{prefix.lower()}_net_ratio")
    
    # 基本的なフロー特徴量
    df = df.with_columns([
        # 外国人・個人のネット比率
        net_ratio("Foreigners"),
        net_ratio("Individuals"),
        
        # 外国人の活動シェア
        (pl.col("ForeignersTotal") / (pl.col("TotalTotal") + 1e-12)).alias("foreign_share_activity"),
        
        # ブレッドス（買い越し部門の割合）
        (
            pl.concat_list([
                (pl.col("ForeignersBalance") > 0),
                (pl.col("IndividualsBalance") > 0),
                pl.when(pl.col("TrustBanksBalance").is_not_null())
                    .then(pl.col("TrustBanksBalance") > 0)
                    .otherwise(False),
                pl.when(pl.col("InvestmentTrustsBalance").is_not_null())
                    .then(pl.col("InvestmentTrustsBalance") > 0)
                    .otherwise(False),
                (pl.col("ProprietaryBalance") > 0),
                (pl.col("BrokerageBalance") > 0),
            ]).list.eval(pl.element().cast(pl.Int8)).list.sum() / 6.0
        ).alias("breadth_pos"),
    ])
    
    # 52週ローリングZ-score（Section内）
    by = ["section"]
    
    def roll_z(col: str, out: str) -> pl.Expr:
        """52週ローリングZ-scoreを計算"""
        return (
            (pl.col(col) - pl.col(col).rolling_mean(52).over(by)) /
            (pl.col(col).rolling_std(52).over(by) + 1e-12)
        ).alias(out)
    
    # Z-score特徴量の追加 + 活況度比率
    df = df.sort(["section", "effective_start"]).with_columns([
        roll_z("ForeignersBalance", "foreign_net_z"),
        roll_z("IndividualsBalance", "individual_net_z"),
        roll_z("TotalTotal", "activity_z"),
        (pl.col("TotalTotal") / (pl.col("TotalTotal").rolling_mean(52).over(by) + 1e-12)).alias("activity_ratio"),
    ])
    
    # スマートマネー指標
    df = df.with_columns([
        # スマートマネーインデックス（外国人 - 個人）
        (pl.col("foreign_net_z") - pl.col("individual_net_z")).alias("smart_money_idx"),
    ]).with_columns([
        # 4週モメンタム
        (
            pl.col("smart_money_idx") - 
            pl.col("smart_money_idx").rolling_mean(4).over(by)
        ).alias("smart_money_mom4"),
        
        # フローショックフラグ
        (pl.col("smart_money_idx").abs() >= 2.0).cast(pl.Int8).alias("flow_shock_flag"),
    ])
    
    # 仕様準拠のflow_* エイリアスを追加（元列は保持）
    df = df.with_columns([
        pl.col("foreigners_net_ratio").alias("flow_foreign_net_ratio"),
        pl.col("individuals_net_ratio").alias("flow_individual_net_ratio"),
        pl.col("activity_ratio").alias("flow_activity_ratio"),
        pl.col("foreign_net_z").alias("flow_foreign_net_z"),
        pl.col("individual_net_z").alias("flow_individual_net_z"),
        pl.col("activity_z").alias("flow_activity_z"),
        pl.col("smart_money_idx").alias("flow_smart_idx"),
        pl.col("smart_money_mom4").alias("flow_smart_mom4"),
    ])

    # 必要な列のみ選択（元列＋flow_*エイリアス）
    keep_cols = [
        "section", "effective_start", "effective_end",
        "foreigners_net_ratio", "individuals_net_ratio",
        "foreign_share_activity", "breadth_pos",
        "activity_ratio", "activity_z", "smart_money_idx", "smart_money_mom4", "flow_shock_flag",
        # flow_* alias (spec)
        "flow_foreign_net_ratio", "flow_individual_net_ratio", "flow_activity_ratio",
        "flow_foreign_net_z", "flow_individual_net_z", "flow_activity_z",
        "flow_smart_idx", "flow_smart_mom4",
    ]
    
    available_cols = [c for c in keep_cols if c in df.columns]
    logger.info(f"Generated {len(available_cols) - 3} flow features")
    
    return df.select(available_cols)


def expand_flow_daily(
    flow_feat: pl.DataFrame, 
    business_days: Iterable
) -> pl.DataFrame:
    """
    P0-2: 区間データを日次データに展開（as-of結合で最適化）
    O(n²) cross join → O(n log n) as-of join
    
    Args:
        flow_feat: 区間ベースのフロー特徴量
        business_days: 営業日リスト
        
    Returns:
        日次展開されたフロー特徴量
    """
    # 営業日カレンダーの作成
    cal = pl.DataFrame({"Date": list(business_days)}).with_columns(
        pl.col("Date").cast(pl.Date)
    )
    
    logger.info(f"P0-2: Expanding flow features to {len(cal)} business days using optimized as-of join")
    
    # P0-2: As-of結合による最適化実装
    sections = flow_feat["section"].unique().to_list()
    daily_parts = []
    
    for section in sections:
        # Section別にas-of結合
        sec_cal = cal.with_columns(pl.lit(section).alias("section"))
        sec_flow = flow_feat.filter(pl.col("section") == section).sort("effective_start")
        
        # As-of結合（backward）でeffective_startを取得
        joined = sec_cal.join_asof(
            sec_flow,
            left_on="Date",
            right_on="effective_start",
            by="section",
            strategy="backward"
        )
        
        # 有効期間内のみ保持
        if "effective_end" in joined.columns:
            joined = joined.filter(
                (pl.col("Date") >= pl.col("effective_start")) &
                (pl.col("Date") <= pl.col("effective_end"))
            )
        
        daily_parts.append(joined)
    
    # 全Sectionを結合
    daily = pl.concat(daily_parts, how="vertical") if daily_parts else pl.DataFrame()
    
    if not daily.is_empty():
        daily = daily.with_columns([
            # フローインパルス（公表初日フラグ）
            (pl.col("Date") == pl.col("effective_start")).cast(pl.Int8).alias("flow_impulse"),
            
            # フロー公表からの経過日数
            (pl.col("Date") - pl.col("effective_start")).dt.days().alias("days_since_flow")
        ]).with_columns([
            # 仕様名のエイリアス
            pl.col("days_since_flow").alias("flow_days_since")
        ]).drop(["effective_start", "effective_end"])
    
    logger.info(f"P0-2 ✅: Created {len(daily)} daily flow records (optimized)")
    return daily


def attach_flow_to_quotes(
    quotes_sec: pl.DataFrame, 
    flow_daily: pl.DataFrame,
    section_col: str = "Section"
) -> pl.DataFrame:
    """
    価格データにフロー特徴量を結合（Section一致）
    
    Args:
        quotes_sec: Section列を持つ価格データ
        flow_daily: 日次展開されたフロー特徴量
        section_col: quotes_sec内のSection列名
        
    Returns:
        フロー特徴量を結合した価格データ
    """
    logger.info(f"Attaching flow features to {len(quotes_sec)} quote records")
    
    # 日付の型を統一
    if quotes_sec["Date"].dtype != flow_daily["Date"].dtype:
        flow_daily = flow_daily.with_columns(
            pl.col("Date").cast(quotes_sec["Date"].dtype)
        )
    
    # Section正規化
    quotes_sec = quotes_sec.with_columns([
        pl.col(section_col).map_elements(normalize_section_name, return_dtype=pl.Utf8).alias(section_col)
    ])

    # Section×Dateで結合
    out = quotes_sec.join(
        flow_daily,
        left_on=[section_col, "Date"],
        right_on=["section", "Date"],
        how="left"
    )
    
    # section列が重複している場合は削除
    if "section" in out.columns and section_col in out.columns:
        out = out.drop("section")
    
    # フロー関連列の特定
    flow_cols = [c for c in out.columns if c.startswith((
        "foreigners_", "individuals_", "smart_money", "activity_z",
        "foreign_share_activity", "breadth_pos", "flow_shock_flag"
    ))]
    
    # 欠損値の処理とバリデーション列の追加（NULLは保持、学習側で制御）
    out = out.with_columns([
        # インパルス・経過日数は明示的に保持しつつ、欠損は符号付きに
        pl.col("flow_impulse").fill_null(0).alias("flow_impulse"),
        pl.col("days_since_flow").fill_null(-1).alias("days_since_flow"),  # -1: 未経験
        (pl.col("activity_z").is_not_null()).cast(pl.Int8).alias("is_flow_valid")
    ])
    
    # カバレッジ統計のログ出力
    coverage = (out["is_flow_valid"] == 1).sum() / len(out) if len(out) > 0 else 0
    logger.info(f"Flow feature coverage: {coverage:.1%}")
    
    return out


def attach_flow_with_fallback(
    quotes: pl.DataFrame,
    flow_daily: pl.DataFrame,
    section_mapper,
    fallback_section: str = "AllMarket"
) -> pl.DataFrame:
    """
    Section不明の銘柄にフォールバック処理を適用して結合
    
    Args:
        quotes: 価格データ
        flow_daily: 日次フロー特徴量
        section_mapper: SectionMapper インスタンス
        fallback_section: フォールバック用のSection名
        
    Returns:
        フロー特徴量を結合した価格データ（フォールバック済み）
    """
    # Section付与
    quotes_sec = quotes.with_columns([
        pl.col("Section").fill_null(fallback_section).alias("Section"),
        pl.col("Section").is_null().cast(pl.Int8).alias("is_section_fallback")
    ])
    
    # フロー結合
    result = attach_flow_to_quotes(quotes_sec, flow_daily, "Section")
    
    # フォールバック統計のログ
    if "is_section_fallback" in result.columns:
        fallback_pct = result["is_section_fallback"].sum() / len(result) if len(result) > 0 else 0
        logger.info(f"Section fallback rate: {fallback_pct:.1%}")
    
    return result