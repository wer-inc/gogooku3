"""Advanced technical features."""
from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl

EPS = 1e-9


@dataclass
class AdvancedFeatureConfig:
    code_column: str = "code"
    date_column: str = "date"
    close_column: str = "close"
    volume_column: str = "volume"
    returns_1d: str = "returns_1d"
    returns_5d: str = "returns_5d"
    dollar_volume: str = "dollar_volume"


class AdvancedFeatureEngineer:
    def __init__(self, config: AdvancedFeatureConfig | None = None) -> None:
        self.config = config or AdvancedFeatureConfig()

    def add_features(self, df: pl.DataFrame) -> pl.DataFrame:
        cfg = self.config
        if df.is_empty():
            return df

        required = {cfg.code_column, cfg.date_column, cfg.close_column}
        if not required.issubset(df.columns):
            return df

        out = df
        if cfg.volume_column in out.columns:
            out = out.with_columns(pl.col(cfg.volume_column).rolling_mean(5).over(cfg.code_column).alias("volume_ma_5"))
            out = out.with_columns(
                pl.col(cfg.volume_column).rolling_mean(20).over(cfg.code_column).alias("volume_ma_20")
            )

        out = self._compute_rsi14(out)
        out = self._compute_realized_vol(out)

        if {"rsi_14", "realized_vol_20"}.issubset(out.columns):
            out = out.with_columns((pl.col("rsi_14") * pl.col("realized_vol_20")).alias("rsi_vol_interact"))

        if {cfg.returns_5d, cfg.volume_column, "volume_ma_20"}.issubset(out.columns):
            out = out.with_columns(
                (pl.col(cfg.returns_5d) * (pl.col(cfg.volume_column) / (pl.col("volume_ma_20") + EPS))).alias(
                    "vol_confirmed_mom"
                )
            )

        out = self._compute_macd_slope(out)

        if {"volume_ma_5", "volume_ma_20"}.issubset(out.columns):
            out = out.with_columns((pl.col("volume_ma_5") / (pl.col("volume_ma_20") + EPS) - 1.0).alias("volume_accel"))

        returns_present = cfg.returns_1d in out.columns
        if returns_present:
            out = out.with_columns(pl.col(cfg.returns_1d).shift(1).over(cfg.code_column).alias("lag_returns_1d"))

            cnt = pl.count().over(cfg.date_column)
            rk = pl.col("lag_returns_1d").rank(method="average").over(cfg.date_column)
            out = out.with_columns(
                pl.when(cnt > 1).then((rk - 1.0) / (cnt - 1.0)).otherwise(0.5).alias("rank_ret_prev_1d")
            )

        if cfg.volume_column in out.columns:
            out = out.with_columns(
                (
                    (pl.col(cfg.volume_column) - pl.col(cfg.volume_column).mean().over(cfg.date_column))
                    / (pl.col(cfg.volume_column).std().over(cfg.date_column) + EPS)
                ).alias("volume_cs_z")
            )

        if returns_present:
            out = out.with_columns((pl.col(cfg.returns_1d) > 0).cast(pl.Int8).alias("_pos"))
            out = out.with_columns(
                (pl.col("_pos") != pl.col("_pos").shift(1).over(cfg.code_column)).cast(pl.Int8).alias("_chg")
            )
            out = out.with_columns(pl.col("_chg").cum_sum().over(cfg.code_column).alias("_grp_id"))
            out = out.with_columns(pl.arange(1, pl.len() + 1).over([cfg.code_column, "_grp_id"]).alias("_seq"))
            out = out.with_columns(
                pl.when(pl.col("_pos") == 1).then(pl.col("_seq")).otherwise(0).alias("up_streak_1d"),
                pl.when(pl.col("_pos") == 0).then(pl.col("_seq")).otherwise(0).alias("down_streak_1d"),
            )
            out = out.with_columns(pl.col("_pos").rolling_mean(5).over(cfg.code_column).alias("mom_persist_5d"))
            out = out.drop([c for c in ("_pos", "_chg", "_grp_id", "_seq") if c in out.columns])

        out = self._add_calendar_flags(out)

        conds = []
        if {"rsi_14", "realized_vol_20"}.issubset(out.columns):
            conds.append(pl.col("rsi_14").is_not_null() & pl.col("realized_vol_20").is_not_null())
        if {cfg.returns_5d, cfg.volume_column, "volume_ma_20"}.issubset(out.columns):
            conds.append(
                pl.col(cfg.returns_5d).is_not_null()
                & pl.col(cfg.volume_column).is_not_null()
                & pl.col("volume_ma_20").is_not_null()
            )
        if conds:
            valid = conds[0]
            for c in conds[1:]:
                valid = valid | c
            out = out.with_columns(valid.cast(pl.Int8).alias("is_adv_valid"))
        else:
            out = out.with_columns(pl.lit(0).cast(pl.Int8).alias("is_adv_valid"))

        return out

    def _compute_rsi14(self, df: pl.DataFrame) -> pl.DataFrame:
        cfg = self.config
        if "rsi_14" in df.columns:
            return df
        delta = pl.col(cfg.close_column).diff().over(cfg.code_column)
        gain = pl.when(delta > 0).then(delta).otherwise(0.0)
        loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
        tmp = df.with_columns([gain.alias("_gain"), loss.alias("_loss")])
        tmp = tmp.with_columns(
            pl.col("_gain").rolling_mean(14).over(cfg.code_column).alias("_avg_gain"),
            pl.col("_loss").rolling_mean(14).over(cfg.code_column).alias("_avg_loss"),
        )
        rs = pl.col("_avg_gain") / (pl.col("_avg_loss") + EPS)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        tmp = tmp.with_columns(rsi.alias("rsi_14"))
        return tmp.drop([c for c in ("_gain", "_loss", "_avg_gain", "_avg_loss") if c in tmp.columns])

    def _compute_realized_vol(self, df: pl.DataFrame) -> pl.DataFrame:
        cfg = self.config
        if "realized_vol_20" in df.columns:
            return df
        try:
            if cfg.returns_1d in df.columns:
                tmp = (
                    df.lazy()
                    .with_columns(
                        pl.col(cfg.returns_1d).rolling_std(20).over(cfg.code_column).alias("_realized_vol_20")
                    )
                    .collect()
                )
                return tmp.with_columns((pl.col("_realized_vol_20") * math.sqrt(252.0)).alias("realized_vol_20")).drop(
                    "_realized_vol_20"
                )
            tmp = (
                df.lazy()
                .with_columns(
                    (
                        (pl.col(cfg.close_column) / pl.col(cfg.close_column).shift(1).over(cfg.code_column) - 1.0)
                        .rolling_std(20)
                        .over(cfg.code_column)
                        .alias("_realized_vol_20")
                    )
                )
                .collect()
            )
            return tmp.with_columns((pl.col("_realized_vol_20") * math.sqrt(252.0)).alias("realized_vol_20")).drop(
                "_realized_vol_20"
            )
        except Exception:
            return df

    def _compute_macd_slope(self, df: pl.DataFrame) -> pl.DataFrame:
        cfg = self.config
        if "macd_hist_slope" in df.columns:
            return df
        try:
            ema12 = pl.col(cfg.close_column).ewm_mean(span=12, adjust=False).over(cfg.code_column)
            ema26 = pl.col(cfg.close_column).ewm_mean(span=26, adjust=False).over(cfg.code_column)
            tmp = df.with_columns((ema12 - ema26).alias("_macd"))
            tmp = tmp.with_columns(
                pl.col("_macd").ewm_mean(span=9, adjust=False).over(cfg.code_column).alias("_signal")
            )
            tmp = tmp.with_columns((pl.col("_macd") - pl.col("_signal")).alias("_hist"))
            tmp = tmp.with_columns(pl.col("_hist").diff().over(cfg.code_column).alias("macd_hist_slope"))
            return tmp.drop([c for c in ("_macd", "_signal", "_hist") if c in tmp.columns])
        except Exception:
            return df

    def _add_calendar_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        cfg = self.config
        if cfg.date_column not in df.columns:
            return df
        cal = df.select(cfg.date_column).unique().sort(cfg.date_column)
        cal = cal.with_columns(
            pl.col(cfg.date_column).dt.weekday().alias("_dow"),
            pl.col(cfg.date_column).dt.year().alias("_yy"),
            pl.col(cfg.date_column).dt.month().alias("_mm"),
        )
        cal = cal.with_columns(
            pl.col(cfg.date_column).cum_count().over(["_yy", "_mm"]).alias("_pos"),
            pl.count().over(["_yy", "_mm"]).alias("_n"),
        )
        cal = cal.with_columns(
            (pl.col("_pos") <= 5).cast(pl.Int8).alias("month_start_flag"),
            ((pl.col("_n") - pl.col("_pos") + 1) <= 5).cast(pl.Int8).alias("month_end_flag"),
            (pl.col("_dow") == 0).cast(pl.Int8).alias("is_mon"),
            (pl.col("_dow") == 1).cast(pl.Int8).alias("is_tue"),
            (pl.col("_dow") == 2).cast(pl.Int8).alias("is_wed"),
            (pl.col("_dow") == 3).cast(pl.Int8).alias("is_thu"),
            (pl.col("_dow") == 4).cast(pl.Int8).alias("is_fri"),
        ).drop(["_dow", "_yy", "_mm", "_pos", "_n"])
        return df.join(cal, on=cfg.date_column, how="left")
