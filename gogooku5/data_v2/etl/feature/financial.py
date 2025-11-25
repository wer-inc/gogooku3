"""Financial statement-based daily features built from /fins/statements and /fins/fs_details."""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

import duckdb
import polars as pl

logger = logging.getLogger(__name__)

IFRS_CURRENT_ASSETS_KEYS = ["Current assets (IFRS)", "Current assets"]
IFRS_INVESTMENT_SECURITIES_KEYS = [
    "Investment securities",
    "Other financial assets - NCA (IFRS)",
    "Other financial assets - CA (IFRS)",
    "Investments and other assets",
]
IFRS_LIABILITIES_KEYS = ["Liabilities (IFRS)", "Liabilities"]
IFRS_ASSETS_KEYS = ["Assets (IFRS)", "Assets"]
IFRS_EQUITY_KEYS = ["Equity (IFRS)", "Equity", "Net assets"]

JGAAP_CURRENT_ASSETS_KEYS = ["Current assets", "Total current assets"]
JGAAP_INVESTMENT_SECURITIES_KEYS = [
    "Investment securities",
    "Other securities",
    "Investments and other assets",
]
JGAAP_LIABILITIES_KEYS = ["Liabilities"]
JGAAP_ASSETS_KEYS = ["Assets"]
JGAAP_EQUITY_KEYS = ["Net assets", "Shareholders' equity"]


def _cast_float(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    exprs = []
    for c in cols:
        if c in df.columns:
            exprs.append(pl.col(c).cast(pl.Float64, strict=False).alias(c))
        else:
            exprs.append(pl.lit(None).cast(pl.Float64).alias(c))
    return df.with_columns(exprs)


def _pick_first_numeric(fs: dict[str, Any], candidates: list[str]) -> float | None:
    for key in candidates:
        if key in fs and fs[key] not in ("", None):
            try:
                return float(fs[key])
            except (TypeError, ValueError):
                continue
    for key, value in fs.items():
        if value in ("", None):
            continue
        for cand in candidates:
            if cand.lower() in key.lower():
                try:
                    return float(value)
                except (TypeError, ValueError):
                    break
    return None


def _extract_k1_components(json_str: str | None) -> tuple[float | None, float | None, float | None]:
    """
    Extract K1-style BS components from fs_details JSON:
    current_assets, investment_securities, total_liabilities.
    """
    if not json_str:
        return None, None, None
    try:
        obj = json.loads(json_str)
    except Exception:
        return None, None, None
    fs = obj.get("FinancialStatement") or obj
    if not isinstance(fs, dict):
        return None, None, None

    std = str(fs.get("Accounting standards, DEI") or "")
    is_ifrs = "IFRS" in std.upper()

    if is_ifrs:
        current_assets = _pick_first_numeric(
            fs,
            IFRS_CURRENT_ASSETS_KEYS,
        )
        invest_sec = _pick_first_numeric(
            fs,
            IFRS_INVESTMENT_SECURITIES_KEYS,
        )
        total_liab = _pick_first_numeric(
            fs,
            IFRS_LIABILITIES_KEYS,
        )
        if total_liab is None:
            assets = _pick_first_numeric(fs, IFRS_ASSETS_KEYS)
            equity = _pick_first_numeric(fs, IFRS_EQUITY_KEYS)
            if assets is not None and equity is not None:
                total_liab = assets - equity
    else:
        current_assets = _pick_first_numeric(fs, JGAAP_CURRENT_ASSETS_KEYS)
        invest_sec = _pick_first_numeric(
            fs,
            JGAAP_INVESTMENT_SECURITIES_KEYS,
        )
        total_liab = _pick_first_numeric(fs, JGAAP_LIABILITIES_KEYS)
        if total_liab is None:
            assets = _pick_first_numeric(fs, JGAAP_ASSETS_KEYS)
            net_assets = _pick_first_numeric(fs, JGAAP_EQUITY_KEYS)
            if assets is not None and net_assets is not None:
                total_liab = assets - net_assets

    if current_assets is None or invest_sec is None or total_liab is None:
        logger.debug("Missing K1 components in fs_details JSON (is_ifrs=%s)", is_ifrs)

    return current_assets, invest_sec, total_liab


def compute_financial_features(
    con: duckdb.DuckDBPyConnection,
    *,
    start: str,
    end: str,
) -> pl.DataFrame:
    """
    Compute financial ratios from /fins/statements and /fins/fs_details and
    project them to daily (date, code).
    """
    start_dt = date.fromisoformat(start)
    end_dt = date.fromisoformat(end)

    pf_arrow = con.execute(
        """
        SELECT
            date,
            code,
            ret_1d,
            ret_5d
        FROM price_flow_features
        WHERE date BETWEEN ? AND ?
        ORDER BY code, date
        """,
        [start, end],
    ).fetch_arrow_table()
    pf = pl.from_arrow(pf_arrow)
    if pf.is_empty():
        return pl.DataFrame()

    stm_arrow = con.execute(
        """
        SELECT
            s.disclosed_date,
            s.local_code AS code,
            s.current_period_end_date,
            s.type_of_current_period,
            s.net_sales,
            s.operating_profit,
            s.profit,
            s.earnings_per_share,
            s.total_assets,
            s.equity,
            s.equity_to_asset_ratio,
            s.book_value_per_share,
            s.cash_flows_from_operating_activities,
            s.cash_and_equivalents,
            s.number_of_issued_and_outstanding_shares_at_the_end_of_fiscal_year_including_treasury_stock,
            s.number_of_treasury_stock_at_the_end_of_fiscal_year,
            fd.financial_statement_json,
            -- Next trading date with a quote (report date ~ DisclosedDate+1営業日)
            (
                SELECT dq.date
                FROM daily_quotes AS dq
                WHERE dq.code = s.local_code
                  AND dq.date > s.disclosed_date
                ORDER BY dq.date
                LIMIT 1
            ) AS effective_date,
            (
                SELECT dq.adjustment_close
                FROM daily_quotes AS dq
                WHERE dq.code = s.local_code
                  AND dq.date > s.disclosed_date
                ORDER BY dq.date
                LIMIT 1
            ) AS price_at_effective_date
        FROM statements AS s
        LEFT JOIN fs_details AS fd
          ON s.disclosed_date = fd.disclosed_date
         AND s.local_code = fd.local_code
         AND s.disclosure_number = fd.disclosure_number
        WHERE s.disclosed_date <= ?
        """,
        [end],
    ).fetch_arrow_table()
    stm = pl.from_arrow(stm_arrow)
    if stm.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "code": pl.Utf8})

    stm = stm.with_columns(
        [
            pl.col("code").cast(pl.Utf8, strict=False),
            pl.col("disclosed_date").cast(pl.Date, strict=False),
            pl.col("current_period_end_date").cast(pl.Date, strict=False),
            pl.col("type_of_current_period").cast(pl.Utf8, strict=False),
            pl.col("effective_date").cast(pl.Date, strict=False),
        ]
    )

    numeric_cols = [
        "net_sales",
        "operating_profit",
        "profit",
        "earnings_per_share",
        "total_assets",
        "equity",
        "equity_to_asset_ratio",
        "book_value_per_share",
        "cash_flows_from_operating_activities",
        "cash_and_equivalents",
        "number_of_issued_and_outstanding_shares_at_the_end_of_fiscal_year_including_treasury_stock",
        "number_of_treasury_stock_at_the_end_of_fiscal_year",
        "price_at_effective_date",
    ]
    stm = _cast_float(stm, numeric_cols)

    if "financial_statement_json" in stm.columns:
        json_list = stm["financial_statement_json"].to_list()
        cur_list: list[float | None] = []
        inv_list: list[float | None] = []
        liab_list: list[float | None] = []
        for raw in json_list:
            c, i, l = _extract_k1_components(raw)
            cur_list.append(c)
            inv_list.append(i)
            liab_list.append(l)
        stm = stm.with_columns(
            [
                pl.Series("fund_current_assets_k1", cur_list),
                pl.Series("fund_investment_securities_k1", inv_list),
                pl.Series("fund_total_liabilities_k1", liab_list),
            ]
        )

    stm = (
        stm.sort(["code", "current_period_end_date", "disclosed_date"])
        .unique(subset=["code", "current_period_end_date"], keep="last")
        .filter(pl.col("code").is_not_null() & pl.col("disclosed_date").is_not_null())
    )
    if stm.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "code": pl.Utf8})

    EPS = 1e-9
    stm = stm.with_columns(
        [
            pl.col("net_sales").alias("fin_sales_ttm"),
            pl.col("operating_profit").alias("fin_op_profit_ttm"),
            pl.col("profit").alias("fin_profit_ttm"),
            pl.col("earnings_per_share").alias("fin_eps_ttm"),
        ]
    )
    stm = stm.with_columns(
        [
            (pl.col("fin_profit_ttm") / (pl.col("fin_sales_ttm") + EPS)).alias("fin_profit_margin"),
            (pl.col("fin_profit_ttm") / (pl.col("equity") + EPS)).alias("fin_roe"),
            (pl.col("fin_profit_ttm") / (pl.col("total_assets") + EPS)).alias("fin_roa"),
            pl.col("cash_flows_from_operating_activities").alias("fin_cfo_ttm"),
            (pl.col("cash_flows_from_operating_activities") / (pl.col("fin_profit_ttm") + EPS)).alias("fin_cfo_to_ni"),
        ]
    )
    stm = stm.with_columns(
        pl.when(pl.col("equity_to_asset_ratio").is_not_null())
        .then(pl.col("equity_to_asset_ratio"))
        .otherwise(pl.col("equity") / (pl.col("total_assets") + EPS))
        .alias("fin_equity_ratio")
    )
    stm = stm.with_columns(
        (
            pl.col("number_of_issued_and_outstanding_shares_at_the_end_of_fiscal_year_including_treasury_stock")
            - pl.col("number_of_treasury_stock_at_the_end_of_fiscal_year").fill_null(0.0)
        ).alias("fin_shares_outstanding_fs")
    )
    stm = stm.with_columns(
        (pl.col("cash_and_equivalents") - (pl.col("total_assets") - pl.col("equity"))).alias("_fin_net_cash_num")
    )

    stm = stm.filter(pl.col("effective_date").is_not_null() & (pl.col("effective_date") <= pl.lit(end_dt)))

    stm = stm.with_columns(
        pl.when(
            pl.col("fund_current_assets_k1").is_null()
            & pl.col("fund_investment_securities_k1").is_null()
            & pl.col("fund_total_liabilities_k1").is_null()
        )
        .then(None)
        .otherwise(
            pl.col("fund_current_assets_k1").fill_null(0.0)
            + 0.7 * pl.col("fund_investment_securities_k1").fill_null(0.0)
            - pl.col("fund_total_liabilities_k1").fill_null(0.0)
        )
        .alias("fund_net_cash_k1")
    )
    stm = stm.with_columns(
        (
            pl.col("fund_current_assets_k1").is_null()
            | pl.col("fund_investment_securities_k1").is_null()
            | pl.col("fund_total_liabilities_k1").is_null()
        ).alias("fund_k1_components_missing")
    )
    stm = stm.with_columns(
        (pl.col("price_at_effective_date") * pl.col("fin_shares_outstanding_fs")).alias("_fund_mcap_at_effective")
    )
    stm = stm.with_columns(
        [
            pl.when(pl.col("fund_net_cash_k1").is_not_null() & (pl.col("_fund_mcap_at_effective").abs() > 0))
            .then(pl.col("fund_net_cash_k1") / (pl.col("_fund_mcap_at_effective") + EPS))
            .otherwise(None)
            .alias("fund_net_cash_ratio_k1"),
        ]
    )
    stm = stm.with_columns(
        [
            pl.when(pl.col("fund_net_cash_ratio_k1").is_not_null())
            .then(pl.col("fund_net_cash_ratio_k1").clip(-5.0, 5.0))
            .otherwise(None)
            .alias("fund_net_cash_ratio_k1_clipped"),
            pl.when(pl.col("fund_net_cash_ratio_k1").is_not_null())
            .then(
                pl.when(pl.col("fund_net_cash_ratio_k1") > 0)
                .then(1.0)
                .when(pl.col("fund_net_cash_ratio_k1") < 0)
                .then(-1.0)
                .otherwise(0.0)
                * (pl.col("fund_net_cash_ratio_k1").abs() + 1.0).log()
            )
            .otherwise(None)
            .alias("fund_net_cash_ratio_k1_log"),
        ]
    )
    stm = stm.with_columns(
        pl.when(pl.col("book_value_per_share").is_not_null())
        .then(pl.col("book_value_per_share"))
        .otherwise(pl.col("equity") / (pl.col("fin_shares_outstanding_fs") + EPS))
        .alias("fund_bps")
    )
    stm = stm.with_columns(
        [
            pl.when(pl.col("fin_eps_ttm").abs() > 0)
            .then(pl.col("price_at_effective_date") / (pl.col("fin_eps_ttm") + EPS))
            .otherwise(None)
            .alias("fund_per_ttm"),
            pl.when(pl.col("fund_bps").abs() > 0)
            .then(pl.col("price_at_effective_date") / (pl.col("fund_bps") + EPS))
            .otherwise(None)
            .alias("fund_pbr"),
        ]
    )
    stm = stm.with_columns(
        pl.when(pl.col("fund_per_ttm").is_not_null() & pl.col("fund_net_cash_ratio_k1").is_not_null())
        .then(pl.col("fund_per_ttm") * (1.0 - pl.col("fund_net_cash_ratio_k1_clipped")))
        .otherwise(None)
        .alias("fund_cash_neutral_per")
    )
    stm = stm.with_columns(
        pl.when(pl.col("fin_shares_outstanding_fs").abs() > 0)
        .then(pl.col("fund_net_cash_k1") / (pl.col("fin_shares_outstanding_fs") + EPS))
        .otherwise(None)
        .alias("fund_net_cash_per_share")
    )

    fy = (
        stm.filter(pl.col("type_of_current_period") == "FY")
        .sort(["code", "current_period_end_date"])
        .with_columns(
            [
                pl.col("current_period_end_date").shift(1).over("code").alias("_prev_period_end_date"),
                pl.col("net_sales").shift(1).over("code").alias("_prev_net_sales"),
                pl.col("profit").shift(1).over("code").alias("_prev_profit"),
                pl.col("cash_flows_from_operating_activities").shift(1).over("code").alias("_prev_cfo"),
                pl.col("fund_net_cash_k1").shift(1).over("code").alias("_prev_net_cash_k1"),
            ]
        )
        .with_columns(
            [
                ((pl.col("current_period_end_date") - pl.col("_prev_period_end_date")).dt.total_days()).alias(
                    "_period_days_delta"
                ),
            ]
        )
        .with_columns(
            [
                pl.when(
                    pl.col("_prev_net_sales").is_not_null()
                    & (pl.col("_prev_net_sales") != 0)
                    & (pl.col("_period_days_delta") >= 0.75 * 365)
                    & (pl.col("_period_days_delta") <= 1.25 * 365)
                )
                .then((pl.col("net_sales") - pl.col("_prev_net_sales")) / pl.col("_prev_net_sales").abs())
                .otherwise(None)
                .clip(-5.0, 5.0)
                .alias("fund_yoy_sales"),
                pl.when(
                    pl.col("_prev_profit").is_not_null()
                    & (pl.col("_prev_profit") != 0)
                    & (pl.col("_period_days_delta") >= 0.75 * 365)
                    & (pl.col("_period_days_delta") <= 1.25 * 365)
                )
                .then((pl.col("profit") - pl.col("_prev_profit")) / pl.col("_prev_profit").abs())
                .otherwise(None)
                .clip(-5.0, 5.0)
                .alias("fund_yoy_profit"),
                pl.when(
                    pl.col("_prev_cfo").is_not_null()
                    & (pl.col("_prev_cfo") != 0)
                    & (pl.col("_period_days_delta") >= 0.75 * 365)
                    & (pl.col("_period_days_delta") <= 1.25 * 365)
                )
                .then(
                    (pl.col("cash_flows_from_operating_activities") - pl.col("_prev_cfo")) / pl.col("_prev_cfo").abs()
                )
                .otherwise(None)
                .clip(-5.0, 5.0)
                .alias("fund_yoy_cfo"),
                pl.when(
                    pl.col("_prev_net_cash_k1").is_not_null()
                    & (pl.col("_prev_net_cash_k1") != 0)
                    & (pl.col("_period_days_delta") >= 0.75 * 365)
                    & (pl.col("_period_days_delta") <= 1.25 * 365)
                )
                .then((pl.col("fund_net_cash_k1") - pl.col("_prev_net_cash_k1")) / pl.col("_prev_net_cash_k1").abs())
                .otherwise(None)
                .clip(-5.0, 5.0)
                .alias("fund_yoy_net_cash"),
            ]
        )
        .select(
            [
                "code",
                "current_period_end_date",
                "fund_yoy_sales",
                "fund_yoy_profit",
                "fund_yoy_cfo",
                "fund_yoy_net_cash",
            ]
        )
    )
    if not fy.is_empty():
        stm = stm.join(
            fy,
            on=["code", "current_period_end_date"],
            how="left",
        )

    stm = stm.with_columns(
        [
            (pl.col("profit") / (pl.col("equity") + EPS)).alias("fund_roe"),
            pl.when(pl.col("net_sales").abs() > 0)
            .then(pl.col("operating_profit") / (pl.col("net_sales") + EPS))
            .otherwise(None)
            .alias("fund_op_margin"),
            pl.when(pl.col("total_assets").abs() > 0)
            .then((pl.col("profit") - pl.col("cash_flows_from_operating_activities")) / (pl.col("total_assets") + EPS))
            .otherwise(None)
            .alias("fund_accruals"),
        ]
    )

    stm = stm.with_columns(
        [
            pl.when(pl.col("fund_net_cash_ratio_k1").is_not_null() & (pl.col("fund_net_cash_ratio_k1") >= 1.0))
            .then(1)
            .otherwise(0)
            .alias("fund_is_cash_rich"),
            pl.when(pl.col("fund_net_cash_ratio_k1").is_not_null() & (pl.col("fund_net_cash_ratio_k1") >= 0.5))
            .then(1)
            .otherwise(0)
            .alias("fund_is_high_net_cash"),
            pl.when(
                pl.col("fund_pbr").is_not_null()
                & (pl.col("fund_pbr") <= 0.5)
                & (pl.col("fund_per_ttm").is_null() | (pl.col("fund_per_ttm") > 30.0) | (pl.col("profit") <= 0))
            )
            .then(1)
            .otherwise(0)
            .alias("fund_value_trap_flag"),
        ]
    )

    stm = stm.sort(["code", "effective_date"]).set_sorted("effective_date")
    pf = pf.sort(["code", "date"]).set_sorted("date")

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sortedness of columns cannot be checked when 'by' groups provided",
        )
        daily = pf.join_asof(
            stm,
            left_on="date",
            right_on="effective_date",
            by="code",
            strategy="backward",
        )

    if daily.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "code": pl.Utf8})

    daily = daily.with_columns(
        [
            pl.when(pl.col("fin_eps_ttm").abs() > 0)
            .then(1.0 / (pl.col("fin_eps_ttm") + EPS))
            .otherwise(None)
            .alias("fin_pe"),
            pl.col("fin_shares_outstanding_fs").alias("_fin_market_cap_fs"),
        ]
    )
    daily = daily.with_columns(
        [
            pl.when(pl.col("_fin_market_cap_fs").abs() > 0)
            .then(pl.col("_fin_net_cash_num") / (pl.col("_fin_market_cap_fs") + EPS))
            .otherwise(None)
            .alias("fin_net_cash_ratio"),
        ]
    )
    daily = daily.with_columns(
        pl.when(pl.col("fin_net_cash_ratio").is_not_null())
        .then(
            pl.when(pl.col("fin_net_cash_ratio") < -10.0)
            .then(-10.0)
            .when(pl.col("fin_net_cash_ratio") > 10.0)
            .then(10.0)
            .otherwise(pl.col("fin_net_cash_ratio"))
            .alias("_fin_net_cash_ratio_clipped")
        )
        .otherwise(pl.lit(None))
    )
    daily = daily.with_columns(
        [
            pl.col("fin_net_cash_ratio").alias("fin_net_cash_per_share"),
        ]
    )
    daily = daily.with_columns(
        pl.when(pl.col("fin_pe").is_not_null() & pl.col("fund_net_cash_ratio_k1_clipped").is_not_null())
        .then(pl.col("fin_pe") * (1.0 - pl.col("fund_net_cash_ratio_k1_clipped")))
        .otherwise(None)
        .alias("fin_cash_neutral_pe")
    )

    result = daily.select(
        [
            "date",
            "code",
            "fin_sales_ttm",
            "fin_op_profit_ttm",
            "fin_profit_ttm",
            "fin_eps_ttm",
            "fin_profit_margin",
            "fin_roe",
            "fin_roa",
            "fin_cfo_ttm",
            "fin_cfo_to_ni",
            "fin_equity_ratio",
            "fin_shares_outstanding_fs",
            "fin_pe",
            "fin_net_cash_ratio",
            "fin_net_cash_per_share",
            "fin_cash_neutral_pe",
            "fund_net_cash_k1",
            "fund_k1_components_missing",
            "fund_net_cash_ratio_k1",
            "fund_net_cash_ratio_k1_clipped",
            "fund_net_cash_ratio_k1_log",
            "fund_net_cash_per_share",
            "fund_per_ttm",
            "fund_pbr",
            "fund_cash_neutral_per",
            "fund_yoy_sales",
            "fund_yoy_profit",
            "fund_yoy_cfo",
            "fund_yoy_net_cash",
            "fund_roe",
            "fund_op_margin",
            "fund_accruals",
            "fund_is_cash_rich",
            "fund_is_high_net_cash",
            "fund_value_trap_flag",
        ]
    )
    return result
