import polars as pl


def add_strict_yoy(stm: pl.DataFrame) -> pl.DataFrame:
    """
    FY×Q厳密YoY（前年同Q）を計算し stmt_yoy_* を付与
    必須列: Code, effective_date, NetSales, OperatingProfit, Profit,
           TypeOfCurrentPeriod, CurrentFiscalYearStartDate
    """
    req = [
        "Code",
        "TypeOfCurrentPeriod",
        "CurrentFiscalYearStartDate",
        "NetSales",
        "OperatingProfit",
        "Profit",
    ]
    if not all(c in stm.columns for c in req):
        return stm

    numeric_cols = ["NetSales", "OperatingProfit", "Profit"]
    cast_exprs = [pl.col(col).cast(pl.Float64, strict=False).alias(col) for col in numeric_cols if col in stm.columns]

    df = stm.with_columns([
        pl.col("Code").cast(pl.Utf8).str.zfill(4),
        pl.col("CurrentFiscalYearStartDate").cast(pl.Utf8).str.slice(0, 4).cast(pl.Int32).alias("FY"),
        pl.col("TypeOfCurrentPeriod").cast(pl.Utf8).alias("Q"),
    ] + cast_exprs)

    prev = df.select([
        pl.col("Code"),
        (pl.col("FY") + 1).alias("FY"),
        pl.col("Q"),
        pl.col("NetSales").alias("_prev_NetSales"),
        pl.col("OperatingProfit").alias("_prev_OperatingProfit"),
        pl.col("Profit").alias("_prev_Profit"),
    ])

    out = df.join(prev, on=["Code", "FY", "Q"], how="left")

    out = out.with_columns([
        ((pl.col("NetSales") - pl.col("_prev_NetSales")) / (pl.col("_prev_NetSales").abs() + 1e-12)).alias("stmt_yoy_sales"),
        ((pl.col("OperatingProfit") - pl.col("_prev_OperatingProfit")) / (pl.col("_prev_OperatingProfit").abs() + 1e-12)).alias("stmt_yoy_op"),
        ((pl.col("Profit") - pl.col("_prev_Profit")) / (pl.col("_prev_Profit").abs() + 1e-12)).alias("stmt_yoy_np"),
    ])

    return out

