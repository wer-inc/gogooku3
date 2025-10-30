#!/usr/bin/env python3
"""
Corporate Actions Adjustment
コーポレートアクション（株式分割、併合等）の調整実装
"""

import json
import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import polars as pl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CorporateActionsAdjuster:
    """コーポレートアクション調整クラス"""

    def __init__(self, actions_file: Path | None = None):
        """
        Initialize with corporate actions data

        Args:
            actions_file: コーポレートアクションデータファイルパス
        """
        self.actions = (
            self._load_actions(actions_file)
            if actions_file
            else self._get_default_actions()
        )

    def _get_default_actions(self) -> pl.DataFrame:
        """デフォルトのコーポレートアクションデータ"""
        # サンプルデータ（実際にはJQuantsや外部ソースから取得）
        actions_data = {
            "Code": ["7203", "9984", "6758"],  # トヨタ、ソフトバンク、ソニー
            "action_date": [
                date(2021, 10, 1),  # トヨタ 5:1分割
                date(2020, 6, 30),  # ソフトバンク 2:1分割
                date(2023, 10, 1),  # ソニー 5:1分割
            ],
            "action_type": ["split", "split", "split"],
            "ratio": [5.0, 2.0, 5.0],
            "description": ["5株分割", "2株分割", "5株分割"],
        }

        return pl.DataFrame(actions_data).with_columns(
            pl.col("action_date").cast(pl.Date)
        )

    def _load_actions(self, file_path: Path) -> pl.DataFrame:
        """外部ファイルからコーポレートアクションを読み込み"""
        if file_path.suffix == ".json":
            with open(file_path) as f:
                data = json.load(f)
            return pl.DataFrame(data)
        elif file_path.suffix == ".csv":
            return pl.read_csv(file_path)
        elif file_path.suffix == ".parquet":
            return pl.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def adjust_for_actions(
        self, df: pl.DataFrame, ticker_col: str = "Code", date_col: str = "Date"
    ) -> pl.DataFrame:
        """
        価格データをコーポレートアクションで調整

        Args:
            df: 価格データ（OHLCV）
            ticker_col: 銘柄コードカラム名
            date_col: 日付カラム名

        Returns:
            調整済みDataFrame
        """
        if self.actions.is_empty():
            logger.info("No corporate actions to apply")
            return df

        df_adjusted = df.clone()

        # Sort actions by date
        actions_sorted = self.actions.sort("action_date")

        for action in actions_sorted.iter_rows(named=True):
            ticker = action["Code"]
            action_date = action["action_date"]
            action_type = action["action_type"]
            ratio = float(action["ratio"])

            logger.info(
                f"Applying {action_type} for {ticker} on {action_date} (ratio: {ratio})"
            )

            # Create mask for affected rows
            if df_adjusted[date_col].dtype == pl.Datetime:
                # Convert action_date to datetime for comparison
                action_datetime = datetime.combine(action_date, datetime.min.time())
                mask = (df_adjusted[ticker_col] == ticker) & (
                    df_adjusted[date_col] < action_datetime
                )
            else:
                mask = (df_adjusted[ticker_col] == ticker) & (
                    df_adjusted[date_col] < action_date
                )

            # Apply adjustment based on action type
            if action_type == "split":
                # 株式分割: 価格を ratio で割る、出来高を ratio 倍する
                df_adjusted = df_adjusted.with_columns(
                    [
                        pl.when(mask)
                        .then(pl.col("Open") / ratio)
                        .otherwise(pl.col("Open"))
                        .alias("Open"),
                        pl.when(mask)
                        .then(pl.col("High") / ratio)
                        .otherwise(pl.col("High"))
                        .alias("High"),
                        pl.when(mask)
                        .then(pl.col("Low") / ratio)
                        .otherwise(pl.col("Low"))
                        .alias("Low"),
                        pl.when(mask)
                        .then(pl.col("Close") / ratio)
                        .otherwise(pl.col("Close"))
                        .alias("Close"),
                        pl.when(mask)
                        .then(pl.col("Volume") * ratio)
                        .otherwise(pl.col("Volume"))
                        .alias("Volume"),
                    ]
                )

            elif action_type == "reverse_split":
                # 株式併合: 価格を ratio 倍する、出来高を ratio で割る
                df_adjusted = df_adjusted.with_columns(
                    [
                        pl.when(mask)
                        .then(pl.col("Open") * ratio)
                        .otherwise(pl.col("Open"))
                        .alias("Open"),
                        pl.when(mask)
                        .then(pl.col("High") * ratio)
                        .otherwise(pl.col("High"))
                        .alias("High"),
                        pl.when(mask)
                        .then(pl.col("Low") * ratio)
                        .otherwise(pl.col("Low"))
                        .alias("Low"),
                        pl.when(mask)
                        .then(pl.col("Close") * ratio)
                        .otherwise(pl.col("Close"))
                        .alias("Close"),
                        pl.when(mask)
                        .then(pl.col("Volume") / ratio)
                        .otherwise(pl.col("Volume"))
                        .alias("Volume"),
                    ]
                )

            elif action_type == "dividend":
                # 配当落ち調整（権利落ち日の価格調整）
                dividend_amount = action.get("dividend_amount", 0)
                if dividend_amount > 0:
                    df_adjusted = df_adjusted.with_columns(
                        [
                            pl.when(mask)
                            .then(pl.col("Open") - dividend_amount)
                            .otherwise(pl.col("Open"))
                            .alias("Open"),
                            pl.when(mask)
                            .then(pl.col("High") - dividend_amount)
                            .otherwise(pl.col("High"))
                            .alias("High"),
                            pl.when(mask)
                            .then(pl.col("Low") - dividend_amount)
                            .otherwise(pl.col("Low"))
                            .alias("Low"),
                            pl.when(mask)
                            .then(pl.col("Close") - dividend_amount)
                            .otherwise(pl.col("Close"))
                            .alias("Close"),
                        ]
                    )

            # Log adjustment statistics
            affected_rows = mask.sum()
            if affected_rows > 0:
                logger.info(f"  Adjusted {affected_rows} rows for {ticker}")

        return df_adjusted

    def add_adjustment_flags(
        self, df: pl.DataFrame, ticker_col: str = "Code", date_col: str = "Date"
    ) -> pl.DataFrame:
        """調整フラグを追加（どのアクションが適用されたか）"""
        df_with_flags = df.with_columns(
            pl.lit("").alias("corporate_action"), pl.lit(1.0).alias("adjustment_factor")
        )

        for action in self.actions.iter_rows(named=True):
            ticker = action["Code"]
            action_date = action["action_date"]
            action_type = action["action_type"]
            ratio = float(action["ratio"])

            # Create mask for action date
            if df_with_flags[date_col].dtype == pl.Datetime:
                action_datetime = datetime.combine(action_date, datetime.min.time())
                mask = (df_with_flags[ticker_col] == ticker) & (
                    df_with_flags[date_col] == action_datetime
                )
            else:
                mask = (df_with_flags[ticker_col] == ticker) & (
                    df_with_flags[date_col] == action_date
                )

            # Add flag for this action
            df_with_flags = df_with_flags.with_columns(
                [
                    pl.when(mask)
                    .then(pl.lit(action_type))
                    .otherwise(pl.col("corporate_action"))
                    .alias("corporate_action"),
                    pl.when(mask)
                    .then(pl.lit(ratio))
                    .otherwise(pl.col("adjustment_factor"))
                    .alias("adjustment_factor"),
                ]
            )

        return df_with_flags

    def get_adjustment_history(self, ticker: str) -> pl.DataFrame:
        """特定銘柄の調整履歴を取得"""
        return self.actions.filter(pl.col("Code") == ticker).sort("action_date")

    def save_actions(self, path: Path):
        """コーポレートアクションデータを保存"""
        if path.suffix == ".json":
            # Convert to dict for JSON serialization
            data = self.actions.to_dict(as_series=False)
            # Convert date objects to strings
            if "action_date" in data:
                data["action_date"] = [
                    d.isoformat() if isinstance(d, date) else str(d)
                    for d in data["action_date"]
                ]
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        elif path.suffix == ".csv":
            self.actions.write_csv(path)
        elif path.suffix == ".parquet":
            self.actions.write_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")


def adjust_for_actions_pandas(df: pd.DataFrame, ca: pd.DataFrame) -> pd.DataFrame:
    """
    Pandas版の調整関数（仕様書の実装）

    Args:
        df: 価格データ
        ca: コーポレートアクションデータ

    Returns:
        調整済みDataFrame
    """
    out = df.copy()

    for _, r in ca.sort_values("action_date").iterrows():
        mask = (out["ticker"] == r["ticker"]) & (
            out["date"] < pd.to_datetime(r["action_date"])
        )

        if r["action_type"] == "split":
            out.loc[mask, ["open", "high", "low", "close"]] /= float(r["ratio"])
            out.loc[mask, "volume"] *= float(r["ratio"])
        elif r["action_type"] == "reverse_split":
            out.loc[mask, ["open", "high", "low", "close"]] *= float(r["ratio"])
            out.loc[mask, "volume"] /= float(r["ratio"])

    return out


# Test function
def test_corporate_actions():
    """コーポレートアクション調整のテスト"""

    # Create sample price data
    price_data = {
        "Code": ["7203"] * 10,
        "Date": pl.date_range(
            date(2021, 9, 27), date(2021, 10, 6), interval="1d", eager=True
        ),
        "Open": [7000.0] * 10,
        "High": [7100.0] * 10,
        "Low": [6900.0] * 10,
        "Close": [7050.0] * 10,
        "Volume": [1000000] * 10,
    }
    df = pl.DataFrame(price_data)

    print("Original data:")
    print(df)

    # Apply adjustments
    adjuster = CorporateActionsAdjuster()
    df_adjusted = adjuster.adjust_for_actions(df)

    print("\nAdjusted data (after 5:1 split on 2021-10-01):")
    print(df_adjusted)

    # Add flags
    df_with_flags = adjuster.add_adjustment_flags(df)
    print("\nData with adjustment flags:")
    print(
        df_with_flags.select(["Date", "Close", "corporate_action", "adjustment_factor"])
    )


if __name__ == "__main__":
    test_corporate_actions()
