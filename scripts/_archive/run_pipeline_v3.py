#!/usr/bin/env python3
"""
Complete ML Dataset Pipeline V3 - 最適化版
効率的なデータフロー: daily_quotes先行取得 → listed_infoでフィルタリング
"""

import os
import sys
import asyncio
import aiohttp
import polars as pl
from datetime import datetime, timedelta
from pathlib import Path
import logging
import time
from typing import List, Optional, Dict, Set
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from data.ml_dataset_builder import MLDatasetBuilder
from components.trading_calendar_fetcher import TradingCalendarFetcher
from components.market_code_filter import MarketCodeFilter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class JQuantsOptimizedFetcher:
    """最適化されたJQuants API fetcher - bulk fetch approach"""

    def __init__(self, email: str, password: str, max_concurrent: int = None):
        self.email = email
        self.password = password
        self.base_url = "https://api.jquants.com/v1"
        self.id_token = None
        # 有料プラン向け設定
        self.max_concurrent = max_concurrent or int(
            os.getenv("MAX_CONCURRENT_FETCH", 75)
        )
        self.semaphore = asyncio.Semaphore(self.max_concurrent)

    async def authenticate(self, session: aiohttp.ClientSession):
        """Authenticate with JQuants API."""
        auth_url = f"{self.base_url}/token/auth_user"
        auth_payload = {"mailaddress": self.email, "password": self.password}

        async with session.post(auth_url, json=auth_payload) as response:
            if response.status != 200:
                text = await response.text()
                raise Exception(f"Auth failed: {response.status} - {text}")
            data = await response.json()
            refresh_token = data["refreshToken"]

        refresh_url = f"{self.base_url}/token/auth_refresh"
        params = {"refreshtoken": refresh_token}

        async with session.post(refresh_url, params=params) as response:
            if response.status != 200:
                raise Exception(f"Failed to get ID token: {response.status}")
            data = await response.json()
            self.id_token = data["idToken"]

        logger.info("✅ JQuants authentication successful")

    async def get_all_listed_info(
        self, session: aiohttp.ClientSession, business_days: List[str]
    ) -> Dict[str, Set[str]]:
        """
        期間内の全上場銘柄情報を取得し、Market Codeでフィルタリング
        
        Returns:
            {code: set(market_codes)} の辞書
        """
        logger.info("全期間の上場銘柄情報を取得中...")
        
        # 銘柄ごとのMarket Code情報を集約
        code_to_markets = {}
        
        # サンプリング: 期間の最初、中間、最後の日付で取得
        sample_dates = []
        if len(business_days) > 0:
            sample_dates.append(business_days[0])  # 最初
            if len(business_days) > 1:
                sample_dates.append(business_days[len(business_days)//2])  # 中間
                sample_dates.append(business_days[-1])  # 最後
        
        for date in sample_dates:
            date_api = date.replace("-", "")
            url = f"{self.base_url}/listed/info"
            headers = {"Authorization": f"Bearer {self.id_token}"}
            params = {"date": date_api}
            
            try:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        info_list = data.get("info", [])
                        
                        for info in info_list:
                            code = info.get("Code")
                            market_code = info.get("MarketCode")
                            
                            if code and market_code:
                                if code not in code_to_markets:
                                    code_to_markets[code] = set()
                                code_to_markets[code].add(market_code)
                        
                        logger.info(f"  {date}: {len(info_list)}銘柄の情報取得")
                        
            except Exception as e:
                logger.warning(f"Failed to fetch listed info for {date}: {e}")
        
        # フィルタリング対象のMarket Codeを持つ銘柄を特定
        target_codes = set()
        excluded_codes = set()
        
        for code, market_codes in code_to_markets.items():
            # いずれかのターゲット市場に上場していればOK
            if any(mc in MarketCodeFilter.TARGET_MARKET_CODES for mc in market_codes):
                target_codes.add(code)
            # 除外市場のみの場合は除外
            elif all(mc in MarketCodeFilter.EXCLUDE_MARKET_CODES for mc in market_codes):
                excluded_codes.add(code)
        
        logger.info(f"✅ 対象銘柄: {len(target_codes)}銘柄, 除外: {len(excluded_codes)}銘柄")
        
        return target_codes

    async def fetch_listed_info_range(
        self, session: aiohttp.ClientSession, business_days: List[str]
    ) -> pl.DataFrame:
        """期間内のlisted/infoを日次で取得して集約（Section interval作成用の素材）。"""
        url = f"{self.base_url}/listed/info"
        headers = {"Authorization": f"Bearer {self.id_token}"}
        records = []
        for date in business_days:
            params = {"date": date.replace("-", "")}
            async with session.get(url, headers=headers, params=params) as resp:
                if resp.status != 200:
                    continue
                data = await resp.json()
                for info in data.get("info", []):
                    rec = {
                        "Code": info.get("Code"),
                        "MarketCode": info.get("MarketCode"),
                        "Date": date,
                    }
                    if rec["Code"] and rec["MarketCode"]:
                        records.append(rec)
        return pl.DataFrame(records) if records else pl.DataFrame({"Code": [], "MarketCode": [], "Date": []})

    async def fetch_daily_quotes_bulk(
        self,
        session: aiohttp.ClientSession,
        business_days: List[str],
        batch_size: int = 30
    ) -> pl.DataFrame:
        """
        営業日を指定してdaily_quotesを一括取得
        
        Args:
            business_days: 営業日リスト
            batch_size: 一度に処理する日数
        """
        logger.info(f"Daily quotes一括取得中 ({len(business_days)}営業日)...")
        
        all_quotes = []
        
        for i in range(0, len(business_days), batch_size):
            batch_days = business_days[i:i+batch_size]
            logger.info(f"  Batch {i//batch_size + 1}: {batch_days[0]} - {batch_days[-1]}")
            
            # バッチ内の日付を並列処理
            tasks = []
            for date in batch_days:
                date_api = date.replace("-", "")
                task = self._fetch_daily_quotes_for_date(session, date, date_api)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            for df in results:
                if not df.is_empty():
                    all_quotes.append(df)
            
            logger.info(f"    累積: {sum(len(df) for df in all_quotes)}レコード")
        
        if all_quotes:
            combined_df = pl.concat(all_quotes)
            logger.info(f"✅ Daily quotes取得完了: {len(combined_df)}レコード")
            return combined_df
        
        return pl.DataFrame()

    async def _fetch_daily_quotes_for_date(
        self, session: aiohttp.ClientSession, date: str, date_api: str
    ) -> pl.DataFrame:
        """特定日の全銘柄のdaily_quotesを取得（ページネーション対応）"""
        url = f"{self.base_url}/prices/daily_quotes"
        headers = {"Authorization": f"Bearer {self.id_token}"}
        
        all_quotes = []
        pagination_key = None
        page_count = 0
        
        while True:
            params = {"date": date_api}
            if pagination_key:
                params["pagination_key"] = pagination_key
            
            try:
                async with self.semaphore:
                    async with session.get(url, headers=headers, params=params) as response:
                        if response.status != 200:
                            logger.debug(f"No quotes for {date}: {response.status}")
                            break
                        
                        data = await response.json()
                        quotes = data.get("daily_quotes", [])
                        
                        if quotes:
                            all_quotes.extend(quotes)
                            page_count += 1
                        
                        pagination_key = data.get("pagination_key")
                        if not pagination_key:
                            break
                            
            except Exception as e:
                logger.error(f"Error fetching quotes for {date}: {e}")
                break
        
        if all_quotes:
            df = pl.DataFrame(all_quotes)
            logger.debug(f"  {date}: {len(all_quotes)}レコード取得 ({page_count}ページ)")
            return df
        
        return pl.DataFrame()

    async def fetch_topix_data(
        self, session: aiohttp.ClientSession, from_date: str, to_date: str
    ) -> pl.DataFrame:
        """Fetch TOPIX index data."""
        url = f"{self.base_url}/indices/topix"
        headers = {"Authorization": f"Bearer {self.id_token}"}
        
        from_api = from_date.replace("-", "")
        to_api = to_date.replace("-", "")

        all_data = []
        pagination_key = None

        while True:
            params = {"from": from_api, "to": to_api}
            if pagination_key:
                params["pagination_key"] = pagination_key

            try:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to fetch TOPIX: {response.status}")
                        break

                    data = await response.json()
                    topix_data = data.get("topix", [])

                    if topix_data:
                        all_data.extend(topix_data)

                    pagination_key = data.get("pagination_key")
                    if not pagination_key:
                        break

            except Exception as e:
                logger.error(f"Error fetching TOPIX: {e}")
                break

        if all_data:
            df = pl.DataFrame(all_data)
            if "Date" in df.columns:
                df = df.with_columns(
                    pl.col("Date").str.strptime(
                        pl.Date, format="%Y-%m-%d", strict=False
                    )
                )
            if "Close" in df.columns:
                df = df.with_columns(pl.col("Close").cast(pl.Float64))

            logger.info(f"✅ Fetched {len(df)} TOPIX records")
            return df.sort("Date")

        return pl.DataFrame()

    async def fetch_trades_spec(
        self, session: aiohttp.ClientSession, from_date: str, to_date: str
    ) -> pl.DataFrame:
        """Fetch trades specification (売買内訳) data."""
        url = f"{self.base_url}/markets/trades_spec"
        headers = {"Authorization": f"Bearer {self.id_token}"}
        
        from_api = from_date.replace("-", "")
        to_api = to_date.replace("-", "")

        all_data = []
        pagination_key = None

        while True:
            params = {"from": from_api, "to": to_api}
            if pagination_key:
                params["pagination_key"] = pagination_key

            try:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to fetch trades_spec: {response.status}")
                        break

                    data = await response.json()
                    trades_data = data.get("trades_spec", [])

                    if trades_data:
                        all_data.extend(trades_data)

                    pagination_key = data.get("pagination_key")
                    if not pagination_key:
                        break

            except Exception as e:
                logger.error(f"Error fetching trades_spec: {e}")
                break

        if all_data:
            df = pl.DataFrame(all_data)
            logger.info(f"✅ Fetched {len(df)} trades_spec records")
            return df

        return pl.DataFrame()

    async def fetch_statements(
        self, session: aiohttp.ClientSession, codes: List[str]
    ) -> pl.DataFrame:
        """Fetch financial statements (財務諸表) data for specified stocks."""
        url = f"{self.base_url}/fins/statements"
        headers = {"Authorization": f"Bearer {self.id_token}"}
        
        all_statements = []
        
        # バッチ処理（一度に複数銘柄を取得）
        batch_size = 100
        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i:i+batch_size]
            code_str = ",".join(batch_codes)
            
            params = {"code": code_str}
            
            try:
                async with self.semaphore:
                    async with session.get(url, headers=headers, params=params) as response:
                        if response.status != 200:
                            logger.warning(f"Failed to fetch statements for batch {i//batch_size}: {response.status}")
                            continue

                        data = await response.json()
                        statements = data.get("statements", [])

                        if statements:
                            all_statements.extend(statements)
                            logger.debug(f"  Batch {i//batch_size + 1}: {len(statements)} statements")

            except Exception as e:
                logger.error(f"Error fetching statements for batch {i//batch_size}: {e}")

        if all_statements:
            df = pl.DataFrame(all_statements)
            logger.info(f"✅ Fetched {len(df)} financial statements")
            
            # 正規化処理
            df = self.normalize_statements(df)
            
            return df

        return pl.DataFrame()
    
    async def fetch_statements_by_date(
        self, session: aiohttp.ClientSession, business_days: List[str]
    ) -> pl.DataFrame:
        """
        Fetch financial statements by date (営業日軸で取得 - 空振り削減版)
        
        Args:
            session: aiohttp ClientSession
            business_days: 営業日リスト (YYYY-MM-DD形式)
            
        Returns:
            財務諸表データのDataFrame
        """
        url = f"{self.base_url}/fins/statements"
        headers = {"Authorization": f"Bearer {self.id_token}"}
        
        all_statements = []
        valid_days = 0
        empty_days = 0
        
        logger.info(f"Fetching statements for {len(business_days)} business days...")
        
        # 営業日ごとに取得（開示があった日のみデータが返る）
        for date in business_days:
            date_api = date.replace("-", "")
            params = {"date": date_api}
            pagination_key = None
            statements_for_date = []
            
            while True:
                if pagination_key:
                    params["pagination_key"] = pagination_key
                
                try:
                    async with self.semaphore:
                        async with session.get(url, headers=headers, params=params) as response:
                            if response.status == 404:
                                # この日は開示なし（正常）
                                empty_days += 1
                                break
                            elif response.status != 200:
                                logger.debug(f"Failed to fetch statements for {date}: {response.status}")
                                break
                            
                            data = await response.json()
                            statements = data.get("statements", [])
                            
                            if statements:
                                statements_for_date.extend(statements)
                            
                            pagination_key = data.get("pagination_key")
                            if not pagination_key:
                                break
                
                except Exception as e:
                    logger.debug(f"Error fetching statements for {date}: {e}")
                    break
            
            if statements_for_date:
                all_statements.extend(statements_for_date)
                valid_days += 1
                logger.debug(f"  {date}: {len(statements_for_date)} statements")
        
        logger.info(f"✅ Statements found on {valid_days}/{len(business_days)} days "
                   f"(empty: {empty_days} days)")
        
        if all_statements:
            df = pl.DataFrame(all_statements)
            logger.info(f"✅ Total {len(df)} financial statements fetched")
            
            # 正規化処理
            df = self.normalize_statements(df)
            
            # 重複除去（LocalCode + DisclosureNumberで一意）
            if "LocalCode" in df.columns and "DisclosureNumber" in df.columns:
                df = df.unique(subset=["LocalCode", "DisclosureNumber"])
            
            return df
        
        return pl.DataFrame()
    
    async def fetch_statements_by_code_for_missing(
        self, session: aiohttp.ClientSession, codes: List[str]
    ) -> pl.DataFrame:
        """
        欠損補完用：特定銘柄の財務諸表を銘柄軸で取得
        
        Args:
            session: aiohttp ClientSession
            codes: 補完が必要な銘柄コードリスト
            
        Returns:
            財務諸表データのDataFrame
        """
        if not codes:
            return pl.DataFrame()
        
        logger.info(f"Fetching missing statements for {len(codes)} stocks...")
        return await self.fetch_statements(session, codes)
    
    def normalize_statements(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        財務諸表データの正規化
        - コード形式の統一（4桁ゼロパディング）
        - 数値列の型変換（文字列→float）
        - 日付/時刻の処理
        """
        logger.info("Normalizing financial statements data...")
        
        # コード正規化（LocalCode → Code）
        if "LocalCode" in df.columns:
            df = df.with_columns([
                pl.col("LocalCode").cast(pl.Utf8).str.zfill(4).alias("Code")
            ])
        elif "Code" in df.columns:
            df = df.with_columns([
                pl.col("Code").cast(pl.Utf8).str.zfill(4)
            ])
        
        # 日付/時刻処理
        if "DisclosedDate" in df.columns:
            df = df.with_columns([
                pl.col("DisclosedDate").cast(pl.Date).alias("disclosed_date")
            ])
        
        if "DisclosedTime" in df.columns:
            # タイムスタンプを作成（時刻情報がある場合）
            df = df.with_columns([
                pl.datetime(
                    pl.col("disclosed_date").dt.year(),
                    pl.col("disclosed_date").dt.month(), 
                    pl.col("disclosed_date").dt.day(),
                    pl.col("DisclosedTime").str.slice(0, 2).cast(pl.Int32, strict=False).fill_null(0),
                    pl.col("DisclosedTime").str.slice(3, 2).cast(pl.Int32, strict=False).fill_null(0),
                    0
                ).alias("disclosed_ts")
            ])
        
        # 数値列の型変換（カンマ除去→float変換）
        num_cols = [
            "NetSales", "OperatingProfit", "Profit", "EarningsPerShare",
            "Equity", "TotalAssets", "CashAndEquivalents",
            "ForecastNetSales", "ForecastOperatingProfit", "ForecastProfit", 
            "ForecastEarningsPerShare",
            "ResultDividendPerShare2ndQuarter", 
            "ForecastDividendPerShareFiscalYearEnd", 
            "ForecastDividendPerShareAnnual",
            "NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock",
            "AverageNumberOfShares"
        ]
        
        for col in num_cols:
            if col in df.columns:
                df = df.with_columns([
                    pl.col(col)
                    .cast(pl.Utf8, strict=False)
                    .str.replace_all(",", "")
                    .cast(pl.Float64, strict=False)
                    .alias(col)
                ])
        
        logger.info(f"✅ Normalized {len(df)} statements")
        return df
    
    def save_statements(self, df: pl.DataFrame, output_path: Path = None) -> Path:
        """
        財務諸表データを保存
        
        Args:
            df: 正規化済み財務諸表データ
            output_path: 保存先パス（省略時は自動生成）
        
        Returns:
            保存したファイルのパス
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"event_raw_statements_{timestamp}.parquet"
        
        # Parquet形式で保存
        df.write_parquet(output_path)
        logger.info(f"✅ Saved statements to {output_path}")
        
        return output_path


class JQuantsPipelineV3:
    """最適化されたパイプライン: bulk fetch → filter approach"""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path(
            "/home/ubuntu/gogooku3-standalone/output"
        )
        self.output_dir.mkdir(exist_ok=True)
        self.fetcher = None
        self.calendar_fetcher = None
        self.builder = MLDatasetBuilder(output_dir=self.output_dir)

    async def fetch_jquants_data(
        self, start_date: str = None, end_date: str = None
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """最適化されたデータ取得フロー"""
        # Get credentials
        email = os.getenv("JQUANTS_AUTH_EMAIL", "")
        password = os.getenv("JQUANTS_AUTH_PASSWORD", "")

        if not email or not password:
            logger.error("JQuants credentials not found in environment")
            return pl.DataFrame(), pl.DataFrame()

        # Initialize fetchers
        self.fetcher = JQuantsOptimizedFetcher(email, password)
        self.calendar_fetcher = TradingCalendarFetcher(self.fetcher)

        # Date range
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = "2021-01-01"

        async with aiohttp.ClientSession() as session:
            # Authenticate
            await self.fetcher.authenticate(session)
            
            # Step 1: 営業日カレンダー取得
            logger.info(f"Step 1: 営業日カレンダー取得中 ({start_date} - {end_date})...")
            calendar_data = await self.calendar_fetcher.get_trading_calendar(
                start_date, end_date, session
            )
            business_days = calendar_data.get("business_days", [])
            logger.info(f"✅ 営業日数: {len(business_days)}日")
            
            if not business_days:
                logger.error("営業日データが取得できませんでした")
                return pl.DataFrame(), pl.DataFrame()

            # Step 2: Daily quotes一括取得（全銘柄×全営業日）
            logger.info("Step 2: Daily quotes一括取得中...")
            price_df = await self.fetcher.fetch_daily_quotes_bulk(
                session, business_days
            )
            
            if price_df.is_empty():
                logger.error("価格データが取得できませんでした")
                return pl.DataFrame(), pl.DataFrame()
            
            logger.info(f"  取得データ: {len(price_df)}レコード, {price_df['Code'].n_unique()}銘柄")

            # Step 3: Listed info取得してMarket Codeでフィルタリング
            logger.info("Step 3: Market Codeベースでフィルタリング中...")
            target_codes = await self.fetcher.get_all_listed_info(session, business_days)
            
            # フィルタリング実行
            original_count = len(price_df)
            price_df = price_df.filter(pl.col("Code").is_in(target_codes))
            filtered_count = len(price_df)
            
            logger.info(f"✅ フィルタリング完了: {original_count} → {filtered_count}レコード")
            logger.info(f"  残存銘柄数: {price_df['Code'].n_unique()}")

            # Step 4: Adjustment列の処理
            columns_to_rename = {}
            
            if "AdjustmentClose" in price_df.columns:
                columns_to_rename["AdjustmentClose"] = "Close"
                if "Close" in price_df.columns:
                    price_df = price_df.drop("Close")
            
            if "AdjustmentOpen" in price_df.columns:
                columns_to_rename["AdjustmentOpen"] = "Open"
                if "Open" in price_df.columns:
                    price_df = price_df.drop("Open")
            
            if "AdjustmentHigh" in price_df.columns:
                columns_to_rename["AdjustmentHigh"] = "High"
                if "High" in price_df.columns:
                    price_df = price_df.drop("High")
            
            if "AdjustmentLow" in price_df.columns:
                columns_to_rename["AdjustmentLow"] = "Low"
                if "Low" in price_df.columns:
                    price_df = price_df.drop("Low")
            
            if "AdjustmentVolume" in price_df.columns:
                columns_to_rename["AdjustmentVolume"] = "Volume"
                if "Volume" in price_df.columns:
                    price_df = price_df.drop("Volume")
            
            if columns_to_rename:
                price_df = price_df.rename(columns_to_rename)
                logger.info(f"Adjustment列を使用: {list(columns_to_rename.keys())}")

            # 必要な列を選択
            required_cols = ["Code", "Date", "Open", "High", "Low", "Close", "Volume"]
            available_cols = [col for col in required_cols if col in price_df.columns]
            price_df = price_df.select(available_cols)

            # 数値型に変換
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in price_df.columns:
                    price_df = price_df.with_columns(pl.col(col).cast(pl.Float64))

            # ソート
            price_df = price_df.sort(["Code", "Date"])

            # Step 5: TOPIXデータ取得（ウォームアップ期間を確保して取得）
            logger.info("Step 5: TOPIXデータ取得中 (with warmup)...")
            from datetime import datetime as _dt, timedelta as _td
            try:
                sd_dt = _dt.strptime(start_date, "%Y-%m-%d")
                warmup_start = (sd_dt - _td(days=1200)).strftime("%Y-%m-%d")
            except Exception:
                warmup_start = start_date
            topix_df = await self.fetcher.fetch_topix_data(
                session, warmup_start, end_date
            )

            # Step 6: 売買内訳データ取得
            logger.info("Step 6: 売買内訳データ取得中...")
            trades_df = await self.fetcher.fetch_trades_spec(
                session, start_date, end_date
            )

            # Step 7: 財務諸表データ取得（date軸で効率的に取得）
            logger.info("Step 7: 財務諸表データ取得中 (date-axis approach)...")
            statements_df = await self.fetcher.fetch_statements_by_date(
                session, business_days
            )

            # Step 8: Listed info（区間作成用の素材）
            logger.info("Step 8: 上場銘柄一覧（日次）取得中...")
            listed_info_df = await self.fetcher.fetch_listed_info_range(session, business_days)

            return price_df, topix_df, trades_df, statements_df, listed_info_df

    def process_pipeline(
        self, 
        price_df: pl.DataFrame, 
        topix_df: Optional[pl.DataFrame] = None,
        trades_df: Optional[pl.DataFrame] = None,
        statements_df: Optional[pl.DataFrame] = None,
        listed_info_df: Optional[pl.DataFrame] = None
    ) -> tuple:
        """Process the complete pipeline with technical indicators."""
        logger.info("=" * 60)
        logger.info("Processing ML Dataset Pipeline")
        logger.info("=" * 60)

        # Apply technical features (topix_dfは不要なので削除)
        df = self.builder.create_technical_features(price_df)

        # Add pandas-ta features
        df = self.builder.add_pandas_ta_features(df)
        
        # Add TOPIX market features and cross features
        if topix_df is not None and not topix_df.is_empty():
            logger.info(f"  Adding TOPIX market features: {len(topix_df)} TOPIX records")
            df = self.builder.add_topix_features(df, topix_df)
            
            # Log feature counts
            market_cols = [c for c in df.columns if c.startswith('mkt_')]
            cross_cols = ['beta_60d', 'alpha_1d', 'alpha_5d', 'rel_strength_5d',
                         'trend_align_mkt', 'alpha_vs_regime', 'idio_vol_ratio', 'beta_stability_60d']
            cross_cols = [c for c in cross_cols if c in df.columns]
            logger.info(f"    → Added {len(market_cols)} market features (mkt_*)")
            logger.info(f"    → Added {len(cross_cols)} cross features")
        else:
            logger.info("  No TOPIX data available, skipping market features")
        
        # Add flow features from trades_spec data
        if trades_df is not None and not trades_df.is_empty():
            logger.info(f"  Adding flow features from trades_spec data: {len(trades_df)} records")
            df = self.builder.add_flow_features(df, trades_df, listed_info_df)
            
            # Log flow feature counts
            flow_cols = [
                "flow_foreign_net_ratio", "flow_individual_net_ratio",
                "flow_smart_money_idx", "flow_activity_z",
                "flow_foreign_share_activity", "flow_breadth_pos",
                "flow_smart_money_mom4", "flow_shock_flag",
                "flow_impulse", "days_since_flow"
            ]
            flow_cols = [c for c in flow_cols if c in df.columns]
            logger.info(f"    → Added {len(flow_cols)} flow features")
        else:
            logger.info("  No trades_spec data available, skipping flow features")
            
        # 財務諸表データを結合（もし存在すれば）
        if statements_df is not None and not statements_df.is_empty():
            logger.info(f"  財務諸表データを結合中: {len(statements_df)}レコード")
            df = self.builder.add_statements_features(df, statements_df)

        # 仕様エイリアスを適用（必須列名の保証）
        try:
            from gogooku3.features.alias_registry import apply_spec_aliases
            df = apply_spec_aliases(df)
            logger.info("  Applied spec aliases for required feature names")
        except Exception as e:
            logger.warning(f"  Failed to apply spec aliases: {e}")

        # Create metadata
        metadata = self.builder.create_metadata(df)

        # Display summary
        logger.info("\nDataset Summary:")
        logger.info(f"  Shape: {len(df)} rows × {len(df.columns)} columns")
        logger.info(f"  Features: {metadata['features']['count']}")
        logger.info(f"  Stocks: {metadata['stocks']}")
        logger.info(
            f"  Date range: {metadata['date_range']['start']} to {metadata['date_range']['end']}"
        )

        # Save dataset
        parquet_path, meta_path = self.builder.save_dataset(df, metadata)
        csv_path = None  # CSV保存はオプション

        return df, metadata, (parquet_path, csv_path, meta_path)

    async def run(
        self, use_jquants: bool = True, start_date: str = None, end_date: str = None
    ):
        """Run the optimized pipeline."""
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("OPTIMIZED ML DATASET PIPELINE V3")
        logger.info("Bulk fetch → Market Code filter approach")
        logger.info("=" * 60)

        # Step 1: Get data
        topix_df = None
        trades_df = None
        statements_df = None
        
        if use_jquants:
            logger.info("Fetching data from JQuants API...")
            price_df, topix_df, trades_df, statements_df, listed_info_df = await self.fetch_jquants_data(start_date, end_date)

            if price_df.is_empty():
                logger.error("Failed to fetch data")
                return None, None
        else:
            logger.info("Creating sample data...")
            from data.ml_dataset_builder import create_sample_data
            price_df = create_sample_data(100, 300)
            listed_info_df = pl.DataFrame()

        logger.info(
            f"Data loaded: {len(price_df)} rows, {price_df['Code'].n_unique()} stocks"
        )

        # Step 2: Process pipeline
        logger.info("\nStep 2: Processing ML features...")
        if topix_df is not None and not topix_df.is_empty():
            logger.info(f"  Including TOPIX data: {len(topix_df)} records")
        if trades_df is not None and not trades_df.is_empty():
            logger.info(f"  Including trades_spec data: {len(trades_df)} records")
        if statements_df is not None and not statements_df.is_empty():
            logger.info(f"  Including financial statements: {len(statements_df)} records")
            
        df, metadata, file_paths = self.process_pipeline(price_df, topix_df, trades_df, statements_df, listed_info_df)

        # Step 3: Validate results
        logger.info("\nStep 3: Validating results...")
        self.validate_dataset(df, metadata)

        # Summary
        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Total time: {elapsed:.2f} seconds")
        logger.info(f"Processing speed: {len(df)/elapsed:.0f} rows/second")
        logger.info("\nOutput files:")
        logger.info(f"  Parquet: {file_paths[0]}")
        logger.info(f"  CSV: {file_paths[1]}")
        logger.info(f"  Metadata: {file_paths[2]}")

        return df, metadata

    def validate_dataset(self, df: pl.DataFrame, metadata: dict):
        """Validate the processed dataset."""
        issues = []

        # Check for NaN columns
        nan_cols = []
        for col in df.columns:
            if df[col].is_null().sum() > 0:
                nan_ratio = df[col].is_null().sum() / len(df)
                if nan_ratio > 0.5:
                    issues.append(f"Column {col} has {nan_ratio:.1%} NaN values")
                nan_cols.append(col)

        # Check feature count
        if metadata["features"]["count"] < 50:
            issues.append(
                f"Only {metadata['features']['count']} features (expected 50+)"
            )

        # Check date range
        if metadata["stocks"] < 10:
            issues.append(f"Only {metadata['stocks']} stocks (expected 10+)")

        if issues:
            logger.warning("Validation issues found:")
            for issue in issues:
                logger.warning(f"  ⚠️  {issue}")
        else:
            logger.info("✅ All validation checks passed")

        # Display fixes applied
        logger.info("\nFixes Applied:")
        for fix in metadata.get("fixes_applied", []):
            logger.info(f"  ✓ {fix}")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Optimized ML Dataset Pipeline V3")
    parser.add_argument(
        "--jquants",
        action="store_true",
        help="Use JQuants API (requires credentials in .env)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2021-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD, default: today)",
    )

    args = parser.parse_args()

    # Set end date to today if not specified
    if args.end_date is None:
        args.end_date = datetime.now().strftime("%Y-%m-%d")

    # Check environment
    if args.jquants:
        if not os.getenv("JQUANTS_AUTH_EMAIL"):
            logger.error("JQuants credentials not found in .env file")
            logger.error("Please check /home/ubuntu/gogooku3-standalone/.env")
            logger.info("\nUsing sample data instead...")
            args.jquants = False
        else:
            logger.info(
                f"Using JQuants API with account: {os.getenv('JQUANTS_AUTH_EMAIL')[:10]}..."
            )

    # Run pipeline
    pipeline = JQuantsPipelineV3()
    df, metadata = await pipeline.run(
        use_jquants=args.jquants, start_date=args.start_date, end_date=args.end_date
    )

    return df, metadata


if __name__ == "__main__":
    # Run the async main function
    df, metadata = asyncio.run(main())