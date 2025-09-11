#!/usr/bin/env python3
"""
gogooku3-standalone メイン実行スクリプト
壊れず・強く・速く を実現する金融ML システムの統合実行

利用可能なワークフロー:
1. 安全学習パイプライン（Safe Training Pipeline）
2. MLデータセット構築（ML Dataset Builder）
3. 直接APIデータセット構築（Direct API Dataset Builder）
4. 完全ATFT学習パイプライン（Complete ATFT Training Pipeline）
5. ATFT推論（ATFT Inference）
"""

import sys
import asyncio
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict

# パス設定
sys.path.append(str(Path(__file__).parent))

from scripts.run_safe_training import SafeTrainingPipeline
from scripts.data.ml_dataset_builder import MLDatasetBuilder
from scripts.data.direct_api_dataset_builder import DirectAPIDatasetBuilder
from scripts.integrated_ml_training_pipeline import CompleteATFTTrainingPipeline

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/main.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GoGooKu3MainRunner:
    """gogooku3-standalone メイン実行クラス"""
    
    def __init__(self):
        self.ensure_directories()
    
    def ensure_directories(self):
        """必要なディレクトリを作成"""
        dirs = [
            "logs",
            "data/processed", 
            "output",
            "output/results",
            "output/checkpoints",
            "output/atft_data"
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def run_safe_training(self, mode: str = "full"):
        """安全学習パイプラインの実行"""
        logger.info("🚀 Starting Safe Training Pipeline...")
        
        pipeline = SafeTrainingPipeline()
        
        # パイプライン実行（同期メソッド）
        result = pipeline.run_full_pipeline()

        # エラーチェック
        if 'error' in result:
            logger.error(f"❌ Safe Training Pipeline failed: {result['error']}")
            return False, result
        else:
            logger.info("✅ Safe Training Pipeline completed successfully!")
            logger.info(f"📊 Report saved: {result.get('report_path', 'N/A')}")
            return True, result

    def run_ml_dataset_builder(self):
        """MLデータセット構築の実行"""
        logger.info("🚀 Starting ML Dataset Builder...")
        
        builder = MLDatasetBuilder()
        result = builder.build_enhanced_dataset()
        
        if result:
            logger.info("✅ ML Dataset Builder completed successfully!")
            logger.info(f"📊 Dataset: {len(result['df'])} rows, {result['metadata']['stocks']} stocks")
            return True, result
        else:
            logger.error("❌ ML Dataset Builder failed")
            return False, {}

    def run_expand_dataset(self, max_stocks: int = 500):
        """データセット拡張の実行（銘柄ベース）"""
        logger.info("🚀 Starting Dataset Expansion...")
        logger.info(f"📊 Target: {max_stocks} new stocks")

        try:
            from scripts.data.expand_dataset import JQuantsDatasetExpander

            expander = JQuantsDatasetExpander()
            success = expander.expand_dataset(max_stocks=max_stocks)

            if success:
                logger.info("✅ Dataset expansion completed successfully!")
                return True, {"message": f"Dataset expanded with {max_stocks} new stocks"}
            else:
                logger.error("❌ Dataset expansion failed")
                return False, {"error": "Dataset expansion failed"}

        except ImportError as e:
            logger.error(f"❌ Failed to import expansion module: {e}")
            return False, {"error": f"Import error: {e}"}
        except Exception as e:
            logger.error(f"❌ Dataset expansion failed with exception: {e}")
            return False, {"error": str(e)}

    def run_expand_dataset_by_date(self, date: str, exclude_market_codes: List[str] = None):
        """日付ベース全銘柄データ取得の実行（MarketCodeフィルタリング対応）"""
        logger.info("🚀 Starting Date-based Dataset Expansion...")
        logger.info(f"📅 Target Date: {date}")
        logger.info(f"📊 Exclude MarketCodes: {exclude_market_codes or ['0105', '0109']}")

        try:
            from scripts.data.expand_dataset import JQuantsDatasetExpander

            expander = JQuantsDatasetExpander()
            all_quotes = expander.expand_dataset_by_date(date, exclude_market_codes=exclude_market_codes)

            if all_quotes:
                logger.info("✅ Date-based dataset expansion completed successfully!")
                return True, {"message": f"Retrieved {len(all_quotes)} filtered stocks for {date}"}
            else:
                logger.error("❌ Date-based dataset expansion failed")
                return False, {"error": "Date-based dataset expansion failed"}

        except ImportError as e:
            logger.error(f"❌ Failed to import expansion module: {e}")
            return False, {"error": f"Import error: {e}"}
        except Exception as e:
            logger.error(f"❌ Date-based dataset expansion failed with exception: {e}")
            return False, {"error": str(e)}

    def run_expand_dataset_by_range(self, start_date: str, end_date: str):
        """期間ベース全銘柄データ取得の実行"""
        logger.info("🚀 Starting Range-based Dataset Expansion...")
        logger.info(f"📅 Date Range: {start_date} ~ {end_date}")

        try:
            from scripts.data.expand_dataset import JQuantsDatasetExpander

            expander = JQuantsDatasetExpander()
            all_quotes = expander.expand_dataset_by_date_range(start_date, end_date)

            if all_quotes:
                logger.info("✅ Range-based dataset expansion completed successfully!")
                return True, {"message": f"Retrieved {len(all_quotes)} records for {start_date} ~ {end_date}"}
            else:
                logger.error("❌ Range-based dataset expansion failed")
                return False, {"error": "Range-based dataset expansion failed"}

        except ImportError as e:
            logger.error(f"❌ Failed to import expansion module: {e}")
            return False, {"error": f"Import error: {e}"}
        except Exception as e:
            logger.error(f"❌ Range-based dataset expansion failed with exception: {e}")
            return False, {"error": str(e)}

    def run_expand_historical_all_stocks(self, years: int = 5, max_days: int = None, exclude_market_codes: List[str] = None):
        """取引カレンダーを使った過去N年分の全銘柄データ取得の実行（MarketCodeフィルタリング対応）"""
        logger.info("🚀 Starting Historical All Stocks Dataset Expansion...")
        logger.info(f"📅 Years: {years}")
        logger.info(f"📊 Max Days: {max_days if max_days else 'No limit'}")
        logger.info(f"📊 Exclude MarketCodes: {exclude_market_codes or ['0105', '0109']}")

        try:
            from scripts.data.expand_dataset import JQuantsDatasetExpander

            expander = JQuantsDatasetExpander()
            all_quotes = expander.get_historical_all_stocks(years=years, max_days=max_days, exclude_market_codes=exclude_market_codes)

            if all_quotes:
                # データを保存
                df = pd.DataFrame(all_quotes)

                # 日付カラムの処理
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"])
                    df = df.rename(columns={"Date": "date"})

                # ファイル名に期間を含める
                output_file = expander.data_dir / f"historical_filtered_stocks_{years}years_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                df.to_parquet(output_file, index=False)

                logger.info(f"💾 過去{years}年分のフィルタリング済みデータを保存: {output_file}")
                logger.info(f"📊 データ: {len(df)}行, {df['Code'].nunique()}銘柄")

                logger.info("✅ Historical all stocks dataset expansion completed successfully!")
                return True, {
                    "message": f"Retrieved {len(all_quotes)} filtered records for {years} years",
                    "file": str(output_file),
                    "stocks": df['Code'].nunique(),
                    "total_records": len(df)
                }
            else:
                logger.error("❌ Historical all stocks dataset expansion failed")
                return False, {"error": "Historical all stocks dataset expansion failed"}

        except ImportError as e:
            logger.error(f"❌ Failed to import expansion module: {e}")
            return False, {"error": f"Import error: {e}"}
        except Exception as e:
            logger.error(f"❌ Historical all stocks dataset expansion failed with exception: {e}")
            return False, {"error": str(e)}

    def create_ml_dataset(self, years: int = 5, exclude_market_codes: List[str] = None, use_existing_data: bool = True):
        """過去取れる全データを取得してML用データセットを作成"""
        logger.info("🚀 Creating ML Dataset from Historical Data...")
        logger.info(f"📅 Years: {years}")
        logger.info(f"📊 Exclude MarketCodes: {exclude_market_codes or ['0105', '0109']}")
        logger.info(f"📊 Use Existing Data: {use_existing_data}")

        try:
            all_quotes = []

            if use_existing_data:
                # Step 1: 既存データの活用
                logger.info("📊 Step 1: 既存データの読み込みを開始")
                existing_quotes = self._load_existing_data()
                if existing_quotes:
                    all_quotes.extend(existing_quotes)
                    logger.info(f"✅ 既存データ読み込み完了: {len(existing_quotes)}件")

            # Step 2: 新規データの取得（APIが利用可能な場合）
            if not use_existing_data or len(all_quotes) < 10000:  # 既存データが少ない場合はAPI取得
                logger.info("📊 Step 2: 新規データの取得を開始")
                try:
                    from scripts.data.expand_dataset import JQuantsDatasetExpander
                    expander = JQuantsDatasetExpander()

                    new_quotes = expander.get_historical_all_stocks(
                        years=min(years, 2),  # API制限を考慮して最大2年に制限
                        max_days=50,  # 最大50日間に制限
                        exclude_market_codes=exclude_market_codes
                    )

                    if new_quotes:
                        all_quotes.extend(new_quotes)
                        logger.info(f"✅ 新規データ取得完了: {len(new_quotes)}件")
                except Exception as e:
                    logger.warning(f"⚠️ 新規データ取得スキップ: {e}")

            if not all_quotes:
                logger.error("❌ 利用可能なデータがありません")
                return False, {"error": "No data available"}

            # Step 3: ML用データセット作成
            logger.info("🔧 Step 3: ML用データセット作成を開始")
            ml_dataset_path = self._create_ml_dataset_from_quotes(all_quotes)

            if not ml_dataset_path:
                logger.error("❌ MLデータセット作成に失敗しました")
                return False, {"error": "Failed to create ML dataset"}

            logger.info("✅ ML Dataset Creation completed successfully!")

            return True, {
                "message": f"Created ML dataset from {len(all_quotes)} historical records",
                "raw_data_records": len(all_quotes),
                "ml_dataset_path": ml_dataset_path,
                "years": years,
                "excluded_markets": exclude_market_codes or ['0105', '0109'],
                "used_existing_data": use_existing_data
            }

        except ImportError as e:
            logger.error(f"❌ Failed to import expansion module: {e}")
            return False, {"error": f"Import error: {e}"}
        except Exception as e:
            logger.error(f"❌ ML Dataset creation failed with exception: {e}")
            return False, {"error": str(e)}

    def _create_ml_dataset_from_quotes(self, all_quotes: List[Dict]) -> str:
        """株価データからML用データセットを作成"""
        logger.info("🔧 MLデータセット作成処理を開始")

        try:
            import pandas as pd
            import numpy as np
            from pathlib import Path
            from datetime import datetime

            # データをDataFrameに変換
            df = pd.DataFrame(all_quotes)

            # 日付カラムの処理
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.rename(columns={"Date": "date"})

            logger.info(f"📊 元データ: {len(df)}行")

            # 基本的なデータクレンジング
            df = self._clean_stock_data(df)

            # 特徴量エンジニアリング
            df = self._create_ml_features(df)

            # MLデータセットとして保存
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = Path("data/processed") / f"ml_dataset_{timestamp}.parquet"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            df.to_parquet(output_path, index=False)

            logger.info(f"💾 MLデータセット保存: {output_path}")
            logger.info(f"📊 MLデータセット: {len(df)}行 × {len(df.columns)}列")

            return str(output_path)

        except Exception as e:
            logger.error(f"❌ MLデータセット作成エラー: {e}")
            return None

    def _load_existing_data(self) -> List[Dict]:
        """既存のデータを読み込んで統合"""
        logger.info("📁 既存データの読み込みを開始")

        try:
            import polars as pl
            import pandas as pd
            from pathlib import Path

            data_dir = Path("data/raw/large_scale")
            all_quotes = []

            # 利用可能なparquetファイルを検索
            parquet_files = list(data_dir.glob("*.parquet"))

            for file_path in parquet_files:
                try:
                    logger.info(f"📄 {file_path.name} を読み込み中...")

                    # Polarsでデータを読み込み
                    df = pl.read_parquet(file_path)

                    # pandas DataFrameに変換して辞書のリストに変換
                    pandas_df = df.to_pandas()

                    # 日付カラムの処理
                    if "Date" in pandas_df.columns:
                        pandas_df["Date"] = pd.to_datetime(pandas_df["Date"])
                        pandas_df = pandas_df.rename(columns={"Date": "date"})

                    # 銘柄コードカラムの統一（Code or code → Code）
                    if "code" in pandas_df.columns and "Code" not in pandas_df.columns:
                        pandas_df = pandas_df.rename(columns={"code": "Code"})

                    # 価格カラムの統一（Close/c_use → Close）
                    if "c_use" in pandas_df.columns and "Close" not in pandas_df.columns:
                        pandas_df = pandas_df.rename(columns={"c_use": "Close"})
                    elif "close" in pandas_df.columns and "Close" not in pandas_df.columns:
                        pandas_df = pandas_df.rename(columns={"close": "Close"})

                    # 必要なカラムがある場合のみ処理
                    if "Code" in pandas_df.columns and "date" in pandas_df.columns and "Close" in pandas_df.columns:
                        # 必要なカラムのみ抽出
                        pandas_df = pandas_df[["Code", "date", "Close"]]
                        quotes = pandas_df.to_dict('records')
                        all_quotes.extend(quotes)
                        logger.info(f"   ✅ {len(quotes)}件読み込み完了")
                    else:
                        logger.warning(f"   ⚠️ 必要なカラムが見つからない: Code={('Code' in pandas_df.columns)}, date={('date' in pandas_df.columns)}, Close={('Close' in pandas_df.columns)}")
                        continue

                    logger.info(f"   📊 現在の累計: {len(all_quotes)}件")

                except Exception as e:
                    logger.warning(f"   ⚠️ {file_path.name} の読み込みスキップ: {e}")
                    continue

            # 重複データの除去（同じ銘柄・日付・ファイルソースのデータを統合）
            if all_quotes:
                logger.info("🔄 重複データ処理を開始")
                temp_df = pd.DataFrame(all_quotes)

                # まず、各ファイルのデータを確認
                logger.info(f"📊 重複処理前: {len(temp_df)}行")

                # Codeとdateで重複を除去（より詳細な条件で）
                if 'Code' in temp_df.columns and 'date' in temp_df.columns:
                    before_count = len(temp_df)
                    # 同じ銘柄・同じ日付のデータを1つにまとめる
                    temp_df = temp_df.sort_values(['Code', 'date'])
                    temp_df = temp_df.drop_duplicates(subset=['Code', 'date'], keep='first')
                    after_count = len(temp_df)

                    duplicates_removed = before_count - after_count
                    logger.info(f"🧹 重複データ除去: {duplicates_removed}件")

                    # 銘柄数の変化を確認
                    unique_stocks_before = before_count  # これは正確ではないが参考値
                    unique_stocks_after = temp_df['Code'].nunique()
                    logger.info(f"📊 銘柄数: {unique_stocks_after}銘柄")
                else:
                    logger.warning("⚠️ Codeまたはdateカラムが見つからないため重複除去をスキップ")

                # 辞書リストに戻す
                all_quotes = temp_df.to_dict('records')

            logger.info(f"📊 既存データ統合完了: {len(all_quotes)}件")
            return all_quotes

        except Exception as e:
            logger.error(f"❌ 既存データ読み込みエラー: {e}")
            return []

    def _clean_stock_data(self, df):
        """株価データのクレンジング（緩和版）"""
        logger.info("🧹 データクレンジングを開始")
        original_count = len(df)

        # 欠損値処理（必須項目のみ）
        df = df.dropna(subset=['Close'])  # 終値は必須

        # 異常値除去（価格が0以下は除外）
        df = df[df['Close'] > 0]

        # OHLCデータの整合性チェック（ある場合のみ）
        if all(col in df.columns for col in ['Open', 'High', 'Low']):
            df = df[(df['Open'] > 0) & (df['High'] > 0) & (df['Low'] > 0)]

        # 出来高のチェック（ある場合のみ）
        if 'Volume' in df.columns:
            # 出来高が0のデータも残す（出来高0でも価格データは有効な場合がある）
            pass

        # 日付でソート
        df = df.sort_values(['Code', 'date']).reset_index(drop=True)

        cleaned_count = len(df)
        removed_count = original_count - cleaned_count

        logger.info(f"✅ クレンジング完了: {cleaned_count}行 (除去: {removed_count}行)")

        return df

    def _create_ml_features(self, df):
        """ML用特徴量の作成"""
        logger.info("🔧 特徴量エンジニアリングを開始")

        try:
            import pandas as pd
        except ImportError:
            logger.error("❌ pandasがimportできません")
            return df

        # 銘柄ごとに特徴量を作成
        df_list = []

        for code in df['Code'].unique():
            stock_df = df[df['Code'] == code].copy()

            if len(stock_df) < 2:  # 最低2日以上のデータが必要（緩和）
                continue

            # 価格変動率
            stock_df['price_change'] = stock_df['Close'].pct_change()

            # データ量に応じた特徴量作成
            data_length = len(stock_df)

            # 移動平均（データ量に応じてウィンドウサイズ調整）
            if data_length >= 5:
                stock_df['ma5'] = stock_df['Close'].rolling(window=min(5, data_length)).mean()
            if data_length >= 10:
                stock_df['ma10'] = stock_df['Close'].rolling(window=min(10, data_length)).mean()
            if data_length >= 20:
                stock_df['ma20'] = stock_df['Close'].rolling(window=min(20, data_length)).mean()
            if data_length >= 60:
                stock_df['ma60'] = stock_df['Close'].rolling(window=min(60, data_length)).mean()

            # ボラティリティ（データ量に応じて調整）
            if data_length >= 5:
                vol_window = min(20, data_length)
                stock_df['volatility'] = stock_df['price_change'].rolling(window=vol_window).std()

            # RSI（データ量に応じて調整）
            if data_length >= 14:
                rsi_window = min(14, data_length)
                delta = stock_df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
                rs = gain / loss
                stock_df['rsi'] = 100 - (100 / (1 + rs))

            # MACD（データ量に応じて調整）
            if data_length >= 26:
                exp1 = stock_df['Close'].ewm(span=min(12, data_length//2), adjust=False).mean()
                exp2 = stock_df['Close'].ewm(span=min(26, data_length), adjust=False).mean()
                stock_df['macd'] = exp1 - exp2
                if data_length >= 35:  # シグナルライン用
                    stock_df['macd_signal'] = stock_df['macd'].ewm(span=min(9, data_length//4), adjust=False).mean()

            # 目的変数: 翌日の価格変動（データが2日以上ある場合のみ）
            if data_length >= 2:
                stock_df['target'] = stock_df['Close'].shift(-1) / stock_df['Close'] - 1

            df_list.append(stock_df)

        if df_list:
            result_df = pd.concat(df_list, ignore_index=True)

            # 利用可能な特徴量を動的に取得
            potential_features = ['price_change', 'ma5', 'ma10', 'ma20', 'ma60',
                                'volatility', 'rsi', 'macd', 'macd_signal']
            available_features = [col for col in potential_features if col in result_df.columns]

            # 特徴量の欠損値を埋める（前値補完）
            if available_features:
                result_df[available_features] = result_df[available_features].fillna(method='ffill')

            # targetがNaNの行を除外
            if 'target' in result_df.columns:
                result_df = result_df.dropna(subset=['target'])

            logger.info(f"✅ 特徴量作成完了: {len(result_df)}行")
            logger.info(f"📊 利用可能特徴量: {available_features}")

            return result_df
        else:
            logger.warning("⚠️ 特徴量作成対象の銘柄がありません")
            return df

    async def run_direct_api_dataset_builder(self):
        """直接APIデータセット構築の実行"""
        logger.info("🚀 Starting Direct API Dataset Builder...")
        
        builder = DirectAPIDatasetBuilder()
        result = await builder.build_direct_api_dataset()
        
        if result:
            logger.info("✅ Direct API Dataset Builder completed successfully!")
            logger.info(f"📊 Dataset: {len(result['df'])} rows, {result['df']['Code'].n_unique()} stocks")
            return True, result
        else:
            logger.error("❌ Direct API Dataset Builder failed")
            return False, {}

    async def run_complete_atft_training(self):
        """完全ATFT学習パイプラインの実行"""
        logger.info("🚀 Starting Complete ATFT Training Pipeline...")
        
        pipeline = CompleteATFTTrainingPipeline()
        success, result = await pipeline.run_complete_training_pipeline()
        
        if success:
            logger.info("✅ Complete ATFT Training Pipeline completed successfully!")
            logger.info(f"🎯 Target Sharpe: 0.849")
            return True, result
        else:
            logger.error(f"❌ Complete ATFT Training Pipeline failed: {result.get('error')}")
            return False, result

    def run_workflow(self, workflow: str, mode: str = "full", stocks: int = 500, date: str = None, start_date: str = None, end_date: str = None, years: int = 5, max_days: int = None, exclude_market_codes: List[str] = None, use_existing_data: bool = True):
        """指定されたワークフローを実行"""
        logger.info(f"🎬 Starting workflow: {workflow}")

        try:
            if workflow == "safe-training":
                return self.run_safe_training(mode)
            elif workflow == "ml-dataset":
                return self.run_ml_dataset_builder()
            elif workflow == "direct-api-dataset":
                return asyncio.run(self.run_direct_api_dataset_builder())
            elif workflow == "complete-atft":
                return asyncio.run(self.run_complete_atft_training())
            elif workflow == "expand-dataset":
                return self.run_expand_dataset(max_stocks=stocks)
            elif workflow == "expand-dataset-by-date":
                if date is None:
                    logger.error("❌ date parameter is required for expand-dataset-by-date")
                    return False, {"error": "date parameter is required"}
                return self.run_expand_dataset_by_date(date, exclude_market_codes=exclude_market_codes)
            elif workflow == "expand-dataset-by-range":
                if start_date is None or end_date is None:
                    logger.error("❌ start_date and end_date parameters are required for expand-dataset-by-range")
                    return False, {"error": "start_date and end_date parameters are required"}
                return self.run_expand_dataset_by_range(start_date, end_date)
            elif workflow == "expand-historical-all-stocks":
                return self.run_expand_historical_all_stocks(years=years, max_days=max_days, exclude_market_codes=exclude_market_codes)
            elif workflow == "create-ml-dataset":
                return self.create_ml_dataset(years=years, exclude_market_codes=exclude_market_codes, use_existing_data=use_existing_data)
            else:
                logger.error(f"❌ Unknown workflow: {workflow}")
                return False, {"error": f"Unknown workflow: {workflow}"}

        except Exception as e:
            logger.error(f"❌ Workflow {workflow} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return False, {"error": str(e)}


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(
        description="gogooku3-standalone - 壊れず・強く・速く を実現する金融ML システム",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
利用可能なワークフロー:
  safe-training               安全学習パイプライン（推奨）
  ml-dataset                  MLデータセット構築
  direct-api-dataset          直接APIデータセット構築
  complete-atft               完全ATFT学習パイプライン
  expand-dataset              J-Quants APIからデータセットを拡張（銘柄ベース）
  expand-dataset-by-date      J-Quants APIから日付ベース全銘柄データを取得
  expand-dataset-by-range     J-Quants APIから期間ベース全銘柄データを取得
  expand-historical-all-stocks J-Quants取引カレンダーAPIを使って過去N年分の全銘柄データを取得
  create-ml-dataset           過去取れる全データを取得してML用データセットを作成

使用例:
  python main.py safe-training --mode full
  python main.py ml-dataset
  python main.py direct-api-dataset
  python main.py complete-atft
  python main.py expand-dataset --stocks 1000
  python main.py expand-dataset-by-date --date 2024-08-29
  python main.py expand-dataset-by-range --start-date 2024-08-01 --end-date 2024-08-29
  python main.py expand-historical-all-stocks --years 5 --max-days 100
  python main.py expand-historical-all-stocks --years 3 --exclude-market-codes 0105 0109 0110
  python main.py create-ml-dataset --years 5 --exclude-market-codes 0105 0109
  python main.py create-ml-dataset --years 2 --use-existing-data --exclude-market-codes 0105 0109
  python main.py create-ml-dataset --years 1 --no-existing-data
        """
    )
    
    parser.add_argument(
        "workflow",
        choices=["safe-training", "ml-dataset", "direct-api-dataset", "complete-atft", "expand-dataset", "expand-dataset-by-date", "expand-dataset-by-range", "expand-historical-all-stocks", "create-ml-dataset"],
        help="実行するワークフロー"
    )

    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="full",
        help="実行モード（safe-trainingのみ：quick=1エポック, full=完全学習）"
    )

    parser.add_argument(
        "--stocks",
        type=int,
        default=500,
        help="expand-dataset時の取得銘柄数（デフォルト: 500）"
    )

    parser.add_argument(
        "--date",
        type=str,
        help="expand-dataset-by-date時の対象日付（YYYY-MM-DD形式）"
    )

    parser.add_argument(
        "--start-date",
        type=str,
        help="expand-dataset-by-range時の開始日付（YYYY-MM-DD形式）"
    )

    parser.add_argument(
        "--end-date",
        type=str,
        help="expand-dataset-by-range時の終了日付（YYYY-MM-DD形式）"
    )

    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="expand-historical-all-stocks時の過去年数（デフォルト: 5）"
    )

    parser.add_argument(
        "--max-days",
        type=int,
        help="expand-historical-all-stocks時の最大取得日数（制限なしの場合は指定なし）"
    )

    parser.add_argument(
        "--exclude-market-codes",
        type=str,
        nargs='*',
        default=['0105', '0109'],
        help="除外するMarketCodeのリスト（デフォルト: 0105 0109）"
    )

    parser.add_argument(
        "--use-existing-data",
        action='store_true',
        default=True,
        help="既存のデータを活用する（デフォルト: True）"
    )

    parser.add_argument(
        "--no-existing-data",
        action='store_false',
        dest='use_existing_data',
        help="既存データを無視して新規取得のみを行う"
    )
    
    args = parser.parse_args()
    
    # バナー表示
    print("=" * 80)
    print("🚀 gogooku3-standalone - 壊れず・強く・速く")
    print("📈 金融ML システム統合実行環境")
    print("=" * 80)
    print(f"Workflow: {args.workflow}")
    print(f"Mode: {args.mode}")
    print("=" * 80)
    
    # メインランナー実行
    runner = GoGooKu3MainRunner()
    
    # 同期実行
    success, result = runner.run_workflow(
        args.workflow,
        args.mode,
        args.stocks,
        args.date,
        args.start_date,
        args.end_date,
        args.years,
        args.max_days,
        args.exclude_market_codes,
        args.use_existing_data
    )
    
    # 結果表示
    print("\n" + "=" * 80)
    if success:
        print("🎉 ワークフロー実行成功!")
        print("=" * 80)
        
        # 結果サマリー
        if args.workflow == "safe-training":
            summary = result.get("summary", {})
            print(f"📊 学習結果:")
            print(f"   - エポック数: {summary.get('epochs', 'N/A')}")
            print(f"   - 最終損失: {summary.get('final_loss', 'N/A')}")
            print(f"   - 実行時間: {summary.get('elapsed_time', 'N/A'):.2f}秒")
        elif args.workflow in ["ml-dataset", "direct-api-dataset"]:
            if "df" in result:
                print(f"📊 データセット構築結果:")
                print(f"   - 行数: {len(result['df']):,}")
                print(f"   - 銘柄数: {result['df']['Code'].n_unique()}")
                if "metadata" in result:
                    print(f"   - 特徴量数: {result['metadata']['features']['count']}")
        elif args.workflow == "complete-atft":
            validation_info = result.get("validation_info", {})
            print(f"🎯 ATFT学習結果:")
            print(f"   - 目標Sharpe: 0.849")
            if validation_info.get('sharpe_ratio') is not None:
                print(f"   - 達成Sharpe: {validation_info.get('sharpe_ratio')}")
            print(f"   - パラメータ数: {validation_info.get('param_count', 'N/A'):,}")
        elif args.workflow == "create-ml-dataset":
            print(f"🤖 MLデータセット作成結果:")
            print(f"   - 元データ件数: {result.get('raw_data_records', 'N/A'):,}")
            print(f"   - 対象年数: {result.get('years', 'N/A')}年")
            print(f"   - 除外市場: {', '.join(result.get('excluded_markets', []))}")
            print(f"   - 既存データ活用: {result.get('used_existing_data', 'N/A')}")
            if "ml_dataset_path" in result:
                print(f"   - 保存先: {result['ml_dataset_path']}")

        print("=" * 80)
        print("✅ 実行完了")
        
    else:
        print("❌ ワークフロー実行失敗")
        print(f"エラー: {result.get('error', 'Unknown error')}")
        print("=" * 80)
        
        sys.exit(1)


if __name__ == "__main__":
    main()
