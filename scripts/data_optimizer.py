#!/usr/bin/env python3
"""
Data Management Best Practices for gogooku3
データ管理のベストプラクティス実装
"""

import sys
import logging
import hashlib
from pathlib import Path
from typing import Dict, List
from functools import lru_cache
import polars as pl
from datetime import datetime, timedelta
import json

# パスを追加
sys.path.append(str(Path(__file__).parent.parent))

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataOptimizer:
    """データ管理のベストプラクティス実装クラス"""

    def __init__(self, base_dir: str = "output"):
        self.base_dir = Path(base_dir)
        self.cache_dir = self.base_dir / "cache"
        self.compressed_dir = self.base_dir / "compressed"
        self.metadata_dir = self.base_dir / "metadata"

        # ディレクトリ作成
        for dir_path in [self.cache_dir, self.compressed_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # 最適化設定
        self.parquet_settings = {
            "compression": "gzip",  # snappy → gzip（圧縮率向上）
            "row_group_size": 100000,
            "use_dictionary": True,
            "use_byte_stream_split": True,
        }

        # キャッシュ設定
        self.cache_settings = {"max_size": 100, "ttl_hours": 24}

    def optimize_parquet_files(self, input_dir: str, output_dir: str = None) -> Dict:
        """parquetファイルの最適化"""
        if output_dir is None:
            output_dir = self.compressed_dir

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        optimization_results = {
            "original_size": 0,
            "optimized_size": 0,
            "compression_ratio": 0,
            "files_processed": 0,
        }

        try:
            # 全parquetファイルを処理
            parquet_files = list(input_path.rglob("*.parquet"))

            for file_path in parquet_files:
                relative_path = file_path.relative_to(input_path)
                output_file = output_path / relative_path
                output_file.parent.mkdir(parents=True, exist_ok=True)

                # ファイル読み込み
                df = pl.read_parquet(file_path)
                original_size = file_path.stat().st_size

                # 最適化された設定で保存
                df.write_parquet(
                    output_file, compression=self.parquet_settings["compression"]
                )

                optimized_size = output_file.stat().st_size
                compression_ratio = (original_size - optimized_size) / original_size

                optimization_results["original_size"] += original_size
                optimization_results["optimized_size"] += optimized_size
                optimization_results["files_processed"] += 1

                logger.info(
                    f"Optimized: {file_path.name} "
                    f"({original_size/1024:.1f}KB → {optimized_size/1024:.1f}KB, "
                    f"compression: {compression_ratio:.1%})"
                )

            optimization_results["compression_ratio"] = (
                optimization_results["original_size"]
                - optimization_results["optimized_size"]
            ) / optimization_results["original_size"]

            logger.info(
                f"Optimization completed: {optimization_results['files_processed']} files, "
                f"compression ratio: {optimization_results['compression_ratio']:.1%}"
            )

            return optimization_results

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise

    @lru_cache(maxsize=100)
    def load_stock_data(self, stock_code: str, date_range: str = None) -> pl.DataFrame:
        """インテリジェントキャッシュ付きデータローダー"""
        cache_key = f"{stock_code}_{date_range or 'all'}"

        # キャッシュファイルパス
        cache_file = (
            self.cache_dir / f"{hashlib.md5(cache_key.encode()).hexdigest()}.parquet"
        )

        # キャッシュが存在し、有効期限内かチェック
        if cache_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(
                cache_file.stat().st_mtime
            )
            if cache_age < timedelta(hours=self.cache_settings["ttl_hours"]):
                logger.info(f"Loading from cache: {stock_code}")
                return pl.read_parquet(cache_file)

        # データ読み込み
        data_file = self.base_dir / "atft_data" / "train" / f"{stock_code}.parquet"
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        df = pl.read_parquet(data_file)

        # 日付範囲フィルタリング
        if date_range:
            start_date, end_date = date_range.split("_")
            df = df.filter(
                (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
            )

        # キャッシュに保存
        df.write_parquet(cache_file)
        logger.info(f"Cached data: {stock_code}")

        return df

    def batch_load_data(
        self, stock_codes: List[str], batch_size: int = 10
    ) -> List[pl.DataFrame]:
        """バッチローディング"""
        results = []

        for i in range(0, len(stock_codes), batch_size):
            batch = stock_codes[i : i + batch_size]
            logger.info(f"Loading batch {i//batch_size + 1}: {len(batch)} stocks")

            batch_data = []
            for stock_code in batch:
                try:
                    df = self.load_stock_data(stock_code)
                    batch_data.append(df)
                except Exception as e:
                    logger.warning(f"Failed to load {stock_code}: {e}")
                    continue

            results.extend(batch_data)

        return results

    def create_data_metadata(self, data_dir: str) -> Dict:
        """データメタデータの作成"""
        data_path = Path(data_dir)
        metadata = {
            "created_at": datetime.now().isoformat(),
            "data_structure": {},
            "file_stats": {},
            "quality_metrics": {},
        }

        # ディレクトリ構造の分析
        for subdir in ["train", "val", "test"]:
            subdir_path = data_path / subdir
            if subdir_path.exists():
                files = list(subdir_path.glob("*.parquet"))
                metadata["data_structure"][subdir] = {
                    "file_count": len(files),
                    "total_size": sum(f.stat().st_size for f in files),
                    "file_names": [f.name for f in files],
                }

        # サンプルファイルの品質チェック
        sample_file = next(data_path.rglob("*.parquet"), None)
        if sample_file:
            df = pl.read_parquet(sample_file)
            metadata["quality_metrics"] = {
                "columns": df.columns,
                "shape": df.shape,
                "null_counts": df.null_count().to_dict(),
                "data_types": df.schema,
            }

        # メタデータ保存
        metadata_file = (
            self.metadata_dir
            / f"data_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Metadata created: {metadata_file}")
        return metadata

    def cleanup_cache(self, max_age_hours: int = 24) -> int:
        """古いキャッシュファイルの削除"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        deleted_count = 0

        for cache_file in self.cache_dir.glob("*.parquet"):
            if datetime.fromtimestamp(cache_file.stat().st_mtime) < cutoff_time:
                cache_file.unlink()
                deleted_count += 1

        logger.info(f"Cleaned up {deleted_count} old cache files")
        return deleted_count


class DataValidator:
    """データ品質検証クラス"""

    def __init__(self):
        self.validation_rules = {
            "required_columns": ["code", "date", "close", "volume"],
            "date_format": "%Y-%m-%d",
            "min_records_per_stock": 20,
            "max_null_ratio": 0.1,
        }

    def validate_dataset(self, df: pl.DataFrame) -> Dict:
        """データセットの品質検証"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "metrics": {},
        }

        # 必須列のチェック
        missing_columns = set(self.validation_rules["required_columns"]) - set(
            df.columns
        )
        if missing_columns:
            validation_results["valid"] = False
            validation_results["errors"].append(
                f"Missing required columns: {missing_columns}"
            )

        # データ型チェック
        if "date" in df.columns:
            try:
                df.select(
                    pl.col("date").str.strptime(
                        pl.Datetime, self.validation_rules["date_format"]
                    )
                )
            except Exception as e:
                validation_results["warnings"].append(f"Date format issues: {e}")

        # 欠損値チェック
        null_counts = df.null_count()
        for col, null_count in null_counts.items():
            null_ratio = null_count / len(df)
            if null_ratio > self.validation_rules["max_null_ratio"]:
                validation_results["warnings"].append(
                    f"High null ratio in {col}: {null_ratio:.1%}"
                )

        # 銘柄ごとのレコード数チェック
        if "code" in df.columns:
            stock_counts = df.groupby("code").count()
            low_record_stocks = stock_counts.filter(
                pl.col("count") < self.validation_rules["min_records_per_stock"]
            )
            if len(low_record_stocks) > 0:
                validation_results["warnings"].append(
                    f"Stocks with few records: {len(low_record_stocks)}"
                )

        # メトリクス計算
        validation_results["metrics"] = {
            "total_records": len(df),
            "total_stocks": df["code"].n_unique() if "code" in df.columns else 0,
            "date_range": {
                "start": df["date"].min() if "date" in df.columns else None,
                "end": df["date"].max() if "date" in df.columns else None,
            },
            "null_ratios": null_counts.to_dict(),
        }

        return validation_results


def main():
    """テスト実行"""
    optimizer = DataOptimizer()
    validator = DataValidator()  # データ検証用

    # データ最適化
    print("🔧 Optimizing parquet files...")
    results = optimizer.optimize_parquet_files("output/atft_data")
    print(f"Optimization results: {results}")

    # メタデータ作成
    print("📊 Creating metadata...")
    metadata = optimizer.create_data_metadata("output/atft_data")
    print(f"Metadata created: {metadata['created_at']}")

    # データ品質検証
    print("🔍 Validating data quality...")
    sample_file = next(Path("output/atft_data").rglob("*.parquet"), None)
    if sample_file:
        df = pl.read_parquet(sample_file)
        validation_results = validator.validate_dataset(df)
        print(f"Data validation results: {validation_results}")

    # キャッシュクリーンアップ
    print("🧹 Cleaning up cache...")
    deleted_count = optimizer.cleanup_cache()
    print(f"Deleted {deleted_count} old cache files")


if __name__ == "__main__":
    main()
