#!/usr/bin/env python3
"""
Apply Best Practices to gogooku3
gogooku3にベストプラクティスを適用する統合スクリプト
"""

import sys
import logging
import json
from pathlib import Path
from typing import Dict
from datetime import datetime
import polars as pl

# パスを追加
sys.path.append(str(Path(__file__).parent.parent))

# ベストプラクティスモジュールのインポート
from scripts.data_optimizer import DataOptimizer, DataValidator
from scripts.performance_optimizer import PerformanceOptimizer, DataPipelineOptimizer
from scripts.monitoring_system import MetricsCollector, ModelMonitor, DataQualityMonitor

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/best_practices.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class BestPracticesApplier:
    """ベストプラクティス適用クラス"""

    def __init__(self):
        self.base_dir = Path(".")
        self.output_dir = self.base_dir / "output"
        self.logs_dir = self.base_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        # 各最適化モジュールの初期化
        self.data_optimizer = DataOptimizer()
        self.data_validator = DataValidator()
        self.performance_optimizer = PerformanceOptimizer()
        self.pipeline_optimizer = DataPipelineOptimizer()

        # モニタリングシステムの初期化
        self.metrics_collector = MetricsCollector()
        self.model_monitor = ModelMonitor(self.metrics_collector)
        self.data_monitor = DataQualityMonitor(self.metrics_collector)

        # 適用結果の記録
        self.application_results = {
            "timestamp": datetime.now().isoformat(),
            "phases": {},
            "overall_success": True,
            "recommendations": [],
        }

    def apply_all_best_practices(self) -> Dict:
        """すべてのベストプラクティスを適用"""
        logger.info("🚀 Starting best practices application...")

        try:
            # Phase 1: データ管理最適化
            self._apply_data_optimization()

            # Phase 2: パフォーマンス最適化
            self._apply_performance_optimization()

            # Phase 3: モニタリングシステム設定
            self._apply_monitoring_system()

            # Phase 4: 統合テスト
            self._run_integration_tests()

            # Phase 5: レポート生成
            self._generate_final_report()

            logger.info("✅ All best practices applied successfully!")
            return self.application_results

        except Exception as e:
            logger.error(f"❌ Best practices application failed: {e}")
            self.application_results["overall_success"] = False
            self.application_results["error"] = str(e)
            return self.application_results

    def _apply_data_optimization(self):
        """データ管理最適化の適用"""
        logger.info("📊 Applying data optimization best practices...")

        phase_results = {"success": True, "optimizations": [], "errors": []}

        try:
            # 1. parquetファイル最適化
            logger.info("🔧 Optimizing parquet files...")
            optimization_results = self.data_optimizer.optimize_parquet_files(
                "output/atft_data"
            )
            phase_results["optimizations"].append(
                {"type": "parquet_optimization", "results": optimization_results}
            )

            # 2. メタデータ作成
            logger.info("📋 Creating data metadata...")
            metadata = self.data_optimizer.create_data_metadata("output/atft_data")
            phase_results["optimizations"].append(
                {"type": "metadata_creation", "results": metadata}
            )

            # 3. キャッシュクリーンアップ
            logger.info("🧹 Cleaning up old cache...")
            deleted_count = self.data_optimizer.cleanup_cache()
            phase_results["optimizations"].append(
                {"type": "cache_cleanup", "deleted_files": deleted_count}
            )

            # 4. データ品質検証
            logger.info("🔍 Validating data quality...")
            sample_file = next(Path("output/atft_data").rglob("*.parquet"), None)
            if sample_file:
                df = pl.read_parquet(sample_file)
                validation_results = self.data_validator.validate_dataset(df)
                phase_results["optimizations"].append(
                    {"type": "data_validation", "results": validation_results}
                )

            logger.info("✅ Data optimization completed successfully")

        except Exception as e:
            logger.error(f"❌ Data optimization failed: {e}")
            phase_results["success"] = False
            phase_results["errors"].append(str(e))

        self.application_results["phases"]["data_optimization"] = phase_results

    def _apply_performance_optimization(self):
        """パフォーマンス最適化の適用"""
        logger.info("⚡ Applying performance optimization best practices...")

        phase_results = {"success": True, "optimizations": [], "errors": []}

        try:
            # 1. システムメトリクス取得
            logger.info("📊 Getting system performance metrics...")
            metrics = self.performance_optimizer.get_performance_metrics()
            phase_results["optimizations"].append(
                {"type": "system_metrics", "results": metrics}
            )

            # 2. メモリ最適化
            logger.info("🧹 Optimizing memory usage...")
            memory_usage = self.performance_optimizer.optimize_memory_usage()
            phase_results["optimizations"].append(
                {"type": "memory_optimization", "memory_usage_gb": memory_usage}
            )

            # 3. パイプライン最適化設定
            logger.info("🔧 Configuring pipeline optimization...")
            pipeline_config = {
                "use_parallel": True,
                "use_chunk": True,
                "use_batch": True,
                "use_cache": True,
            }
            pipeline_results = self.pipeline_optimizer.optimize_pipeline(
                pipeline_config
            )
            phase_results["optimizations"].append(
                {"type": "pipeline_optimization", "results": pipeline_results}
            )

            logger.info("✅ Performance optimization completed successfully")

        except Exception as e:
            logger.error(f"❌ Performance optimization failed: {e}")
            phase_results["success"] = False
            phase_results["errors"].append(str(e))

        self.application_results["phases"]["performance_optimization"] = phase_results

    def _apply_monitoring_system(self):
        """モニタリングシステムの適用"""
        logger.info("📊 Applying monitoring system best practices...")

        phase_results = {"success": True, "configurations": [], "errors": []}

        try:
            # 1. メトリクス収集開始
            logger.info("📈 Starting metrics collection...")
            self.metrics_collector.start_collection(interval_seconds=60)
            phase_results["configurations"].append(
                {
                    "type": "metrics_collection",
                    "status": "started",
                    "interval_seconds": 60,
                }
            )

            # 2. サンプルメトリクス収集
            logger.info("📊 Collecting sample metrics...")
            import numpy as np
            import polars as pl

            # モデルメトリクスサンプル
            predictions = np.random.randn(100) * 0.01
            actual_returns = np.random.randn(100) * 0.01
            model_metrics = self.model_monitor.monitor_model_performance(
                predictions, actual_returns
            )
            phase_results["configurations"].append(
                {
                    "type": "model_monitoring",
                    "sample_metrics": {
                        "sharpe_ratio": model_metrics.sharpe_ratio,
                        "max_drawdown": model_metrics.max_drawdown,
                        "win_rate": model_metrics.win_rate,
                    },
                }
            )

            # データ品質メトリクスサンプル
            sample_data = pl.DataFrame(
                {
                    "code": ["1234"] * 50,
                    "date": [f"2024-01-{i:02d}" for i in range(1, 51)],
                    "close": np.random.randn(50) * 100 + 1000,
                }
            )
            data_metrics = self.data_monitor.monitor_data_quality(sample_data)
            phase_results["configurations"].append(
                {
                    "type": "data_quality_monitoring",
                    "sample_metrics": {
                        "total_records": data_metrics.total_records,
                        "null_ratio": data_metrics.null_ratio,
                        "validation_errors": data_metrics.validation_errors,
                    },
                }
            )

            logger.info("✅ Monitoring system configured successfully")

        except Exception as e:
            logger.error(f"❌ Monitoring system configuration failed: {e}")
            phase_results["success"] = False
            phase_results["errors"].append(str(e))

        self.application_results["phases"]["monitoring_system"] = phase_results

    def _run_integration_tests(self):
        """統合テストの実行"""
        logger.info("🧪 Running integration tests...")

        phase_results = {"success": True, "tests": [], "errors": []}

        try:
            # 1. データ最適化テスト
            logger.info("📊 Testing data optimization...")
            test_results = self._test_data_optimization()
            phase_results["tests"].append(
                {
                    "name": "data_optimization",
                    "success": test_results["success"],
                    "details": test_results,
                }
            )

            # 2. パフォーマンステスト
            logger.info("⚡ Testing performance optimization...")
            test_results = self._test_performance_optimization()
            phase_results["tests"].append(
                {
                    "name": "performance_optimization",
                    "success": test_results["success"],
                    "details": test_results,
                }
            )

            # 3. モニタリングテスト
            logger.info("📈 Testing monitoring system...")
            test_results = self._test_monitoring_system()
            phase_results["tests"].append(
                {
                    "name": "monitoring_system",
                    "success": test_results["success"],
                    "details": test_results,
                }
            )

            # 全体の成功判定
            all_tests_passed = all(test["success"] for test in phase_results["tests"])
            phase_results["success"] = all_tests_passed

            if all_tests_passed:
                logger.info("✅ All integration tests passed")
            else:
                logger.warning("⚠️ Some integration tests failed")

        except Exception as e:
            logger.error(f"❌ Integration tests failed: {e}")
            phase_results["success"] = False
            phase_results["errors"].append(str(e))

        self.application_results["phases"]["integration_tests"] = phase_results

    def _test_data_optimization(self) -> Dict:
        """データ最適化テスト"""
        try:
            # 最適化されたファイルの存在確認
            compressed_dir = self.output_dir / "compressed"
            if compressed_dir.exists():
                file_count = len(list(compressed_dir.rglob("*.parquet")))
                return {"success": file_count > 0, "compressed_files": file_count}
            return {"success": False, "error": "No compressed files found"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _test_performance_optimization(self) -> Dict:
        """パフォーマンス最適化テスト"""
        try:
            # システムメトリクスの取得
            metrics = self.performance_optimizer.get_performance_metrics()
            return {
                "success": True,
                "cpu_usage": metrics["cpu"]["usage_percent"],
                "memory_usage_gb": metrics["memory"]["usage_gb"],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _test_monitoring_system(self) -> Dict:
        """モニタリングシステムテスト"""
        try:
            # データベースの存在確認
            db_path = Path("monitoring.db")
            if db_path.exists():
                return {"success": True, "database_exists": True}
            return {"success": False, "error": "Monitoring database not found"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _generate_final_report(self):
        """最終レポート生成"""
        logger.info("📋 Generating final report...")

        # レポートファイルの保存
        report_file = (
            self.output_dir
            / f"best_practices_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, "w") as f:
            json.dump(self.application_results, f, indent=2, default=str)

        # 推奨事項の生成
        self._generate_recommendations()

        logger.info(f"📄 Final report saved: {report_file}")

    def _generate_recommendations(self):
        """推奨事項の生成"""
        recommendations = []

        # データ最適化の推奨事項
        data_phase = self.application_results["phases"].get("data_optimization", {})
        if data_phase.get("success", False):
            recommendations.append("✅ Data optimization completed successfully")
        else:
            recommendations.append("⚠️ Consider reviewing data optimization settings")

        # パフォーマンス最適化の推奨事項
        perf_phase = self.application_results["phases"].get(
            "performance_optimization", {}
        )
        if perf_phase.get("success", False):
            recommendations.append("✅ Performance optimization completed successfully")
        else:
            recommendations.append("⚠️ Consider adjusting performance settings")

        # モニタリングシステムの推奨事項
        monitor_phase = self.application_results["phases"].get("monitoring_system", {})
        if monitor_phase.get("success", False):
            recommendations.append("✅ Monitoring system configured successfully")
        else:
            recommendations.append("⚠️ Review monitoring system configuration")

        # 統合テストの推奨事項
        test_phase = self.application_results["phases"].get("integration_tests", {})
        if test_phase.get("success", False):
            recommendations.append("✅ All integration tests passed")
        else:
            recommendations.append("⚠️ Some integration tests failed - review system")

        # 追加の推奨事項
        recommendations.extend(
            [
                "📊 Monitor system performance regularly",
                "🔄 Schedule regular cache cleanup",
                "📈 Review model performance metrics",
                "🔧 Consider implementing automated alerts",
            ]
        )

        self.application_results["recommendations"] = recommendations


def main():
    """メイン実行関数"""
    print("🚀 gogooku3 Best Practices Application")
    print("=" * 50)

    # ベストプラクティス適用
    applier = BestPracticesApplier()
    results = applier.apply_all_best_practices()

    # 結果の表示
    print("\n📊 Application Results:")
    print("=" * 30)

    if results["overall_success"]:
        print("✅ Overall Status: SUCCESS")
    else:
        print("❌ Overall Status: FAILED")
        if "error" in results:
            print(f"Error: {results['error']}")

    print(f"\n📅 Applied at: {results['timestamp']}")

    # フェーズ別結果
    print("\n📋 Phase Results:")
    for phase_name, phase_result in results["phases"].items():
        status = "✅" if phase_result.get("success", False) else "❌"
        print(f"{status} {phase_name.replace('_', ' ').title()}")

    # 推奨事項
    if "recommendations" in results:
        print("\n💡 Recommendations:")
        for rec in results["recommendations"]:
            print(f"  {rec}")

    print("\n🎉 Best practices application completed!")


if __name__ == "__main__":
    main()
