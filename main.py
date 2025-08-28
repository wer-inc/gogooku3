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
from typing import Optional

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

    async def run_safe_training(self, mode: str = "full"):
        """安全学習パイプラインの実行"""
        logger.info("🚀 Starting Safe Training Pipeline...")
        
        pipeline = SafeTrainingPipeline()
        
        if mode == "quick":
            # クイックテスト（1エポック）
            success, result = await pipeline.run_quick_test()
        else:
            # フル学習パイプライン
            success, result = await pipeline.run_full_pipeline()
        
        if success:
            logger.info("✅ Safe Training Pipeline completed successfully!")
            logger.info(f"📊 Results: {result.get('summary', {})}")
            return True, result
        else:
            logger.error(f"❌ Safe Training Pipeline failed: {result.get('error')}")
            return False, result

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

    async def run_workflow(self, workflow: str, mode: str = "full"):
        """指定されたワークフローを実行"""
        logger.info(f"🎬 Starting workflow: {workflow}")
        
        try:
            if workflow == "safe-training":
                return await self.run_safe_training(mode)
            elif workflow == "ml-dataset":
                return self.run_ml_dataset_builder()
            elif workflow == "direct-api-dataset":
                return await self.run_direct_api_dataset_builder()
            elif workflow == "complete-atft":
                return await self.run_complete_atft_training()
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
  safe-training       安全学習パイプライン（推奨）
  ml-dataset          MLデータセット構築
  direct-api-dataset  直接APIデータセット構築
  complete-atft       完全ATFT学習パイプライン

使用例:
  python main.py safe-training --mode full
  python main.py ml-dataset
  python main.py direct-api-dataset
  python main.py complete-atft
        """
    )
    
    parser.add_argument(
        "workflow",
        choices=["safe-training", "ml-dataset", "direct-api-dataset", "complete-atft"],
        help="実行するワークフロー"
    )
    
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="full",
        help="実行モード（safe-trainingのみ：quick=1エポック, full=完全学習）"
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
    
    async def run():
        return await runner.run_workflow(args.workflow, args.mode)
    
    # 非同期実行
    success, result = asyncio.run(run())
    
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
            print(f"   - 達成Sharpe: {validation_info.get('sharpe_ratio', 'N/A')}")
            print(f"   - パラメータ数: {validation_info.get('param_count', 'N/A'):,}")
        
        print("=" * 80)
        print("✅ 実行完了")
        
    else:
        print("❌ ワークフロー実行失敗")
        print(f"エラー: {result.get('error', 'Unknown error')}")
        print("=" * 80)
        
        sys.exit(1)


if __name__ == "__main__":
    main()