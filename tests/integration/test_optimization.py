#!/usr/bin/env python3
"""
Test script for ATFT-GAT-FAN optimization system
最適化システムの動作確認用スクリプト
"""

import os
import sys
from pathlib import Path
import logging

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """依存関係チェック"""
    logger.info("Checking dependencies...")
    
    # Required packages
    required_packages = [
        ('optuna', 'optuna'),
        ('mlflow', 'mlflow'), 
        ('torch', 'torch'),
        ('hydra-core', 'hydra'),
        ('omegaconf', 'omegaconf')
    ]
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            logger.info(f"✅ {package_name} - OK")
        except ImportError:
            logger.error(f"❌ {package_name} - Missing")
            missing_packages.append(package_name)
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.info("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_data_availability():
    """データ可用性チェック"""
    logger.info("Checking data availability...")
    
    data_paths = [
        'output/atft_data/train',
        'output/ml_dataset_latest_full.parquet'
    ]
    
    for data_path in data_paths:
        full_path = project_root / data_path
        if full_path.exists():
            logger.info(f"✅ {data_path} - Available")
            
            if data_path.endswith('.parquet'):
                # Check file size
                size_mb = full_path.stat().st_size / (1024 * 1024)
                logger.info(f"   File size: {size_mb:.1f} MB")
            elif full_path.is_dir():
                # Count files in directory
                files = list(full_path.glob('*.parquet'))
                logger.info(f"   Files: {len(files)} parquet files")
        else:
            logger.warning(f"⚠️  {data_path} - Not found")
    
    return True

def check_configs():
    """設定ファイルチェック"""
    logger.info("Checking configuration files...")
    
    config_files = [
        'configs/atft/config.yaml',
        'configs/atft/train/production.yaml',
        'configs/atft/train/optimized_phases.yaml',
        'scripts/hyperparameter_tuning_real.py',
        'scripts/optimize_atft.sh',
        '.env.optuna'
    ]
    
    for config_file in config_files:
        full_path = project_root / config_file
        if full_path.exists():
            logger.info(f"✅ {config_file} - Available")
        else:
            logger.error(f"❌ {config_file} - Missing")
            return False
    
    return True

def run_quick_test():
    """クイックテスト実行"""
    logger.info("Running quick optimization test...")
    
    try:
        from scripts.hyperparameter_tuning_real import RealHyperparameterTuner
        
        # 最小限のテスト
        tuner = RealHyperparameterTuner(
            data_path="output/atft_data/train",
            n_epochs_trial=1,  # 1エポックのみ
            max_data_files=10  # 10ファイルのみ
        )
        
        # Phase 1の探索空間をテスト
        search_space = tuner.get_phase_search_space(1)
        logger.info(f"✅ Phase 1 search space: {list(search_space.keys())}")
        
        # Hydraオーバーライド生成テスト
        test_params = {
            'lr': 0.001,
            'batch_size': 512,
            'weight_decay': 0.01
        }
        overrides = tuner.create_hydra_overrides(test_params)
        logger.info(f"✅ Hydra overrides: {overrides}")
        
        logger.info("✅ Quick test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Quick test failed: {e}")
        return False

def show_usage_examples():
    """使用例表示"""
    logger.info("Usage examples:")
    
    examples = [
        # Quick test
        "# Quick test (1 trial, 1 epoch)",
        "python scripts/hyperparameter_tuning_real.py --phase 1 --trials 1 --epochs 1",
        "",
        
        # Phase-by-phase
        "# Phase 1: Basic parameters (10 trials, 5 epochs each)",
        "python scripts/hyperparameter_tuning_real.py --phase 1 --trials 10 --epochs 5",
        "",
        "# Phase 2: Graph parameters", 
        "python scripts/hyperparameter_tuning_real.py --phase 2 --trials 10 --epochs 5",
        "",
        "# Phase 3: FAN/TFT fusion",
        "python scripts/hyperparameter_tuning_real.py --phase 3 --trials 10 --epochs 5",
        "",
        
        # All phases at once
        "# All phases sequentially",
        "./scripts/optimize_atft.sh",
        "",
        "# Custom settings",
        "./scripts/optimize_atft.sh output/atft_data/train 15 7 150",
        "",
        
        # Environment setup
        "# Load environment for optimization",
        "source .env.optuna",
        "",
        
        # MLflow dashboard
        "# View results in MLflow (separate terminal)",
        "mlflow ui --host 0.0.0.0 --port 5000",
    ]
    
    for example in examples:
        if example.startswith('#'):
            logger.info(f"\033[92m{example}\033[0m")  # Green for comments
        elif example.strip():
            logger.info(f"\033[93m{example}\033[0m")  # Yellow for commands
        else:
            logger.info("")

def main():
    """メイン関数"""
    logger.info("🚀 ATFT-GAT-FAN Optimization System Test")
    logger.info("=" * 60)
    
    # Step 1: Dependencies
    if not check_dependencies():
        logger.error("❌ Dependency check failed")
        return 1
    
    # Step 2: Data
    check_data_availability()
    
    # Step 3: Configs
    if not check_configs():
        logger.error("❌ Configuration check failed") 
        return 1
    
    # Step 4: Quick test
    if not run_quick_test():
        logger.error("❌ Quick test failed")
        return 1
    
    # Step 5: Usage examples
    logger.info("=" * 60)
    show_usage_examples()
    
    logger.info("=" * 60)
    logger.info("✅ All checks passed! Optimization system is ready.")
    logger.info("💡 Start with a quick test: python scripts/hyperparameter_tuning_real.py --phase 1 --trials 1 --epochs 1")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())