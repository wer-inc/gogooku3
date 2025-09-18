#!/usr/bin/env python3
"""
A+実装のテストスクリプト
各コンポーネントが正しく動作するか確認
"""

import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(str(Path(__file__).parent))


def test_fan_san_enabled():
    """Test 1: FAN/SANが有効化されているか確認"""
    logger.info("=" * 60)
    logger.info("Test 1: FAN/SAN有効化確認")
    logger.info("=" * 60)
    
    try:
        from src.models.architectures.atft_gat_fan import ATFT_GAT_FAN
        from types import SimpleNamespace
        
        # 最小設定でモデル初期化
        config = SimpleNamespace(
            model=SimpleNamespace(
                hidden_size=64,
                input_projection=SimpleNamespace(dropout=0.1),
                # adaptive_normalizationを意図的に省略
            ),
            data=SimpleNamespace(
                input_features=SimpleNamespace(
                    dynamic_features=10,
                    static_features=5,
                    use_static_features=False
                ),
                time_series=SimpleNamespace(
                    sequence_length=20,
                    prediction_horizons=[1, 5]
                )
            )
        )
        
        # モデル作成（デフォルトでFAN/SANが有効になるはず）
        model = ATFT_GAT_FAN(config)
        
        # FAN/SANのチェック
        import torch.nn as nn
        fan_enabled = not isinstance(model.fan, nn.Identity)
        san_enabled = not isinstance(model.san, nn.Identity)
        
        logger.info(f"✅ FAN enabled: {fan_enabled}")
        logger.info(f"✅ SAN enabled: {san_enabled}")
        
        if fan_enabled and san_enabled:
            logger.info("✅ Test 1 PASSED: FAN/SAN are enabled by default")
            return True
        else:
            logger.error("❌ Test 1 FAILED: FAN/SAN are not enabled")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test 1 FAILED: {e}")
        return False


def test_phase_runner():
    """Test 2: Phaseランナーが存在するか確認"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Phaseランナー確認")
    logger.info("=" * 60)
    
    try:
        # train_atft.pyからrun_phase_training関数をインポート
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "train_atft",
            "scripts/train_atft.py"
        )
        train_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_module)
        
        # 関数の存在確認
        has_phase_runner = hasattr(train_module, 'run_phase_training')
        
        if has_phase_runner:
            logger.info("✅ run_phase_training function found")
            
            # 関数のシグネチャ確認
            import inspect
            sig = inspect.signature(train_module.run_phase_training)
            params = list(sig.parameters.keys())
            expected_params = ['model', 'train_loader', 'val_loader', 'config', 'device']
            
            if params == expected_params:
                logger.info(f"✅ Function signature correct: {params}")
                logger.info("✅ Test 2 PASSED: Phase runner implemented correctly")
                return True
            else:
                logger.warning(f"⚠️ Function signature mismatch: {params} vs {expected_params}")
                return False
        else:
            logger.error("❌ Test 2 FAILED: run_phase_training not found")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test 2 FAILED: {e}")
        return False


def test_correlation_edges():
    """Test 3: 相関エッジ構築が動作するか確認"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: 相関エッジ構築確認")
    logger.info("=" * 60)
    
    try:
        import torch
        from src.graph.graph_builder import GraphBuilder, GBConfig
        
        # テストデータ作成
        batch_size = 30
        seq_len = 20
        n_features = 10
        
        # 時系列データ（最初の特徴量をリターンと仮定）
        features = torch.randn(batch_size, seq_len, n_features)
        
        # GraphBuilder初期化
        config = GBConfig(max_nodes=100, edge_threshold=0.3)
        graph_builder = GraphBuilder(config)
        
        # グラフ構築
        edge_index, edge_attr = graph_builder.build_graph(features)
        
        logger.info(f"✅ Graph built: {edge_index.shape[1]} edges")
        logger.info(f"✅ Edge attributes shape: {edge_attr.shape}")
        
        # 相関エッジが構築されているか確認
        if hasattr(graph_builder, 'build_correlation_edges'):
            logger.info("✅ build_correlation_edges method exists")
            
            # 直接相関エッジを構築
            edge_index_corr, edge_attr_corr = graph_builder.build_correlation_edges(features)
            logger.info(f"✅ Correlation edges: {edge_index_corr.shape[1]} edges")
            
            # エッジ属性が[0, 1]範囲か確認
            if edge_attr_corr.min() >= 0 and edge_attr_corr.max() <= 1:
                logger.info(f"✅ Edge attributes in [0, 1] range: [{edge_attr_corr.min():.3f}, {edge_attr_corr.max():.3f}]")
                logger.info("✅ Test 3 PASSED: Correlation edges working correctly")
                return True
            else:
                logger.warning(f"⚠️ Edge attributes out of range: [{edge_attr_corr.min():.3f}, {edge_attr_corr.max():.3f}]")
                return False
        else:
            logger.error("❌ build_correlation_edges method not found")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test 3 FAILED: {e}")
        return False


def test_wf_evaluation():
    """Test 4: WF評価スクリプトが存在し実行可能か確認"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 4: WF評価スクリプト確認")
    logger.info("=" * 60)
    
    try:
        script_path = Path("scripts/evaluate_with_wf.py")
        
        if not script_path.exists():
            logger.error(f"❌ Script not found: {script_path}")
            return False
        
        logger.info(f"✅ Script exists: {script_path}")
        
        # スクリプトのインポート確認
        import importlib.util
        spec = importlib.util.spec_from_file_location("evaluate_wf", script_path)
        eval_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eval_module)
        
        # 必要な関数の存在確認
        required_functions = [
            'compute_hit_rate',
            'compute_max_drawdown',
            'compute_rank_ic',
            'load_model',
            'evaluate_fold',
            'main'
        ]
        
        missing = []
        for func in required_functions:
            if not hasattr(eval_module, func):
                missing.append(func)
        
        if missing:
            logger.warning(f"⚠️ Missing functions: {missing}")
            return False
        
        logger.info(f"✅ All required functions found: {required_functions}")
        
        # WalkForwardSplitterV2のインポート確認
        from src.data.safety.walk_forward_v2 import WalkForwardSplitterV2
        logger.info("✅ WalkForwardSplitterV2 imported successfully")
        
        logger.info("✅ Test 4 PASSED: WF evaluation script ready")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test 4 FAILED: {e}")
        return False


def main():
    """全テスト実行"""
    logger.info("🚀 A+ Implementation Test Suite")
    logger.info("=" * 60)
    
    results = {
        "FAN/SAN有効化": test_fan_san_enabled(),
        "Phaseランナー": test_phase_runner(),
        "相関エッジ構築": test_correlation_edges(),
        "WF評価スクリプト": test_wf_evaluation()
    }
    
    # 結果サマリー
    logger.info("\n" + "=" * 60)
    logger.info("Test Results Summary")
    logger.info("=" * 60)
    
    passed = 0
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info("-" * 60)
    logger.info(f"Total: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("\n🎉 All tests passed! A+ implementation is ready.")
        logger.info("\n📝 Next steps:")
        logger.info("1. Run training with Phase Training:")
        logger.info("   python scripts/integrated_ml_training_pipeline.py \\")
        logger.info("     --data-path output/ml_dataset_*.parquet \\")
        logger.info("     --max-epochs 45 --batch-size 512")
        logger.info("\n2. Evaluate with Walk-Forward:")
        logger.info("   python scripts/evaluate_with_wf.py \\")
        logger.info("     --model-path output/checkpoints/best_model.pth \\")
        logger.info("     --data-path output/ml_dataset_*.parquet")
    else:
        logger.warning(f"\n⚠️ {len(results) - passed} tests failed. Please check the implementation.")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)