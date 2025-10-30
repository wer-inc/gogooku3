#!/usr/bin/env python3
# DEPRECATED: Moved to scripts/_archive on 2025-09-04
"""
ATFT-GAT-FAN Production Deployment Script
本番環境移行支援スクリプト
"""

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.validate_improvements import run_detailed_comparison, validate_performance

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProductionDeploymentManager:
    """本番移行管理クラス"""

    def __init__(self, config_path: str = "configs/atft/config.yaml"):
        self.config_path = Path(config_path)
        self.backup_dir = Path("backups")
        self.backup_dir.mkdir(exist_ok=True)

        # 現在のタイムスタンプ
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def backup_current_config(self):
        """現在の設定ファイルをバックアップ"""
        logger.info("Backing up current configuration...")

        backup_path = self.backup_dir / f"config_backup_{self.timestamp}.yaml"
        shutil.copy2(self.config_path, backup_path)

        logger.info(f"Configuration backed up to: {backup_path}")
        return backup_path

    def create_phase_config(self, phase: int):
        """各Phaseの設定ファイル作成"""
        logger.info(f"Creating Phase {phase} configuration...")

        base_config = self._load_config_template()

        if phase == 1:
            # Phase 1: 基本改善有効化
            config = self._apply_phase1_settings(base_config)
        elif phase == 2:
            # Phase 2: 学習安定化有効化
            config = self._apply_phase2_settings(base_config)
        elif phase == 3:
            # Phase 3: 高度機能有効化
            config = self._apply_phase3_settings(base_config)
        else:
            raise ValueError(f"Unknown phase: {phase}")

        # 設定ファイル保存
        phase_config_path = self.config_path.parent / f"config_phase{phase}.yaml"
        self._save_config(config, phase_config_path)

        logger.info(f"Phase {phase} configuration created: {phase_config_path}")
        return phase_config_path

    def _load_config_template(self):
        """設定テンプレート読み込み"""
        import yaml
        with open(self.config_path, encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _apply_phase1_settings(self, config):
        """Phase 1設定適用"""
        config['improvements'].update({
            'output_head_small_init': True,
            'gat_temperature': 1.0,
            'huber_loss': True,
            'use_ema': False,
            'online_normalization': False,
            'freq_dropout_p': 0.0,
            'enable_wandb': False,
            'auto_recover_oom': False
        })
        return config

    def _apply_phase2_settings(self, config):
        """Phase 2設定適用"""
        config['improvements'].update({
            'output_head_small_init': True,
            'gat_temperature': 1.0,
            'huber_loss': True,
            'use_ema': True,
            'online_normalization': True,
            'freq_dropout_p': 0.0,
            'enable_wandb': False,
            'auto_recover_oom': False
        })
        return config

    def _apply_phase3_settings(self, config):
        """Phase 3設定適用"""
        config['improvements'].update({
            'output_head_small_init': True,
            'gat_temperature': 1.0,
            'huber_loss': True,
            'use_ema': True,
            'online_normalization': True,
            'freq_dropout_p': 0.1,
            'enable_wandb': True,
            'auto_recover_oom': True
        })
        return config

    def _save_config(self, config, path):
        """設定ファイル保存"""
        import yaml
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    def run_phase_test(self, phase: int, data_path: str):
        """各Phaseのテスト実行"""
        logger.info(f"Running Phase {phase} tests...")

        # Phase設定適用
        phase_config = self.create_phase_config(phase)

        try:
            # 基本性能テスト
            logger.info("Running basic performance test...")
            basic_result = validate_performance(data_path)
            logger.info(f"Basic test result: {basic_result}")

            # 詳細比較テスト（Phase 3のみ）
            if phase == 3:
                logger.info("Running detailed comparison...")
                detailed_result = run_detailed_comparison(data_path)
                logger.info(f"Detailed comparison result: {detailed_result}")

            # テスト結果保存
            test_results = {
                'phase': phase,
                'timestamp': self.timestamp,
                'basic_test': basic_result,
                'config_path': str(phase_config)
            }

            if phase == 3:
                test_results['detailed_comparison'] = detailed_result

            results_path = self.backup_dir / f"phase{phase}_test_results_{self.timestamp}.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, indent=2, default=str)

            logger.info(f"Phase {phase} test completed successfully")
            logger.info(f"Results saved to: {results_path}")

            return True

        except Exception as e:
            logger.error(f"Phase {phase} test failed: {e}")
            return False

    def deploy_to_production(self, final_config_path: str = None):
        """本番環境への最終デプロイ"""
        logger.info("Deploying to production environment...")

        if final_config_path is None:
            final_config_path = self.create_phase_config(3)  # Phase 3を最終設定として使用

        # 最終設定を本番設定としてコピー
        production_config = self.config_path.parent / "config_production.yaml"
        shutil.copy2(final_config_path, production_config)

        # デプロイ完了ログ
        deployment_log = {
            'timestamp': self.timestamp,
            'final_config': str(production_config),
            'backup_config': str(self.backup_current_config()),
            'status': 'deployed'
        }

        log_path = self.backup_dir / f"production_deployment_{self.timestamp}.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(deployment_log, f, indent=2, default=str)

        logger.info("Production deployment completed!")
        logger.info(f"Production config: {production_config}")
        logger.info(f"Deployment log: {log_path}")

        return production_config

    def rollback_to_previous(self, backup_path: str = None):
        """ロールバック実行"""
        logger.info("Rolling back to previous configuration...")

        if backup_path is None:
            # 最新のバックアップを探す
            backups = list(self.backup_dir.glob("config_backup_*.yaml"))
            if backups:
                backup_path = max(backups, key=lambda p: p.stat().st_mtime)
            else:
                logger.error("No backup configuration found")
                return False

        # バックアップから復元
        shutil.copy2(backup_path, self.config_path)

        logger.info(f"Rolled back to: {backup_path}")
        return True


def main():
    parser = argparse.ArgumentParser(description="ATFT-GAT-FAN Production Deployment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/atft/config.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Test data file path"
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        help="Deployment phase (1: Basic, 2: Stability, 3: Advanced)"
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy to production"
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback to previous configuration"
    )
    parser.add_argument(
        "--full-test",
        action="store_true",
        help="Run all phases test"
    )

    args = parser.parse_args()

    # デプロイマネージャー初期化
    manager = ProductionDeploymentManager(args.config)

    if args.rollback:
        # ロールバック実行
        success = manager.rollback_to_previous()
        if success:
            logger.info("✅ Rollback completed successfully")
        else:
            logger.error("❌ Rollback failed")
        return 0 if success else 1

    if args.deploy:
        # 本番デプロイ実行
        production_config = manager.deploy_to_production()
        logger.info(f"✅ Deployed to production with config: {production_config}")
        return 0

    if args.phase:
        # 指定Phaseのテスト実行
        success = manager.run_phase_test(args.phase, args.data)
        if success:
            logger.info(f"✅ Phase {args.phase} test completed successfully")
        else:
            logger.error(f"❌ Phase {args.phase} test failed")
        return 0 if success else 1

    if args.full_test:
        # 全Phaseテスト実行
        logger.info("Running full deployment test (all phases)...")

        results = []
        for phase in [1, 2, 3]:
            logger.info(f"\n{'='*50}")
            logger.info(f"PHASE {phase} TEST")
            logger.info(f"{'='*50}")

            success = manager.run_phase_test(phase, args.data)
            results.append(success)

            if not success:
                logger.error(f"Phase {phase} failed, stopping full test")
                break

        overall_success = all(results)
        if overall_success:
            logger.info("✅ All phases completed successfully!")
            logger.info("Ready for production deployment")
        else:
            logger.error("❌ Some phases failed, check logs for details")

        return 0 if overall_success else 1

    # デフォルト: ヘルプ表示
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
