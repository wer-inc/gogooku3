#!/usr/bin/env python3
"""
ATFT-GAT-FAN Monitoring Dashboard
本番運用時の監視ダッシュボード
"""

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from tensorboard import program
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import GPUtil
    GPU_UTIL_AVAILABLE = True
except ImportError:
    GPU_UTIL_AVAILABLE = False

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MonitoringDashboard:
    """監視ダッシュボードクラス"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.logs_dir = self.project_root / "logs"
        self.models_dir = self.project_root / "models"
        self.monitoring_dir = self.project_root / "monitoring"
        self.monitoring_dir.mkdir(exist_ok=True)

        # メトリクス履歴
        self.metrics_history = []
        self.alerts_history = []

        # アラートしきい値
        self.alert_thresholds = {
            'rankic_h1': 0.05,  # RankIC@1dが0.05以下でアラート
            'loss': 0.1,        # 損失が0.1以上でアラート
            'memory_usage': 0.9, # メモリ使用率90%以上でアラート
            'gpu_memory': 0.95,  # GPUメモリ95%以上でアラート
        }

    def start_tensorboard(self, port: int = 6006) -> subprocess.Popen | None:
        """TensorBoardサーバーを起動"""
        if not TENSORBOARD_AVAILABLE:
            logger.warning("TensorBoard is not available")
            return None

        try:
            # ログディレクトリの確認
            if not self.logs_dir.exists():
                logger.warning(f"Logs directory not found: {self.logs_dir}")
                return None

            logger.info(f"Starting TensorBoard on port {port}")
            tb = program.TensorBoard()
            tb.configure(argv=[
                None,
                '--logdir', str(self.logs_dir),
                '--port', str(port),
                '--host', '0.0.0.0',
                '--reload_interval', '30'
            ])

            # 非同期で起動
            process = subprocess.Popen(
                ['tensorboard', '--logdir', str(self.logs_dir), '--port', str(port), '--host', '0.0.0.0'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            logger.info(f"TensorBoard started on http://localhost:{port}")
            return process

        except Exception as e:
            logger.error(f"Failed to start TensorBoard: {e}")
            return None

    def check_wandb_status(self) -> dict[str, Any]:
        """W&Bの状態を確認"""
        status = {
            'available': WANDB_AVAILABLE,
            'logged_in': False,
            'current_run': None,
            'api_key_set': False
        }

        if not WANDB_AVAILABLE:
            return status

        try:
            # APIキーの確認
            api_key = os.getenv('WANDB_API_KEY')
            status['api_key_set'] = api_key is not None

            # ログイン状態の確認
            if wandb.run is not None:
                status['current_run'] = wandb.run.name
                status['logged_in'] = True
            elif api_key:
                wandb.login(key=api_key)
                status['logged_in'] = True

        except Exception as e:
            logger.error(f"W&B status check failed: {e}")

        return status

    def collect_system_metrics(self) -> dict[str, Any]:
        """システムメトリクスを収集"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        }

        # GPUメトリクス（利用可能な場合）
        if GPU_UTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # 最初のGPUを使用
                    # GPUtilのAPIに応じて属性名を調整
                    gpu_memory_percent = getattr(gpu, 'memoryPercent', None)
                    if gpu_memory_percent is None:
                        # 代替方法でメモリ使用率を計算
                        gpu_memory_used = getattr(gpu, 'memoryUsed', 0)
                        gpu_memory_total = getattr(gpu, 'memoryTotal', 1)
                        gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100 if gpu_memory_total > 0 else 0

                    metrics.update({
                        'gpu_memory_percent': gpu_memory_percent,
                        'gpu_memory_used_mb': getattr(gpu, 'memoryUsed', 0),
                        'gpu_memory_total_mb': getattr(gpu, 'memoryTotal', 0),
                        'gpu_utilization': getattr(gpu, 'load', 0) * 100,
                        'gpu_temperature': getattr(gpu, 'temperature', 0)
                    })
            except Exception as e:
                logger.warning(f"GPU metrics collection failed: {e}")

        # ディスク使用量
        disk = psutil.disk_usage('/')
        metrics.update({
            'disk_percent': disk.percent,
            'disk_used_gb': disk.used / (1024**3),
            'disk_total_gb': disk.total / (1024**3)
        })

        return metrics

    def check_training_status(self) -> dict[str, Any]:
        """トレーニング状態を確認"""
        status = {
            'active_runs': [],
            'last_model': None,
            'recent_logs': []
        }

        # アクティブなトレーニングプロセスを確認
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'train' in cmdline and 'atft_gat_fan' in cmdline:
                        status['active_runs'].append({
                            'pid': proc.info['pid'],
                            'command': cmdline[:100] + '...' if len(cmdline) > 100 else cmdline
                        })
        except Exception as e:
            logger.warning(f"Process check failed: {e}")

        # 最新のモデルファイルを確認
        try:
            if self.models_dir.exists():
                model_files = list(self.models_dir.glob("*.ckpt"))
                if model_files:
                    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
                    status['last_model'] = {
                        'path': str(latest_model),
                        'modified': datetime.fromtimestamp(latest_model.stat().st_mtime).isoformat(),
                        'size_mb': latest_model.stat().st_size / (1024**2)
                    }
        except Exception as e:
            logger.warning(f"Model check failed: {e}")

        # 最近のログファイルを確認
        try:
            if self.logs_dir.exists():
                log_files = list(self.logs_dir.rglob("*.log"))
                recent_logs = sorted(log_files, key=lambda p: p.stat().st_mtime, reverse=True)[:5]
                for log_file in recent_logs:
                    status['recent_logs'].append({
                        'path': str(log_file),
                        'modified': datetime.fromtimestamp(log_file.stat().st_mtime).isoformat(),
                        'size_kb': log_file.stat().st_size / 1024
                    })
        except Exception as e:
            logger.warning(f"Log check failed: {e}")

        return status

    def check_alerts(self, metrics: dict[str, Any]) -> list[dict[str, Any]]:
        """アラート条件を確認"""
        alerts = []

        # RankICアラート
        if 'rankic_h1' in metrics:
            if metrics['rankic_h1'] < self.alert_thresholds['rankic_h1']:
                alerts.append({
                    'type': 'warning',
                    'metric': 'rankic_h1',
                    'value': metrics['rankic_h1'],
                    'threshold': self.alert_thresholds['rankic_h1'],
                    'message': f'RankIC@1d がしきい値以下: {metrics["rankic_h1"]:.3f}'
                })

        # 損失アラート
        if 'loss' in metrics:
            if metrics['loss'] > self.alert_thresholds['loss']:
                alerts.append({
                    'type': 'warning',
                    'metric': 'loss',
                    'value': metrics['loss'],
                    'threshold': self.alert_thresholds['loss'],
                    'message': f'損失がしきい値以上: {metrics["loss"]:.3f}'
                })

        # メモリアラート
        if metrics['memory_percent'] > self.alert_thresholds['memory_usage'] * 100:
            alerts.append({
                'type': 'critical',
                'metric': 'memory_percent',
                'value': metrics['memory_percent'],
                'threshold': self.alert_thresholds['memory_usage'] * 100,
                'message': f'メモリ使用率がしきい値以上: {metrics["memory_percent"]:.1f}%'
            })

        # GPUメモリアラート
        if 'gpu_memory_percent' in metrics:
            if metrics['gpu_memory_percent'] > self.alert_thresholds['gpu_memory'] * 100:
                alerts.append({
                    'type': 'critical',
                    'metric': 'gpu_memory_percent',
                    'value': metrics['gpu_memory_percent'],
                    'threshold': self.alert_thresholds['gpu_memory'] * 100,
                    'message': f'GPUメモリ使用率がしきい値以上: {metrics["gpu_memory_percent"]:.1f}%'
                })

        return alerts

    def generate_report(self) -> dict[str, Any]:
        """総合レポートを生成"""
        system_metrics = self.collect_system_metrics()
        training_status = self.check_training_status()
        wandb_status = self.check_wandb_status()
        alerts = self.check_alerts(system_metrics)

        # アラート履歴に追加
        if alerts:
            self.alerts_history.extend(alerts)

        # メトリクス履歴に追加
        self.metrics_history.append(system_metrics)

        # 最新の10件のみ保持
        self.metrics_history = self.metrics_history[-10:]
        self.alerts_history = self.alerts_history[-50:]

        report = {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': system_metrics,
            'training_status': training_status,
            'wandb_status': wandb_status,
            'active_alerts': alerts,
            'alerts_count': len(self.alerts_history),
            'metrics_count': len(self.metrics_history),
            'overall_status': 'healthy' if not alerts else 'warning'
        }

        return report

    def save_report(self, report: dict[str, Any]):
        """レポートを保存"""
        report_file = self.monitoring_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Report saved: {report_file}")

    def display_dashboard(self, report: dict[str, Any]):
        """ダッシュボードを表示"""
        print("\n" + "="*80)
        print("ATFT-GAT-FAN MONITORING DASHBOARD")
        print("="*80)

        # システムステータス
        print(f"\n📊 SYSTEM STATUS: {report['overall_status'].upper()}")
        sys_metrics = report['system_metrics']
        print(f"  CPU: {sys_metrics['cpu_percent']:.1f}%")
        print(f"  Memory: {sys_metrics['memory_percent']:.1f}% ({sys_metrics['memory_used_gb']:.1f}/{sys_metrics['memory_total_gb']:.1f} GB)")

        if 'gpu_memory_percent' in sys_metrics:
            print(f"  GPU Memory: {sys_metrics['gpu_memory_percent']:.1f}% ({sys_metrics['gpu_memory_used_mb']:.0f}/{sys_metrics['gpu_memory_total_mb']:.0f} MB)")
            print(f"  GPU Utilization: {sys_metrics['gpu_utilization']:.1f}%")

        # トレーニングステータス
        train_status = report['training_status']
        print("\n🔄 TRAINING STATUS")
        print(f"  Active Runs: {len(train_status['active_runs'])}")
        for run in train_status['active_runs'][:3]:  # 最大3件表示
            print(f"    PID {run['pid']}: {run['command']}")

        if train_status['last_model']:
            model = train_status['last_model']
            print(f"  Last Model: {Path(model['path']).name} ({model['size_mb']:.1f} MB)")

        # W&Bステータス
        wandb_status = report['wandb_status']
        print("\n📈 W&B STATUS")
        print(f"  Available: {'✅' if wandb_status['available'] else '❌'}")
        print(f"  Logged In: {'✅' if wandb_status['logged_in'] else '❌'}")
        print(f"  API Key Set: {'✅' if wandb_status['api_key_set'] else '❌'}")

        if wandb_status['current_run']:
            print(f"  Current Run: {wandb_status['current_run']}")

        # アラート
        alerts = report['active_alerts']
        if alerts:
            print(f"\n🚨 ACTIVE ALERTS ({len(alerts)})")
            for alert in alerts:
                icon = "⚠️" if alert['type'] == 'warning' else "🚨"
                print(f"  {icon} {alert['message']}")
        else:
            print("\n✅ NO ACTIVE ALERTS")

        print("\n📋 SUMMARY")
        print(f"  Total Alerts: {report['alerts_count']}")
        print(f"  Metrics History: {report['metrics_count']} entries")
        print(f"  Report Time: {report['timestamp']}")

        print("\n" + "="*80)


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="ATFT-GAT-FAN Monitoring Dashboard")
    parser.add_argument(
        "--tensorboard-port",
        type=int,
        default=6006,
        help="TensorBoard port"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Monitoring interval in seconds"
    )
    parser.add_argument(
        "--start-tensorboard",
        action="store_true",
        help="Start TensorBoard server"
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuous monitoring"
    )

    args = parser.parse_args()

    dashboard = MonitoringDashboard()

    # TensorBoard起動
    if args.start_tensorboard:
        logger.info("Starting TensorBoard...")
        tb_process = dashboard.start_tensorboard(args.tensorboard_port)
        if tb_process:
            logger.info(f"TensorBoard started on port {args.tensorboard_port}")
            time.sleep(2)  # 起動待機

    # 初回レポート
    logger.info("Generating initial report...")
    report = dashboard.generate_report()
    dashboard.display_dashboard(report)
    dashboard.save_report(report)

    # 継続監視
    if args.continuous:
        logger.info(f"Starting continuous monitoring (interval: {args.interval}s)...")
        try:
            while True:
                time.sleep(args.interval)
                report = dashboard.generate_report()
                dashboard.display_dashboard(report)
                dashboard.save_report(report)

        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitoring failed: {e}")
    else:
        logger.info("One-time monitoring completed")


if __name__ == "__main__":
    main()
