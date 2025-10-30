#!/usr/bin/env python3
"""
ATFT-GAT-FAN Monitoring Setup
監視システムの初期設定スクリプト
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MonitoringSetup:
    """監視システムセットアップクラス"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.requirements_file = self.project_root / "requirements.txt"

    def install_monitoring_dependencies(self):
        """監視に必要な依存関係をインストール"""
        logger.info("Installing monitoring dependencies...")

        packages = [
            "wandb",
            "tensorboard",
            "psutil",
            "gputil",
            "plotly",
            "slack-sdk",  # アラート通知用
        ]

        for package in packages:
            try:
                logger.info(f"Installing {package}...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                logger.info(f"✅ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to install {package}: {e}")
            except Exception as e:
                logger.warning(f"Error installing {package}: {e}")

    def setup_wandb(self, api_key: str | None = None, project_name: str = "ATFT-GAT-FAN"):
        """W&Bのセットアップ"""
        logger.info("Setting up Weights & Biases...")

        try:
            import wandb

            # APIキーの設定
            if api_key:
                os.environ['WANDB_API_KEY'] = api_key
                logger.info("W&B API key set from parameter")

            # 環境変数からのAPIキー取得
            api_key = os.getenv('WANDB_API_KEY')
            if not api_key:
                logger.warning("W&B API key not found. Please set WANDB_API_KEY environment variable")
                logger.info("You can get API key from: https://wandb.ai/authorize")
                return False

            # ログイン
            wandb.login(key=api_key)
            logger.info("✅ W&B login successful")

            # プロジェクト設定
            wandb.init(
                project=project_name,
                name=f"setup_{wandb.util.generate_id()}",
                config={
                    "setup": True,
                    "timestamp": "2024-08-29"
                }
            )

            wandb.finish()
            logger.info(f"✅ W&B project '{project_name}' initialized")

            return True

        except ImportError:
            logger.error("wandb package not found. Run: pip install wandb")
            return False
        except Exception as e:
            logger.error(f"W&B setup failed: {e}")
            return False

    def create_monitoring_config(self):
        """監視設定ファイルを作成"""
        logger.info("Creating monitoring configuration...")

        config_dir = self.project_root / "configs" / "monitoring"
        config_dir.mkdir(exist_ok=True)

        monitoring_config = {
            "monitoring": {
                "enabled": True,
                "interval_seconds": 300,  # 5分間隔
                "alerts": {
                    "enabled": True,
                    "rankic_threshold": 0.05,
                    "loss_threshold": 0.1,
                    "memory_threshold": 0.9,
                    "gpu_memory_threshold": 0.95
                },
                "notifications": {
                    "slack": {
                        "enabled": False,
                        "webhook_url": "",
                        "channel": "#alerts"
                    },
                    "email": {
                        "enabled": False,
                        "smtp_server": "",
                        "recipients": []
                    }
                },
                "wandb": {
                    "project": "ATFT-GAT-FAN",
                    "entity": "",
                    "tags": ["production", "monitoring"]
                },
                "tensorboard": {
                    "port": 6006,
                    "log_dir": "logs"
                }
            },
            "dashboard": {
                "port": 8050,
                "host": "0.0.0.0",
                "debug": False
            }
        }

        import yaml
        config_file = config_dir / "monitoring.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(monitoring_config, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"✅ Monitoring config created: {config_file}")
        return config_file

    def create_alert_system(self):
        """アラートシステムを作成"""
        logger.info("Creating alert system...")

        alert_script = self.project_root / "scripts" / "alert_system.py"

        alert_code = '''#!/usr/bin/env python3
"""
ATFT-GAT-FAN Alert System
監視アラート通知システム
"""

import os
import sys
import logging
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

try:
    from slack_sdk import WebClient
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AlertSystem:
    """アラートシステムクラス"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or (project_root / "configs" / "monitoring" / "monitoring.yaml")
        self.config = self._load_config()
        self.alert_history = []

    def _load_config(self) -> Dict:
        """設定ファイルを読み込み"""
        try:
            import yaml
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            return {}

    def send_alert(self, alert_type: str, title: str, message: str, details: Optional[Dict] = None):
        """アラートを送信"""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "type": alert_type,
            "title": title,
            "message": message,
            "details": details or {}
        }

        self.alert_history.append(alert)
        logger.warning(f"🚨 ALERT: {title} - {message}")

        # Slack通知
        if self.config.get("monitoring", {}).get("notifications", {}).get("slack", {}).get("enabled", False):
            self._send_slack_alert(alert)

        # メール通知
        if self.config.get("monitoring", {}).get("notifications", {}).get("email", {}).get("enabled", False):
            self._send_email_alert(alert)

        # アラート履歴を保存（最新100件）
        self._save_alert_history()

    def _send_slack_alert(self, alert: Dict):
        """Slackにアラートを送信"""
        if not SLACK_AVAILABLE:
            logger.warning("Slack SDK not available")
            return

        try:
            slack_config = self.config["monitoring"]["notifications"]["slack"]
            client = WebClient(token=slack_config["webhook_url"])  # Webhook URLを使用

            icon = "🚨" if alert["type"] == "critical" else "⚠️"
            message = f"{icon} *{alert['title']}*\\n{alert['message']}"

            if alert["details"]:
                details_text = "\\n".join([f"• {k}: {v}" for k, v in alert["details"].items()])
                message += f"\\n\\n{details_text}"

            client.chat_postMessage(
                channel=slack_config["channel"],
                text=message,
                username="ATFT-GAT-FAN Monitor"
            )

            logger.info("Slack alert sent successfully")

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

    def _send_email_alert(self, alert: Dict):
        """メールでアラートを送信"""
        try:
            email_config = self.config["monitoring"]["notifications"]["email"]
            smtp_server = email_config["smtp_server"]
            recipients = email_config["recipients"]

            msg = MIMEMultipart()
            msg['From'] = "monitor@atft-gat-fan.com"
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = f"ATFT-GAT-FAN Alert: {alert['title']}"

            body = f"""
ATFT-GAT-FAN Monitoring Alert

Type: {alert['type'].upper()}
Title: {alert['title']}
Message: {alert['message']}
Timestamp: {alert['timestamp']}

Details:
{json.dumps(alert['details'], indent=2, ensure_ascii=False) if alert['details'] else 'None'}
            """

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(smtp_server)
            server.send_message(msg)
            server.quit()

            logger.info("Email alert sent successfully")

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    def _save_alert_history(self):
        """アラート履歴を保存"""
        try:
            history_file = project_root / "monitoring" / "alert_history.json"

            # 最新100件のみ保持
            recent_alerts = self.alert_history[-100:]

            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(recent_alerts, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Failed to save alert history: {e}")

    def get_alert_summary(self) -> Dict:
        """アラート要約を取得"""
        total_alerts = len(self.alert_history)
        critical_alerts = len([a for a in self.alert_history if a["type"] == "critical"])
        warning_alerts = len([a for a in self.alert_history if a["type"] == "warning"])

        return {
            "total_alerts": total_alerts,
            "critical_alerts": critical_alerts,
            "warning_alerts": warning_alerts,
            "last_alert": self.alert_history[-1] if self.alert_history else None
        }


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="ATFT-GAT-FAN Alert System")
    parser.add_argument("--test", action="store_true", help="Send test alert")
    parser.add_argument("--summary", action="store_true", help="Show alert summary")

    args = parser.parse_args()

    alert_system = AlertSystem()

    if args.test:
        # テストアラート送信
        alert_system.send_alert(
            "warning",
            "Test Alert",
            "This is a test alert from the monitoring system",
            {"test": True, "timestamp": datetime.now().isoformat()}
        )
        print("✅ Test alert sent")

    elif args.summary:
        # アラート要約表示
        summary = alert_system.get_alert_summary()
        print("\\n📊 ALERT SUMMARY")
        print(f"Total Alerts: {summary['total_alerts']}")
        print(f"Critical: {summary['critical_alerts']}")
        print(f"Warning: {summary['warning_alerts']}")
        if summary['last_alert']:
            print(f"Last Alert: {summary['last_alert']['title']} ({summary['last_alert']['timestamp']})")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
'''

        with open(alert_script, 'w', encoding='utf-8') as f:
            f.write(alert_code)

        # 実行権限付与
        alert_script.chmod(0o755)

        logger.info(f"✅ Alert system created: {alert_script}")
        return alert_script

    def setup_systemd_service(self):
        """SystemDサービスを作成（オプション）"""
        logger.info("Setting up systemd service for monitoring...")

        service_content = f'''[Unit]
Description=ATFT-GAT-FAN Monitoring Dashboard
After=network.target

[Service]
Type=simple
User={os.getenv('USER', 'ubuntu')}
WorkingDirectory={self.project_root}
ExecStart={sys.executable} {self.project_root}/scripts/monitoring_dashboard.py --continuous --interval 300
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
'''

        service_file = Path("/tmp/atft-monitoring.service")
        with open(service_file, 'w', encoding='utf-8') as f:
            f.write(service_content)

        logger.info(f"✅ SystemD service template created: {service_file}")
        logger.info("To install: sudo cp /tmp/atft-monitoring.service /etc/systemd/system/")
        logger.info("To enable: sudo systemctl enable atft-monitoring")
        logger.info("To start: sudo systemctl start atft-monitoring")

        return service_file


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="ATFT-GAT-FAN Monitoring Setup")
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install monitoring dependencies"
    )
    parser.add_argument(
        "--setup-wandb",
        action="store_true",
        help="Setup Weights & Biases"
    )
    parser.add_argument(
        "--wandb-key",
        type=str,
        help="W&B API key"
    )
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create monitoring configuration"
    )
    parser.add_argument(
        "--create-alerts",
        action="store_true",
        help="Create alert system"
    )
    parser.add_argument(
        "--systemd-service",
        action="store_true",
        help="Create systemd service"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Setup everything"
    )

    args = parser.parse_args()

    setup = MonitoringSetup()

    if args.all or args.install_deps:
        setup.install_monitoring_dependencies()

    if args.all or args.setup_wandb:
        api_key = args.wandb_key or os.getenv('WANDB_API_KEY')
        setup.setup_wandb(api_key)

    if args.all or args.create_config:
        setup.create_monitoring_config()

    if args.all or args.create_alerts:
        setup.create_alert_system()

    if args.systemd_service:
        setup.setup_systemd_service()

    if not any([args.install_deps, args.setup_wandb, args.create_config,
                args.create_alerts, args.systemd_service, args.all]):
        parser.print_help()
        print("\\n" + "="*60)
        print("QUICK SETUP COMMANDS:")
        print("="*60)
        print("# Install all dependencies")
        print("python scripts/setup_monitoring.py --install-deps")
        print()
        print("# Setup W&B (set API key first)")
        print("export WANDB_API_KEY='your-api-key'")
        print("python scripts/setup_monitoring.py --setup-wandb")
        print()
        print("# Create monitoring configuration")
        print("python scripts/setup_monitoring.py --create-config")
        print()
        print("# Setup everything")
        print("python scripts/setup_monitoring.py --all")


if __name__ == "__main__":
    main()
