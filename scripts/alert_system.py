#!/usr/bin/env python3
"""
ATFT-GAT-FAN Alert System
監視アラート通知システム
"""

import json
import logging
import smtplib
import sys
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

try:
    from slack_sdk import WebClient

    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ロギング設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AlertSystem:
    """アラートシステムクラス"""

    def __init__(self, config_path: str | None = None):
        self.config_path = config_path or (
            project_root / "configs" / "monitoring" / "monitoring.yaml"
        )
        self.config = self._load_config()
        self.alert_history = []

    def _load_config(self) -> dict:
        """設定ファイルを読み込み"""
        try:
            import yaml

            with open(self.config_path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            return {}

    def send_alert(
        self, alert_type: str, title: str, message: str, details: dict | None = None
    ):
        """アラートを送信"""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "type": alert_type,
            "title": title,
            "message": message,
            "details": details or {},
        }

        self.alert_history.append(alert)
        logger.warning(f"🚨 ALERT: {title} - {message}")

        # Slack通知
        if (
            self.config.get("monitoring", {})
            .get("notifications", {})
            .get("slack", {})
            .get("enabled", False)
        ):
            self._send_slack_alert(alert)

        # メール通知
        if (
            self.config.get("monitoring", {})
            .get("notifications", {})
            .get("email", {})
            .get("enabled", False)
        ):
            self._send_email_alert(alert)

        # アラート履歴を保存（最新100件）
        self._save_alert_history()

    def _send_slack_alert(self, alert: dict):
        """Slackにアラートを送信"""
        if not SLACK_AVAILABLE:
            logger.warning("Slack SDK not available")
            return

        try:
            slack_config = self.config["monitoring"]["notifications"]["slack"]
            client = WebClient(token=slack_config["webhook_url"])  # Webhook URLを使用

            icon = "🚨" if alert["type"] == "critical" else "⚠️"
            message = f"{icon} *{alert['title']}*\n{alert['message']}"

            if alert["details"]:
                details_text = "\n".join(
                    [f"• {k}: {v}" for k, v in alert["details"].items()]
                )
                message += f"\n\n{details_text}"

            client.chat_postMessage(
                channel=slack_config["channel"],
                text=message,
                username="ATFT-GAT-FAN Monitor",
            )

            logger.info("Slack alert sent successfully")

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

    def _send_email_alert(self, alert: dict):
        """メールでアラートを送信"""
        try:
            email_config = self.config["monitoring"]["notifications"]["email"]
            smtp_server = email_config["smtp_server"]
            recipients = email_config["recipients"]

            msg = MIMEMultipart()
            msg["From"] = "monitor@atft-gat-fan.com"
            msg["To"] = ", ".join(recipients)
            msg["Subject"] = f"ATFT-GAT-FAN Alert: {alert['title']}"

            body = f"""
ATFT-GAT-FAN Monitoring Alert

Type: {alert['type'].upper()}
Title: {alert['title']}
Message: {alert['message']}
Timestamp: {alert['timestamp']}

Details:
{json.dumps(alert['details'], indent=2, ensure_ascii=False) if alert['details'] else 'None'}
            """

            msg.attach(MIMEText(body, "plain"))

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

            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(recent_alerts, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Failed to save alert history: {e}")

    def get_alert_summary(self) -> dict:
        """アラート要約を取得"""
        total_alerts = len(self.alert_history)
        critical_alerts = len(
            [a for a in self.alert_history if a["type"] == "critical"]
        )
        warning_alerts = len([a for a in self.alert_history if a["type"] == "warning"])

        return {
            "total_alerts": total_alerts,
            "critical_alerts": critical_alerts,
            "warning_alerts": warning_alerts,
            "last_alert": self.alert_history[-1] if self.alert_history else None,
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
            {"test": True, "timestamp": datetime.now().isoformat()},
        )
        print("✅ Test alert sent")

    elif args.summary:
        # アラート要約表示
        summary = alert_system.get_alert_summary()
        print("\n📊 ALERT SUMMARY")
        print(f"Total Alerts: {summary['total_alerts']}")
        print(f"Critical: {summary['critical_alerts']}")
        print(f"Warning: {summary['warning_alerts']}")
        if summary["last_alert"]:
            print(
                f"Last Alert: {summary['last_alert']['title']} ({summary['last_alert']['timestamp']})"
            )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
