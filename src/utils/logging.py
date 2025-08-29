"""
Gogooku3 統一ログ設定モジュール

JSON Lines形式・JST時刻・自動ローテーション対応の統一ロガー
全スクリプトでの一貫したログ出力を提供
"""

import logging
import logging.handlers
import os
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import json
import subprocess


class JSKFormatter(logging.Formatter):
    """JST時刻・JSON Lines形式のカスタムフォーマッター"""
    
    def __init__(self, service: str = "app", extra_fields: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.service = service
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self.extra_fields = extra_fields or {}
        
        # Git SHA取得（エラーでも継続）
        try:
            self.git_sha = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=Path(__file__).parent.parent.parent,
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except:
            self.git_sha = "unknown"
    
    def format(self, record: logging.LogRecord) -> str:
        """ログレコードをJSON Lines形式にフォーマット"""
        
        # JST時刻
        jst = timezone.utc.offset(datetime.fromtimestamp(record.created))
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        jst_time = dt.replace(tzinfo=timezone.utc).astimezone(
            timezone(offset=jst + timezone.utc.utcoffset(None) or timezone.utc.utcoffset(None))
        )
        
        # 基本フィールド
        log_entry = {
            "ts": jst_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+09:00",
            "lvl": record.levelname,
            "msg": record.getMessage(),
            "svc": self.service,
            "mod": record.module,
            "file": record.filename,
            "line": record.lineno,
            "pid": self.pid,
            "tid": threading.get_ident(),
            "host": self.hostname,
            "git_sha": self.git_sha
        }
        
        # 追加フィールド
        log_entry.update(self.extra_fields)
        
        # レコード固有の属性
        for key, value in record.__dict__.items():
            if key.startswith(('run_id', 'fold', 'horizon', 'ticker', 'seed', 'duration_ms', 'container_id')):
                log_entry[key] = value
        
        # 例外情報
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, ensure_ascii=False)


def setup_gogooku_logger(
    service: str = "app",
    level: str = "INFO", 
    log_dir: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True,
    max_bytes: int = 50 * 1024 * 1024,  # 50MB
    backup_count: int = 5,
    extra_fields: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """
    Gogooku3統一ロガーの設定
    
    Args:
        service: サービス名 (app, dagster, mlflow, etc.)
        level: ログレベル (DEBUG, INFO, WARNING, ERROR)
        log_dir: ログディレクトリ (None時は_logs/dev/{service}を使用)
        enable_console: コンソール出力の有効化
        enable_file: ファイル出力の有効化
        max_bytes: ファイルローテーションサイズ
        backup_count: バックアップファイル数
        extra_fields: 追加フィールド辞書
    
    Returns:
        設定済みLogger
    """
    
    logger_name = f"gogooku3.{service}"
    logger = logging.getLogger(logger_name)
    
    # 既存ハンドラークリア（重複回避）
    if logger.handlers:
        logger.handlers.clear()
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # フォーマッター作成
    json_formatter = JSKFormatter(service=service, extra_fields=extra_fields)
    console_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # ファイルハンドラー
    if enable_file:
        if log_dir is None:
            # デフォルト: _logs/dev/{service}/YYYY/MM/DD/
            base_dir = Path(__file__).parent.parent.parent / "_logs" / "dev" / service
            today = datetime.now()
            log_dir = base_dir / str(today.year) / f"{today.month:02d}" / f"{today.day:02d}"
        
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"{socket.gethostname()}_{service}.jsonl"
        
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(json_formatter)
        logger.addHandler(file_handler)
    
    # コンソールハンドラー
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # 親ロガーへの伝播無効化（重複回避）
    logger.propagate = False
    
    return logger


def get_ml_logger(
    run_id: Optional[str] = None,
    experiment_name: str = "default",
    **kwargs
) -> logging.Logger:
    """
    ML実験用の拡張ロガー
    
    Args:
        run_id: MLflow実行ID
        experiment_name: 実験名
        **kwargs: setup_gogooku_loggerへの追加引数
    
    Returns:
        ML実験用Logger
    """
    extra_fields = kwargs.pop('extra_fields', {})
    extra_fields.update({
        'run_id': run_id,
        'experiment': experiment_name
    })
    
    return setup_gogooku_logger(
        service="training",
        extra_fields=extra_fields,
        **kwargs
    )


def get_dagster_logger(**kwargs) -> logging.Logger:
    """Dagster用ロガー"""
    return setup_gogooku_logger(service="dagster", **kwargs)


def get_feast_logger(**kwargs) -> logging.Logger:
    """Feast Feature Store用ロガー"""
    return setup_gogooku_logger(service="feast", **kwargs)


# 後方互換性のためのヘルパー関数
def setup_logging(
    level: str = "INFO",
    service: str = "app",
    **kwargs
) -> logging.Logger:
    """
    後方互換性のための旧形式サポート
    
    Warning:
        この関数は非推奨です。setup_gogooku_loggerを直接使用してください。
    """
    import warnings
    warnings.warn(
        "setup_logging() is deprecated. Use setup_gogooku_logger() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return setup_gogooku_logger(service=service, level=level, **kwargs)


if __name__ == "__main__":
    # テスト実行
    logger = setup_gogooku_logger(service="test", level="DEBUG")
    
    logger.debug("デバッグメッセージ")
    logger.info("情報メッセージ", extra={
        'run_id': 'test_run_123',
        'fold': 1,
        'duration_ms': 1500
    })
    logger.warning("警告メッセージ")
    logger.error("エラーメッセージ")
    
    try:
        raise ValueError("テスト例外")
    except Exception:
        logger.exception("例外発生")
    
    print("✅ 統一ロガーのテストが完了しました")