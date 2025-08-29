#!/usr/bin/env python3
"""
Gogooku3 ログ移設ツール

散在しているログファイルを統一ディレクトリ構造 (_logs/) に安全に移設する
既存ログファイルの形式変換・メタデータ付与・JST正規化も実行
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import gzip

# JST timezone
JST = timezone(timedelta(hours=9))

class LogMigrator:
    """ログ移設・変換クラス"""
    
    def __init__(self, repo_root: Path, dry_run: bool = False):
        self.repo_root = Path(repo_root)
        self.dry_run = dry_run
        self.hostname = socket.gethostname()
        
        # Git SHA取得
        try:
            self.git_sha = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except:
            self.git_sha = "unknown"
        
        # 統計情報
        self.stats = {
            'files_found': 0,
            'files_migrated': 0,
            'files_converted': 0,
            'files_skipped': 0,
            'errors': []
        }
        
        # ログ設定
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def find_log_files(self) -> List[Path]:
        """散在ログファイルを検索"""
        log_patterns = ['*.log', '*.out', '*.err']
        exclude_dirs = {'.git', '_logs', '__pycache__', '.vscode', '.idea', 'node_modules'}
        
        found_files = []
        
        for pattern in log_patterns:
            for path in self.repo_root.rglob(pattern):
                # 除外ディレクトリのスキップ
                if any(part in exclude_dirs for part in path.parts):
                    continue
                
                found_files.append(path)
                self.stats['files_found'] += 1
        
        return sorted(found_files)
    
    def classify_service(self, file_path: Path) -> str:
        """ファイルパスからサービスを推定"""
        path_str = str(file_path).lower()
        
        if any(x in path_str for x in ['dagster', 'orchestration']):
            return 'dagster'
        elif any(x in path_str for x in ['mlflow', 'ml_training', 'atft']):
            return 'mlflow'
        elif any(x in path_str for x in ['feast', 'feature_store']):
            return 'feast'
        elif any(x in path_str for x in ['clickhouse', 'db']):
            return 'clickhouse'
        elif any(x in path_str for x in ['redis', 'cache']):
            return 'redis'
        elif any(x in path_str for x in ['postgres', 'sql']):
            return 'postgres'
        elif 'docker' in path_str:
            return 'docker'
        else:
            return 'app'
    
    def get_file_date(self, file_path: Path) -> datetime:
        """ファイルの日付を取得（名前から推定 or mtime使用）"""
        
        # ファイル名から日付抽出を試行
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',          # YYYY-MM-DD
            r'(\d{8})',                       # YYYYMMDD
            r'(\d{4})(\d{2})(\d{2})',        # YYYYMMDD (分割)
        ]
        
        filename = file_path.name
        for pattern in date_patterns:
            match = re.search(pattern, filename)
            if match:
                try:
                    if len(match.groups()) == 1:
                        date_str = match.group(1)
                        if len(date_str) == 8:  # YYYYMMDD
                            return datetime.strptime(date_str, '%Y%m%d')
                        else:  # YYYY-MM-DD
                            return datetime.strptime(date_str, '%Y-%m-%d')
                    else:  # (YYYY)(MM)(DD)
                        year, month, day = match.groups()
                        return datetime(int(year), int(month), int(day))
                except ValueError:
                    continue
        
        # ファイル更新日時を使用
        return datetime.fromtimestamp(file_path.stat().st_mtime)
    
    def create_target_path(self, file_path: Path, service: str, file_date: datetime) -> Path:
        """移設先パスを生成"""
        env = 'dev'  # デフォルト環境
        
        target_dir = (
            self.repo_root / '_logs' / env / service / 
            f"{file_date.year}" / f"{file_date.month:02d}" / f"{file_date.day:02d}"
        )
        
        # ファイル名: hostname_service_original.jsonl.gz
        original_stem = file_path.stem
        if file_path.suffix in ['.log', '.out', '.err']:
            new_name = f"{self.hostname}_{service}_{original_stem}.jsonl.gz"
        else:
            new_name = f"{self.hostname}_{service}_{file_path.name}.gz"
            
        return target_dir / new_name
    
    def convert_to_jsonl(self, source_path: Path, target_path: Path, service: str) -> bool:
        """プレーンテキストログをJSON Lines形式に変換"""
        
        try:
            with open(source_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            with gzip.open(target_path, 'wt', encoding='utf-8') as f:
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # ログレベル推定
                    level = 'INFO'
                    if any(keyword in line.upper() for keyword in ['ERROR', 'CRITICAL', 'FATAL']):
                        level = 'ERROR'
                    elif any(keyword in line.upper() for keyword in ['WARN', 'WARNING']):
                        level = 'WARNING'
                    elif any(keyword in line.upper() for keyword in ['DEBUG']):
                        level = 'DEBUG'
                    
                    # タイムスタンプ抽出試行
                    timestamp = datetime.now(JST).isoformat()
                    ts_patterns = [
                        r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})',
                        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})',
                    ]
                    
                    for pattern in ts_patterns:
                        match = re.search(pattern, line)
                        if match:
                            try:
                                dt = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
                                timestamp = dt.replace(tzinfo=JST).isoformat()
                                break
                            except ValueError:
                                try:
                                    dt = datetime.strptime(match.group(1), '%Y-%m-%dT%H:%M:%S')
                                    timestamp = dt.replace(tzinfo=JST).isoformat()
                                    break
                                except ValueError:
                                    continue
                    
                    # JSON Lines エントリ作成
                    log_entry = {
                        "ts": timestamp,
                        "lvl": level,
                        "msg": line,
                        "svc": service,
                        "mod": "migrated",
                        "file": source_path.name,
                        "line": line_num,
                        "host": self.hostname,
                        "git_sha": self.git_sha,
                        "migrated_from": str(source_path.relative_to(self.repo_root)),
                        "migration_ts": datetime.now(JST).isoformat()
                    }
                    
                    f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
            return True
            
        except Exception as e:
            self.stats['errors'].append(f"Conversion failed for {source_path}: {e}")
            self.logger.error(f"Failed to convert {source_path}: {e}")
            return False
    
    def copy_binary_file(self, source_path: Path, target_path: Path) -> bool:
        """バイナリファイル（JSON等）のコピー・圧縮"""
        
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(source_path, 'rb') as f_in:
                with gzip.open(target_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            return True
            
        except Exception as e:
            self.stats['errors'].append(f"Copy failed for {source_path}: {e}")
            self.logger.error(f"Failed to copy {source_path}: {e}")
            return False
    
    def migrate_file(self, source_path: Path) -> bool:
        """単一ファイルの移設処理"""
        
        # サービス分類
        service = self.classify_service(source_path)
        
        # 日付取得
        file_date = self.get_file_date(source_path)
        
        # 移設先パス生成
        target_path = self.create_target_path(source_path, service, file_date)
        
        self.logger.info(f"Migrating: {source_path} -> {target_path}")
        
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would migrate {source_path} to {target_path}")
            return True
        
        # 既存ファイルのスキップ
        if target_path.exists():
            self.logger.warning(f"Target exists, skipping: {target_path}")
            self.stats['files_skipped'] += 1
            return False
        
        # ファイル形式に応じた処理
        if source_path.suffix in ['.log', '.out', '.err'] and source_path.stat().st_size > 0:
            # テキストログの変換
            success = self.convert_to_jsonl(source_path, target_path, service)
            if success:
                self.stats['files_converted'] += 1
        else:
            # その他ファイルのコピー・圧縮
            success = self.copy_binary_file(source_path, target_path)
        
        if success:
            self.stats['files_migrated'] += 1
            
            # 元ファイルのバックアップ移動（削除はしない）
            backup_dir = self.repo_root / '_logs_backup'
            backup_dir.mkdir(exist_ok=True)
            
            backup_path = backup_dir / source_path.name
            if not backup_path.exists():
                shutil.copy2(source_path, backup_path)
            
        return success
    
    def migrate_all(self) -> Dict:
        """全ログファイルの移設実行"""
        
        self.logger.info("🚀 Starting log migration...")
        self.logger.info(f"Repository: {self.repo_root}")
        self.logger.info(f"Dry run: {self.dry_run}")
        
        # ログファイル検索
        log_files = self.find_log_files()
        self.logger.info(f"Found {len(log_files)} log files")
        
        if not log_files:
            self.logger.info("No log files found to migrate")
            return self.stats
        
        # 移設実行
        for file_path in log_files:
            try:
                self.migrate_file(file_path)
            except Exception as e:
                error_msg = f"Migration error for {file_path}: {e}"
                self.stats['errors'].append(error_msg)
                self.logger.error(error_msg)
        
        # レポート生成
        self.generate_report()
        
        return self.stats
    
    def generate_report(self):
        """移設レポート生成"""
        
        report_path = self.repo_root / 'logs_migration_report.md'
        
        report_content = f"""# Gogooku3 ログ移設レポート

**実行日時**: {datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S JST')}  
**実行モード**: {'DRY RUN' if self.dry_run else 'ACTUAL MIGRATION'}  
**Git SHA**: {self.git_sha}

## 📊 移設統計

- **発見ファイル数**: {self.stats['files_found']}
- **移設済みファイル数**: {self.stats['files_migrated']}
- **変換済みファイル数**: {self.stats['files_converted']}
- **スキップファイル数**: {self.stats['files_skipped']}
- **エラー数**: {len(self.stats['errors'])}

## 📁 移設先構造

```
_logs/
├── dev/
│   ├── app/           # Python scripts
│   ├── dagster/       # Orchestration
│   ├── mlflow/        # ML lifecycle
│   ├── feast/         # Feature store
│   ├── clickhouse/    # OLAP database
│   ├── redis/         # Cache
│   ├── postgres/      # RDBMS
│   └── docker/        # Container logs
└── prd/               # Production (same structure)
```

## 🔧 変換仕様

- **形式**: Plain text → JSON Lines (gzip圧縮)
- **タイムゾーン**: JST (Asia/Tokyo)
- **メタデータ**: git_sha, hostname, migration_timestamp 付与
- **命名**: `{{hostname}}_{{service}}_{{original}}.jsonl.gz`

## ❌ エラー一覧

"""

        if self.stats['errors']:
            for i, error in enumerate(self.stats['errors'], 1):
                report_content += f"{i}. {error}\n"
        else:
            report_content += "エラーはありませんでした。\n"

        report_content += f"""

## 🎯 次のステップ

1. **動作確認**: 移設されたログファイルの内容確認
2. **統一ロガー適用**: scripts/*.py への setup_gogooku_logger 適用
3. **元ファイル削除**: バックアップ確認後の cleanup
4. **Docker設定**: docker-compose.yml のログローテーション設定

## 📞 トラブル時

- **ロールバック**: `_logs_backup/` からの復旧
- **再実行**: `python tools/logs_migrate.py --force`
- **確認**: `find _logs -name "*.jsonl.gz" | head -10`

---

*このレポートは tools/logs_migrate.py により自動生成されました*
"""
        
        if not self.dry_run:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            self.logger.info(f"Report saved: {report_path}")
        else:
            self.logger.info("Report content (dry run):")
            print(report_content)


def main():
    parser = argparse.ArgumentParser(description="Gogooku3 ログファイル移設ツール")
    parser.add_argument(
        '--repo-root', 
        type=Path, 
        default=Path(__file__).parent.parent,
        help="リポジトリルートパス"
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help="実際の移設をせず、変更予定を表示"
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="既存の移設先ファイルを上書き"
    )
    
    args = parser.parse_args()
    
    if not args.repo_root.exists():
        print(f"❌ Repository root not found: {args.repo_root}")
        return 1
    
    migrator = LogMigrator(args.repo_root, dry_run=args.dry_run)
    stats = migrator.migrate_all()
    
    print("\n🎯 Migration Summary:")
    print(f"  Files found: {stats['files_found']}")
    print(f"  Files migrated: {stats['files_migrated']}")
    print(f"  Files converted: {stats['files_converted']}")
    print(f"  Files skipped: {stats['files_skipped']}")
    print(f"  Errors: {len(stats['errors'])}")
    
    if stats['errors']:
        print("\n❌ Errors occurred:")
        for error in stats['errors'][:5]:  # Show first 5 errors
            print(f"  - {error}")
        if len(stats['errors']) > 5:
            print(f"  ... and {len(stats['errors']) - 5} more")
    
    return 0 if len(stats['errors']) == 0 else 1


if __name__ == "__main__":
    exit(main())