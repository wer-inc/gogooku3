#!/usr/bin/env python3
"""
Gogooku3 ログアーカイブツール

_logs/ 配下の古いログファイルをMinIOにアーカイブし、
ローカルストレージを効率的に管理する
"""

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional
import logging
import subprocess
import tempfile

# JST timezone  
JST = timezone(timedelta(hours=9))

class LogArchiver:
    """ログアーカイブ管理クラス"""
    
    def __init__(
        self, 
        repo_root: Path, 
        minio_endpoint: str = "localhost:9000",
        minio_bucket: str = "gogooku",
        minio_access_key: str = "minioadmin",
        minio_secret_key: str = "minioadmin123",
        dry_run: bool = False
    ):
        self.repo_root = Path(repo_root)
        self.logs_dir = self.repo_root / "_logs"
        self.minio_endpoint = minio_endpoint
        self.minio_bucket = minio_bucket
        self.minio_access_key = minio_access_key
        self.minio_secret_key = minio_secret_key
        self.dry_run = dry_run
        
        # 統計情報
        self.stats = {
            'files_found': 0,
            'files_uploaded': 0,
            'files_deleted': 0,
            'bytes_uploaded': 0,
            'bytes_freed': 0,
            'errors': []
        }
        
        # ログ設定
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def check_minio_connection(self) -> bool:
        """MinIO接続確認"""
        try:
            # mc alias設定
            subprocess.run([
                "mc", "alias", "set", "gogooku3-local",
                f"http://{self.minio_endpoint}",
                self.minio_access_key,
                self.minio_secret_key
            ], check=True, capture_output=True)
            
            # バケット存在確認・作成
            result = subprocess.run([
                "mc", "ls", f"gogooku3-local/{self.minio_bucket}"
            ], capture_output=True)
            
            if result.returncode != 0:
                self.logger.info(f"Creating bucket: {self.minio_bucket}")
                subprocess.run([
                    "mc", "mb", f"gogooku3-local/{self.minio_bucket}"
                ], check=True)
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.stats['errors'].append(f"MinIO connection failed: {e}")
            self.logger.error(f"MinIO connection failed: {e}")
            return False
        except FileNotFoundError:
            self.stats['errors'].append("mc command not found. Please install MinIO Client.")
            self.logger.error("mc command not found. Please install MinIO Client.")
            return False
    
    def find_archivable_files(self, retention_days: int = 14) -> List[Path]:
        """アーカイブ対象ファイル検索"""
        if not self.logs_dir.exists():
            return []
        
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        archivable_files = []
        
        for file_path in self.logs_dir.rglob("*.jsonl.gz"):
            try:
                file_date = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_date < cutoff_date:
                    archivable_files.append(file_path)
                    self.stats['files_found'] += 1
            except OSError as e:
                self.logger.warning(f"Cannot access {file_path}: {e}")
        
        return sorted(archivable_files)
    
    def get_s3_path(self, local_path: Path) -> str:
        """ローカルパス → S3パス変換"""
        # _logs/dev/app/2025/08/29/hostname_app_file.jsonl.gz
        # → logs/dev/app/2025/08/29/hostname_app_file.jsonl.gz
        
        relative_path = local_path.relative_to(self.logs_dir)
        return f"logs/{relative_path}"
    
    def upload_file(self, local_path: Path) -> bool:
        """単一ファイルのアップロード"""
        s3_path = self.get_s3_path(local_path)
        s3_url = f"gogooku3-local/{self.minio_bucket}/{s3_path}"
        
        self.logger.info(f"Uploading: {local_path} -> {s3_url}")
        
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would upload {local_path}")
            return True
        
        try:
            subprocess.run([
                "mc", "cp", str(local_path), s3_url
            ], check=True, capture_output=True)
            
            # アップロード確認
            result = subprocess.run([
                "mc", "stat", s3_url
            ], capture_output=True)
            
            if result.returncode == 0:
                file_size = local_path.stat().st_size
                self.stats['files_uploaded'] += 1
                self.stats['bytes_uploaded'] += file_size
                return True
            else:
                self.stats['errors'].append(f"Upload verification failed: {s3_url}")
                return False
                
        except subprocess.CalledProcessError as e:
            error_msg = f"Upload failed for {local_path}: {e}"
            self.stats['errors'].append(error_msg)
            self.logger.error(error_msg)
            return False
    
    def delete_local_file(self, local_path: Path) -> bool:
        """ローカルファイル削除"""
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would delete {local_path}")
            return True
        
        try:
            file_size = local_path.stat().st_size
            local_path.unlink()
            
            self.stats['files_deleted'] += 1
            self.stats['bytes_freed'] += file_size
            return True
            
        except OSError as e:
            error_msg = f"Delete failed for {local_path}: {e}"
            self.stats['errors'].append(error_msg)
            self.logger.error(error_msg)
            return False
    
    def archive_files(self, retention_days: int = 14) -> Dict:
        """ファイルアーカイブ実行"""
        
        self.logger.info("🚀 Starting log archival...")
        self.logger.info(f"Repository: {self.repo_root}")
        self.logger.info(f"MinIO endpoint: {self.minio_endpoint}")
        self.logger.info(f"Retention: {retention_days} days")
        self.logger.info(f"Dry run: {self.dry_run}")
        
        # MinIO接続確認
        if not self.check_minio_connection():
            return self.stats
        
        # アーカイブ対象ファイル検索
        archivable_files = self.find_archivable_files(retention_days)
        self.logger.info(f"Found {len(archivable_files)} files to archive")
        
        if not archivable_files:
            self.logger.info("No files to archive")
            return self.stats
        
        # アーカイブ実行
        for file_path in archivable_files:
            try:
                # アップロード
                if self.upload_file(file_path):
                    # ローカル削除
                    self.delete_local_file(file_path)
                    
            except Exception as e:
                error_msg = f"Archival error for {file_path}: {e}"
                self.stats['errors'].append(error_msg)
                self.logger.error(error_msg)
        
        # レポート生成
        self.generate_report()
        
        return self.stats
    
    def list_archived_files(self, days: int = 30) -> List[Dict]:
        """アーカイブファイル一覧表示"""
        try:
            result = subprocess.run([
                "mc", "ls", "--recursive", f"gogooku3-local/{self.minio_bucket}/logs/"
            ], capture_output=True, text=True, check=True)
            
            files = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    # Parse mc ls output: [date] [time] [size] [path]
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        date_str = parts[0]
                        time_str = parts[1]
                        size_str = parts[2] 
                        path = parts[3]
                        
                        files.append({
                            'date': date_str,
                            'time': time_str,
                            'size': size_str,
                            'path': path
                        })
            
            return files
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to list archived files: {e}")
            return []
    
    def restore_file(self, s3_path: str, local_path: Optional[Path] = None) -> bool:
        """アーカイブファイル復旧"""
        if local_path is None:
            # s3_path から local_path を推定
            # logs/dev/app/2025/08/29/file.jsonl.gz → _logs/dev/app/2025/08/29/file.jsonl.gz
            relative_path = s3_path.replace("logs/", "", 1)
            local_path = self.logs_dir / relative_path
        
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        s3_url = f"gogooku3-local/{self.minio_bucket}/{s3_path}"
        
        self.logger.info(f"Restoring: {s3_url} -> {local_path}")
        
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would restore {s3_path}")
            return True
        
        try:
            subprocess.run([
                "mc", "cp", s3_url, str(local_path)
            ], check=True)
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Restore failed: {e}")
            return False
    
    def generate_report(self):
        """アーカイブレポート生成"""
        
        report_path = self.repo_root / 'logs_archive_report.md'
        
        report_content = f"""# Gogooku3 ログアーカイブレポート

**実行日時**: {datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S JST')}  
**実行モード**: {'DRY RUN' if self.dry_run else 'ACTUAL ARCHIVAL'}  
**MinIO エンドポイント**: {self.minio_endpoint}
**バケット**: {self.minio_bucket}

## 📊 アーカイブ統計

- **対象ファイル数**: {self.stats['files_found']}
- **アップロード済み**: {self.stats['files_uploaded']}
- **削除済み**: {self.stats['files_deleted']}
- **アップロード容量**: {self.stats['bytes_uploaded'] / 1024 / 1024:.1f} MB
- **解放容量**: {self.stats['bytes_freed'] / 1024 / 1024:.1f} MB
- **エラー数**: {len(self.stats['errors'])}

## 🗂️ MinIOパス構造

```
s3://{self.minio_bucket}/
└── logs/
    ├── dev/
    │   ├── app/YYYY/MM/DD/
    │   ├── dagster/YYYY/MM/DD/
    │   └── ...
    └── prd/
        └── (同構造)
```

## 📋 復旧方法

```bash
# 特定ファイルの復旧
python tools/ship_logs_to_minio.py --restore "logs/dev/app/2025/08/29/hostname_app_file.jsonl.gz"

# アーカイブ一覧表示
python tools/ship_logs_to_minio.py --list

# MinIO Web UI確認
# http://localhost:9001 (minioadmin/minioadmin123)
```

## ❌ エラー一覧

"""

        if self.stats['errors']:
            for i, error in enumerate(self.stats['errors'], 1):
                report_content += f"{i}. {error}\n"
        else:
            report_content += "エラーはありませんでした。\n"

        report_content += """

## 🎯 運用推奨事項

1. **定期実行**: crontab で週次実行 (日曜 3:00AM 等)
2. **監視**: アーカイブ成功・失敗アラート設定
3. **容量監視**: MinIO使用容量・ローカル解放容量の監視
4. **復旧テスト**: 月次でランダムファイルの復旧テスト

## 🔧 トラブルシューティング

- **MinIO接続エラー**: docker-compose でMinIOサービス確認
- **mc command**: `brew install minio/stable/mc` (Mac) / `apt install minio-client` (Ubuntu)
- **権限エラー**: MinIOアクセスキー・シークレットキー確認
- **容量不足**: MinIO側のストレージ容量確認

---

*このレポートは tools/ship_logs_to_minio.py により自動生成されました*
"""
        
        if not self.dry_run:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            self.logger.info(f"Report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Gogooku3 ログアーカイブツール")
    parser.add_argument(
        '--repo-root', 
        type=Path, 
        default=Path(__file__).parent.parent,
        help="リポジトリルートパス"
    )
    parser.add_argument(
        '--minio-endpoint',
        default="localhost:9000",
        help="MinIOエンドポイント"
    )
    parser.add_argument(
        '--minio-bucket',
        default="gogooku",
        help="MinIOバケット名"
    )
    parser.add_argument(
        '--retention-days',
        type=int,
        default=14,
        help="ローカル保持日数（デフォルト: 14日）"
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help="実際のアーカイブをせず、動作予定を表示"
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help="アーカイブファイル一覧表示"
    )
    parser.add_argument(
        '--restore',
        type=str,
        help="指定ファイルをアーカイブから復旧"
    )
    
    args = parser.parse_args()
    
    if not args.repo_root.exists():
        print(f"❌ Repository root not found: {args.repo_root}")
        return 1
    
    archiver = LogArchiver(
        repo_root=args.repo_root,
        minio_endpoint=args.minio_endpoint,
        minio_bucket=args.minio_bucket,
        dry_run=args.dry_run
    )
    
    # コマンド分岐
    if args.list:
        print("📋 Archived files:")
        files = archiver.list_archived_files()
        for f in files[:20]:  # Show first 20 files
            print(f"  {f['date']} {f['time']} {f['size']:>10} {f['path']}")
        if len(files) > 20:
            print(f"  ... and {len(files) - 20} more files")
        return 0
        
    elif args.restore:
        success = archiver.restore_file(args.restore)
        if success:
            print(f"✅ Restored: {args.restore}")
            return 0
        else:
            print(f"❌ Restore failed: {args.restore}")
            return 1
            
    else:
        # アーカイブ実行
        stats = archiver.archive_files(args.retention_days)
        
        print("\n🎯 Archive Summary:")
        print(f"  Files found: {stats['files_found']}")
        print(f"  Files uploaded: {stats['files_uploaded']}")
        print(f"  Files deleted: {stats['files_deleted']}")
        print(f"  Bytes uploaded: {stats['bytes_uploaded'] / 1024 / 1024:.1f} MB")
        print(f"  Bytes freed: {stats['bytes_freed'] / 1024 / 1024:.1f} MB")
        print(f"  Errors: {len(stats['errors'])}")
        
        return 0 if len(stats['errors']) == 0 else 1


if __name__ == "__main__":
    exit(main())