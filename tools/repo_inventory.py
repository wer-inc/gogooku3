#!/usr/bin/env python3
"""
Gogooku3 リポジトリ在庫調査ツール

リポジトリ内の全ファイルを走査し、詳細なメタデータと参照関係を調査
クリーンアップ計画の基礎資料として repo_inventory.csv を生成
"""

import argparse
import csv
import hashlib
import mimetypes
import os
import re
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import logging
import json

# JST timezone
JST = timezone(timedelta(hours=9))

class RepoInventory:
    """リポジトリ在庫調査クラス"""
    
    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        self.git_root = self._find_git_root()
        
        # 除外パターン
        self.exclude_dirs = {
            '.git', '_logs', '.venv', 'venv', '__pycache__', 
            'node_modules', 'mlruns', '.pytest_cache', '.mypy_cache',
            '.idea', '.vscode', 'output/batch', '.temp'
        }
        
        self.exclude_patterns = [
            r'.*\.pyc$', r'.*\.pyo$', r'.*\.pyd$',
            r'.*/__pycache__/.*', r'.*\.egg-info/.*',
            r'.*\.DS_Store$', r'.*\.tmp$', r'.*\.temp$'
        ]
        
        # 生成物パターン
        self.generated_patterns = [
            r'^output/', r'^reports/', r'^results/',
            r'.*\.(csv|parquet|html|png|jpg|pdf|pkl|joblib)$',
            r'.*_report\..*', r'.*_results\..*', r'.*_output\..*'
        ]
        
        # レポートパターン
        self.report_patterns = [
            r'.*report.*', r'.*summary.*', r'.*analysis.*',
            r'.*結果.*', r'.*検証.*', r'.*評価.*'
        ]
        
        # 秘密ファイルパターン
        self.secret_patterns = [
            r'.*\.env$', r'.*\.key$', r'.*\.pem$',
            r'.*credentials.*', r'.*secret.*', r'.*password.*',
            r'.*id_rsa.*', r'.*service-account.*\.json$'
        ]
        
        # 参照カウンター
        self.reference_map: Dict[str, int] = {}
        
        # ログ設定
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def _find_git_root(self) -> Optional[Path]:
        """Gitルートディレクトリを検索"""
        current = self.repo_root
        while current != current.parent:
            if (current / '.git').exists():
                return current
            current = current.parent
        return None

    def _is_excluded_path(self, path: Path) -> bool:
        """除外パス判定"""
        # 除外ディレクトリ
        if any(exclude_dir in path.parts for exclude_dir in self.exclude_dirs):
            return True
            
        # 除外パターン
        path_str = str(path.relative_to(self.repo_root))
        return any(re.match(pattern, path_str) for pattern in self.exclude_patterns)

    def _get_file_hash(self, file_path: Path) -> Optional[str]:
        """ファイルのSHA256ハッシュを取得"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return None

    def _is_binary_file(self, file_path: Path) -> bool:
        """バイナリファイル判定"""
        try:
            # mimetypeによる判定
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type and not mime_type.startswith('text/'):
                return True
                
            # バイト列による判定（最初の1024バイトをチェック）
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                if b'\0' in chunk:
                    return True
                    
            return False
        except Exception:
            return True  # 読めない場合はバイナリと判定

    def _guess_file_type(self, file_path: Path, content_sample: str = "") -> Dict[str, bool]:
        """ファイルタイプ推定"""
        path_str = str(file_path.relative_to(self.repo_root))
        filename = file_path.name.lower()
        
        return {
            'binary_guess': self._is_binary_file(file_path),
            'generated_guess': any(re.match(pattern, path_str) for pattern in self.generated_patterns),
            'notebook_guess': file_path.suffix == '.ipynb',
            'report_guess': any(re.search(pattern, filename) for pattern in self.report_patterns),
            'secret_guess': any(re.match(pattern, path_str) for pattern in self.secret_patterns)
        }

    def _get_git_info(self, file_path: Path) -> Dict[str, Optional[str]]:
        """Gitメタデータ取得"""
        if not self.git_root:
            return {
                'git_tracked': False,
                'git_lfs': False,
                'last_commit_date': None,
                'author': None,
                'first_seen_commit': None
            }
            
        try:
            relative_path = file_path.relative_to(self.git_root)
            
            # Git追跡状況
            result = subprocess.run(
                ['git', 'ls-files', '--', str(relative_path)],
                cwd=self.git_root,
                capture_output=True,
                text=True
            )
            git_tracked = bool(result.stdout.strip())
            
            # Git LFS確認
            if git_tracked:
                lfs_result = subprocess.run(
                    ['git', 'lfs', 'ls-files', '--name-only'],
                    cwd=self.git_root,
                    capture_output=True,
                    text=True
                )
                git_lfs = str(relative_path) in lfs_result.stdout
            else:
                git_lfs = False
            
            # 最終コミット情報
            last_commit_date = None
            author = None
            first_seen_commit = None
            
            if git_tracked:
                # 最終コミット
                log_result = subprocess.run([
                    'git', 'log', '-1', '--format=%ci|%an', '--', str(relative_path)
                ], cwd=self.git_root, capture_output=True, text=True)
                
                if log_result.stdout.strip():
                    parts = log_result.stdout.strip().split('|')
                    last_commit_date = parts[0] if len(parts) > 0 else None
                    author = parts[1] if len(parts) > 1 else None
                
                # 初回コミット
                first_result = subprocess.run([
                    'git', 'log', '--format=%H', '--diff-filter=A', '--', str(relative_path)
                ], cwd=self.git_root, capture_output=True, text=True)
                
                commits = first_result.stdout.strip().split('\n')
                first_seen_commit = commits[-1] if commits and commits[0] else None
            
            return {
                'git_tracked': git_tracked,
                'git_lfs': git_lfs,
                'last_commit_date': last_commit_date,
                'author': author,
                'first_seen_commit': first_seen_commit
            }
            
        except Exception as e:
            self.logger.warning(f"Git info error for {file_path}: {e}")
            return {
                'git_tracked': False,
                'git_lfs': False,
                'last_commit_date': None,
                'author': None,
                'first_seen_commit': None
            }

    def _analyze_references(self):
        """参照解析（import、相対パス、CLI呼び出し）"""
        self.logger.info("Analyzing file references...")
        
        # Python imports
        for py_file in self.repo_root.rglob("*.py"):
            if self._is_excluded_path(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                # import文の解析
                import_patterns = [
                    r'from\s+([\w.]+)\s+import',
                    r'import\s+([\w.]+)',
                    r'importlib\.import_module\(["\']([^"\']+)["\']'
                ]
                
                for pattern in import_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        # モジュールパスをファイルパスに変換
                        possible_paths = [
                            f"{match.replace('.', '/')}.py",
                            f"{match.replace('.', '/')}/__init__.py"
                        ]
                        for path in possible_paths:
                            if path in self.reference_map:
                                self.reference_map[path] += 1
                            else:
                                self.reference_map[path] = 1
                
                # ファイルパス参照
                path_patterns = [
                    r'["\']([^"\']+\.(py|yaml|json|csv|md))["\']',
                    r'Path\(["\']([^"\']+)["\']',
                    r'open\(["\']([^"\']+)["\']'
                ]
                
                for pattern in path_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        ref_path = match[0] if isinstance(match, tuple) else match
                        if ref_path in self.reference_map:
                            self.reference_map[ref_path] += 1
                        else:
                            self.reference_map[ref_path] = 1
                            
            except Exception as e:
                self.logger.warning(f"Reference analysis error for {py_file}: {e}")
        
        # YAML/JSON設定ファイル
        for config_file in list(self.repo_root.rglob("*.yaml")) + list(self.repo_root.rglob("*.yml")) + list(self.repo_root.rglob("*.json")):
            if self._is_excluded_path(config_file):
                continue
                
            try:
                with open(config_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    # パス参照を検索
                    matches = re.findall(r'["\']([^"\']+\.(py|sh|sql|md))["\']', content)
                    for match in matches:
                        ref_path = match[0] if isinstance(match, tuple) else match
                        if ref_path in self.reference_map:
                            self.reference_map[ref_path] += 1
                        else:
                            self.reference_map[ref_path] = 1
            except Exception:
                pass

    def _scan_files(self) -> List[Dict]:
        """ファイル走査とメタデータ収集"""
        self.logger.info(f"Scanning repository: {self.repo_root}")
        
        files_data = []
        total_files = 0
        
        for file_path in self.repo_root.rglob("*"):
            if not file_path.is_file():
                continue
                
            if self._is_excluded_path(file_path):
                continue
                
            total_files += 1
            if total_files % 100 == 0:
                self.logger.info(f"Processed {total_files} files...")
            
            try:
                stat = file_path.stat()
                relative_path = str(file_path.relative_to(self.repo_root))
                
                # 基本情報
                file_info = {
                    'path': relative_path,
                    'ext': file_path.suffix,
                    'size_bytes': stat.st_size,
                    'mtime': datetime.fromtimestamp(stat.st_mtime, JST).isoformat(),
                    'sha256': self._get_file_hash(file_path),
                    'referenced_by_count': self.reference_map.get(relative_path, 0)
                }
                
                # Git情報
                git_info = self._get_git_info(file_path)
                file_info.update(git_info)
                
                # ファイルタイプ推定
                type_info = self._guess_file_type(file_path)
                file_info.update(type_info)
                
                files_data.append(file_info)
                
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                
        self.logger.info(f"Scan complete: {len(files_data)} files processed")
        return files_data

    def generate_inventory(self, output_path: Path) -> Dict:
        """在庫調査実行"""
        start_time = datetime.now(JST)
        
        self.logger.info("Starting repository inventory...")
        
        # 1. 参照解析
        self._analyze_references()
        
        # 2. ファイル走査
        files_data = self._scan_files()
        
        # 3. CSV出力
        fieldnames = [
            'path', 'ext', 'size_bytes', 'mtime', 'sha256', 
            'git_tracked', 'git_lfs', 'last_commit_date', 'author', 
            'referenced_by_count', 'first_seen_commit', 
            'binary_guess', 'generated_guess', 'notebook_guess', 
            'report_guess', 'secret_guess'
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(files_data)
        
        # 4. 統計情報
        end_time = datetime.now(JST)
        stats = {
            'total_files': len(files_data),
            'total_size_bytes': sum(f['size_bytes'] for f in files_data),
            'git_tracked_files': sum(1 for f in files_data if f['git_tracked']),
            'git_lfs_files': sum(1 for f in files_data if f['git_lfs']),
            'binary_files': sum(1 for f in files_data if f['binary_guess']),
            'generated_files': sum(1 for f in files_data if f['generated_guess']),
            'notebook_files': sum(1 for f in files_data if f['notebook_guess']),
            'report_files': sum(1 for f in files_data if f['report_guess']),
            'secret_files': sum(1 for f in files_data if f['secret_guess']),
            'large_files': sum(1 for f in files_data if f['size_bytes'] > 50_000_000),
            'referenced_files': sum(1 for f in files_data if f['referenced_by_count'] > 0),
            'scan_duration': str(end_time - start_time),
            'scan_timestamp': end_time.isoformat()
        }
        
        self.logger.info(f"Inventory saved: {output_path}")
        self.logger.info(f"Total files: {stats['total_files']}")
        self.logger.info(f"Total size: {stats['total_size_bytes'] / 1024 / 1024:.1f} MB")
        
        return stats

def main():
    parser = argparse.ArgumentParser(description="Repository inventory tool")
    parser.add_argument('--repo-root', type=Path, default=Path('.'), help='Repository root path')
    parser.add_argument('--output', type=Path, default='repo_inventory.csv', help='Output CSV file')
    parser.add_argument('--stats-output', type=Path, default='inventory_stats.json', help='Stats JSON file')
    
    args = parser.parse_args()
    
    inventory = RepoInventory(args.repo_root)
    stats = inventory.generate_inventory(args.output)
    
    # 統計情報をJSONで出力
    with open(args.stats_output, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 Repository Inventory Summary:")
    print(f"  Files scanned: {stats['total_files']}")
    print(f"  Total size: {stats['total_size_bytes'] / 1024 / 1024:.1f} MB")
    print(f"  Git tracked: {stats['git_tracked_files']}")
    print(f"  Large files (>50MB): {stats['large_files']}")
    print(f"  Generated files: {stats['generated_files']}")
    print(f"  Report files: {stats['report_files']}")
    print(f"  Secret files: {stats['secret_files']}")
    print(f"  Referenced files: {stats['referenced_files']}")
    print(f"\n✅ Inventory saved: {args.output}")
    print(f"📈 Stats saved: {args.stats_output}")

if __name__ == "__main__":
    main()