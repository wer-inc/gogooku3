#!/usr/bin/env python3
"""
Gogooku3 ファイル分類・処理判定ツール

在庫データと重複情報を基に、各ファイルの処理方針を自動判定
DISPOSITION.csv として出力し、クリーンアップ計画の基礎とする
"""

import argparse
import csv
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Set
import logging

# JST timezone
JST = timezone(timedelta(hours=9))

class FileClassifier:
    """ファイル分類・処理判定クラス"""
    
    def __init__(self, inventory_path: Path, duplicates_path: Path = None):
        self.inventory_path = inventory_path
        self.duplicates_path = duplicates_path
        
        self.files_data = self._load_inventory()
        self.duplicates_data = self._load_duplicates() if duplicates_path else []
        
        # 分類スコア重み
        self.score_weights = {
            'referenced_by_count': 2.0,
            'recent_activity': 1.5,      # 180日以内の活動
            'tests_presence': 1.0,
            'dup_penalty': -2.0,
            'generated_penalty': -1.5
        }
        
        # 処理判定閾値
        self.thresholds = {
            'keep': 0.7,
            'move_min': 0.4,
            'archive_min': 0.2
        }
        
        # 重要ファイルパターン（必ず keep）
        self.critical_files = {
            'pyproject.toml', 'setup.py', 'setup.cfg', 'requirements.txt',
            'Makefile', 'docker-compose.yml', 'Dockerfile',
            '.gitignore', '.gitattributes', 'README.md', 'LICENSE',
            '.pre-commit-config.yaml', '.github/workflows'
        }
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')

    def _load_inventory(self) -> List[Dict]:
        """在庫データ読み込み"""
        files_data = []
        with open(self.inventory_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 数値・論理型変換
                row['size_bytes'] = int(row['size_bytes']) if row['size_bytes'].isdigit() else 0
                row['referenced_by_count'] = int(row['referenced_by_count']) if row['referenced_by_count'].isdigit() else 0
                row['git_tracked'] = row['git_tracked'].lower() == 'true'
                row['binary_guess'] = row['binary_guess'].lower() == 'true'
                row['generated_guess'] = row['generated_guess'].lower() == 'true'
                row['notebook_guess'] = row['notebook_guess'].lower() == 'true'
                row['report_guess'] = row['report_guess'].lower() == 'true'
                row['secret_guess'] = row['secret_guess'].lower() == 'true'
                files_data.append(row)
        return files_data

    def _load_duplicates(self) -> List[Dict]:
        """重複データ読み込み"""
        if not self.duplicates_path or not self.duplicates_path.exists():
            return []
            
        duplicates_data = []
        with open(self.duplicates_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            duplicates_data = list(reader)
        return duplicates_data

    def _calculate_keep_score(self, file_info: Dict) -> float:
        """ファイル保持スコア計算"""
        score = 0.0
        path = file_info['path']
        
        # 参照数
        score += file_info['referenced_by_count'] * self.score_weights['referenced_by_count']
        
        # 最近の活動（180日以内）
        if file_info.get('last_commit_date'):
            try:
                commit_date = datetime.fromisoformat(file_info['last_commit_date'].replace('Z', '+00:00'))
                days_ago = (datetime.now().astimezone() - commit_date).days
                if days_ago <= 180:
                    score += self.score_weights['recent_activity'] * (1.0 - days_ago / 180)
            except:
                pass
        
        # テストの存在
        if 'test' in path.lower() or '/tests/' in path:
            score += self.score_weights['tests_presence']
        
        # 重複ペナルティ
        is_duplicate_target = any(
            d['path_b'] == path and d['path_a'] != d['path_b'] 
            for d in self.duplicates_data
        )
        if is_duplicate_target:
            score += self.score_weights['dup_penalty']
        
        # 生成物ペナルティ
        if file_info['generated_guess']:
            score += self.score_weights['generated_penalty']
        
        return score

    def _classify_file(self, file_info: Dict) -> Dict[str, str]:
        """単一ファイルの処理方針決定"""
        path = file_info['path']
        score = self._calculate_keep_score(file_info)
        
        # 重要ファイルは強制 keep
        if any(critical in path for critical in self.critical_files):
            return {
                'action': 'keep',
                'target_path': path,
                'rationale': 'critical_system_file',
                'risk': 'low'
            }
        
        # 秘密ファイルは即座削除
        if file_info['secret_guess']:
            return {
                'action': 'delete',
                'target_path': '',
                'rationale': 'security_risk_secret_file',
                'risk': 'high'
            }
        
        # 大容量ファイルはLFS化
        if file_info['size_bytes'] > 50_000_000:  # 50MB
            return {
                'action': 'lfs',
                'target_path': path,
                'rationale': 'large_file_lfs_candidate',
                'risk': 'medium'
            }
        
        # 重複ファイル（非canonical）は統合
        duplicate_entry = next((
            d for d in self.duplicates_data 
            if d['path_b'] == path and d['path_a'] != d['path_b']
        ), None)
        
        if duplicate_entry:
            return {
                'action': 'consolidate',
                'target_path': duplicate_entry['path_a'],
                'rationale': f'duplicate_of_{duplicate_entry["path_a"]}',
                'risk': 'low'
            }
        
        # scripts/ のロジックはsrc/へ移設
        if (path.startswith('scripts/') and 
            path.endswith('.py') and 
            not path.endswith('__init__.py') and
            file_info['size_bytes'] > 1000):  # 1KB以上の実装
            target = f"src/gogooku3/{path.replace('scripts/', '').replace('.py', '.py')}"
            return {
                'action': 'move',
                'target_path': target,
                'rationale': 'logic_to_src_package',
                'risk': 'medium'
            }
        
        # ログファイルは _logs/ へ移設
        if re.search(r'\.(log|out|err)$', path):
            return {
                'action': 'move',
                'target_path': f'_logs/dev/app/{datetime.now(JST).strftime("%Y/%m/%d")}/{Path(path).name}',
                'rationale': 'log_file_to_unified_structure',
                'risk': 'low'
            }
        
        # レポート・生成物の判定
        if (file_info['report_guess'] or 
            file_info['generated_guess'] or
            path.startswith('output/') or
            path.startswith('reports/')):
            
            # 再生成可能か判定
            if re.search(r'\.(html|png|jpg|pdf)$', path) or 'output/' in path:
                return {
                    'action': 'regenerate',
                    'target_path': '',
                    'rationale': 'regenerable_output_file',
                    'risk': 'low'
                }
            else:
                return {
                    'action': 'archive',
                    'target_path': f'_archive/{datetime.now(JST).strftime("%Y%m")}/{path}',
                    'rationale': 'generated_content_archive',
                    'risk': 'low'
                }
        
        # ノートブックの変換
        if file_info['notebook_guess']:
            if file_info['size_bytes'] > 1_000_000:  # 1MB以上の重いノートブック
                return {
                    'action': 'convert',
                    'target_path': path.replace('.ipynb', '.py'),
                    'rationale': 'heavy_notebook_to_python',
                    'risk': 'medium'
                }
            else:
                return {
                    'action': 'keep',
                    'target_path': path,
                    'rationale': 'lightweight_notebook',
                    'risk': 'low'
                }
        
        # スコアによる最終判定
        if score >= self.thresholds['keep']:
            return {
                'action': 'keep',
                'target_path': path,
                'rationale': f'high_score_{score:.2f}',
                'risk': 'low'
            }
        elif score >= self.thresholds['move_min']:
            # 適切なディレクトリに移設
            if path.endswith('.md') and not path.startswith('docs/'):
                target = f"docs/{Path(path).name}"
            elif path.endswith('.py') and not any(path.startswith(p) for p in ['src/', 'tests/', 'scripts/']):
                target = f"src/gogooku3/{Path(path).name}"
            else:
                target = path
                
            return {
                'action': 'move',
                'target_path': target,
                'rationale': f'medium_score_relocate_{score:.2f}',
                'risk': 'medium'
            }
        elif score >= self.thresholds['archive_min']:
            return {
                'action': 'archive',
                'target_path': f'_archive/{datetime.now(JST).strftime("%Y%m")}/{path}',
                'rationale': f'low_score_archive_{score:.2f}',
                'risk': 'medium'
            }
        else:
            return {
                'action': 'delete',
                'target_path': '',
                'rationale': f'very_low_score_{score:.2f}',
                'risk': 'high'
            }

    def _determine_rollback_strategy(self, action: str, path: str) -> str:
        """ロールバック戦略決定"""
        if action in ['delete', 'consolidate']:
            return f'restore_from_git_or_archive'
        elif action in ['move', 'convert']:
            return f'reverse_operation'
        elif action == 'lfs':
            return 'git_lfs_unlock'
        elif action in ['archive', 'regenerate']:
            return 'manual_restore_if_needed'
        else:
            return 'no_rollback_needed'

    def generate_disposition(self, output_path: Path) -> Dict:
        """処理方針決定・DISPOSITION.csv生成"""
        self.logger.info(f"Classifying {len(self.files_data)} files...")
        
        dispositions = []
        action_counts = defaultdict(int)
        
        for file_info in self.files_data:
            classification = self._classify_file(file_info)
            rollback = self._determine_rollback_strategy(classification['action'], file_info['path'])
            
            disposition = {
                'path': file_info['path'],
                'action': classification['action'],
                'target_path': classification['target_path'],
                'rationale': classification['rationale'],
                'risk': classification['risk'],
                'rollback': rollback,
                'size_bytes': file_info['size_bytes'],
                'git_tracked': file_info['git_tracked'],
                'referenced_by_count': file_info['referenced_by_count']
            }
            
            dispositions.append(disposition)
            action_counts[classification['action']] += 1
        
        # CSV出力
        fieldnames = [
            'path', 'action', 'target_path', 'rationale', 'risk', 'rollback',
            'size_bytes', 'git_tracked', 'referenced_by_count'
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dispositions)
        
        # 統計情報
        stats = {
            'total_files': len(dispositions),
            'action_counts': dict(action_counts),
            'high_risk_actions': len([d for d in dispositions if d['risk'] == 'high']),
            'space_to_free_mb': sum(d['size_bytes'] for d in dispositions 
                                  if d['action'] in ['delete', 'archive', 'consolidate']) / 1024 / 1024,
            'files_to_relocate': len([d for d in dispositions if d['action'] == 'move']),
            'lfs_candidates': len([d for d in dispositions if d['action'] == 'lfs'])
        }
        
        self.logger.info(f"Classification complete: {output_path}")
        for action, count in action_counts.items():
            self.logger.info(f"  {action}: {count} files")
        
        return stats

def main():
    parser = argparse.ArgumentParser(description="Classify files for cleanup")
    parser.add_argument('--inventory', type=Path, default='repo_inventory.csv', help='Inventory CSV file')
    parser.add_argument('--duplicates', type=Path, default='duplicates.csv', help='Duplicates CSV file')
    parser.add_argument('--output', type=Path, default='DISPOSITION.csv', help='Output disposition CSV')
    
    args = parser.parse_args()
    
    classifier = FileClassifier(args.inventory, args.duplicates)
    stats = classifier.generate_disposition(args.output)
    
    print(f"\n📊 File Classification Summary:")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Actions breakdown:")
    for action, count in stats['action_counts'].items():
        print(f"    {action}: {count} files")
    print(f"  High-risk actions: {stats['high_risk_actions']} files")
    print(f"  Space to free: {stats['space_to_free_mb']:.1f} MB")
    print(f"  Files to relocate: {stats['files_to_relocate']}")
    print(f"  LFS candidates: {stats['lfs_candidates']}")
    print(f"\n✅ Disposition saved: {args.output}")

if __name__ == "__main__":
    main()