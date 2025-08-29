#!/usr/bin/env python3
"""
Gogooku3 重複ファイル検出ツール

SHA256ハッシュベースの完全一致と、内容の類似性による重複検出
Canonical選定基準に基づいて統合対象を提案
"""

import argparse
import csv
import difflib
import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
import logging

class DuplicateFinder:
    """重複ファイル検出・解析クラス"""
    
    def __init__(self, inventory_path: Path):
        self.inventory_path = inventory_path
        self.files_data = self._load_inventory()
        
        # Canonical選定基準の重み
        self.weights = {
            'has_tests': 5.0,      # テスト存在
            'references': 2.0,      # 参照数
            'recency': 1.0,        # 更新日時
            'naming_clarity': 1.5,  # 命名明確性
            'path_appropriateness': 1.2,  # 配置適切性
            'git_history': 0.8     # Git履歴の長さ
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
                # 数値型フィールドの変換
                row['size_bytes'] = int(row['size_bytes']) if row['size_bytes'].isdigit() else 0
                row['referenced_by_count'] = int(row['referenced_by_count']) if row['referenced_by_count'].isdigit() else 0
                row['git_tracked'] = row['git_tracked'].lower() == 'true'
                row['binary_guess'] = row['binary_guess'].lower() == 'true'
                files_data.append(row)
        return files_data

    def _calculate_canonical_score(self, file_info: Dict, duplicates: List[Dict]) -> float:
        """Canonical選定スコア計算"""
        score = 0.0
        path = file_info['path']
        
        # 1. テスト存在（test_*.py, tests/, conftest.py）
        has_tests = any('test' in file_info['path'].lower() for file_info in duplicates) or \
                   Path(path).parent.name == 'tests' or \
                   'conftest.py' in path
        if has_tests:
            score += self.weights['has_tests']
        
        # 2. 参照数
        score += file_info['referenced_by_count'] * self.weights['references']
        
        # 3. 更新日時（180日以内なら加点）
        if file_info.get('last_commit_date'):
            try:
                from datetime import datetime, timedelta
                commit_date = datetime.fromisoformat(file_info['last_commit_date'].replace('Z', '+00:00'))
                if (datetime.now().astimezone() - commit_date).days <= 180:
                    score += self.weights['recency']
            except:
                pass
        
        # 4. 命名明確性（main, core, primary, base などの優先）
        filename = Path(path).name.lower()
        clear_names = ['main', 'core', 'primary', 'base', '__init__']
        if any(name in filename for name in clear_names):
            score += self.weights['naming_clarity']
        
        # 5. パス適切性（src/ > scripts/ > tests/ > 他）
        path_preferences = {
            'src/': 3.0,
            'scripts/': 2.0,
            'tests/': 1.5,
            'tools/': 1.0
        }
        for prefix, weight in path_preferences.items():
            if path.startswith(prefix):
                score += weight * self.weights['path_appropriateness']
                break
        
        # 6. Git履歴（初回コミットが古い = 歴史が長い）
        if file_info.get('first_seen_commit'):
            score += self.weights['git_history']
        
        return score

    def find_exact_duplicates(self) -> List[Dict]:
        """SHA256ハッシュによる完全一致重複検出"""
        hash_groups = defaultdict(list)
        
        for file_info in self.files_data:
            if file_info['sha256'] and file_info['size_bytes'] > 0:
                hash_groups[file_info['sha256']].append(file_info)
        
        # 重複グループのみ抽出
        exact_duplicates = []
        group_id = 1
        
        for hash_value, files in hash_groups.items():
            if len(files) > 1:
                # Canonical選定
                scored_files = [(self._calculate_canonical_score(f, files), f) for f in files]
                scored_files.sort(reverse=True, key=lambda x: x[0])
                canonical = scored_files[0][1]
                
                for i, file_info in enumerate(files):
                    exact_duplicates.append({
                        'group_id': group_id,
                        'type': 'exact',
                        'path_a': canonical['path'],
                        'path_b': file_info['path'],
                        'similarity_or_hash': hash_value[:16],  # 短縮ハッシュ
                        'size': file_info['size_bytes'],
                        'preferred': canonical['path'],
                        'canonical_score_a': scored_files[0][0],
                        'canonical_score_b': next((score for score, f in scored_files if f['path'] == file_info['path']), 0)
                    })
                
                group_id += 1
        
        return exact_duplicates

    def find_similar_configs(self, similarity_threshold: float = 0.8) -> List[Dict]:
        """設定ファイルの類似重複検出"""
        config_extensions = {'.yaml', '.yml', '.json', '.toml', '.ini', '.cfg'}
        config_files = [f for f in self.files_data 
                       if Path(f['path']).suffix in config_extensions and not f['binary_guess']]
        
        similar_duplicates = []
        group_id = 1000  # exactと区別するためのオフセット
        processed = set()
        
        for i, file_a in enumerate(config_files):
            if file_a['path'] in processed:
                continue
                
            similar_group = [file_a]
            
            for j, file_b in enumerate(config_files[i+1:], i+1):
                if file_b['path'] in processed:
                    continue
                
                try:
                    # ファイル内容の類似度計算
                    with open(file_a['path'], 'r', encoding='utf-8', errors='ignore') as fa:
                        content_a = fa.read().strip()
                    with open(file_b['path'], 'r', encoding='utf-8', errors='ignore') as fb:
                        content_b = fb.read().strip()
                    
                    similarity = difflib.SequenceMatcher(None, content_a, content_b).ratio()
                    
                    if similarity >= similarity_threshold:
                        similar_group.append(file_b)
                        processed.add(file_b['path'])
                        
                except Exception as e:
                    self.logger.warning(f"Content comparison failed for {file_a['path']} vs {file_b['path']}: {e}")
            
            if len(similar_group) > 1:
                # Canonical選定
                scored_files = [(self._calculate_canonical_score(f, similar_group), f) for f in similar_group]
                scored_files.sort(reverse=True, key=lambda x: x[0])
                canonical = scored_files[0][1]
                
                for file_info in similar_group:
                    similar_duplicates.append({
                        'group_id': group_id,
                        'type': 'config_similar',
                        'path_a': canonical['path'],
                        'path_b': file_info['path'],
                        'similarity_or_hash': f"{similarity:.3f}",
                        'size': file_info['size_bytes'],
                        'preferred': canonical['path'],
                        'canonical_score_a': scored_files[0][0],
                        'canonical_score_b': next((score for score, f in scored_files if f['path'] == file_info['path']), 0)
                    })
                
                processed.add(file_a['path'])
                group_id += 1
        
        return similar_duplicates

    def find_notebook_duplicates(self) -> List[Dict]:
        """ノートブック重複検出（名前パターンベース）"""
        notebooks = [f for f in self.files_data if f['notebook_guess']]
        
        # 名前パターンによるグループ化
        pattern_groups = defaultdict(list)
        for nb in notebooks:
            basename = Path(nb['path']).stem
            # バージョン番号、日付、_copy等を除去して正規化
            normalized = re.sub(r'[-_](v?\d+|copy|backup|\d{4}\d{2}\d{2}|\d{8}|old)$', '', basename, flags=re.IGNORECASE)
            pattern_groups[normalized].append(nb)
        
        notebook_duplicates = []
        group_id = 2000
        
        for pattern, files in pattern_groups.items():
            if len(files) > 1:
                # Canonical選定（最新＞小サイズ＞明確命名）
                scored_files = [(self._calculate_canonical_score(f, files), f) for f in files]
                scored_files.sort(reverse=True, key=lambda x: x[0])
                canonical = scored_files[0][1]
                
                for file_info in files:
                    notebook_duplicates.append({
                        'group_id': group_id,
                        'type': 'notebook_pattern',
                        'path_a': canonical['path'],
                        'path_b': file_info['path'],
                        'similarity_or_hash': pattern,
                        'size': file_info['size_bytes'],
                        'preferred': canonical['path'],
                        'canonical_score_a': scored_files[0][0],
                        'canonical_score_b': next((score for score, f in scored_files if f['path'] == file_info['path']), 0)
                    })
                
                group_id += 1
        
        return notebook_duplicates

    def generate_duplicates_report(self, output_path: Path) -> Dict:
        """重複検出実行・レポート生成"""
        self.logger.info("Finding duplicates...")
        
        # 各種重複検出
        exact_duplicates = self.find_exact_duplicates()
        config_duplicates = self.find_similar_configs()
        notebook_duplicates = self.find_notebook_duplicates()
        
        all_duplicates = exact_duplicates + config_duplicates + notebook_duplicates
        
        # CSV出力
        fieldnames = [
            'group_id', 'type', 'path_a', 'path_b', 'similarity_or_hash', 
            'size', 'preferred', 'canonical_score_a', 'canonical_score_b'
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_duplicates)
        
        # 統計
        stats = {
            'total_duplicate_groups': len(set(d['group_id'] for d in all_duplicates)),
            'exact_duplicate_groups': len(set(d['group_id'] for d in exact_duplicates)),
            'config_duplicate_groups': len(set(d['group_id'] for d in config_duplicates)),
            'notebook_duplicate_groups': len(set(d['group_id'] for d in notebook_duplicates)),
            'total_redundant_files': len([d for d in all_duplicates if d['path_a'] != d['path_b']]),
            'potential_space_savings_mb': sum(d['size'] for d in all_duplicates if d['path_a'] != d['path_b']) / 1024 / 1024
        }
        
        self.logger.info(f"Duplicate analysis complete: {stats['total_duplicate_groups']} groups found")
        self.logger.info(f"Potential space savings: {stats['potential_space_savings_mb']:.1f} MB")
        
        return stats

def main():
    parser = argparse.ArgumentParser(description="Find duplicate files")
    parser.add_argument('--inventory', type=Path, default='repo_inventory.csv', help='Inventory CSV file')
    parser.add_argument('--output', type=Path, default='duplicates.csv', help='Output duplicates CSV')
    parser.add_argument('--similarity-threshold', type=float, default=0.8, help='Similarity threshold for config files')
    
    args = parser.parse_args()
    
    finder = DuplicateFinder(args.inventory)
    stats = finder.generate_duplicates_report(args.output)
    
    print(f"\n📊 Duplicate Detection Summary:")
    print(f"  Duplicate groups: {stats['total_duplicate_groups']}")
    print(f"  - Exact duplicates: {stats['exact_duplicate_groups']}")
    print(f"  - Similar configs: {stats['config_duplicate_groups']}")
    print(f"  - Similar notebooks: {stats['notebook_duplicate_groups']}")
    print(f"  Redundant files: {stats['total_redundant_files']}")
    print(f"  Potential space savings: {stats['potential_space_savings_mb']:.1f} MB")
    print(f"\n✅ Duplicates report saved: {args.output}")

if __name__ == "__main__":
    main()