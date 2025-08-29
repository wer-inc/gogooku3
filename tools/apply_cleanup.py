#!/usr/bin/env python3
"""
Gogooku3 クリーンアップ実行ツール

DISPOSITION.csv に基づいて段階的・安全にクリーンアップを実行
DRY RUN機能、ロールバック機能、進捗レポート機能を提供
"""

import argparse
import csv
import shutil
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Set
import logging
import json

# JST timezone
JST = timezone(timedelta(hours=9))

class CleanupApplicator:
    """クリーンアップ実行クラス"""
    
    def __init__(self, disposition_path: Path, dry_run: bool = True):
        self.disposition_path = disposition_path
        self.dry_run = dry_run
        self.dispositions = self._load_dispositions()
        
        # 実行統計
        self.stats = {
            'processed': 0,
            'succeeded': 0,
            'failed': 0,
            'skipped': 0,
            'space_freed_bytes': 0,
            'actions_taken': defaultdict(int),
            'errors': []
        }
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')

    def _load_dispositions(self) -> List[Dict]:
        """処理方針データ読み込み"""
        dispositions = []
        with open(self.disposition_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['size_bytes'] = int(row['size_bytes']) if row['size_bytes'].isdigit() else 0
                row['git_tracked'] = row['git_tracked'].lower() == 'true'
                dispositions.append(row)
        return dispositions

    def _safe_file_operation(self, operation: str, source: Path, target: Path = None) -> bool:
        """安全なファイル操作実行"""
        try:
            if not source.exists():
                self.logger.warning(f"Source file not found: {source}")
                return False
            
            if self.dry_run:
                self.logger.info(f"[DRY RUN] Would {operation}: {source} -> {target or 'DELETE'}")
                return True
            
            if operation == 'move':
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(target))
                self.logger.info(f"Moved: {source} -> {target}")
                
            elif operation == 'copy':
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(source), str(target))
                self.logger.info(f"Copied: {source} -> {target}")
                
            elif operation == 'delete':
                source.unlink()
                self.logger.info(f"Deleted: {source}")
                
            elif operation == 'archive':
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(target))
                self.logger.info(f"Archived: {source} -> {target}")
                
            return True
            
        except Exception as e:
            error_msg = f"Operation {operation} failed for {source}: {e}"
            self.stats['errors'].append(error_msg)
            self.logger.error(error_msg)
            return False

    def _apply_keep(self, disposition: Dict) -> bool:
        """keep: 何もしない（統計のみ）"""
        return True

    def _apply_delete(self, disposition: Dict) -> bool:
        """delete: ファイル削除"""
        source = Path(disposition['path'])
        
        # 高リスク確認
        if disposition['risk'] == 'high':
            if not self.dry_run:
                self.logger.warning(f"HIGH RISK DELETE: {source}")
                # 実際の削除では追加確認が必要
                
        success = self._safe_file_operation('delete', source)
        if success:
            self.stats['space_freed_bytes'] += disposition['size_bytes']
        return success

    def _apply_move(self, disposition: Dict) -> bool:
        """move: ファイル移動"""
        source = Path(disposition['path'])
        target = Path(disposition['target_path'])
        
        return self._safe_file_operation('move', source, target)

    def _apply_consolidate(self, disposition: Dict) -> bool:
        """consolidate: 重複統合（非canonicalファイル削除）"""
        source = Path(disposition['path'])
        canonical = Path(disposition['target_path'])
        
        # canonical自身の場合はスキップ
        if source == canonical:
            return True
            
        # canonicalが存在することを確認
        if not canonical.exists():
            self.logger.error(f"Canonical file not found: {canonical}")
            return False
            
        success = self._safe_file_operation('delete', source)
        if success:
            self.stats['space_freed_bytes'] += disposition['size_bytes']
            self.logger.info(f"Consolidated: {source} -> canonical: {canonical}")
        return success

    def _apply_archive(self, disposition: Dict) -> bool:
        """archive: アーカイブディレクトリへ移動"""
        source = Path(disposition['path'])
        target = Path(disposition['target_path'])
        
        return self._safe_file_operation('archive', source, target)

    def _apply_regenerate(self, disposition: Dict) -> bool:
        """regenerate: 削除＋再現手順記録"""
        source = Path(disposition['path'])
        
        # 再現手順を記録（runbooks.md に追記）
        if not self.dry_run:
            runbooks_path = Path('docs/operations/runbooks.md')
            runbooks_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(runbooks_path, 'a', encoding='utf-8') as f:
                f.write(f"\n## Regenerate {source.name}\n")
                f.write(f"Original path: `{source}`\n")
                f.write(f"To regenerate, run the appropriate script or pipeline.\n\n")
        
        success = self._safe_file_operation('delete', source)
        if success:
            self.stats['space_freed_bytes'] += disposition['size_bytes']
        return success

    def _apply_lfs(self, disposition: Dict) -> bool:
        """lfs: Git LFS追加（手動操作が必要）"""
        source = Path(disposition['path'])
        
        self.logger.info(f"LFS candidate: {source} (requires manual git lfs track)")
        # LFSは手動操作が必要なのでここでは記録のみ
        return True

    def _apply_convert(self, disposition: Dict) -> bool:
        """convert: 形式変換（例：notebook→py）"""
        source = Path(disposition['path'])
        target = Path(disposition['target_path'])
        
        if source.suffix == '.ipynb':
            # ノートブック変換は複雑なので、今回は移動のみ
            return self._safe_file_operation('move', source, target)
        else:
            return self._safe_file_operation('move', source, target)

    def apply_action(self, disposition: Dict) -> bool:
        """単一アクション実行"""
        action = disposition['action']
        path = disposition['path']
        
        self.stats['processed'] += 1
        self.logger.info(f"Processing ({self.stats['processed']}): {action} -> {path}")
        
        # アクション分岐
        action_map = {
            'keep': self._apply_keep,
            'delete': self._apply_delete,
            'move': self._apply_move,
            'consolidate': self._apply_consolidate,
            'archive': self._apply_archive,
            'regenerate': self._apply_regenerate,
            'lfs': self._apply_lfs,
            'convert': self._apply_convert
        }
        
        if action not in action_map:
            self.logger.error(f"Unknown action: {action}")
            self.stats['failed'] += 1
            return False
        
        try:
            success = action_map[action](disposition)
            if success:
                self.stats['succeeded'] += 1
                self.stats['actions_taken'][action] += 1
            else:
                self.stats['failed'] += 1
            return success
            
        except Exception as e:
            error_msg = f"Action {action} failed for {path}: {e}"
            self.stats['errors'].append(error_msg)
            self.logger.error(error_msg)
            self.stats['failed'] += 1
            return False

    def apply_by_action(self, target_action: str, max_files: int = None) -> Dict:
        """特定アクションのファイルのみ処理"""
        self.logger.info(f"Applying action: {target_action}")
        
        target_dispositions = [d for d in self.dispositions if d['action'] == target_action]
        
        if max_files:
            target_dispositions = target_dispositions[:max_files]
        
        self.logger.info(f"Processing {len(target_dispositions)} files with action: {target_action}")
        
        for disposition in target_dispositions:
            self.apply_action(disposition)
        
        return self.stats

    def apply_by_risk(self, max_risk: str = 'medium') -> Dict:
        """リスクレベル以下のファイルのみ処理"""
        risk_order = {'low': 0, 'medium': 1, 'high': 2}
        max_risk_level = risk_order[max_risk]
        
        target_dispositions = [
            d for d in self.dispositions 
            if risk_order.get(d['risk'], 2) <= max_risk_level
        ]
        
        self.logger.info(f"Processing {len(target_dispositions)} files with risk <= {max_risk}")
        
        for disposition in target_dispositions:
            self.apply_action(disposition)
        
        return self.stats

    def apply_all(self) -> Dict:
        """全ファイル処理（段階的）"""
        
        # 段階1: 低リスクアクション
        safe_actions = ['keep', 'archive', 'consolidate']
        self.logger.info("Phase 1: Safe actions")
        for action in safe_actions:
            self.apply_by_action(action)
        
        # 段階2: 中リスクアクション
        medium_actions = ['move', 'convert', 'lfs']
        self.logger.info("Phase 2: Medium risk actions")
        for action in medium_actions:
            self.apply_by_action(action)
        
        # 段階3: 削除系アクション
        delete_actions = ['regenerate', 'delete']
        self.logger.info("Phase 3: Deletion actions")
        for action in delete_actions:
            self.apply_by_action(action)
        
        return self.stats

    def generate_report(self, output_path: Path) -> None:
        """実行レポート生成"""
        report_content = f"""# Gogooku3 クリーンアップ実行レポート

**実行日時**: {datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S JST')}  
**実行モード**: {'DRY RUN' if self.dry_run else 'ACTUAL EXECUTION'}  
**処理方針**: {self.disposition_path}

## 📊 実行統計

- **処理ファイル数**: {self.stats['processed']}
- **成功**: {self.stats['succeeded']}
- **失敗**: {self.stats['failed']}
- **スキップ**: {self.stats['skipped']}
- **解放容量**: {self.stats['space_freed_bytes'] / 1024 / 1024:.1f} MB

## 🎯 アクション別実行数

"""
        for action, count in self.stats['actions_taken'].items():
            report_content += f"- **{action}**: {count} files\n"

        report_content += f"""

## ❌ エラー一覧

"""
        if self.stats['errors']:
            for i, error in enumerate(self.stats['errors'], 1):
                report_content += f"{i}. {error}\n"
        else:
            report_content += "エラーはありませんでした。\n"

        report_content += f"""

## 🔄 ロールバック情報

実行済み操作のロールバック手順は ROLLBACK_GUIDE.md を参照してください。

---

*このレポートは tools/apply_cleanup.py により自動生成されました*
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"Execution report saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Apply repository cleanup")
    parser.add_argument('--disposition', type=Path, default='DISPOSITION.csv', help='Disposition CSV file')
    parser.add_argument('--action', type=str, help='Apply specific action only')
    parser.add_argument('--max-risk', type=str, choices=['low', 'medium', 'high'], default='medium', help='Maximum risk level')
    parser.add_argument('--max-files', type=int, help='Maximum files to process')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode (default)')
    parser.add_argument('--execute', action='store_true', help='Execute actual changes')
    parser.add_argument('--report', type=Path, default='cleanup_execution_report.md', help='Report output path')
    
    args = parser.parse_args()
    
    # デフォルトはdry-run（安全）
    dry_run = not args.execute
    
    applicator = CleanupApplicator(args.disposition, dry_run=dry_run)
    
    if args.action:
        stats = applicator.apply_by_action(args.action, args.max_files)
    else:
        stats = applicator.apply_by_risk(args.max_risk)
    
    applicator.generate_report(args.report)
    
    print(f"\n🎯 Cleanup Execution Summary:")
    print(f"  Mode: {'DRY RUN' if dry_run else 'ACTUAL EXECUTION'}")
    print(f"  Processed: {stats['processed']} files")
    print(f"  Succeeded: {stats['succeeded']} files")
    print(f"  Failed: {stats['failed']} files")
    print(f"  Space freed: {stats['space_freed_bytes'] / 1024 / 1024:.1f} MB")
    print(f"  Errors: {len(stats['errors'])}")
    print(f"\n📊 Actions taken:")
    for action, count in stats['actions_taken'].items():
        print(f"    {action}: {count}")
    print(f"\n📝 Report saved: {args.report}")

if __name__ == "__main__":
    main()