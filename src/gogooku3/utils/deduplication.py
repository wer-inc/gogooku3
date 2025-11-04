"""
Duplicate File Detection and Removal Utility
Safe deduplication with backup and verification
"""

import hashlib
import json
import logging
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class SafeDeduplicator:
    """Safe file deduplication with comprehensive backup and verification."""

    def __init__(self, backup_dir: Path | None = None):
        self.backup_dir = backup_dir or Path("output/deduplication_backup")
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Stats tracking
        self.stats = {
            "files_scanned": 0,
            "duplicates_found": 0,
            "files_removed": 0,
            "space_saved_bytes": 0,
            "scan_start": None,
            "scan_end": None,
        }

        # Safety settings
        self.dry_run = True  # Default to dry run mode
        self.min_file_size = 100  # Don't process files smaller than 100 bytes
        self.create_backup = True  # Always backup before deletion

    def calculate_file_hash(self, filepath: Path) -> str:
        """Calculate SHA-256 hash of file content."""
        hash_sha256 = hashlib.sha256()

        try:
            with open(filepath, 'rb') as f:
                # Process in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Failed to hash {filepath}: {e}")
            return ""

    def scan_directory(self, directory: Path, patterns: list[str] = None) -> dict[str, list[Path]]:
        """Scan directory and identify duplicate files by hash.

        Args:
            directory: Directory to scan
            patterns: List of glob patterns to match (e.g., ['*.parquet', '*.json'])

        Returns:
            Dictionary mapping hash -> list of file paths with that hash
        """
        if patterns is None:
            patterns = ['*.parquet']  # Default to parquet files

        self.stats["scan_start"] = datetime.now()
        hash_to_files: dict[str, list[Path]] = defaultdict(list)

        print(f"🔍 Scanning directory: {directory}")
        print(f"📋 Patterns: {patterns}")

        for pattern in patterns:
            for filepath in directory.rglob(pattern):
                if not filepath.is_file():
                    continue

                # Skip small files
                if filepath.stat().st_size < self.min_file_size:
                    continue

                self.stats["files_scanned"] += 1

                # Calculate hash
                file_hash = self.calculate_file_hash(filepath)
                if file_hash:
                    hash_to_files[file_hash].append(filepath)

                if self.stats["files_scanned"] % 100 == 0:
                    print(f"  📊 Scanned {self.stats['files_scanned']} files...")

        self.stats["scan_end"] = datetime.now()

        # Filter to only duplicates
        duplicates = {h: files for h, files in hash_to_files.items() if len(files) > 1}
        self.stats["duplicates_found"] = sum(len(files) - 1 for files in duplicates.values())

        print(f"✅ Scan complete: {self.stats['files_scanned']} files, {len(duplicates)} duplicate groups")
        return duplicates

    def analyze_duplicates(self, duplicates: dict[str, list[Path]]) -> dict:
        """Analyze duplicate files and create removal plan."""
        analysis = {
            "duplicate_groups": len(duplicates),
            "total_duplicates": sum(len(files) - 1 for files in duplicates.values()),
            "potential_space_saved": 0,
            "removal_plan": [],
            "preservation_plan": [],
        }

        for file_hash, files in duplicates.items():
            # Sort files by modification time (keep newest) and path length (prefer shorter paths)
            files_with_info = []
            for filepath in files:
                stat = filepath.stat()
                files_with_info.append({
                    "path": filepath,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "path_depth": len(filepath.parts),
                })

            # Sort: prefer newer files, then shorter paths
            files_with_info.sort(key=lambda x: (-x["mtime"], x["path_depth"]))

            # Keep the first (newest/shortest path), remove the rest
            keep_file = files_with_info[0]
            remove_files = files_with_info[1:]

            analysis["preservation_plan"].append(keep_file["path"])

            for remove_file in remove_files:
                analysis["removal_plan"].append({
                    "remove_path": remove_file["path"],
                    "keep_path": keep_file["path"],
                    "size_bytes": remove_file["size"],
                    "hash": file_hash
                })
                analysis["potential_space_saved"] += remove_file["size"]

        return analysis

    def create_backup(self, filepath: Path) -> Path:
        """Create backup of file before removal."""
        if not self.create_backup:
            return None

        # Create backup path maintaining directory structure
        relative_path = filepath.relative_to(Path.cwd())
        backup_path = self.backup_dir / relative_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file to backup
        shutil.copy2(filepath, backup_path)
        return backup_path

    def execute_removal_plan(self, removal_plan: list[dict], dry_run: bool = True) -> dict:
        """Execute the file removal plan.

        Args:
            removal_plan: List of files to remove
            dry_run: If True, only simulate removal

        Returns:
            Execution results
        """
        results = {
            "files_processed": 0,
            "files_removed": 0,
            "files_backed_up": 0,
            "space_saved": 0,
            "errors": [],
            "dry_run": dry_run,
        }

        print(f"🗑️ Executing removal plan (dry_run={dry_run})")
        print(f"📁 Backup directory: {self.backup_dir}")

        for item in removal_plan:
            results["files_processed"] += 1
            remove_path = item["remove_path"]
            keep_path = item["keep_path"]

            try:
                if not dry_run:
                    # Create backup first
                    if self.create_backup:
                        backup_path = self.create_backup(remove_path)
                        results["files_backed_up"] += 1
                        print(f"  💾 Backed up: {remove_path} -> {backup_path}")

                    # Remove the duplicate
                    remove_path.unlink()
                    results["files_removed"] += 1
                    results["space_saved"] += item["size_bytes"]
                    print(f"  🗑️ Removed: {remove_path}")
                else:
                    print(f"  [DRY RUN] Would remove: {remove_path}")
                    print(f"  [DRY RUN] Would keep: {keep_path}")

            except Exception as e:
                error_msg = f"Failed to process {remove_path}: {e}"
                results["errors"].append(error_msg)
                logger.error(error_msg)

        return results

    def save_deduplication_report(self, analysis: dict, results: dict) -> Path:
        """Save comprehensive deduplication report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(f"output/deduplication_report_{timestamp}.json")

        report = {
            "timestamp": timestamp,
            "stats": self.stats,
            "analysis": analysis,
            "execution_results": results,
            "settings": {
                "dry_run": self.dry_run,
                "backup_dir": str(self.backup_dir),
                "min_file_size": self.min_file_size,
                "create_backup": self.create_backup,
            }
        }

        # Convert Path objects and datetime to strings for JSON serialization
        def convert_objects(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_objects(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_objects(item) for item in obj]
            else:
                return obj

        # Deep convert all objects
        report_serializable = convert_objects(report)

        with open(report_path, 'w') as f:
            json.dump(report_serializable, f, indent=2)

        print(f"📊 Report saved: {report_path}")
        return report_path

    def deduplicate_directory(
        self,
        directory: Path,
        patterns: list[str] = None,
        dry_run: bool = True
    ) -> dict:
        """Complete deduplication workflow for a directory.

        Args:
            directory: Directory to deduplicate
            patterns: File patterns to process
            dry_run: If True, only simulate removal

        Returns:
            Complete deduplication results
        """
        print(f"🚀 Starting deduplication of {directory}")
        print(f"⚠️ Dry run mode: {dry_run}")

        # Step 1: Scan for duplicates
        duplicates = self.scan_directory(directory, patterns)

        if not duplicates:
            print("✅ No duplicates found!")
            return {"duplicates": 0, "message": "No duplicates found"}

        # Step 2: Analyze and create removal plan
        analysis = self.analyze_duplicates(duplicates)

        print("\n📊 DEDUPLICATION ANALYSIS")
        print(f"  🔍 Duplicate groups: {analysis['duplicate_groups']}")
        print(f"  📁 Files to remove: {analysis['total_duplicates']}")
        print(f"  💾 Space to save: {analysis['potential_space_saved'] / 1024 / 1024:.1f} MB")

        # Step 3: Execute removal plan
        results = self.execute_removal_plan(analysis["removal_plan"], dry_run=dry_run)

        # Step 4: Save report
        report_path = self.save_deduplication_report(analysis, results)

        print("\n✅ Deduplication complete!")
        if not dry_run:
            print(f"  🗑️ Files removed: {results['files_removed']}")
            print(f"  💾 Space saved: {results['space_saved'] / 1024 / 1024:.1f} MB")

        return {
            "duplicates_found": len(duplicates),
            "files_to_remove": analysis["total_duplicates"],
            "space_saved_bytes": analysis["potential_space_saved"],
            "execution_results": results,
            "report_path": report_path,
        }


def main():
    """CLI entry point for deduplication utility."""
    import argparse

    parser = argparse.ArgumentParser(description="Safe file deduplication utility")
    parser.add_argument("directory", type=Path, help="Directory to deduplicate")
    parser.add_argument("--patterns", nargs="+", default=["*.parquet"], help="File patterns to process")
    parser.add_argument("--execute", action="store_true", help="Execute removal (default: dry run)")
    parser.add_argument("--backup-dir", type=Path, help="Backup directory")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup creation")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create deduplicator
    deduplicator = SafeDeduplicator(backup_dir=args.backup_dir)
    deduplicator.dry_run = not args.execute
    deduplicator.create_backup = not args.no_backup

    # Run deduplication
    results = deduplicator.deduplicate_directory(
        directory=args.directory,
        patterns=args.patterns,
        dry_run=not args.execute
    )

    return results


if __name__ == "__main__":
    main()
