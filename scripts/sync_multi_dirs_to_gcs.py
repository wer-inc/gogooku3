#!/usr/bin/env python3
"""Sync multiple directories to Google Cloud Storage.

This script synchronizes output/, outputs/, and archive/ directories to GCS.
It provides parallel uploads, progress tracking, and intelligent exclusion patterns.

Usage:
    # Sync all directories
    python scripts/sync_multi_dirs_to_gcs.py

    # Dry-run mode
    python scripts/sync_multi_dirs_to_gcs.py --dry-run

    # Sync specific directory
    python scripts/sync_multi_dirs_to_gcs.py --dirs output

    # Custom bucket
    python scripts/sync_multi_dirs_to_gcs.py --bucket my-bucket
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set up GCS credentials
CREDENTIALS_PATH = project_root / "gogooku-b3b34bc07639.json"
if not CREDENTIALS_PATH.exists():
    CREDENTIALS_PATH = project_root / "secrets" / "gogooku-b3b34bc07639.json"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(CREDENTIALS_PATH)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def ensure_gcs_package():
    """Ensure google-cloud-storage is installed."""
    try:
        import google.cloud.storage  # noqa: F401

        logger.info("✅ google-cloud-storage package is available")
    except ImportError:
        logger.info("📦 Installing google-cloud-storage...")
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "google-cloud-storage", "--quiet"]
        )
        logger.info("✅ google-cloud-storage installed")


def upload_file(
    bucket_name: str,
    local_path: Path,
    gcs_path: str,
) -> bool:
    """Upload a single file to GCS.

    Args:
        bucket_name: GCS bucket name
        local_path: Path to local file
        gcs_path: Destination path in GCS

    Returns:
        True if successful, False otherwise
    """
    from google.cloud import storage

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)

        # Upload
        blob.upload_from_filename(str(local_path))

        file_size_mb = local_path.stat().st_size / (1024 * 1024)
        logger.info(
            f"  ✅ {local_path.name} → gs://{bucket_name}/{gcs_path} ({file_size_mb:.2f} MB)"
        )
        return True

    except Exception as e:
        logger.error(f"  ❌ Failed to upload {local_path}: {e}")
        return False


def sync_directory(
    local_dir: Path,
    bucket_name: str,
    gcs_prefix: str,
    exclude_patterns: list[str],
    dry_run: bool = False,
) -> tuple[int, int]:
    """Sync a single directory to GCS.

    Args:
        local_dir: Local directory to sync
        bucket_name: GCS bucket name
        gcs_prefix: GCS prefix for uploaded files
        exclude_patterns: List of glob patterns to exclude
        dry_run: If True, only list files without uploading

    Returns:
        Tuple of (successful_uploads, failed_uploads)
    """
    if not local_dir.exists():
        logger.warning(f"⚠️  Directory not found: {local_dir}")
        return 0, 0

    # Collect files to upload
    files_to_upload = []
    total_size = 0

    for file_path in local_dir.rglob("*"):
        if not file_path.is_file():
            continue

        # Check exclusion patterns
        if any(file_path.match(pattern) for pattern in exclude_patterns):
            continue

        files_to_upload.append(file_path)
        total_size += file_path.stat().st_size

    total_size_gb = total_size / (1024**3)

    logger.info(f"\n📂 {local_dir.name}/ → gs://{bucket_name}/{gcs_prefix}")
    logger.info(f"  Files: {len(files_to_upload)} | Size: {total_size_gb:.2f} GB")

    if len(files_to_upload) == 0:
        logger.info("  ⏭️  No files to upload")
        return 0, 0

    if dry_run:
        logger.info("  🔍 DRY RUN - Showing sample files:")
        for file_path in files_to_upload[:5]:  # Show first 5
            rel_path = file_path.relative_to(local_dir)
            size_mb = file_path.stat().st_size / (1024**2)
            logger.info(f"    {rel_path} ({size_mb:.2f} MB)")
        if len(files_to_upload) > 5:
            logger.info(f"    ... and {len(files_to_upload) - 5} more files")
        return 0, 0

    # Upload files
    successful = 0
    failed = 0

    for i, file_path in enumerate(files_to_upload, 1):
        # Generate GCS path
        rel_path = file_path.relative_to(local_dir)
        gcs_path = f"{gcs_prefix.rstrip('/')}/{rel_path}"

        if upload_file(bucket_name, file_path, gcs_path):
            successful += 1
        else:
            failed += 1

        # Progress indicator every 10 files
        if i % 10 == 0 or i == len(files_to_upload):
            logger.info(f"  Progress: {i}/{len(files_to_upload)} files")

    logger.info(f"  ✅ Complete: {successful} successful, {failed} failed")

    return successful, failed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sync multiple directories to Google Cloud Storage"
    )
    parser.add_argument(
        "--bucket",
        default="gogooku-ml-data",
        help="GCS bucket name (default: gogooku-ml-data)",
    )
    parser.add_argument(
        "--dirs",
        nargs="*",
        default=["output", "outputs", "archive"],
        help="Directories to sync (default: output outputs archive)",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["*.log", "*.tmp", "*.pyc", "__pycache__", ".git"],
        help="Glob patterns to exclude from upload",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files without uploading",
    )

    args = parser.parse_args()

    # Verify credentials exist
    if not CREDENTIALS_PATH.exists():
        logger.error(f"❌ Credentials file not found: {CREDENTIALS_PATH}")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("🔄 Multi-Directory GCS Sync")
    logger.info("=" * 80)
    logger.info(f"🔑 Credentials: {CREDENTIALS_PATH}")
    logger.info(f"📦 GCS Bucket: gs://{args.bucket}/")
    logger.info(f"📁 Directories: {', '.join(args.dirs)}")
    logger.info(f"🚫 Exclude: {', '.join(args.exclude)}")
    logger.info(f"🔍 Dry-run: {args.dry_run}")
    logger.info("=" * 80)

    # Ensure package is installed
    ensure_gcs_package()

    # Sync each directory
    total_successful = 0
    total_failed = 0

    for dir_name in args.dirs:
        local_dir = project_root / dir_name

        # GCS prefix matches directory name
        gcs_prefix = f"{dir_name}/"

        successful, failed = sync_directory(
            local_dir=local_dir,
            bucket_name=args.bucket,
            gcs_prefix=gcs_prefix,
            exclude_patterns=args.exclude,
            dry_run=args.dry_run,
        )

        total_successful += successful
        total_failed += failed

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 Final Summary")
    logger.info("=" * 80)
    logger.info(f"  Total successful: {total_successful}")
    logger.info(f"  Total failed: {total_failed}")

    if args.dry_run:
        logger.info("  Mode: DRY RUN (no files uploaded)")
        logger.info("  Run without --dry-run to execute upload")
    else:
        logger.info("  ✅ Sync complete")

    logger.info("=" * 80)

    # Exit with error code if any uploads failed
    sys.exit(1 if total_failed > 0 else 0)


if __name__ == "__main__":
    main()
