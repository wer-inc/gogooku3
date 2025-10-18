#!/usr/bin/env python3
"""Upload output files to Google Cloud Storage.

This script uploads all files from the local output directory to GCS bucket.
"""

from __future__ import annotations

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
os.environ["GCS_ENABLED"] = "1"
os.environ["GCS_BUCKET"] = "gogooku-ml-data"

import logging

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


def upload_output_to_gcs(
    local_output_dir: Path,
    bucket_name: str = "gogooku-ml-data",
    gcs_prefix: str = "output/",
    exclude_patterns: list[str] | None = None,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Upload all files from local output directory to GCS.

    Args:
        local_output_dir: Local output directory
        bucket_name: GCS bucket name
        gcs_prefix: GCS prefix for uploaded files
        exclude_patterns: List of glob patterns to exclude
        dry_run: If True, only list files without uploading

    Returns:
        Tuple of (successful_uploads, failed_uploads)
    """
    local_output_dir = Path(local_output_dir)

    if not local_output_dir.exists():
        logger.error(f"❌ Output directory not found: {local_output_dir}")
        return 0, 0

    # Verify credentials exist
    if not CREDENTIALS_PATH.exists():
        logger.error(f"❌ Credentials file not found: {CREDENTIALS_PATH}")
        return 0, 0

    logger.info(f"🔑 Using credentials: {CREDENTIALS_PATH}")

    # Default exclusions
    exclude_patterns = exclude_patterns or ["*.log", "*.tmp", "__pycache__"]

    # Collect files to upload
    files_to_upload = []
    total_size = 0

    for file_path in local_output_dir.rglob("*"):
        if not file_path.is_file():
            continue

        # Check exclusion patterns
        if any(file_path.match(pattern) for pattern in exclude_patterns):
            continue

        files_to_upload.append(file_path)
        total_size += file_path.stat().st_size

    total_size_gb = total_size / (1024**3)

    logger.info("📦 Upload Summary:")
    logger.info(f"  Local dir: {local_output_dir.absolute()}")
    logger.info(f"  GCS bucket: gs://{bucket_name}/{gcs_prefix}")
    logger.info(f"  Files to upload: {len(files_to_upload)}")
    logger.info(f"  Total size: {total_size_gb:.2f} GB")
    logger.info(f"  Exclude patterns: {exclude_patterns}")

    if dry_run:
        logger.info("🔍 DRY RUN MODE - No files will be uploaded")
        for file_path in files_to_upload[:20]:  # Show first 20
            rel_path = file_path.relative_to(local_output_dir)
            size_mb = file_path.stat().st_size / (1024**2)
            logger.info(f"  Would upload: {rel_path} ({size_mb:.2f} MB)")
        if len(files_to_upload) > 20:
            logger.info(f"  ... and {len(files_to_upload) - 20} more files")
        return 0, 0

    # Upload files
    logger.info(f"\n📤 Uploading to gs://{bucket_name}/{gcs_prefix}...")

    successful = 0
    failed = 0

    for file_path in files_to_upload:
        # Generate GCS path
        rel_path = file_path.relative_to(local_output_dir)
        gcs_path = f"{gcs_prefix.rstrip('/')}/{rel_path}"

        if upload_file(bucket_name, file_path, gcs_path):
            successful += 1
        else:
            failed += 1

    logger.info(f"\n✅ Upload complete: {successful} successful, {failed} failed")

    return successful, failed


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload output files to Google Cloud Storage"
    )
    parser.add_argument(
        "--bucket",
        default="gogooku-ml-data",
        help="GCS bucket name (default: gogooku-ml-data)",
    )
    parser.add_argument(
        "--prefix",
        default="output/",
        help="GCS prefix for uploaded files (default: output/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Local output directory (default: project_root/output)",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["*.log", "*.tmp", "__pycache__"],
        help="Glob patterns to exclude from upload",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files without uploading",
    )

    args = parser.parse_args()

    # Default output directory
    output_dir = args.output_dir or (project_root / "output")

    # Ensure package is installed
    ensure_gcs_package()

    # Upload files
    successful, failed = upload_output_to_gcs(
        local_output_dir=output_dir,
        bucket_name=args.bucket,
        gcs_prefix=args.prefix,
        exclude_patterns=args.exclude,
        dry_run=args.dry_run,
    )

    # Exit with error code if any uploads failed
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
