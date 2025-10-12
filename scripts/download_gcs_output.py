#!/usr/bin/env python3
"""Download output files from Google Cloud Storage.

This script uses the service account credentials to download all files
from the GCS bucket's output directory to the local output directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set up GCS credentials
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
        subprocess.check_call([sys.executable, "-m", "pip", "install", "google-cloud-storage", "--quiet"])
        logger.info("✅ google-cloud-storage installed")


def list_gcs_output_files(bucket_name: str = "gogooku-ml-data", prefix: str = "output/") -> list[dict]:
    """List all files in the GCS bucket's output directory.

    Args:
        bucket_name: GCS bucket name
        prefix: Prefix to filter files (default: "output/")

    Returns:
        List of dicts with file info (name, size, updated)
    """
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    logger.info(f"🔍 Listing files in gs://{bucket_name}/{prefix}...")

    blobs = bucket.list_blobs(prefix=prefix)
    files = []

    for blob in blobs:
        # Skip directory markers
        if blob.name.endswith("/"):
            continue

        files.append({
            "name": blob.name,
            "size": blob.size,
            "size_mb": blob.size / (1024 * 1024),
            "updated": blob.updated,
        })

    return files


def download_file(bucket_name: str, gcs_path: str, local_path: Path) -> bool:
    """Download a single file from GCS.

    Args:
        bucket_name: GCS bucket name
        gcs_path: Path in GCS
        local_path: Destination local path

    Returns:
        True if successful, False otherwise
    """
    from google.cloud import storage

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)

        # Create parent directory
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Download
        blob.download_to_filename(str(local_path))

        file_size_mb = local_path.stat().st_size / (1024 * 1024)
        logger.info(f"  ✅ {local_path.name} ({file_size_mb:.2f} MB)")
        return True

    except Exception as e:
        logger.error(f"  ❌ Failed to download {gcs_path}: {e}")
        return False


def download_gcs_output(
    bucket_name: str = "gogooku-ml-data",
    prefix: str = "output/",
    local_output_dir: Path | None = None,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Download all files from GCS output directory.

    Args:
        bucket_name: GCS bucket name
        prefix: GCS prefix to download from
        local_output_dir: Local output directory (default: project_root/output)
        dry_run: If True, only list files without downloading

    Returns:
        Tuple of (successful_downloads, failed_downloads)
    """
    if local_output_dir is None:
        local_output_dir = project_root / "output"

    local_output_dir = Path(local_output_dir)

    # Verify credentials exist
    if not CREDENTIALS_PATH.exists():
        logger.error(f"❌ Credentials file not found: {CREDENTIALS_PATH}")
        return 0, 0

    logger.info(f"🔑 Using credentials: {CREDENTIALS_PATH}")

    # List files
    files = list_gcs_output_files(bucket_name, prefix)

    if not files:
        logger.warning(f"⚠️  No files found in gs://{bucket_name}/{prefix}")
        return 0, 0

    logger.info(f"📋 Found {len(files)} files:")
    total_size_mb = sum(f["size_mb"] for f in files)
    for f in files:
        logger.info(f"   • {f['name']} ({f['size_mb']:.2f} MB) - {f['updated']}")

    logger.info(f"📊 Total size: {total_size_mb:.2f} MB")

    if dry_run:
        logger.info("🏃 Dry-run mode: No files will be downloaded")
        return 0, 0

    # Download files
    logger.info(f"\n📥 Downloading to {local_output_dir}...")

    successful = 0
    failed = 0

    for file_info in files:
        gcs_path = file_info["name"]

        # Generate local path (strip prefix)
        rel_path = gcs_path
        if rel_path.startswith(prefix):
            rel_path = rel_path[len(prefix):]

        local_path = local_output_dir / rel_path

        if download_file(bucket_name, gcs_path, local_path):
            successful += 1
        else:
            failed += 1

    logger.info(f"\n✅ Download complete: {successful} successful, {failed} failed")

    return successful, failed


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download output files from Google Cloud Storage"
    )
    parser.add_argument(
        "--bucket",
        default="gogooku-ml-data",
        help="GCS bucket name (default: gogooku-ml-data)",
    )
    parser.add_argument(
        "--prefix",
        default="output/",
        help="GCS prefix to download from (default: output/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Local output directory (default: project_root/output)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files without downloading",
    )

    args = parser.parse_args()

    # Ensure package is installed
    ensure_gcs_package()

    # Download files
    successful, failed = download_gcs_output(
        bucket_name=args.bucket,
        prefix=args.prefix,
        local_output_dir=args.output_dir,
        dry_run=args.dry_run,
    )

    # Exit with error code if any downloads failed
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
