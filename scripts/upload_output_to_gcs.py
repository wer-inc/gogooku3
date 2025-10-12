#!/usr/bin/env python3
"""Upload output directory to Google Cloud Storage.

This script uploads all files in the output directory to GCS bucket.
It uses the GCS storage module for seamless integration.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gogooku3.utils.gcs_storage import sync_directory_to_gcs

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main function to upload output directory to GCS."""
    # Set environment variables
    os.environ["GCS_ENABLED"] = "1"
    os.environ["GCS_BUCKET"] = os.getenv("GCS_BUCKET", "gogooku-ml-data")

    # Set Google Cloud credentials
    credentials_path = project_root / "gogooku-b3b34bc07639.json"
    if credentials_path.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)
        logger.info(f"Using credentials: {credentials_path}")
    else:
        logger.error(f"Credentials file not found: {credentials_path}")
        logger.error("Please ensure gogooku-b3b34bc07639.json exists in project root")
        return 1

    # Define output directory
    output_dir = project_root / "output"
    if not output_dir.exists():
        logger.error(f"Output directory not found: {output_dir}")
        return 1

    logger.info(f"Starting upload of {output_dir} to GCS bucket: {os.environ['GCS_BUCKET']}")
    logger.info("This may take several minutes depending on the data size...")

    # Sync directory to GCS
    # Exclude patterns: temporary files, hidden files, etc.
    exclude_patterns = [
        "*.tmp",
        "*.lock",
        "*.swp",
        "*~",
        ".DS_Store",
        "__pycache__",
    ]

    uploaded, skipped = sync_directory_to_gcs(
        local_dir=output_dir,
        gcs_prefix="output/",  # Store under "output/" prefix in GCS
        bucket=os.environ["GCS_BUCKET"],
        exclude_patterns=exclude_patterns,
    )

    logger.info("=" * 60)
    logger.info("Upload Summary:")
    logger.info(f"  Files uploaded: {uploaded}")
    logger.info(f"  Files skipped: {skipped}")
    logger.info(f"  Bucket: gs://{os.environ['GCS_BUCKET']}/output/")
    logger.info("=" * 60)

    if uploaded > 0:
        logger.info("✅ Upload completed successfully!")
        return 0
    else:
        logger.warning("⚠️  No files were uploaded")
        return 1


if __name__ == "__main__":
    sys.exit(main())
