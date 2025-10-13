#!/usr/bin/env python3
"""
Download output directory from GCS bucket to local project root.
"""
import os
from pathlib import Path
from google.cloud import storage

def download_directory(bucket_name: str, source_prefix: str, destination_dir: str):
    """
    Download all files from a GCS bucket prefix to a local directory.

    Args:
        bucket_name: GCS bucket name
        source_prefix: Prefix/directory in the bucket (e.g., 'output')
        destination_dir: Local directory to save files
    """
    # Set up authentication
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/root/gogooku3/secrets/gogooku-b3b34bc07639.json'

    # Initialize client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # List all blobs with the prefix
    blobs = bucket.list_blobs(prefix=source_prefix)

    # Create destination directory
    dest_path = Path(destination_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    downloaded_count = 0
    total_size = 0

    print(f"📥 Downloading from gs://{bucket_name}/{source_prefix} to {destination_dir}")
    print("-" * 80)

    for blob in blobs:
        # Skip directory markers
        if blob.name.endswith('/'):
            continue

        # Calculate relative path
        relative_path = blob.name
        local_file_path = dest_path / relative_path

        # Create parent directories if needed
        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Download the file
        print(f"⬇️  {blob.name} ({blob.size / 1024 / 1024:.2f} MB)")
        blob.download_to_filename(str(local_file_path))

        downloaded_count += 1
        total_size += blob.size

    print("-" * 80)
    print(f"✅ Downloaded {downloaded_count} files ({total_size / 1024 / 1024:.2f} MB total)")
    print(f"📂 Files saved to: {destination_dir}")

if __name__ == "__main__":
    bucket_name = "gogooku-ml-data"
    source_prefix = "output"
    destination_dir = "/root/gogooku3"

    download_directory(bucket_name, source_prefix, destination_dir)
