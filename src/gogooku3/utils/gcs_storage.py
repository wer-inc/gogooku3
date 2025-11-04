"""Google Cloud Storage utility for hybrid local+cloud storage.

This module provides seamless integration between local storage and GCS,
enabling automatic backup and synchronization of ML datasets and artifacts.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_CREDENTIAL_HINT = (
    "Set either GCS_CREDENTIALS_PATH, GOOGLE_APPLICATION_CREDENTIALS, "
    "GCS_SERVICE_ACCOUNT_JSON, or GCS_SERVICE_ACCOUNT_JSON_B64."
)


def is_gcs_enabled() -> bool:
    """Check if GCS integration is enabled via environment variable."""
    return os.getenv("GCS_ENABLED", "0") == "1"


def get_gcs_bucket() -> str:
    """Get GCS bucket name from environment."""
    bucket = os.getenv("GCS_BUCKET", "gogooku-ml-data")
    return bucket


def _create_storage_client():
    """Create a google.cloud.storage.Client using explicit credentials when provided."""
    from google.cloud import storage

    # 1) JSON string (plain)
    cred_json = os.getenv("GCS_SERVICE_ACCOUNT_JSON")
    if cred_json:
        try:
            info = json.loads(cred_json)
            logger.debug("Creating GCS client from GCS_SERVICE_ACCOUNT_JSON env.")
            return storage.Client.from_service_account_info(info)
        except Exception as exc:
            raise RuntimeError(f"GCS_SERVICE_ACCOUNT_JSON invalid: {exc}") from exc

    # 2) JSON string (base64)
    cred_b64 = os.getenv("GCS_SERVICE_ACCOUNT_JSON_B64")
    if cred_b64:
        try:
            decoded = base64.b64decode(cred_b64)
            info = json.loads(decoded.decode("utf-8"))
            logger.debug("Creating GCS client from GCS_SERVICE_ACCOUNT_JSON_B64 env.")
            return storage.Client.from_service_account_info(info)
        except Exception as exc:
            raise RuntimeError(f"GCS_SERVICE_ACCOUNT_JSON_B64 invalid: {exc}") from exc

    # 3) Explicit credential path
    cred_path = (
        os.getenv("GCS_CREDENTIALS_PATH")
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    )
    if cred_path:
        path = Path(cred_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"GCS credentials file not found at {path}. {_CREDENTIAL_HINT}"
            )
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(path)
        logger.debug("Creating GCS client from %s", path)
        return storage.Client.from_service_account_json(str(path))

    # 4) Fallback to default credentials (metadata server / ADC)
    try:
        logger.debug("Creating GCS client using default credentials.")
        return storage.Client()
    except Exception as exc:
        raise RuntimeError(
            f"Unable to obtain Google Cloud credentials. {_CREDENTIAL_HINT}"
        ) from exc


def validate_gcs_credentials() -> bool:
    """Return True if a storage client can be created (when GCS is enabled)."""
    if not is_gcs_enabled():
        return True
    try:
        _create_storage_client()
        return True
    except Exception as exc:
        logger.error("GCS credential validation failed: %s", exc)
        return False

def upload_to_gcs(
    local_path: Path | str,
    gcs_path: str | None = None,
    bucket: str | None = None,
) -> bool:
    """Upload a local file to Google Cloud Storage.

    Args:
        local_path: Path to local file
        gcs_path: Destination path in GCS (e.g., "raw/flow/data.parquet")
                  If None, mirrors local directory structure
        bucket: GCS bucket name (defaults to GCS_BUCKET env var)

    Returns:
        True if upload successful, False otherwise
    """
    if not is_gcs_enabled():
        logger.debug("GCS not enabled, skipping upload")
        return False

    local_path = Path(local_path)
    if not local_path.exists():
        logger.warning(f"Local file not found: {local_path}")
        return False

    try:
        bucket_name = bucket or get_gcs_bucket()
        client = _create_storage_client()
        bucket_obj = client.bucket(bucket_name)

        # Auto-generate GCS path from local path structure
        if gcs_path is None:
            # Extract relative path from output directory
            output_dir = Path(os.getenv("LOCAL_CACHE_DIR", "output"))
            try:
                rel_path = local_path.relative_to(output_dir)
                gcs_path = str(rel_path)
            except ValueError:
                # File is outside output dir, use filename only
                gcs_path = local_path.name

        blob = bucket_obj.blob(gcs_path)
        blob.upload_from_filename(str(local_path))

        file_size_mb = local_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Uploaded to GCS: gs://{bucket_name}/{gcs_path} ({file_size_mb:.2f} MB)")
        return True

    except Exception as e:
        logger.error(f"Failed to upload to GCS: {e}")
        return False


def download_from_gcs(
    gcs_path: str,
    local_path: Path | str | None = None,
    bucket: str | None = None,
) -> Path | None:
    """Download a file from GCS to local storage.

    Args:
        gcs_path: Path in GCS (e.g., "raw/flow/data.parquet")
        local_path: Destination local path (defaults to LOCAL_CACHE_DIR/gcs_path)
        bucket: GCS bucket name (defaults to GCS_BUCKET env var)

    Returns:
        Path to downloaded file, or None if failed
    """
    if not is_gcs_enabled():
        logger.debug("GCS not enabled, skipping download")
        return None

    try:
        bucket_name = bucket or get_gcs_bucket()
        client = _create_storage_client()
        bucket_obj = client.bucket(bucket_name)

        # Auto-generate local path from GCS path
        if local_path is None:
            output_dir = Path(os.getenv("LOCAL_CACHE_DIR", "output"))
            local_path = output_dir / gcs_path

        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        blob = bucket_obj.blob(gcs_path)
        if not blob.exists():
            logger.warning(f"GCS file not found: gs://{bucket_name}/{gcs_path}")
            return None

        blob.download_to_filename(str(local_path))

        file_size_mb = local_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Downloaded from GCS: {local_path} ({file_size_mb:.2f} MB)")
        return local_path

    except Exception as e:
        logger.error(f"Failed to download from GCS: {e}")
        return None


def sync_directory_to_gcs(
    local_dir: Path | str,
    gcs_prefix: str | None = None,
    bucket: str | None = None,
    exclude_patterns: list[str] | None = None,
) -> tuple[int, int]:
    """Sync entire local directory to GCS.

    Args:
        local_dir: Local directory to sync
        gcs_prefix: Prefix for GCS paths (e.g., "raw/")
        bucket: GCS bucket name (defaults to GCS_BUCKET env var)
        exclude_patterns: List of glob patterns to exclude

    Returns:
        Tuple of (files_uploaded, files_skipped)
    """
    if not is_gcs_enabled():
        logger.info("GCS not enabled, skipping sync")
        return 0, 0

    local_dir = Path(local_dir)
    if not local_dir.exists():
        logger.warning(f"Local directory not found: {local_dir}")
        return 0, 0

    exclude_patterns = exclude_patterns or []
    uploaded = 0
    skipped = 0

    try:
        # Recursively find all files
        for file_path in local_dir.rglob("*"):
            if not file_path.is_file():
                continue

            # Check exclusion patterns
            if any(file_path.match(pattern) for pattern in exclude_patterns):
                skipped += 1
                continue

            # Generate GCS path
            rel_path = file_path.relative_to(local_dir)
            if gcs_prefix:
                gcs_path = f"{gcs_prefix.rstrip('/')}/{rel_path}"
            else:
                gcs_path = str(rel_path)

            # Upload
            if upload_to_gcs(file_path, gcs_path, bucket):
                uploaded += 1
            else:
                skipped += 1

        logger.info(f"✅ Sync complete: {uploaded} uploaded, {skipped} skipped")
        return uploaded, skipped

    except Exception as e:
        logger.error(f"Failed to sync directory: {e}")
        return uploaded, skipped


def list_gcs_files(
    prefix: str = "",
    bucket: str | None = None,
) -> list[str]:
    """List files in GCS bucket with optional prefix filter.

    Args:
        prefix: Prefix to filter files (e.g., "raw/flow/")
        bucket: GCS bucket name (defaults to GCS_BUCKET env var)

    Returns:
        List of file paths in GCS
    """
    if not is_gcs_enabled():
        return []

    try:
        from google.cloud import storage

        bucket_name = bucket or get_gcs_bucket()
        client = storage.Client()
        bucket_obj = client.bucket(bucket_name)

        blobs = bucket_obj.list_blobs(prefix=prefix)
        files = [blob.name for blob in blobs if not blob.name.endswith("/")]

        logger.info(f"Found {len(files)} files in gs://{bucket_name}/{prefix}")
        return files

    except Exception as e:
        logger.error(f"Failed to list GCS files: {e}")
        return []


def find_latest_in_gcs(
    pattern: str,
    bucket: str | None = None,
) -> str | None:
    """Find the latest file matching pattern in GCS (by lexicographic sort).

    Args:
        pattern: Glob-like pattern (simple wildcard support)
        bucket: GCS bucket name (defaults to GCS_BUCKET env var)

    Returns:
        Latest matching file path in GCS, or None
    """
    if not is_gcs_enabled():
        return None

    try:
        # Convert simple glob pattern to prefix
        # e.g., "short_selling_*.parquet" -> "short_selling_"
        prefix = pattern.split("*")[0] if "*" in pattern else pattern

        files = list_gcs_files(prefix=prefix, bucket=bucket)

        # Filter by pattern (simple wildcard matching)
        import fnmatch
        matching = [f for f in files if fnmatch.fnmatch(f, pattern)]

        if not matching:
            return None

        # Return latest (lexicographically sorted, assumes YYYYMMDD in filename)
        latest = sorted(matching)[-1]
        logger.info(f"Latest in GCS: {latest}")
        return latest

    except Exception as e:
        logger.error(f"Failed to find latest in GCS: {e}")
        return None


def save_parquet_with_gcs(
    df,  # pl.DataFrame or pd.DataFrame
    local_path: Path | str,
    gcs_path: str | None = None,
    auto_sync: bool = True,
) -> Path:
    """Save parquet file locally and optionally upload to GCS.

    This is a convenience function for raw data and cache files that need
    both local storage and cloud backup.

    Args:
        df: Polars or Pandas DataFrame to save
        local_path: Local file path to save to
        gcs_path: Optional GCS path (auto-generated if None)
        auto_sync: Upload to GCS if GCS_SYNC_AFTER_SAVE=1 (default: True)

    Returns:
        Path to the saved local file
    """
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Save locally
    try:
        # Try Polars first, then Pandas
        if hasattr(df, 'write_parquet'):
            df.write_parquet(local_path)
        elif hasattr(df, 'to_parquet'):
            df.to_parquet(local_path)
        else:
            raise ValueError(f"Unsupported DataFrame type: {type(df)}")

        file_size_mb = local_path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved to local: {local_path} ({file_size_mb:.2f} MB)")
    except Exception as e:
        logger.error(f"Failed to save parquet locally: {e}")
        raise

    # Upload to GCS if enabled and auto_sync is True
    if auto_sync and os.getenv("GCS_SYNC_AFTER_SAVE") == "1":
        try:
            upload_to_gcs(local_path, gcs_path)
        except Exception as e:
            logger.warning(f"GCS sync failed (non-blocking): {e}")

    return local_path
