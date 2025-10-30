"""Storage helpers for uploading artifacts."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..config import DatasetBuilderSettings, get_settings
from .logger import get_logger

LOGGER = get_logger("storage")


@dataclass
class StorageClient:
    """Placeholder storage client (extend with GCS integration)."""

    bucket: Optional[str] = None
    settings: DatasetBuilderSettings = field(default_factory=get_settings)

    def upload_file(self, path: Path, *, destination: Optional[str] = None) -> None:
        """Upload `path` to remote storage (currently a no-op).

        Implement GCS or S3 upload logic here during later phases.
        """

        dest = destination or path.name
        LOGGER.info("[noop] Upload %s to bucket=%s destination=%s", path, self.bucket, dest)

    def ensure_remote_symlink(self, *, target: str) -> None:
        """Placeholder for maintaining the `ml_dataset_latest` link remotely."""

        LOGGER.info("[noop] Ensure remote symlink points to %s", target)
