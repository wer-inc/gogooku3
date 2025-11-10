"""
Status management for chunk execution.

Manages status.json files for tracking chunk build progress:
- State tracking (running, completed, failed)
- Atomic file operations
- Resume/force logic
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ChunkStatus:
    """Chunk execution status constants."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StatusManager:
    """
    Manager for chunk status.json files.

    Provides atomic read/write operations and resume logic.
    """

    def __init__(self):
        """Initialize status manager."""
        pass

    def read_status(self, status_file: Path) -> Optional[Dict[str, Any]]:
        """
        Read status from status.json file.

        Args:
            status_file: Path to status.json

        Returns:
            Status dict or None if file doesn't exist
        """
        if not status_file.exists():
            return None

        try:
            with open(status_file, "r") as f:
                status = json.load(f)
            logger.debug(f"Read status from {status_file}: {status}")
            return status
        except Exception as e:
            logger.warning(f"Failed to read status from {status_file}: {e}")
            return None

    def write_status(self, status_file: Path, chunk_id: str, state: str, **kwargs) -> None:
        """
        Write status to status.json file (atomic).

        Args:
            status_file: Path to status.json
            chunk_id: Chunk identifier
            state: Chunk state (running, completed, failed)
            **kwargs: Additional status fields (rows, error, etc.)
        """
        # Ensure parent directory exists
        status_file.parent.mkdir(parents=True, exist_ok=True)

        # Build status payload
        payload = {"chunk_id": chunk_id, "state": state, "timestamp": datetime.utcnow().isoformat() + "Z", **kwargs}

        # Atomic write: write to temp file, then rename
        temp_file = status_file.with_suffix(".json.tmp")
        try:
            with open(temp_file, "w") as f:
                json.dump(payload, f, indent=2)
            temp_file.replace(status_file)
            logger.debug(f"Wrote status to {status_file}: {payload}")
        except Exception as e:
            logger.error(f"Failed to write status to {status_file}: {e}")
            if temp_file.exists():
                temp_file.unlink()
            raise

    def should_skip_chunk(
        self,
        status_file: Path,
        resume: bool,
        force: bool,
    ) -> bool:
        """
        Determine if chunk should be skipped.

        Args:
            status_file: Path to status.json
            resume: Resume mode (skip completed)
            force: Force mode (never skip)

        Returns:
            True if chunk should be skipped, False otherwise
        """
        # Force mode: never skip
        if force:
            logger.debug(f"Force mode: will not skip {status_file.parent.name}")
            return False

        # No resume mode: never skip
        if not resume:
            logger.debug(f"No resume: will not skip {status_file.parent.name}")
            return False

        # Read status
        status = self.read_status(status_file)

        # No status file: execute
        if status is None:
            logger.debug(f"No status file: will execute {status_file.parent.name}")
            return False

        # Check state
        state = status.get("state")
        if state == ChunkStatus.COMPLETED:
            logger.info(f"⏭️  Skipping completed chunk: {status_file.parent.name}")
            return True
        elif state == ChunkStatus.RUNNING:
            logger.warning(f"⚠️  Chunk {status_file.parent.name} was interrupted (state=running). " "Will re-execute.")
            return False
        elif state == ChunkStatus.FAILED:
            logger.info(f"🔄 Retrying failed chunk: {status_file.parent.name}")
            return False
        else:
            logger.warning(f"Unknown state '{state}' for {status_file.parent.name}, will execute")
            return False

    def mark_running(self, status_file: Path, chunk_id: str) -> None:
        """
        Mark chunk as running.

        Args:
            status_file: Path to status.json
            chunk_id: Chunk identifier
        """
        self.write_status(
            status_file,
            chunk_id=chunk_id,
            state=ChunkStatus.RUNNING,
        )

    def mark_completed(self, status_file: Path, chunk_id: str, rows: Optional[int] = None, **kwargs) -> None:
        """
        Mark chunk as completed.

        Args:
            status_file: Path to status.json
            chunk_id: Chunk identifier
            rows: Number of rows in output dataset
            **kwargs: Additional metadata
        """
        payload = {"state": ChunkStatus.COMPLETED}
        if rows is not None:
            payload["rows"] = rows
        payload.update(kwargs)

        self.write_status(status_file, chunk_id=chunk_id, **payload)

    def mark_failed(self, status_file: Path, chunk_id: str, error: str, **kwargs) -> None:
        """
        Mark chunk as failed.

        Args:
            status_file: Path to status.json
            chunk_id: Chunk identifier
            error: Error message
            **kwargs: Additional metadata
        """
        self.write_status(status_file, chunk_id=chunk_id, state=ChunkStatus.FAILED, error=error, **kwargs)

    def get_chunk_summary(self, chunks_dir: Path) -> Dict[str, int]:
        """
        Get summary of chunk states.

        Args:
            chunks_dir: Base chunks directory

        Returns:
            Dict with counts: {completed, failed, running, pending}
        """
        if not chunks_dir.exists():
            return {
                "completed": 0,
                "failed": 0,
                "running": 0,
                "pending": 0,
                "total": 0,
            }

        summary = {
            "completed": 0,
            "failed": 0,
            "running": 0,
            "pending": 0,
        }

        for chunk_dir in chunks_dir.iterdir():
            if not chunk_dir.is_dir():
                continue

            status_file = chunk_dir / "status.json"
            status = self.read_status(status_file)

            if status is None:
                summary["pending"] += 1
            else:
                state = status.get("state", "pending")
                summary[state] = summary.get(state, 0) + 1

        summary["total"] = sum(summary.values())
        return summary
