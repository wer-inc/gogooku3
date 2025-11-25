"""
Progress tracking for chunk execution.

Provides real-time progress updates using tqdm or simple logging.
"""

import logging

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

logger = logging.getLogger(__name__)


class ProgressTracker:
    """
    Progress tracker for chunk execution.

    Uses tqdm if available, falls back to logging.
    """

    def __init__(
        self,
        total: int,
        desc: str = "Building chunks",
        use_tqdm: bool = True,
    ):
        """
        Initialize progress tracker.

        Args:
            total: Total number of chunks
            desc: Description text
            use_tqdm: Use tqdm if available
        """
        self.total = total
        self.current = 0
        self.desc = desc
        self.use_tqdm = use_tqdm and TQDM_AVAILABLE

        if self.use_tqdm:
            self.pbar = tqdm(total=total, desc=desc, unit="chunk")
        else:
            self.pbar = None
            logger.info(f"{desc}: 0/{total}")

    def update(self, n: int = 1) -> None:
        """
        Update progress by n chunks.

        Args:
            n: Number of chunks completed
        """
        self.current += n

        if self.pbar:
            self.pbar.update(n)
        else:
            logger.info(f"{self.desc}: {self.current}/{self.total}")

    def set_description(self, desc: str) -> None:
        """
        Update description.

        Args:
            desc: New description
        """
        self.desc = desc
        if self.pbar:
            self.pbar.set_description(desc)

    def close(self) -> None:
        """Close progress bar."""
        if self.pbar:
            self.pbar.close()
        else:
            logger.info(f"{self.desc}: {self.current}/{self.total} (done)")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
