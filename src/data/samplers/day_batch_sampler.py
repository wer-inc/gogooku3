"""Day batch sampler for backward compatibility."""

import logging
from typing import Iterator, List, Any

logger = logging.getLogger(__name__)


class DayBatchSampler:
    """Day Batch Sampler for backward compatibility."""
    
    def __init__(self, *args, **kwargs):
        """Initialize DayBatchSampler."""
        logger.info("DayBatchSampler initialized")
        self.args = args
        self.kwargs = kwargs
    
    def __iter__(self) -> Iterator[List[int]]:
        """Iterate over batches."""
        logger.info("DayBatchSampler iteration started")
        return iter([])
    
    def __len__(self) -> int:
        """Get length of sampler."""
        return 0
