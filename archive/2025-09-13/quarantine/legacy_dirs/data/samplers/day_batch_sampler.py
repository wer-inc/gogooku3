"""Day Batch Sampler - Placeholder implementation"""

from torch.utils.data import Sampler


class DayBatchSampler(Sampler):
    """Placeholder sampler that groups samples by day"""
    
    def __init__(self, *args, **kwargs):
        self.indices = []
        
    def __iter__(self):
        return iter(self.indices)
        
    def __len__(self):
        return len(self.indices)


__all__ = ["DayBatchSampler"]
