"""Production Data Loader V2

Direct implementation that loads data from parquet files.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class ProductionDatasetV2(Dataset):
    """Production dataset for financial time series data"""
    
    def __init__(self, data_dir: str, split: str = 'train', **kwargs):
        super().__init__()
        logger.info(f"ProductionDatasetV2.__init__ called with data_dir type: {type(data_dir)}")
        logger.info(f"data_dir value: {data_dir}")
        
        # Handle both string and DictConfig inputs
        if hasattr(data_dir, 'data_dir'):
            data_dir = data_dir.data_dir
            logger.info(f"Extracted from data_dir.data_dir: {data_dir}")
        elif hasattr(data_dir, 'source') and hasattr(data_dir.source, 'data_dir'):
            data_dir = data_dir.source.data_dir
            logger.info(f"Extracted from data_dir.source.data_dir: {data_dir}")
        elif isinstance(data_dir, dict) and 'source' in data_dir and 'data_dir' in data_dir['source']:
            data_dir = data_dir['source']['data_dir']
            logger.info(f"Extracted from dict['source']['data_dir']: {data_dir}")
        elif isinstance(data_dir, str) and data_dir.startswith('{') and 'data_dir' in data_dir:
            # Handle stringified dict case
            try:
                import ast
                parsed = ast.literal_eval(data_dir)
                if 'data' in parsed and 'source' in parsed['data'] and 'data_dir' in parsed['data']['source']:
                    data_dir = parsed['data']['source']['data_dir']
                    logger.info(f"Extracted from stringified dict: {data_dir}")
            except (ValueError, SyntaxError, KeyError) as e:
                logger.warning(f"Failed to parse stringified dict: {e}")
        
        # Extract the actual path string
        data_dir_str = str(data_dir)
        logger.info(f"Final data_dir_str: {data_dir_str}")
        self.data_dir = Path(data_dir_str)
        self.split = split
        self.data_files = []
        
        # Load data files
        split_dir = self.data_dir / split
        if split_dir.exists():
            self.data_files = sorted(list(split_dir.glob("*.parquet")))
            logger.info(f"Found {len(self.data_files)} files in {split_dir}")
        else:
            logger.warning(f"Split directory not found: {split_dir}")
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        if idx >= len(self.data_files):
            return {}
            
        file_path = self.data_files[idx]
        try:
            data = pd.read_parquet(file_path)
            
            # Convert to PyTorch-compatible types
            for col in data.columns:
                if data[col].dtype == 'object' or str(data[col].dtype) == 'object_':
                    # Try to convert object columns to numeric
                    try:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                    except:
                        # If conversion fails, drop the column
                        data = data.drop(columns=[col])
                        logger.warning(f"Dropped non-numeric column: {col}")
            
            # Ensure all remaining columns are numeric
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data = data[numeric_cols]
            
            # Convert to float32 for PyTorch compatibility
            data = data.astype(np.float32)
            
            # Fill any remaining NaN values with 0
            data = data.fillna(0.0)
            
            # Convert to tensor
            features = torch.tensor(data.values, dtype=torch.float32)
            
            # Pad or truncate to fixed sequence length if needed
            target_length = 20  # Standard sequence length
            if features.shape[0] > target_length:
                # Truncate to target length
                features = features[:target_length]
            elif features.shape[0] < target_length:
                # Pad with zeros
                padding = torch.zeros(target_length - features.shape[0], features.shape[1])
                features = torch.cat([features, padding], dim=0)
            
            # Pad or truncate features to fixed dimension
            target_features = 300  # Standard feature dimension
            if features.shape[1] > target_features:
                # Truncate to target features
                features = features[:, :target_features]
            elif features.shape[1] < target_features:
                # Pad with zeros
                padding = torch.zeros(features.shape[0], target_features - features.shape[1])
                features = torch.cat([features, padding], dim=1)
            
            # Extract actual target return columns from the data
            # Map horizon to actual column names in the data
            target_mapping = {
                'return_1d': 'label_ret_1_bps',
                'return_2d': 'label_ret_5_bps',  # Use 5d as closest to 2d
                'return_3d': 'label_ret_5_bps',  # Use 5d as closest to 3d
                'return_5d': 'label_ret_5_bps',
                'return_10d': 'label_ret_10_bps'
            }
            
            targets = {}
            
            for horizon_key, actual_col in target_mapping.items():
                if actual_col in data.columns:
                    # Use actual return data (convert from bps to decimal)
                    target_values = data[actual_col].values / 10000.0  # Convert bps to decimal
                    if len(target_values) >= features.shape[0]:
                        # Use the last N values for the sequence
                        targets[horizon_key] = torch.tensor(target_values[-features.shape[0]:], dtype=torch.float32)
                    else:
                        # Pad if necessary
                        padded = np.pad(target_values, (features.shape[0] - len(target_values), 0), 'constant', constant_values=0.0)
                        targets[horizon_key] = torch.tensor(padded, dtype=torch.float32)
                    # logger.info(f"Using actual data for {horizon_key} from {actual_col}")  # Too verbose
                else:
                    # Create small random targets as fallback (avoid zero gradients)
                    targets[horizon_key] = torch.randn(features.shape[0]) * 0.01
                    logger.warning(f"Column {actual_col} not found for {horizon_key}, using random fallback")
            
            # Ensure we have all required horizons
            for h in [1, 2, 3, 5, 10]:
                horizon_key = f'return_{h}d'
                if horizon_key not in targets:
                    targets[horizon_key] = torch.randn(features.shape[0]) * 0.01
                    logger.warning(f"Missing {horizon_key}, using random fallback")
            
            return {
                'features': features,
                'targets': targets,
                'file_path': str(file_path)
            }
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return {}


class ProductionDataLoaderV2:
    """Production data loader with configuration support"""
    
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4, **kwargs):
        # Handle both string and DictConfig inputs
        if hasattr(data_dir, 'data_dir'):
            data_dir = data_dir.data_dir
        elif hasattr(data_dir, 'source') and hasattr(data_dir.source, 'data_dir'):
            data_dir = data_dir.source.data_dir
        elif isinstance(data_dir, dict) and 'source' in data_dir and 'data_dir' in data_dir['source']:
            data_dir = data_dir['source']['data_dir']
        elif isinstance(data_dir, str) and data_dir.startswith('{') and 'data_dir' in data_dir:
            # Handle stringified dict case
            try:
                import ast
                parsed = ast.literal_eval(data_dir)
                if 'data' in parsed and 'source' in parsed['data'] and 'data_dir' in parsed['data']['source']:
                    data_dir = parsed['data']['source']['data_dir']
            except (ValueError, SyntaxError, KeyError):
                pass
        
        self.data_dir = str(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs
    
    def get_dataloader(self, split: str = 'train', shuffle: bool = True) -> DataLoader:
        dataset = ProductionDatasetV2(self.data_dir, split=split, **self.kwargs)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )


class ProductionDataModuleV2(pl.LightningDataModule):
    """Lightning data module for production data"""
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        **kwargs
    ):
        super().__init__()
        # Handle both string and DictConfig inputs
        if hasattr(data_dir, 'data_dir'):
            data_dir = data_dir.data_dir
        elif hasattr(data_dir, 'source') and hasattr(data_dir.source, 'data_dir'):
            data_dir = data_dir.source.data_dir
        elif isinstance(data_dir, dict) and 'source' in data_dir and 'data_dir' in data_dir['source']:
            data_dir = data_dir['source']['data_dir']
        elif isinstance(data_dir, str) and data_dir.startswith('{') and 'data_dir' in data_dir:
            # Handle stringified dict case
            try:
                import ast
                parsed = ast.literal_eval(data_dir)
                if 'data' in parsed and 'source' in parsed['data'] and 'data_dir' in parsed['data']['source']:
                    data_dir = parsed['data']['source']['data_dir']
            except (ValueError, SyntaxError, KeyError):
                pass
        
        self.data_dir = str(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs
        
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for train/val/test"""
        if stage == 'fit' or stage is None:
            self.train_dataset = ProductionDatasetV2(
                self.data_dir, split='train', **self.kwargs
            )
            self.val_dataset = ProductionDatasetV2(
                self.data_dir, split='val', **self.kwargs
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = ProductionDatasetV2(
                self.data_dir, split='test', **self.kwargs
            )
    
    def train_dataloader(self):
        if hasattr(self, 'train_dataset') and len(self.train_dataset) > 0:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            )
        return None
    
    def val_dataloader(self):
        if hasattr(self, 'val_dataset') and len(self.val_dataset) > 0:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
        return None
    
    def test_dataloader(self):
        if hasattr(self, 'test_dataset') and len(self.test_dataset) > 0:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
        return None


# Export required classes
__all__ = ["ProductionDatasetV2", "ProductionDataLoaderV2", "ProductionDataModuleV2"]