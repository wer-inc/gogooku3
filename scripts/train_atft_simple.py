#!/usr/bin/env python3
"""
Simplified ATFT-GAT-FAN training script
Direct training without complex pipeline
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from types import SimpleNamespace

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleMLDataset(Dataset):
    """Simple dataset for ML training"""
    
    def __init__(self, df: pl.DataFrame, seq_len: int = 60):
        # Filter numeric columns only
        numeric_cols = []
        for col in df.columns:
            if col not in ["Code", "Date"]:
                try:
                    # Try to convert to numeric
                    _ = df[col].cast(pl.Float32)
                    numeric_cols.append(col)
                except:
                    pass
        
        logger.info(f"Using {len(numeric_cols)} numeric columns out of {len(df.columns)}")
        
        # Convert to numpy
        self.features = df.select(numeric_cols).fill_null(0).to_numpy().astype(np.float32)
        
        # Find target column
        target_cols = ["feat_ret_1d", "returns_1d", "target"]
        self.target_col = None
        for col in target_cols:
            if col in numeric_cols:
                self.target_col = col
                self.target_idx = numeric_cols.index(col)
                break
        
        if self.target_col:
            logger.info(f"Using target column: {self.target_col}")
            self.targets = self.features[:, self.target_idx]
        else:
            logger.warning("No target column found, using random targets")
            self.targets = np.random.randn(len(self.features)).astype(np.float32) * 0.01
        
        self.seq_len = seq_len
        self.n_features = self.features.shape[1]
        
    def __len__(self):
        return max(1, len(self.features) - self.seq_len)
    
    def __getitem__(self, idx):
        # Create sequence
        if idx + self.seq_len < len(self.features):
            seq = self.features[idx:idx+self.seq_len]
            target = self.targets[idx+self.seq_len-1]
        else:
            # Pad if needed
            seq = np.zeros((self.seq_len, self.n_features), dtype=np.float32)
            available = min(self.seq_len, len(self.features) - idx)
            seq[:available] = self.features[idx:idx+available]
            target = self.targets[min(idx+available-1, len(self.targets)-1)]
        
        return torch.from_numpy(seq), torch.tensor(target)


def train_simple(args):
    """Simple training function"""
    
    logger.info("=" * 60)
    logger.info("Simple ATFT-GAT-FAN Training")
    logger.info("=" * 60)
    
    # Load data
    logger.info(f"Loading dataset: {args.data_path}")
    df = pl.read_parquet(args.data_path)
    logger.info(f"Dataset shape: {df.shape}")
    
    # Split data
    n_samples = len(df)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size+val_size]
    test_df = df[train_size+val_size:]
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create datasets
    train_dataset = SimpleMLDataset(train_df, seq_len=60)
    val_dataset = SimpleMLDataset(val_df, seq_len=60)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    logger.info("Creating ATFT-GAT-FAN model...")
    from src.models.architectures.atft_gat_fan import ATFT_GAT_FAN
    
    # More complete config structure
    config = SimpleNamespace(
        model=SimpleNamespace(
            hidden_size=128,
            num_layers=3,
            num_heads=4,
            dropout=0.2,
            input_projection=SimpleNamespace(
                hidden_size=128,
                dropout=0.1
            ),
            adaptive_normalization=SimpleNamespace(
                fan=SimpleNamespace(
                    enabled=True,
                    window_sizes=[5, 10, 20]
                ),
                san=SimpleNamespace(
                    enabled=True,
                    num_slices=3
                )
            ),
            graph_attention=SimpleNamespace(
                enabled=True,
                num_layers=2,
                hidden_channels=[64, 32],
                heads=[4, 2],
                dropout=0.2,
                edge_dropout=0.1
            )
        ),
        data=SimpleNamespace(
            features=SimpleNamespace(
                basic=SimpleNamespace(
                    price_volume=['Open', 'High', 'Low', 'Close', 'Volume'],
                    flags=[]
                ),
                ta=SimpleNamespace(
                    returns=['returns_1d', 'returns_5d', 'returns_10d', 'returns_20d'],
                    moving_averages=['ema_5', 'ema_10', 'ema_20', 'ema_60', 'ema_200'],
                    momentum=['rsi_14', 'rsi_2'],
                    volatility=['volatility_20d'],
                    trend=['macd_signal', 'macd_histogram'],
                    volume=[],
                    custom=['sharpe_1d', 'bb_pct_b', 'bb_bandwidth']
                ),
                market=SimpleNamespace(
                    topix=[],
                    sector=[]
                ),
                flow=SimpleNamespace(
                    proprietary=[],
                    foreign=[],
                    individual=[]
                ),
                statements=SimpleNamespace(
                    fundamental=[]
                ),
                dynamic_raw_features=train_dataset.n_features,
                static_features=0,
                use_static_features=False
            ),
            input_features=SimpleNamespace(
                dynamic_features=train_dataset.n_features,
                static_features=0,
                use_static_features=False
            ),
            time_series=SimpleNamespace(
                sequence_length=60,
                prediction_horizons=[1, 5, 10, 20]
            )
        ),
        training=SimpleNamespace(
            learning_rate=1e-4,
            weight_decay=1e-5
        )
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = ATFT_GAT_FAN(config).to(device)
    
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {param_count:,}")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.max_epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            if batch_idx >= 10:  # Limit batches for quick test
                break
                
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            # Build graph (simple version)
            from src.graph.graph_builder import GraphBuilder, GBConfig
            graph_builder = GraphBuilder(GBConfig())
            edge_index, edge_attr = graph_builder.build_graph(sequences)
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Model expects different input format
            outputs = model(sequences, None, edge_index, edge_attr)
            
            # Extract predictions for horizon 1
            if isinstance(outputs, dict):
                if 'point_horizon_1' in outputs:
                    predictions = outputs['point_horizon_1'].squeeze()
                else:
                    predictions = list(outputs.values())[0].squeeze()
            else:
                predictions = outputs.squeeze()
            
            # Calculate loss
            loss = criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            if batch_idx % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{args.max_epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}")
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_idx, (sequences, targets) in enumerate(val_loader):
                if batch_idx >= 5:  # Limit validation batches
                    break
                    
                sequences = sequences.to(device)
                targets = targets.to(device)
                
                edge_index, edge_attr = graph_builder.build_graph(sequences)
                edge_index = edge_index.to(device)
                edge_attr = edge_attr.to(device)
                
                outputs = model(sequences, None, edge_index, edge_attr)
                
                if isinstance(outputs, dict):
                    if 'point_horizon_1' in outputs:
                        predictions = outputs['point_horizon_1'].squeeze()
                    else:
                        predictions = list(outputs.values())[0].squeeze()
                else:
                    predictions = outputs.squeeze()
                
                loss = criterion(predictions, targets)
                val_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses) if train_losses else 0
        avg_val_loss = np.mean(val_losses) if val_losses else 0
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            checkpoint_dir = Path("output/checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = checkpoint_dir / "simple_best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            
            logger.info(f"✅ Saved best model to {checkpoint_path}")
    
    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Simple ATFT-GAT-FAN Training")
    parser.add_argument("--data-path", type=str, required=True, help="Path to ML dataset")
    parser.add_argument("--max-epochs", type=int, default=5, help="Maximum epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    
    train_simple(args)


if __name__ == "__main__":
    main()
