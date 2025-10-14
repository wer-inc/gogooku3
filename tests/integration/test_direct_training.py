#!/usr/bin/env python3
"""Direct training test - simplest possible setup"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

class SimpleDataset(Dataset):
    """Simple dataset that directly loads ML dataset"""

    def __init__(self, data_path, min_date="2020-01-01"):
        print(f"Loading data from {data_path}")
        self.df = pd.read_parquet(data_path)

        # Filter by date
        if 'Date' in self.df.columns:
            self.df = self.df[pd.to_datetime(self.df['Date']) >= min_date]
        elif 'date' in self.df.columns:
            self.df = self.df[pd.to_datetime(self.df['date']) >= min_date]

        print(f"Loaded {len(self.df)} samples after date filter")

        # Find feature columns (exclude metadata and targets)
        exclude_cols = ['Date', 'date', 'Code', 'code', 'row_idx',
                       'returns_1d', 'returns_5d', 'returns_10d', 'returns_20d',
                       'Section', 'MarketCode', 'sector17_name', 'sector33_name',
                       'sector17_code', 'sector33_code']

        # Get numeric columns only
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        # Filter to get feature columns
        self.feature_cols = [c for c in numeric_cols if c not in exclude_cols and not c.startswith('returns_')]
        self.target_cols = ['returns_1d', 'returns_5d', 'returns_10d', 'returns_20d']

        # Ensure target columns exist
        self.target_cols = [c for c in self.target_cols if c in self.df.columns]

        print(f"Features: {len(self.feature_cols)}, Targets: {len(self.target_cols)}")
        print(f"Feature columns (first 10): {self.feature_cols[:10]}")

        # Convert to numpy
        self.features = self.df[self.feature_cols].values.astype(np.float32)
        self.targets = self.df[self.target_cols].values.astype(np.float32)

        # Replace NaN with 0
        self.features = np.nan_to_num(self.features, 0.0)
        self.targets = np.nan_to_num(self.targets, 0.0)

        print(f"Feature shape: {self.features.shape}, Target shape: {self.targets.shape}")

        # Check target statistics
        for i, col in enumerate(self.target_cols):
            valid = np.isfinite(self.targets[:, i])
            nonzero = (self.targets[:, i] != 0) & valid
            print(f"  {col}: {nonzero.sum()} non-zero ({100*nonzero.sum()/len(self.targets):.1f}%)")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class SimpleModel(nn.Module):
    """Simple MLP model for testing"""

    def __init__(self, input_dim, hidden_dim=128, output_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


def train_step(model, dataloader, optimizer, criterion, device):
    """Single training step"""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, (features, targets) in enumerate(dataloader):
        features = features.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}")

            # Check if loss is zero
            if loss.item() == 0.0:
                print("  ⚠️ WARNING: Zero loss detected!")
                print(f"    Output sample: {outputs[0].detach().cpu().numpy()}")
                print(f"    Target sample: {targets[0].detach().cpu().numpy()}")

    return total_loss / num_batches


def main():
    print("=" * 60)
    print("DIRECT TRAINING TEST")
    print("=" * 60)

    # Configuration
    data_path = "output/ml_dataset_latest_full.parquet"
    batch_size = 256
    learning_rate = 1e-3
    hidden_dim = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice: {device}")

    # Load dataset
    dataset = SimpleDataset(data_path, min_date=os.getenv("MIN_TRAINING_DATE", "2020-01-01"))

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Single process for stability
    )

    # Create model
    input_dim = dataset.features.shape[1]
    output_dim = dataset.targets.shape[1]
    model = SimpleModel(input_dim, hidden_dim, output_dim).to(device)

    print(f"\nModel: {input_dim} -> {hidden_dim} -> {output_dim}")

    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Train for 1 epoch
    print("\nTraining for 1 epoch...")
    avg_loss = train_step(model, dataloader, optimizer, criterion, device)

    print(f"\nAverage loss: {avg_loss:.6f}")

    if avg_loss == 0.0:
        print("❌ Training failed - zero loss throughout")
    else:
        print("✅ Training successful - non-zero loss achieved")

    return avg_loss > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)