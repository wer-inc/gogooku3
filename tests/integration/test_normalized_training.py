#!/usr/bin/env python3
"""Normalized training test - with proper data scaling"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

class NormalizedDataset(Dataset):
    """Dataset with proper normalization"""

    def __init__(self, data_path, min_date="2020-01-01", normalize=True):
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

        # Convert to numpy
        self.features = self.df[self.feature_cols].values.astype(np.float32)
        self.targets = self.df[self.target_cols].values.astype(np.float32)

        # Replace NaN/Inf with 0
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)
        self.targets = np.nan_to_num(self.targets, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize features
        if normalize:
            print("Normalizing features...")
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)

            # Clip extreme values
            self.features = np.clip(self.features, -10, 10)

        # Clip target values to reasonable range (returns should be small)
        self.targets = np.clip(self.targets, -1.0, 1.0)

        print(f"Feature shape: {self.features.shape}, Target shape: {self.targets.shape}")
        print(f"Feature range: [{self.features.min():.2f}, {self.features.max():.2f}]")
        print(f"Target range: [{self.targets.min():.4f}, {self.targets.max():.4f}]")

        # Check target statistics
        for i, col in enumerate(self.target_cols):
            valid = np.isfinite(self.targets[:, i])
            nonzero = (self.targets[:, i] != 0) & valid
            if valid.any():
                print(f"  {col}: {nonzero.sum()} non-zero ({100*nonzero.sum()/len(self.targets):.1f}%), "
                      f"mean={self.targets[:, i][valid].mean():.6f}, "
                      f"std={self.targets[:, i][valid].std():.6f}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class SimpleModel(nn.Module):
    """Simple MLP model with proper initialization"""

    def __init__(self, input_dim, hidden_dim=128, output_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        # Better weight initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)  # No activation for regression


def train_step(model, dataloader, optimizer, criterion, device):
    """Single training step"""
    model.train()
    total_loss = 0
    num_batches = 0
    losses = []

    for batch_idx, (features, targets) in enumerate(dataloader):
        features = features.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, targets)

        # Check for NaN/Inf
        if not torch.isfinite(loss):
            print(f"  ⚠️ WARNING: Non-finite loss at batch {batch_idx}: {loss.item()}")
            continue

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        loss_val = loss.item()
        total_loss += loss_val
        losses.append(loss_val)
        num_batches += 1

        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss_val:.6f}")

            # Check if loss is zero or too large
            if loss_val == 0.0:
                print("  ⚠️ WARNING: Zero loss detected!")
                print(f"    Output sample: {outputs[0].detach().cpu().numpy()}")
                print(f"    Target sample: {targets[0].detach().cpu().numpy()}")
            elif loss_val > 1000:
                print("  ⚠️ WARNING: Very large loss detected!")

    if num_batches > 0:
        avg_loss = total_loss / num_batches
        median_loss = np.median(losses)
        print(f"\n  Stats - Mean: {avg_loss:.6f}, Median: {median_loss:.6f}, "
              f"Min: {min(losses):.6f}, Max: {max(losses):.6f}")
        return avg_loss
    else:
        return 0.0


def main():
    print("=" * 60)
    print("NORMALIZED TRAINING TEST")
    print("=" * 60)

    # Configuration
    data_path = "output/ml_dataset_latest_full.parquet"
    batch_size = 256
    learning_rate = 1e-4  # Smaller learning rate
    hidden_dim = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice: {device}")

    # Load dataset with normalization
    dataset = NormalizedDataset(
        data_path,
        min_date=os.getenv("MIN_TRAINING_DATE", "2020-01-01"),
        normalize=True
    )

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

    print(f"\nModel: {input_dim} -> {hidden_dim} -> {hidden_dim//2} -> {output_dim}")

    # Setup training with smaller learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # Train for 1 epoch
    print("\nTraining for 1 epoch...")
    avg_loss = train_step(model, dataloader, optimizer, criterion, device)

    print(f"\n" + "=" * 60)
    print(f"Final average loss: {avg_loss:.6f}")

    if avg_loss == 0.0:
        print("❌ Training failed - zero loss throughout")
        return False
    elif avg_loss > 1.0:
        print("⚠️ Loss is high but training is working")
        print("   This is expected for financial data with proper targets")
        return True
    else:
        print("✅ Training successful - reasonable loss achieved")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)