#!/usr/bin/env python3
"""Analyze training results from log file"""

import pandas as pd
import json
from pathlib import Path

# Find latest log
log_file = 'logs/ml_training.log'
if Path(log_file).exists():
    with open(log_file, 'r') as f:
        lines = f.readlines()

    # Extract metrics
    epochs = []
    for line in lines:
        if 'Epoch' in line and 'Train Loss' in line:
            try:
                parts = line.split('Epoch ')[1].split(':')[0].split('/')
                epoch = int(parts[0])

                # Extract losses
                train_loss = float(line.split('Train Loss=')[1].split(',')[0])
                val_loss = float(line.split('Val Loss=')[1].split(',')[0])

                # Extract RankIC if present
                rank_ic = 0.0
                if 'RankIC:' in line:
                    rank_ic = float(line.split('RankIC:')[1].split(',')[0].split()[0])

                epochs.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'rank_ic': rank_ic
                })
            except:
                pass

    if epochs:
        df = pd.DataFrame(epochs)
        print('📊 Training Summary:')
        print(f'  Total epochs: {len(df)}')
        print(f'  Best val_loss: {df["val_loss"].min():.4f} (epoch {df["val_loss"].idxmin() + 1})')
        print(f'  Final train_loss: {df["train_loss"].iloc[-1]:.4f}')
        print(f'  Final val_loss: {df["val_loss"].iloc[-1]:.4f}')
        if df["rank_ic"].any():
            print(f'  Best RankIC: {df["rank_ic"].max():.4f}')
else:
    print('❌ No log file found')