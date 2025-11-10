#!/usr/bin/env python
"""
Minimal PyTorch training template with:
 - full RNG seeding (Python / NumPy / Torch / CUDA)
 - deterministic toggle
 - AMP (autocast + GradScaler)
 - DataLoader tuned for high throughput
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset


# --------------------------------------------------------------------------- #
#  Reproducibility helpers
# --------------------------------------------------------------------------- #
def set_global_seed(seed: int, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


# --------------------------------------------------------------------------- #
#  Dummy dataset/model for demonstration
# --------------------------------------------------------------------------- #
class DummyDataset(Dataset):
    def __init__(self, size: int = 10_000) -> None:
        self.x = torch.randn(size, 32)
        self.y = torch.randint(0, 2, (size,))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
#  Training config
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    seed: int = int(os.getenv("SEED", "42"))
    deterministic: bool = os.getenv("DETERMINISTIC", "0") == "1"
    use_amp: bool = os.getenv("USE_AMP", "1") == "1"
    batch_size: int = int(os.getenv("BATCH_SIZE", "512"))
    num_workers: int = int(os.getenv("NUM_WORKERS", "8"))
    pin_memory: bool = os.getenv("PIN_MEMORY", "1") == "1"
    prefetch_factor: int | None = int(os.getenv("PREFETCH_FACTOR", "3"))
    persistent_workers: bool = os.getenv("PERSISTENT_WORKERS", "1") == "1"
    epochs: int = int(os.getenv("EPOCHS", "3"))


def main() -> None:
    cfg = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_global_seed(cfg.seed, cfg.deterministic)

    dataset = DummyDataset()
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers if cfg.num_workers > 0 else 0,
        pin_memory=cfg.pin_memory if device.type == "cuda" else False,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
    )

    model = SimpleModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    scaler = GradScaler(device.type) if cfg.use_amp and device.type == "cuda" else GradScaler(enabled=False)

    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    for epoch in range(cfg.epochs):
        model.train()
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=autocast_device, enabled=cfg.use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        print(f"[epoch {epoch+1}] loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
