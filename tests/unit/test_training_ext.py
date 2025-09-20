import numpy as np
import polars as pl
import torch

from gogooku3.training.cv_purged import purged_kfold_indices
from gogooku3.training.losses import HuberMultiHorizon
from gogooku3.training.model_multihead import MultiHeadRegressor


def test_purged_kfold_indices_embargo():
    days = np.array([f"2024-01-{d:02d}" for d in range(1, 11)], dtype="datetime64[D]")
    folds = purged_kfold_indices(days, n_splits=2, embargo_days=2)
    # Ensure no val indices appear in train masks
    for f in folds:
        assert set(f.train_idx).isdisjoint(set(f.val_idx))


def test_multihead_forward_and_loss():
    torch.manual_seed(0)
    X = torch.randn(8, 10)
    y = torch.randn(8)
    model = MultiHeadRegressor(in_dim=10, hidden=32, out_heads=(1, 1, 1))
    outs = model(X)
    assert len(outs) == 3
    assert outs[0].shape == (8, 1)
    crit = HuberMultiHorizon(deltas=(0.01, 0.02, 0.03), horizon_w=(1.0, 0.8, 0.6))
    loss = crit([o.squeeze(-1) for o in outs], [y, y, y])
    assert loss.ndim == 0

