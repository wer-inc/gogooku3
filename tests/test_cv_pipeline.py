"""
Cross-validation pipeline integration tests.

Tests the complete ML pipeline with feature extensions:
- Data loading and transformation
- Cross-validation with embargo
- Model training with feature groups
- Evaluation metrics
"""

from __future__ import annotations

import pytest
import polars as pl
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any

from gogooku3.features_ext.sector_loo import add_sector_loo
from gogooku3.features_ext.scale_unify import add_ratio_adv_z
from gogooku3.features_ext.outliers import winsorize
from gogooku3.features_ext.interactions import add_interactions
from gogooku3.training.cv_purged import purged_kfold_indices
from gogooku3.training.datamodule import PanelDataModule
from gogooku3.training.model_multihead import MultiHeadRegressor
from gogooku3.training.losses import HuberMultiHorizon


class TestPipelineIntegration:
    """Test the complete ML pipeline integration."""

    @pytest.fixture
    def create_full_dataset(self) -> pl.DataFrame:
        """Create a complete dataset with all features."""
        np.random.seed(42)
        n_stocks = 50
        n_days = 100

        # Base data
        data = {
            "Date": [],
            "Code": [],
            "returns_1d": [],
            "returns_5d": [],
            "sector33_id": [],
            "volatility_20d": [],
            "dollar_volume_ma20": [],
        }

        # Technical indicators
        tech_cols = [
            "ma_gap_5_20", "ema_12", "bb_upper", "rsi_14",
            "volume_ratio_5", "z_close_20", "alpha_1d", "beta_stability_60d"
        ]

        # Market features
        mkt_cols = ["mkt_gap_5_20", "mkt_high_vol", "rel_to_sec_5d", "sec_mom_20", "rel_strength_5d"]

        # Flow/margin features
        flow_cols = ["flow_foreign_net_value", "flow_smart_idx", "margin_long_tot", "dmi_long", "dmi_short_to_adv20", "dmi_credit_ratio"]

        # Statement features
        stmt_cols = ["stmt_rev_fore_op", "stmt_progress_op", "stmt_days_since_statement"]

        for d in range(n_days):
            date = f"2024-{(d//30)+1:02d}-{(d%30)+1:02d}"
            for s in range(n_stocks):
                data["Date"].append(date)
                data["Code"].append(f"STOCK_{s:04d}")
                data["returns_1d"].append(np.random.randn() * 0.02)
                data["returns_5d"].append(np.random.randn() * 0.05)
                data["sector33_id"].append(s // 5)  # 10 sectors
                data["volatility_20d"].append(abs(np.random.randn()) * 0.1 + 0.01)
                data["dollar_volume_ma20"].append(abs(np.random.randn()) * 1e6 + 1e5)

        df = pl.DataFrame(data)

        # Add synthetic technical indicators
        for col in tech_cols:
            df = df.with_columns(pl.Series(col, np.random.randn(len(df))))

        # Add market features
        for col in mkt_cols:
            df = df.with_columns(pl.Series(col, np.random.randn(len(df))))

        # Add flow/margin features
        for col in flow_cols:
            df = df.with_columns(pl.Series(col, np.random.randn(len(df)) * 100))

        # Add statement features
        for col in stmt_cols:
            if "days_since" in col:
                df = df.with_columns(pl.Series(col, np.random.randint(1, 90, len(df))))
            else:
                df = df.with_columns(pl.Series(col, np.random.randn(len(df))))

        # Add target
        df = df.with_columns(
            (pl.col("returns_1d") + np.random.randn(len(df)) * 0.01).alias("target_1d")
        )

        return df

    def test_feature_extension_pipeline(self, create_full_dataset: pl.DataFrame) -> None:
        """Test the feature extension pipeline."""
        df = create_full_dataset

        initial_cols = len(df.columns)

        # 1. Add sector LOO
        df = add_sector_loo(df, ret_col="returns_1d", sec_col="sector33_id")
        assert "sec_ret_1d_eq_loo" in df.columns

        # 2. Add ratio/ADV/Z features
        df = add_ratio_adv_z(df, "margin_long_tot", "dollar_volume_ma20", prefix="margin_long")
        assert "margin_long_to_adv20" in df.columns
        assert "margin_long_z260" in df.columns

        # 3. Winsorize outliers
        outlier_cols = ["returns_1d", "returns_5d"]
        df = winsorize(df, outlier_cols, k=5.0)

        # 4. Add interactions
        df = add_interactions(df)
        interaction_cols = [c for c in df.columns if c.startswith("x_")]
        assert len(interaction_cols) == 10, f"Expected 10 interaction features, got {len(interaction_cols)}"

        # Should have added multiple columns
        assert len(df.columns) > initial_cols + 10

    def test_cv_split_with_embargo(self, create_full_dataset: pl.DataFrame) -> None:
        """Test cross-validation splitting with embargo."""
        df = create_full_dataset

        dates = df["Date"].to_numpy()
        folds = purged_kfold_indices(dates, n_splits=3, embargo_days=5)

        assert len(folds) == 3

        for i, fold in enumerate(folds):
            # Check sizes
            assert len(fold.train_idx) > 0, f"Fold {i} has no training data"
            assert len(fold.val_idx) > 0, f"Fold {i} has no validation data"

            # Check no overlap
            overlap = set(fold.train_idx) & set(fold.val_idx)
            assert len(overlap) == 0, f"Fold {i} has overlap"

    def test_datamodule_setup(self, create_full_dataset: pl.DataFrame) -> None:
        """Test data module setup for training."""
        df = create_full_dataset

        # Select feature columns
        feature_cols = [c for c in df.columns if c not in ["Date", "Code", "target_1d", "sector33_id"]]

        dm = PanelDataModule(
            df,
            feature_cols=feature_cols,
            target_col="target_1d",
            date_col="Date",
            by_cols=["sector33_id"],
            outlier_cols=["returns_1d", "returns_5d"],
            vol_col="volatility_20d"
        )

        # Setup first fold
        dates = df["Date"].to_numpy()
        folds = purged_kfold_indices(dates, n_splits=3, embargo_days=5)
        train_ds, val_ds, train_df, val_df = dm.setup_fold(folds[0])

        # Check dataset sizes
        assert len(train_ds) > 0
        assert len(val_ds) > 0

        # Check tensor shapes
        X_sample, y_sample, vol_sample = train_ds[0]
        assert X_sample.shape[0] == len(feature_cols)  # Features
        assert y_sample.shape == torch.Size([])  # Scalar target
        assert vol_sample is not None

    def test_model_training_smoke(self, create_full_dataset: pl.DataFrame) -> None:
        """Smoke test for model training."""
        df = create_full_dataset

        # Prepare data
        feature_cols = [c for c in df.columns if c not in ["Date", "Code", "target_1d", "sector33_id"]]

        dm = PanelDataModule(
            df,
            feature_cols=feature_cols,
            target_col="target_1d",
            date_col="Date"
        )

        # Setup fold
        dates = df["Date"].to_numpy()
        folds = purged_kfold_indices(dates, n_splits=3, embargo_days=5)
        train_ds, val_ds, _, _ = dm.setup_fold(folds[0])

        # Create model
        model = MultiHeadRegressor(
            in_dim=len(feature_cols),
            hidden=64,  # Small for testing
            out_heads=(1, 1, 1, 1, 1)
        )

        # Create loss and optimizer
        criterion = HuberMultiHorizon()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Training step
        model.train()
        X, y, vol = train_ds[0:32]  # Small batch
        X = X.unsqueeze(0) if X.dim() == 1 else X
        y = y.unsqueeze(0) if y.dim() == 0 else y
        if vol is not None:
            vol = vol.unsqueeze(0) if vol.dim() == 0 else vol

        optimizer.zero_grad()
        outputs = model(X)
        # All outputs predict same target for testing
        loss = criterion(outputs, [y for _ in outputs], vol20=vol)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0, "Loss should be positive"

    def test_feature_groups_dropout(self) -> None:
        """Test feature group dropout functionality."""
        from gogooku3.training.model_multihead import FeatureGroupDropout

        # Mock feature groups
        groups = {
            "MA": [0, 1, 2],
            "EMA": [3, 4, 5],
            "VOL": [6, 7],
        }

        dropout = FeatureGroupDropout(groups, p=0.5)

        # Test training mode
        dropout.train()
        X = torch.randn(32, 10)  # Batch of 32, 10 features
        X_dropped = dropout(X)

        # Some features should be zeroed (probabilistically)
        # We can't test exact behavior due to randomness,
        # but shape should be preserved
        assert X_dropped.shape == X.shape

        # Test eval mode (no dropout)
        dropout.eval()
        X_eval = dropout(X)
        assert torch.allclose(X_eval, X), "No dropout should occur in eval mode"

    def test_multi_horizon_loss(self) -> None:
        """Test multi-horizon loss calculation."""
        criterion = HuberMultiHorizon(
            deltas=(0.01, 0.015, 0.02, 0.025, 0.03),
            horizon_w=(1.0, 0.9, 0.8, 0.7, 0.6)
        )

        # Mock predictions and targets
        batch_size = 32
        preds = [torch.randn(batch_size, 1) for _ in range(5)]
        targets = [torch.randn(batch_size) for _ in range(5)]
        vol = torch.abs(torch.randn(batch_size)) + 0.01

        # Calculate loss
        loss = criterion(preds, targets, vol20=vol)

        assert loss.item() > 0, "Loss should be positive"
        assert loss.requires_grad, "Loss should have gradients"

    def test_ablation_stages(self, create_full_dataset: pl.DataFrame) -> None:
        """Test that ablation stages produce progressively more features."""
        df = create_full_dataset

        # Stage 1: Base
        base_cols = set(df.columns)

        # Stage 2: +LOO
        df_loo = add_sector_loo(df.clone(), ret_col="returns_1d", sec_col="sector33_id")
        loo_cols = set(df_loo.columns)
        assert len(loo_cols) > len(base_cols)

        # Stage 3: +ScaleUnify
        df_scale = add_ratio_adv_z(df_loo.clone(), "margin_long_tot", "dollar_volume_ma20", prefix="margin_long")
        scale_cols = set(df_scale.columns)
        assert len(scale_cols) > len(loo_cols)

        # Stage 4: +Outlier (doesn't add columns, but transforms)
        df_outlier = winsorize(df_scale.clone(), ["returns_1d"], k=5.0)
        outlier_cols = set(df_outlier.columns)
        assert len(outlier_cols) == len(scale_cols)  # Same columns, different values

        # Stage 5: +Interactions
        df_interact = add_interactions(df_outlier.clone())
        interact_cols = set(df_interact.columns)
        assert len(interact_cols) > len(outlier_cols)

        # Verify progression
        print(f"Feature progression: {len(base_cols)} → {len(loo_cols)} → {len(scale_cols)} → {len(interact_cols)}")


class TestMemoryAndPerformance:
    """Test memory efficiency and performance."""

    def test_pipeline_memory_usage(self, create_full_dataset: pl.DataFrame) -> None:
        """Test that pipeline doesn't exceed memory limits."""
        import psutil
        process = psutil.Process()

        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run feature extensions
        df = create_full_dataset
        df = add_sector_loo(df, ret_col="returns_1d", sec_col="sector33_id")
        df = add_ratio_adv_z(df, "margin_long_tot", "dollar_volume_ma20")
        df = add_interactions(df)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Should not increase by more than 100MB for test data
        assert memory_increase < 100, f"Memory increased by {memory_increase:.1f}MB"

    @pytest.mark.slow
    def test_pipeline_execution_time(self, create_full_dataset: pl.DataFrame) -> None:
        """Test that pipeline completes in reasonable time."""
        import time

        df = create_full_dataset

        start = time.time()

        # Full pipeline
        df = add_sector_loo(df, ret_col="returns_1d", sec_col="sector33_id")
        df = add_ratio_adv_z(df, "margin_long_tot", "dollar_volume_ma20")
        df = winsorize(df, ["returns_1d", "returns_5d"], k=5.0)
        df = add_interactions(df)

        # Setup data module
        feature_cols = [c for c in df.columns if c not in ["Date", "Code", "target_1d", "sector33_id"]]
        dm = PanelDataModule(df, feature_cols=feature_cols, target_col="target_1d")

        # Get folds
        dates = df["Date"].to_numpy()
        folds = purged_kfold_indices(dates, n_splits=3, embargo_days=5)

        # Setup one fold
        train_ds, val_ds, _, _ = dm.setup_fold(folds[0])

        elapsed = time.time() - start

        # Should complete in under 5 seconds for test data
        assert elapsed < 5.0, f"Pipeline took {elapsed:.1f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])