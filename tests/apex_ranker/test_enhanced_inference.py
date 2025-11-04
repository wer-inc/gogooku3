"""Unit tests for enhanced inference improvements (A.1-A.4)."""
import numpy as np
import pandas as pd
import pytest

from apex_ranker.backtest.enhanced_inference import (
    filter_uncertain,
    hysteresis_selection,
    rank_ensemble,
    risk_neutralize,
)


class TestRankEnsemble:
    """Test A.1: Rank ensemble functionality."""

    def test_basic_ensemble(self):
        """Test basic rank averaging across 2 folds."""
        fold_scores = {
            "fold_0": {"5d": np.array([0.5, 0.8, 0.3], dtype=np.float32)},
            "fold_1": {"5d": np.array([0.6, 0.7, 0.4], dtype=np.float32)},
        }

        mean_rank = rank_ensemble(fold_scores, "5d")

        # Check output shape
        assert mean_rank.shape == (3,)
        assert mean_rank.dtype == np.float32

        # Check that highest consensus stock has lowest rank
        # Stock 1 has highest scores in both folds (0.8, 0.7)
        assert mean_rank[1] < mean_rank[0]
        assert mean_rank[1] < mean_rank[2]

    def test_empty_dict_raises(self):
        """Test that empty fold dict raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            rank_ensemble({}, "5d")

    def test_missing_horizon_raises(self):
        """Test that missing horizon key raises KeyError."""
        fold_scores = {
            "fold_0": {"10d": np.array([0.5, 0.8])},
        }
        with pytest.raises(KeyError, match="5d"):
            rank_ensemble(fold_scores, "5d")

    def test_single_fold(self):
        """Test that single fold returns normalized ranks."""
        fold_scores = {
            "fold_0": {"5d": np.array([0.5, 0.8, 0.3], dtype=np.float32)},
        }

        mean_rank = rank_ensemble(fold_scores, "5d")
        assert mean_rank.shape == (3,)
        # Normalized ranks should have ~zero mean
        assert abs(mean_rank.mean()) < 1e-6


class TestFilterUncertain:
    """Test A.2: Uncertainty filter functionality."""

    def test_basic_filter(self):
        """Test basic uncertainty filtering with 2 folds."""
        fold_scores = {
            "fold_0": {"5d": np.array([0.5, 0.8, 0.3], dtype=np.float32)},
            "fold_1": {"5d": np.array([0.6, 0.2, 0.4], dtype=np.float32)},
        }

        # Exclude top 30% most uncertain
        mask = filter_uncertain(fold_scores, "5d", top_pct=0.3)

        # Check output shape and type
        assert mask.shape == (3,)
        assert mask.dtype == np.bool_

        # Stock 1 has high variance (0.8 → 0.2), should be excluded
        assert not mask[1], "High variance stock should be excluded"

        # At least 1 stock should be kept (70% retention)
        assert mask.sum() >= 1

    def test_empty_dict_raises(self):
        """Test that empty fold dict raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            filter_uncertain({}, "5d")

    def test_invalid_top_pct(self):
        """Test that invalid top_pct raises ValueError."""
        fold_scores = {"fold_0": {"5d": np.array([0.5, 0.8])}}

        with pytest.raises(ValueError, match="top_pct"):
            filter_uncertain(fold_scores, "5d", top_pct=1.5)

        with pytest.raises(ValueError, match="top_pct"):
            filter_uncertain(fold_scores, "5d", top_pct=-0.1)

    def test_zero_variance_all_kept(self):
        """Test that zero variance (identical ranks) keeps all stocks."""
        fold_scores = {
            "fold_0": {"5d": np.array([0.5, 0.8, 0.3], dtype=np.float32)},
            "fold_1": {"5d": np.array([0.5, 0.8, 0.3], dtype=np.float32)},
        }

        mask = filter_uncertain(fold_scores, "5d", top_pct=0.2)
        # All stocks have zero variance, all should be kept
        assert mask.all()


class TestHysteresisSelection:
    """Test A.3: Exit hysteresis functionality."""

    def test_basic_hysteresis(self):
        """Test basic hysteresis selection."""
        scores = np.array([0.9, 0.7, 0.5, 0.3, 0.1], dtype=np.float32)
        current = [1, 2]  # Currently hold stocks 1, 2

        holdings = hysteresis_selection(scores, current, entry_k=2, exit_k=3)

        # Should return list of stock indices
        assert isinstance(holdings, list)
        assert len(holdings) == 2  # Exactly entry_k stocks

        # Stock 1 (rank 2) should be kept (within exit_k=3)
        assert 1 in holdings

        # Stock 2 (rank 3) should be kept (within exit_k=3)
        assert 2 in holdings

        # Stock 0 is NOT added because we're already at entry_k=2
        # This is correct hysteresis behavior - keep existing holdings within exit_k

    def test_initial_selection_no_holdings(self):
        """Test initial selection with no current holdings."""
        scores = np.array([0.9, 0.7, 0.5, 0.3, 0.1], dtype=np.float32)

        holdings = hysteresis_selection(scores, None, entry_k=3, exit_k=5)

        # Should return top entry_k stocks
        assert len(holdings) == 3
        assert sorted(holdings) == [0, 1, 2]

    def test_empty_current_holdings(self):
        """Test with empty current holdings list."""
        scores = np.array([0.9, 0.7, 0.5], dtype=np.float32)

        holdings = hysteresis_selection(scores, [], entry_k=2, exit_k=3)

        # Should return top entry_k stocks
        assert len(holdings) == 2
        assert sorted(holdings) == [0, 1]

    def test_invalid_k_values(self):
        """Test that invalid k values raise ValueError."""
        scores = np.array([0.9, 0.7, 0.5], dtype=np.float32)

        with pytest.raises(ValueError, match="must be > 0"):
            hysteresis_selection(scores, None, entry_k=0, exit_k=3)

        with pytest.raises(ValueError, match="exit_k must be >= entry_k"):
            hysteresis_selection(scores, None, entry_k=5, exit_k=3)

    def test_exit_drops_low_ranked(self):
        """Test that low-ranked holdings are dropped."""
        scores = np.array([0.9, 0.7, 0.5, 0.3, 0.1], dtype=np.float32)
        current = [3, 4]  # Hold low-ranked stocks

        holdings = hysteresis_selection(scores, current, entry_k=2, exit_k=3)

        # Low-ranked stocks (3, 4) should be dropped (outside exit_k=3)
        assert 3 not in holdings
        assert 4 not in holdings

        # Top stocks should be added
        assert 0 in holdings
        assert 1 in holdings


class TestRiskNeutralize:
    """Test A.4: Risk neutralization functionality."""

    def test_basic_neutralization(self):
        """Test basic risk neutralization with beta and size."""
        scores = np.array([0.5, 0.8, 0.3, 0.6], dtype=np.float32)
        df = pd.DataFrame({
            "beta_60d": [1.2, 0.8, 1.0, 1.1],
            "log_mktcap": [10.5, 11.2, 9.8, 10.0],
        })

        residuals = risk_neutralize(
            scores, df, factors=["beta_60d", "log_mktcap"], alpha=0.1
        )

        # Check output shape and type
        assert residuals.shape == scores.shape
        assert residuals.dtype == np.float32

        # Residuals should have lower correlation with factors
        beta_corr_orig = np.corrcoef(scores, df["beta_60d"])[0, 1]
        beta_corr_resid = np.corrcoef(residuals, df["beta_60d"])[0, 1]
        assert abs(beta_corr_resid) < abs(beta_corr_orig) or np.isclose(beta_corr_orig, 0)

    def test_with_sector(self):
        """Test neutralization with categorical sector factor."""
        scores = np.array([0.5, 0.8, 0.3, 0.6], dtype=np.float32)
        df = pd.DataFrame({
            "beta_60d": [1.2, 0.8, 1.0, 1.1],
            "log_mktcap": [10.5, 11.2, 9.8, 10.0],
            "Sector33Code": ["17", "50", "17", "50"],
        })

        residuals = risk_neutralize(
            scores, df, factors=["beta_60d", "log_mktcap", "Sector33Code"]
        )

        # Should work without errors
        assert residuals.shape == scores.shape

    def test_missing_factors_raises(self):
        """Test that missing factors raise KeyError."""
        scores = np.array([0.5, 0.8], dtype=np.float32)
        df = pd.DataFrame({"beta_60d": [1.2, 0.8]})

        with pytest.raises(KeyError, match="Missing factors"):
            risk_neutralize(scores, df, factors=["beta_60d", "nonexistent"])

    def test_length_mismatch_raises(self):
        """Test that length mismatch raises ValueError."""
        scores = np.array([0.5, 0.8, 0.3], dtype=np.float32)
        df = pd.DataFrame({"beta_60d": [1.2, 0.8]})  # Only 2 rows

        with pytest.raises(ValueError, match="Length mismatch"):
            risk_neutralize(scores, df, factors=["beta_60d"])

    def test_handles_nan_values(self):
        """Test that NaN values in factors are handled gracefully."""
        scores = np.array([0.5, 0.8, 0.3, 0.6], dtype=np.float32)
        df = pd.DataFrame({
            "beta_60d": [1.2, np.nan, 1.0, 1.1],
            "log_mktcap": [10.5, 11.2, np.nan, 10.0],
        })

        residuals = risk_neutralize(scores, df, factors=["beta_60d", "log_mktcap"])

        # Should complete without errors
        assert residuals.shape == scores.shape
        assert not np.any(np.isnan(residuals))

    def test_default_factors(self):
        """Test that default factors work when present."""
        scores = np.array([0.5, 0.8, 0.3], dtype=np.float32)
        df = pd.DataFrame({
            "beta_60d": [1.2, 0.8, 1.0],
            "log_mktcap": [10.5, 11.2, 9.8],
            "Sector33Code": ["17", "50", "17"],
        })

        # Should use default factors
        residuals = risk_neutralize(scores, df)
        assert residuals.shape == scores.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
