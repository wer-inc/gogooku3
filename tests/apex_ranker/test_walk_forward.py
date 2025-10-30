"""
Unit tests for WalkForwardSplitter

Tests cover:
- Basic splitting functionality
- Edge cases (insufficient data, empty dates)
- Rolling vs expanding window modes
- Gap days configuration
- Date extraction methods
"""

from datetime import date, timedelta

import polars as pl
import pytest
from apex_ranker.backtest.walk_forward import WalkForwardFold, WalkForwardSplitter


class TestWalkForwardSplitter:
    """Test suite for WalkForwardSplitter"""

    @pytest.fixture
    def trading_dates(self):
        """Generate 500 weekday dates for testing"""
        all_dates = pl.date_range(
            date(2020, 1, 1),
            date(2022, 12, 31),
            interval="1d",
            eager=True,
        )
        return all_dates.filter(all_dates.dt.weekday() < 5)  # Weekdays only

    def test_basic_split_rolling(self, trading_dates):
        """Test basic rolling window split"""
        splitter = WalkForwardSplitter(
            train_days=100,
            test_days=20,
            step_days=10,
            mode="rolling",
        )

        folds = splitter.split(trading_dates)

        # Should generate multiple folds
        assert len(folds) > 0

        # Check first fold
        first = folds[0]
        assert first.fold_id == 1
        assert first.train_days == 100
        assert first.test_days == 20

        # Check train/test don't overlap
        assert first.test_start > first.train_end

    def test_basic_split_expanding(self, trading_dates):
        """Test basic expanding window split"""
        splitter = WalkForwardSplitter(
            train_days=100,
            test_days=20,
            step_days=10,
            mode="expanding",
        )

        folds = splitter.split(trading_dates)

        assert len(folds) > 0

        # First fold should have exactly train_days
        assert folds[0].train_days == 100

        # Subsequent folds should have more training days
        if len(folds) > 1:
            assert folds[1].train_days > folds[0].train_days
            assert folds[1].train_days == 100 + 10  # initial + step_days

    def test_gap_days(self, trading_dates):
        """Test gap between train and test"""
        splitter_no_gap = WalkForwardSplitter(
            train_days=100,
            test_days=20,
            step_days=10,
            gap_days=0,
        )

        splitter_with_gap = WalkForwardSplitter(
            train_days=100,
            test_days=20,
            step_days=10,
            gap_days=5,
        )

        folds_no_gap = splitter_no_gap.split(trading_dates)
        folds_with_gap = splitter_with_gap.split(trading_dates)

        # Both should generate folds
        assert len(folds_no_gap) > 0
        assert len(folds_with_gap) > 0

        # With gap, test_start should be further from train_end
        first_no_gap = folds_no_gap[0]
        first_with_gap = folds_with_gap[0]

        # Calculate gap in indices (not calendar days)
        date_list = trading_dates.to_list()
        train_end_idx_no_gap = date_list.index(first_no_gap.train_end)
        test_start_idx_no_gap = date_list.index(first_no_gap.test_start)
        gap_no_gap = test_start_idx_no_gap - train_end_idx_no_gap

        train_end_idx_with_gap = date_list.index(first_with_gap.train_end)
        test_start_idx_with_gap = date_list.index(first_with_gap.test_start)
        gap_with_gap = test_start_idx_with_gap - train_end_idx_with_gap

        assert gap_with_gap > gap_no_gap
        assert gap_with_gap == 5 + 1  # gap_days + 1 (next day)

    def test_insufficient_data(self):
        """Test error when insufficient dates"""
        short_dates = pl.date_range(
            date(2020, 1, 1),
            date(2020, 1, 10),  # Only 10 days
            interval="1d",
            eager=True,
        )

        splitter = WalkForwardSplitter(
            train_days=100,  # More than available
            test_days=20,
            step_days=10,
        )

        with pytest.raises(ValueError, match="Insufficient dates"):
            splitter.split(short_dates)

    def test_empty_dates(self):
        """Test error when dates is empty"""
        empty_dates = pl.Series("dates", [], dtype=pl.Date)

        splitter = WalkForwardSplitter()

        with pytest.raises(ValueError, match="dates cannot be empty"):
            splitter.split(empty_dates)

    def test_invalid_parameters(self):
        """Test parameter validation"""
        # train_days < min_train_days
        with pytest.raises(ValueError, match="train_days.*must be"):
            WalkForwardSplitter(train_days=50, min_train_days=126)

        # test_days < 1
        with pytest.raises(ValueError, match="test_days.*must be"):
            WalkForwardSplitter(test_days=0)

        # step_days < 1
        with pytest.raises(ValueError, match="step_days.*must be"):
            WalkForwardSplitter(step_days=0)

        # gap_days < 0
        with pytest.raises(ValueError, match="gap_days.*must be"):
            WalkForwardSplitter(gap_days=-1)

    def test_fold_progression(self, trading_dates):
        """Test that folds progress correctly"""
        splitter = WalkForwardSplitter(
            train_days=100,
            test_days=20,
            step_days=10,
            mode="rolling",
        )

        folds = splitter.split(trading_dates)

        for i in range(len(folds) - 1):
            fold1 = folds[i]
            fold2 = folds[i + 1]

            # Fold IDs should increment
            assert fold2.fold_id == fold1.fold_id + 1

            # Train windows should shift by step_days
            date_list = trading_dates.to_list()
            train_start_idx1 = date_list.index(fold1.train_start)
            train_start_idx2 = date_list.index(fold2.train_start)
            assert train_start_idx2 - train_start_idx1 == 10  # step_days

    def test_get_train_dates(self, trading_dates):
        """Test extraction of training dates"""
        splitter = WalkForwardSplitter(
            train_days=100,
            test_days=20,
            step_days=10,
        )

        folds = splitter.split(trading_dates)
        first_fold = folds[0]

        train_dates = splitter.get_train_dates(first_fold, trading_dates)

        # Should have exactly train_days dates
        assert len(train_dates) == first_fold.train_days

        # All dates should be within train window
        assert min(train_dates) == first_fold.train_start
        assert max(train_dates) == first_fold.train_end

    def test_get_test_dates(self, trading_dates):
        """Test extraction of test dates"""
        splitter = WalkForwardSplitter(
            train_days=100,
            test_days=20,
            step_days=10,
        )

        folds = splitter.split(trading_dates)
        first_fold = folds[0]

        test_dates = splitter.get_test_dates(first_fold, trading_dates)

        # Should have exactly test_days dates
        assert len(test_dates) == first_fold.test_days

        # All dates should be within test window
        assert min(test_dates) == first_fold.test_start
        assert max(test_dates) == first_fold.test_end

    def test_summary(self, trading_dates):
        """Test summary statistics generation"""
        splitter = WalkForwardSplitter(
            train_days=100,
            test_days=20,
            step_days=10,
        )

        summary = splitter.summary(trading_dates)

        # Check required keys
        assert "total_folds" in summary
        assert "mode" in summary
        assert "date_range" in summary
        assert "first_fold" in summary
        assert "last_fold" in summary
        assert "coverage" in summary

        # Check values
        assert summary["total_folds"] > 0
        assert summary["mode"] == "rolling"
        assert summary["train_days_config"] == 100
        assert summary["test_days_config"] == 20

    def test_list_input(self):
        """Test that splitter works with list input"""
        date_list = [date(2020, 1, 1) + timedelta(days=i) for i in range(500)]

        splitter = WalkForwardSplitter(
            train_days=100,
            test_days=20,
            step_days=10,
        )

        folds = splitter.split(date_list)

        assert len(folds) > 0
        assert folds[0].train_days == 100

    def test_fold_repr(self):
        """Test WalkForwardFold string representation"""
        fold = WalkForwardFold(
            fold_id=1,
            train_start=date(2020, 1, 1),
            train_end=date(2020, 6, 1),
            test_start=date(2020, 6, 2),
            test_end=date(2020, 9, 1),
            train_days=100,
            test_days=20,
        )

        repr_str = repr(fold)

        # Should contain key information
        assert "Fold 1" in repr_str
        assert "2020-01-01" in repr_str
        assert "2020-06-01" in repr_str
        assert "100d" in repr_str
        assert "20d" in repr_str


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
