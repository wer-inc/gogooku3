"""Tests for walk-forward splitter."""
import pytest
from datetime import date, timedelta

import polars as pl

from apex_ranker.backtest.splitter import WalkForwardSplitter, Split


def generate_trading_dates(start: date, end: date, skip_weekends: bool = True):
    """Generate list of trading dates (skip weekends if specified)."""
    dates = []
    current = start
    while current <= end:
        if not skip_weekends or current.weekday() < 5:  # Mon-Fri
            dates.append(current)
        current += timedelta(days=1)
    return dates


def test_splitter_basic():
    """Test basic splitter functionality."""
    # Generate 2 years of trading days (~500 days)
    start = date(2023, 1, 1)
    end = date(2024, 12, 31)
    trading_dates = generate_trading_dates(start, end)

    splitter = WalkForwardSplitter(
        train_days=252,  # 1 year
        val_days=63,  # 3 months
        step_days=21,  # 1 month
    )

    splits = splitter.split(trading_dates)

    # Should have multiple folds
    assert len(splits) > 0, "Should generate at least one split"

    # Check first split
    first_split = splits[0]
    assert first_split.fold_id == 1
    assert first_split.train_start == trading_dates[0]
    assert first_split.val_start > first_split.train_end

    # Check splits don't overlap
    for i in range(len(splits) - 1):
        assert splits[i].val_end < splits[i + 1].train_start or \
               splits[i].train_end < splits[i + 1].train_start


def test_splitter_insufficient_data():
    """Test error handling with insufficient data."""
    # Only 100 days (less than min requirement)
    trading_dates = generate_trading_dates(date(2023, 1, 1), date(2023, 4, 10))

    splitter = WalkForwardSplitter(
        train_days=252,
        val_days=63,
    )

    with pytest.raises(ValueError, match="Insufficient data"):
        splitter.split(trading_dates)


def test_splitter_with_date_filter():
    """Test splitter with start/end date filters."""
    trading_dates = generate_trading_dates(date(2020, 1, 1), date(2025, 12, 31))

    splitter = WalkForwardSplitter(
        train_days=100,
        val_days=20,
        step_days=10,
    )

    # Filter to specific range
    splits = splitter.split(
        trading_dates,
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
    )

    # All splits should be within range
    for split in splits:
        assert split.train_start >= date(2023, 1, 1)
        assert split.val_end <= date(2023, 12, 31)


def test_splitter_with_gaps():
    """Test splitter with missing dates (holidays, data gaps)."""
    # Create dates with gaps (skip some weeks)
    trading_dates = []
    current = date(2023, 1, 1)
    while current < date(2024, 1, 1):
        # Add 4 days, skip 3 days (simulate gaps)
        for _ in range(4):
            if current.weekday() < 5:
                trading_dates.append(current)
            current += timedelta(days=1)
        current += timedelta(days=3)  # Gap

    splitter = WalkForwardSplitter(
        train_days=100,
        val_days=20,
        step_days=10,
    )

    splits = splitter.split(trading_dates)

    # Should still generate valid splits
    assert len(splits) > 0
    for split in splits:
        assert split.train_start in trading_dates
        assert split.train_end in trading_dates
        assert split.val_start in trading_dates
        assert split.val_end in trading_dates


def test_splitter_from_dataset():
    """Test splitter with polars DataFrame."""
    # Create mock dataset
    dates = generate_trading_dates(date(2023, 1, 1), date(2024, 12, 31))
    df = pl.DataFrame({
        "Date": dates,
        "value": range(len(dates)),
    })

    splitter = WalkForwardSplitter(
        train_days=100,
        val_days=20,
        step_days=10,
    )

    splits = splitter.split_from_dataset(df)

    assert len(splits) > 0
    assert splits[0].train_start == dates[0]


def test_get_date_masks():
    """Test getting train/val data masks."""
    dates = generate_trading_dates(date(2023, 1, 1), date(2023, 12, 31))
    df = pl.DataFrame({
        "Date": dates,
        "value": range(len(dates)),
    })

    splitter = WalkForwardSplitter(train_days=100, val_days=20, step_days=50)
    splits = splitter.split_from_dataset(df)

    # Get first split
    split = splits[0]
    train_data, val_data = splitter.get_date_masks(split, df)

    # Check data separation
    assert len(train_data) > 0
    assert len(val_data) > 0
    assert train_data["Date"].max() <= split.train_end
    assert val_data["Date"].min() >= split.val_start


def test_splitter_summary():
    """Test summary statistics generation."""
    dates = generate_trading_dates(date(2023, 1, 1), date(2024, 12, 31))

    splitter = WalkForwardSplitter(train_days=100, val_days=20, step_days=10)
    splits = splitter.split(dates)

    summary = splitter.summary(splits)

    assert summary["num_splits"] == len(splits)
    assert "date_range" in summary
    assert "train_window" in summary
    assert "val_window" in summary
    assert summary["step_size"] == 10


def test_edge_case_exact_fit():
    """Test when data exactly fits one split."""
    # Exactly 252 train + 63 val = 315 days
    dates = generate_trading_dates(date(2023, 1, 1), date(2023, 1, 1) + timedelta(days=400))
    dates = dates[:315]  # Exactly 315 trading days

    splitter = WalkForwardSplitter(
        train_days=252,
        val_days=63,
        step_days=1000,  # Large step to get only one split
    )

    splits = splitter.split(dates)

    assert len(splits) == 1
    assert splits[0].fold_id == 1


def test_min_requirements():
    """Test minimum day requirements."""
    dates = generate_trading_dates(date(2023, 1, 1), date(2024, 1, 1))

    splitter = WalkForwardSplitter(
        train_days=100,
        val_days=20,
        step_days=10,
        min_train_days=80,
        min_val_days=15,
    )

    splits = splitter.split(dates)

    # All splits should meet minimum requirements
    for split in splits:
        train_size = (split.train_end - split.train_start).days + 1
        val_size = (split.val_end - split.val_start).days + 1
        assert train_size >= 80
        assert val_size >= 15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
