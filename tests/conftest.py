"""
Pytest configuration and shared fixtures
"""

import pytest
import polars as pl
from pathlib import Path
from datetime import datetime
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


@pytest.fixture
def sample_price_data():
    """Sample price data for testing"""
    return pl.DataFrame(
        {
            "Code": ["7203", "7203", "7203", "9984", "9984", "9984"],
            "Date": [
                datetime(2024, 1, 4),
                datetime(2024, 1, 5),
                datetime(2024, 1, 9),
                datetime(2024, 1, 4),
                datetime(2024, 1, 5),
                datetime(2024, 1, 9),
            ],
            "Open": [2500.0, 2510.0, 2520.0, 5000.0, 5050.0, 5100.0],
            "High": [2550.0, 2560.0, 2570.0, 5100.0, 5150.0, 5200.0],
            "Low": [2490.0, 2500.0, 2510.0, 4950.0, 5000.0, 5050.0],
            "Close": [2530.0, 2540.0, 2550.0, 5080.0, 5130.0, 5180.0],
            "Volume": [1000000, 1100000, 1200000, 500000, 550000, 600000],
        }
    )


@pytest.fixture
def sample_topix_data():
    """Sample TOPIX data for testing"""
    return pl.DataFrame(
        {
            "Date": [
                datetime(2024, 1, 4),
                datetime(2024, 1, 5),
                datetime(2024, 1, 9),
            ],
            "Close": [2000.0, 2010.0, 2020.0],
        }
    )


@pytest.fixture
def test_config():
    """Test configuration"""
    return {
        "output_dir": "/tmp/gogooku3_test",
        "use_sample": True,
        "max_workers": 2,
    }
