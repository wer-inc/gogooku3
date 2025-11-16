#!/usr/bin/env python3
"""Simple test of J-Quants short_selling API."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "gogooku5" / "data" / "src"))

from builder.api.data_sources import DataSourceManager
from builder.config.settings import DatasetBuilderSettings

settings = DatasetBuilderSettings()
data_sources = DataSourceManager(settings=settings)

print("Testing short_selling API...")
print("=" * 80)

# Test 1: Recent week
print("\n1. Recent week (2025-11-01 to 2025-11-08)")
try:
    df = data_sources.short_selling(start="2025-11-01", end="2025-11-08")
    print(f"   Rows: {len(df) if df is not None else 0}")
    if df is not None and len(df) > 0:
        print(f"   Columns: {df.columns}")
        print(f"   Sample:\n{df.head(3)}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 2: 2024-01-15 to 2024-01-19
print("\n2. Sample week 2024 (2024-01-15 to 2024-01-19)")
try:
    df = data_sources.short_selling(start="2024-01-15", end="2024-01-19")
    print(f"   Rows: {len(df) if df is not None else 0}")
    if df is not None and len(df) > 0:
        print(f"   Columns: {df.columns}")
        print(f"   Sample:\n{df.head(3)}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 3: 2025Q1 (2025-01-06 to 2025-01-10)
print("\n3. Early 2025Q1 (2025-01-06 to 2025-01-10)")
try:
    df = data_sources.short_selling(start="2025-01-06", end="2025-01-10")
    print(f"   Rows: {len(df) if df is not None else 0}")
    if df is not None and len(df) > 0:
        print(f"   Columns: {df.columns}")
        print(f"   Sample:\n{df.head(3)}")
except Exception as e:
    print(f"   ERROR: {e}")

print("\n" + "=" * 80)
