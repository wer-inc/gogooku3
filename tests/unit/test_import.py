#!/usr/bin/env python3
"""Test imports and basic functionality"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

print("Testing imports...")

try:
    from scripts.models.unified_feature_converter import UnifiedFeatureConverter

    print("✅ UnifiedFeatureConverter imported")
except ImportError as e:
    print(f"❌ UnifiedFeatureConverter import failed: {e}")

try:
    from src.models.architectures.atft_gat_fan import ATFT_GAT_FAN

    print("✅ ATFT_GAT_FAN imported")
except ImportError as e:
    print(f"❌ ATFT_GAT_FAN import failed: {e}")

try:
    from src.graph.graph_builder import GraphBuilder

    print("✅ GraphBuilder imported")
except ImportError as e:
    print(f"❌ GraphBuilder import failed: {e}")

print("\nTesting basic model creation...")

try:
    from types import SimpleNamespace

    from src.models.architectures.atft_gat_fan import ATFT_GAT_FAN

    # Minimal config
    config = SimpleNamespace(
        model=SimpleNamespace(hidden_size=64), data=SimpleNamespace()
    )

    print("Attempting to create model with minimal config...")
    # This will likely fail but show us what's needed
    model = ATFT_GAT_FAN(config)
    print("✅ Model created successfully")
except Exception as e:
    print(f"❌ Model creation failed: {e}")
    print("This is expected - shows what config is needed")

print("\nImport test complete.")
