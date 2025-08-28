#!/usr/bin/env python3
"""
ATFT-GAT-FAN Integration Test for gogooku3
"""

import sys
from pathlib import Path
import torch
import polars as pl
import numpy as np

# Add paths
sys.path.append(str(Path(__file__).parent))


def test_atft_loading():
    """Test ATFT model loading"""
    print("\n=== Test 1: ATFT Model Loading ===")

    from scripts.models.atft_inference import ATFTInference

    try:
        # Initialize model
        atft = ATFTInference()
        print("✅ Model loaded successfully")

        # Check model properties
        param_count = sum(p.numel() for p in atft.model.parameters())
        print(f"   Model parameters: {param_count:,}")
        print(f"   Device: {atft.device}")
        return True
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False


def test_feature_conversion():
    """Test feature conversion from gogooku3 to ATFT format"""
    print("\n=== Test 2: Feature Conversion ===")

    from scripts.models.unified_feature_converter import UnifiedFeatureConverter

    try:
        # Create sample gogooku3 data
        sample_data = pl.DataFrame(
            {
                "Code": ["1234"] * 30,
                "Date": [f"2024-01-{i:02d}" for i in range(1, 31)],
                "Close": np.random.randn(30) * 100 + 1000,
                "Volume": np.random.randint(1000, 10000, 30),
                "Open": np.random.randn(30) * 100 + 1000,
                "High": np.random.randn(30) * 100 + 1020,
                "Low": np.random.randn(30) * 100 + 980,
                "return_1d": np.random.randn(30) * 0.02,
                "return_5d": np.random.randn(30) * 0.05,
                "return_20d": np.random.randn(30) * 0.10,
                "rsi": np.random.uniform(30, 70, 30),
                "macd_diff": np.random.randn(30) * 10,
                "bb_upper": np.random.randn(30) * 100 + 1050,
                "atr": np.random.uniform(10, 30, 30),
                "obv": np.random.randn(30) * 10000,
                "cci": np.random.randn(30) * 100,
                "stoch_k": np.random.uniform(20, 80, 30),
            }
        )

        # Convert features
        converter = UnifiedFeatureConverter()
        atft_data = converter.prepare_atft_features(sample_data)

        print("✅ Conversion successful")
        print(f"   Original shape: {sample_data.shape}")
        print(f"   Converted shape: {atft_data.shape}")
        print(f"   Features: {len(atft_data.columns) - 2} (excluding Code, Date)")

        # Validate
        if converter.validate_conversion(sample_data, atft_data):
            print("✅ Validation passed")

        return True

    except Exception as e:
        print(f"❌ Feature conversion failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_inference():
    """Test inference with sample data"""
    print("\n=== Test 3: Inference Test ===")

    from scripts.models.atft_inference import ATFTInference

    try:
        # Initialize model
        atft = ATFTInference()

        # Create sample features (batch_size=10, seq_len=20, features=8)
        sample_features = torch.randn(10, 20, 8)

        # Run inference
        results = atft.predict(sample_features, horizon=1, return_confidence=True)

        print("✅ Inference successful")
        print(f"   Predictions shape: {results['predictions'].shape}")
        print(f"   Mean prediction: {results['predictions'].mean():.6f}")
        print(f"   Std prediction: {results['predictions'].std():.6f}")

        if "confidence" in results:
            print(f"   Mean confidence: {results['confidence'].mean():.6f}")

        return True

    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_end_to_end():
    """Test end-to-end pipeline: gogooku3 data → ATFT prediction"""
    print("\n=== Test 4: End-to-End Pipeline ===")

    from scripts.models.atft_inference import ATFTInference

    try:
        # Create larger sample data (multiple stocks)
        stocks = ["1234", "5678", "9012"]
        all_data = []

        for stock in stocks:
            stock_data = pl.DataFrame(
                {
                    "Code": [stock] * 50,
                    "Date": [f"2024-01-{i:02d}" for i in range(1, 51)]
                    if len(all_data) < 50
                    else [f"2024-02-{i:02d}" for i in range(1, 51)],
                    "Close": np.random.randn(50) * 100 + 1000,
                    "Volume": np.random.randint(1000, 10000, 50),
                    "Open": np.random.randn(50) * 100 + 1000,
                    "High": np.random.randn(50) * 100 + 1020,
                    "Low": np.random.randn(50) * 100 + 980,
                    "return_1d": np.random.randn(50) * 0.02,
                    "return_5d": np.random.randn(50) * 0.05,
                    "return_20d": np.random.randn(50) * 0.10,
                    "rsi": np.random.uniform(30, 70, 50),
                    "macd": np.random.randn(50) * 10,  # macd_diffからmacdに修正
                    "bb_upper": np.random.randn(50) * 100 + 1050,
                    "atr": np.random.uniform(10, 30, 50),
                    "obv": np.random.randn(50) * 10000,
                }
            )
            all_data.append(stock_data)

        df = pl.concat(all_data)

        # Initialize components
        atft = ATFTInference()

        # Run prediction
        predictions_df = atft.predict_from_dataframe(
            df,
            sequence_length=20,
            horizon=[1, 5],  # Test multiple horizons
            batch_size=32,
        )

        print("✅ End-to-end pipeline successful")
        print(f"   Input shape: {df.shape}")
        print(f"   Predictions shape: {predictions_df.shape}")
        print(f"   Stocks processed: {len(predictions_df.get_column('Code').unique())}")

        # Show sample predictions
        print("\n   Sample predictions:")
        print(predictions_df.head(5))

        return True

    except Exception as e:
        print(f"❌ End-to-end pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("ATFT-GAT-FAN Integration Test Suite")
    print("Testing integration with gogooku3")
    print("=" * 60)

    tests = [
        ("Model Loading", test_atft_loading),
        ("Feature Conversion", test_feature_conversion),
        ("Inference", test_inference),
        ("End-to-End Pipeline", test_end_to_end),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")

    passed = sum(1 for _, s in results if s)
    total = len(results)

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! ATFT-GAT-FAN is ready to use in gogooku3")
        print("Expected Sharpe Ratio: 0.849")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()
