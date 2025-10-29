#!/usr/bin/env python3
"""
Verify dataset features against documentation specification.

This script checks if expected features from docs/ml/dataset.md
are present in the generated dataset.

Note: Theoretical maximum is 395 features with all data sources enabled.
Currently ~303-307 features are generated (88-92 futures features disabled).
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import polars as pl


def load_dataset_columns(dataset_path: Path) -> List[str]:
    """Load column names from a parquet dataset."""
    try:
        df = pl.scan_parquet(dataset_path)
        return sorted(df.columns)
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        sys.exit(1)


def categorize_features(columns: List[str]) -> Dict[str, List[str]]:
    """Categorize features by their prefix/type."""
    categories = {
        # Identifiers and metadata
        "identifiers": [],

        # Price and volume
        "returns": [],
        "volume": [],
        "moving_averages": [],
        "volatility": [],

        # Technical indicators
        "rsi": [],
        "macd": [],
        "bollinger": [],
        "other_technical": [],

        # Market features
        "market": [],
        "market_z": [],

        # Cross features
        "beta_alpha": [],
        "relative": [],

        # Sector features
        "sector_base": [],
        "sector_aggregates": [],
        "sector_relative": [],

        # Flow features
        "flow": [],

        # Margin features
        "margin_weekly": [],
        "margin_daily": [],

        # Statement features
        "statements": [],

        # Interaction features
        "interactions": [],

        # Target features
        "targets": [],

        # Validity flags
        "validity": [],

        # Other
        "other": []
    }

    for col in columns:
        # Skip internal columns
        if col.startswith("_"):
            continue

        # Categorize based on prefix/pattern
        if col in ["Code", "Date", "Section", "MarketCode", "row_idx", "SharesOutstanding", "shares_outstanding"]:
            categories["identifiers"].append(col)
        elif col.startswith("sector"):
            if any(x in col for x in ["_eq", "_mom", "_vol", "_z"]):
                categories["sector_aggregates"].append(col)
            elif "rel_to_sec" in col or "z_in_sec" in col:
                categories["sector_relative"].append(col)
            else:
                categories["sector_base"].append(col)
        elif col.startswith("returns_") or col == "returns_1d":
            categories["returns"].append(col)
        elif col.startswith("log_returns"):
            categories["returns"].append(col)
        elif "volume" in col.lower() or col == "Volume" or col == "TurnoverValue":
            categories["volume"].append(col)
        elif any(x in col for x in ["sma", "ema", "ma_"]):
            categories["moving_averages"].append(col)
        elif "volatility" in col or "realized_vol" in col:
            categories["volatility"].append(col)
        elif col.startswith("rsi"):
            categories["rsi"].append(col)
        elif "macd" in col.lower():
            categories["macd"].append(col)
        elif col.startswith("bb_") or "bollinger" in col:
            categories["bollinger"].append(col)
        elif any(x in col for x in ["atr", "adx", "stoch"]):
            categories["other_technical"].append(col)
        elif col.startswith("mkt_") and "_z" in col:
            categories["market_z"].append(col)
        elif col.startswith("mkt_"):
            categories["market"].append(col)
        elif col.startswith("beta") or col.startswith("alpha"):
            categories["beta_alpha"].append(col)
        elif col.startswith("rel_") or "relative" in col:
            categories["relative"].append(col)
        elif col.startswith("sec_"):
            categories["sector_aggregates"].append(col)
        elif col.startswith("flow_"):
            categories["flow"].append(col)
        elif col.startswith("margin_"):
            categories["margin_weekly"].append(col)
        elif col.startswith("dmi_"):
            categories["margin_daily"].append(col)
        elif col.startswith("stmt_"):
            categories["statements"].append(col)
        elif col.startswith("x_"):
            categories["interactions"].append(col)
        elif col.startswith("feat_ret_") or col.startswith("target_"):
            categories["targets"].append(col)
        elif col.startswith("is_") and "_valid" in col:
            categories["validity"].append(col)
        elif any(x in col for x in ["Open", "High", "Low", "Close", "high_low", "close_to"]):
            categories["other_technical"].append(col)
        else:
            categories["other"].append(col)

    return categories


def check_required_interactions() -> List[str]:
    """Return the list of required interaction features from documentation."""
    # High priority interactions (Section A)
    high_priority = [
        "x_trend_intensity",
        "x_trend_intensity_g",
        "x_rel_sec_mom",
        "x_mom_sh_5",
        "x_mom_sh_5_mktneu",
        "x_rvol5_dir",
        "x_rvol5_bb",
        "x_squeeze_pressure",
        "x_credit_rev_bias",
        "x_pead_effect",
        "x_rev_gate",
        "x_bo_gate",
        "x_alpha_meanrev_stable",
        "x_flow_smart_rel",
        "x_foreign_relsec",
    ]

    # Medium priority interactions (Section B)
    medium_priority = [
        "x_tri_align",
        "x_bbpos_rvol5",
        "x_bbneg_rvol5",
        "x_liquidityshock_mom",
        "x_pead_times_mkt",
        "x_dmi_impulse_dir",
        "x_breadth_rel",
    ]

    return high_priority + medium_priority


def verify_dataset(dataset_path: Path) -> Tuple[bool, Dict]:
    """Verify dataset completeness and return results."""
    columns = load_dataset_columns(dataset_path)
    categories = categorize_features(columns)

    # Count features
    total_features = len(columns)

    # Check required interactions
    required_interactions = check_required_interactions()
    found_interactions = set(categories["interactions"])
    missing_interactions = [x for x in required_interactions if x not in found_interactions]

    # Check daily margin features
    has_daily_margin = len(categories["margin_daily"]) > 0

    # Compile results
    results = {
        "total_features": total_features,
        "categories": {k: len(v) for k, v in categories.items()},
        "interaction_features": {
            "required": len(required_interactions),
            "found": len(found_interactions),
            "missing": missing_interactions
        },
        "daily_margin": {
            "enabled": has_daily_margin,
            "count": len(categories["margin_daily"]),
            "features": categories["margin_daily"][:5] if has_daily_margin else []
        },
        "target_395": total_features >= 395,
        "expected_range": 303 <= total_features <= 395,  # 303-307 with futures disabled
        "details": categories
    }

    # Check metadata if available
    metadata_path = dataset_path.parent / f"{dataset_path.stem}_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                results["metadata"] = {
                    "columns_total": metadata.get("columns", {}).get("total"),
                    "rows": metadata.get("rows"),
                    "date_range": metadata.get("date_range")
                }
        except Exception:
            pass

    # Determine pass/fail
    # Accept 303-307 range (futures disabled) or full 395
    passed = (
        (303 <= total_features <= 395) and
        len(missing_interactions) == 0
    )

    return passed, results


def print_results(results: Dict) -> None:
    """Print verification results in a formatted way."""
    print("\n" + "="*60)
    print("📊 DATASET FEATURE VERIFICATION REPORT")
    print("="*60)

    # Overall status
    total = results["total_features"]
    target_met = results["target_395"]
    in_expected_range = results["expected_range"]

    if target_met:
        print(f"\n✅ Total Features: {total} (theoretical max: 395) - FULL DATASET")
    elif in_expected_range:
        print(f"\n✅ Total Features: {total} (expected: 303-307 with futures disabled) - PASSED")
    else:
        print(f"\n❌ Total Features: {total} (expected: 303-395) - NEEDS REVIEW")

    # Category breakdown
    print("\n📁 Feature Categories:")
    print("-" * 40)
    categories = results["categories"]
    for category, count in sorted(categories.items(), key=lambda x: -x[1]):
        if count > 0:
            emoji = "✅" if count > 0 else "❌"
            print(f"  {emoji} {category:20s}: {count:3d} features")

    # Interaction features
    print("\n🔄 Interaction Features (x_*):")
    print("-" * 40)
    inter = results["interaction_features"]
    print(f"  Required: {inter['required']}")
    print(f"  Found:    {inter['found']}")
    if inter["missing"]:
        print(f"  ❌ Missing: {', '.join(inter['missing'])}")
    else:
        print(f"  ✅ All required interactions present!")

    # Daily margin features
    print("\n💹 Daily Margin Features (dmi_*):")
    print("-" * 40)
    dmi = results["daily_margin"]
    if dmi["enabled"]:
        print(f"  ✅ Enabled: {dmi['count']} features")
        if dmi["features"]:
            print(f"  Examples: {', '.join(dmi['features'][:3])}...")
    else:
        print(f"  ❌ Not enabled - daily margin features missing!")

    # Metadata info
    if "metadata" in results:
        print("\n📋 Metadata:")
        print("-" * 40)
        meta = results["metadata"]
        print(f"  Columns: {meta.get('columns_total', 'N/A')}")
        print(f"  Rows:    {meta.get('rows', 'N/A')}")
        if meta.get("date_range"):
            dr = meta["date_range"]
            print(f"  Dates:   {dr.get('min', 'N/A')} to {dr.get('max', 'N/A')}")

    # Key missing features
    print("\n🔍 Key Observations:")
    print("-" * 40)

    observations = []

    if not dmi["enabled"]:
        observations.append("❌ Daily margin (dmi_*) features are missing - need to enable in pipeline")

    if inter["missing"]:
        observations.append(f"❌ Missing {len(inter['missing'])} interaction features")

    if categories.get("sector_aggregates", 0) < 10:
        observations.append("⚠️  Sector aggregate features may be incomplete")

    if total < 303:
        shortfall = 303 - total
        observations.append(f"📈 Need {shortfall} more features to reach expected minimum")
    elif 303 <= total < 395:
        futures_gap = 395 - total
        observations.append(f"ℹ️  {futures_gap} features short of theoretical max (likely futures: 88-92 disabled)")

    if not observations:
        observations.append("✅ Dataset appears complete with all required features!")

    for obs in observations:
        print(f"  {obs}")

    print("\n" + "="*60)


def main():
    """Main execution function."""
    # Check for dataset path argument or use default
    if len(sys.argv) > 1:
        dataset_path = Path(sys.argv[1])
    else:
        # Try to find latest dataset
        dataset_path = Path("output/ml_dataset_latest_full.parquet")
        if not dataset_path.exists():
            # Try alternate location
            alt_path = Path("output/datasets/ml_dataset_latest_full.parquet")
            if alt_path.exists():
                dataset_path = alt_path

    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("\nUsage: python verify_dataset_features.py [dataset_path]")
        sys.exit(1)

    print(f"📂 Verifying dataset: {dataset_path}")

    # Run verification
    passed, results = verify_dataset(dataset_path)

    # Print results
    print_results(results)

    # Exit with appropriate code
    if passed:
        print("\n🎉 VERIFICATION PASSED - Dataset is complete!")
        sys.exit(0)
    else:
        print("\n⚠️  VERIFICATION INCOMPLETE - See observations above")
        sys.exit(1)


if __name__ == "__main__":
    main()
