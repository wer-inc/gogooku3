"""
P0-2: Feature Manifest Contract Test
306列のFeature ABIを検証
"""
import hashlib
from pathlib import Path

import yaml

p = Path("output/reports/feature_manifest_306.yaml")
assert p.exists(), f"Feature manifest not found: {p}"

man = yaml.safe_load(p.read_text())
feats = man["features"]
assert len(feats) == 306, f"len={len(feats)}, expected 306"

abi = hashlib.sha1(",".join(feats).encode()).hexdigest()
stored_abi = man.get("meta", {}).get("abi_sha1")

print("✓ Feature manifest validated")
print(f"  - Features: {len(feats)}")
print(f"  - ABI (computed): {abi}")
print(f"  - ABI (stored): {stored_abi}")

if stored_abi:
    assert abi == stored_abi, f"ABI mismatch: computed={abi}, stored={stored_abi}"
    print("  - ABI match: ✓")

print(f"\nOK: Feature manifest 306, sha1={abi}")
