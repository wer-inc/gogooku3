# Test Files Cleanup Plan

**Date**: 2025-10-13
**Status**: 📋 Plan
**Issue**: 24 test files scattered outside tests/ directory

## Current State

### Scattered Test Files (24 files - PROBLEM)

#### Root Directory (7 files)
```
/root/gogooku3/
├── test_data_loading.py
├── test_date_filtering.py
├── test_env_settings.py
├── test_normalization.py
├── test_phase2_dataloader.py
├── test_phase2_simple.py
└── test_phase2_verification.py
```

#### scripts/ Directory (17 files)
```
scripts/
├── test_atft_training.py
├── test_baseline_rankic.py
├── test_cache_cpu_fallback.py
├── test_default_features.py
├── test_direct_training.py
├── test_earnings_events.py
├── test_full_integration.py
├── test_futures_integration.py
├── test_graph_cache_effectiveness.py
├── test_multi_horizon.py
├── test_normalized_training.py
├── test_optimization.py
├── test_phase1_features.py
├── test_phase2_features.py
├── test_regime_moe.py
├── train_simple_test.py
└── smoke_test.py (keep in scripts/)
```

### Properly Organized (OK)
```
tests/
├── unit/           # Unit tests (27 files) ✅
├── integration/    # Integration tests (7 files) ✅
├── exploratory/    # Exploratory/ad-hoc tests (14 files) ✅
├── api/            # API tests (2 files) ✅
└── ...
```

## Analysis

### File Classification

#### Type 1: Exploratory/Debugging Tests (Root directory - 7 files)
**Files**:
- test_data_loading.py - Diagnose zero loss issue
- test_date_filtering.py - Debug date filtering
- test_env_settings.py - Debug environment
- test_normalization.py - Debug normalization
- test_phase2_dataloader.py - Debug Phase 2
- test_phase2_simple.py - Debug Phase 2
- test_phase2_verification.py - Verify Phase 2

**Reason**: Ad-hoc debugging scripts created during development

**Recommendation**: Move to `tests/exploratory/`

#### Type 2: Integration Tests (scripts/ - most files)
**Files**:
- test_atft_training.py - ATFT training integration test
- test_full_integration.py - Full pipeline integration test
- test_phase1_features.py - J-Quants Phase 1 API integration test
- test_phase2_features.py - J-Quants Phase 2 API integration test
- test_futures_integration.py - Futures features integration test
- test_earnings_events.py - Earnings events test
- test_multi_horizon.py - Multi-horizon prediction test
- test_baseline_rankic.py - Baseline RankIC test
- test_optimization.py - Optimization test
- test_direct_training.py - Direct training test
- test_normalized_training.py - Normalized training test
- test_regime_moe.py - Regime MoE test
- train_simple_test.py - Simple training test

**Reason**: Integration tests that verify component interactions

**Recommendation**: Move to `tests/integration/`

#### Type 3: Performance/Cache Tests (scripts/)
**Files**:
- test_cache_cpu_fallback.py - Cache fallback behavior
- test_graph_cache_effectiveness.py - Graph cache effectiveness

**Recommendation**: Move to `tests/integration/`

#### Type 4: Feature Tests (scripts/)
**Files**:
- test_default_features.py - Default feature set

**Recommendation**: Move to `tests/unit/` or `tests/integration/`

#### Type 5: Keep in scripts/ (1 file)
**Files**:
- smoke_test.py - Quick validation script

**Reason**: Executable script, not pytest-style test

**Recommendation**: Keep in `scripts/testing/` (already moved during shell cleanup)

## Proposed Directory Structure

```
tests/
├── unit/                          # Unit tests (existing + 1 new)
│   ├── test_default_features.py   # (moved)
│   └── ... (27 existing files)
│
├── integration/                   # Integration tests (existing + 15 new)
│   ├── test_atft_training.py      # (moved)
│   ├── test_baseline_rankic.py    # (moved)
│   ├── test_cache_cpu_fallback.py # (moved)
│   ├── test_direct_training.py    # (moved)
│   ├── test_earnings_events.py    # (moved)
│   ├── test_full_integration.py   # (moved)
│   ├── test_futures_integration.py # (moved)
│   ├── test_graph_cache_effectiveness.py # (moved)
│   ├── test_multi_horizon.py      # (moved)
│   ├── test_normalized_training.py # (moved)
│   ├── test_optimization.py       # (moved)
│   ├── test_phase1_features.py    # (moved)
│   ├── test_phase2_features.py    # (moved)
│   ├── test_regime_moe.py         # (moved)
│   ├── train_simple_test.py       # (moved)
│   └── ... (7 existing files)
│
├── exploratory/                   # Exploratory tests (existing + 7 new)
│   ├── test_data_loading.py       # (moved from root)
│   ├── test_date_filtering.py     # (moved from root)
│   ├── test_env_settings.py       # (moved from root)
│   ├── test_normalization.py      # (moved from root)
│   ├── test_phase2_dataloader.py  # (moved from root)
│   ├── test_phase2_simple.py      # (moved from root)
│   ├── test_phase2_verification.py # (moved from root)
│   └── ... (14 existing files)
│
├── api/                           # API tests (existing)
│   └── ... (2 existing files)
│
└── components/                    # Component tests (existing)
    └── ... (existing files)

scripts/
└── testing/                       # Executable test scripts
    ├── smoke_test.py              # (keep here)
    └── e2e_incremental_test.py    # (existing)
```

## Cleanup Actions

### Phase 1: Move Root Directory Tests
```bash
# Move from root to tests/exploratory/
mv test_data_loading.py tests/exploratory/
mv test_date_filtering.py tests/exploratory/
mv test_env_settings.py tests/exploratory/
mv test_normalization.py tests/exploratory/
mv test_phase2_dataloader.py tests/exploratory/
mv test_phase2_simple.py tests/exploratory/
mv test_phase2_verification.py tests/exploratory/
```

### Phase 2: Move scripts/ Integration Tests
```bash
# Move from scripts/ to tests/integration/
mv scripts/test_atft_training.py tests/integration/
mv scripts/test_baseline_rankic.py tests/integration/
mv scripts/test_cache_cpu_fallback.py tests/integration/
mv scripts/test_direct_training.py tests/integration/
mv scripts/test_earnings_events.py tests/integration/
mv scripts/test_full_integration.py tests/integration/
mv scripts/test_futures_integration.py tests/integration/
mv scripts/test_graph_cache_effectiveness.py tests/integration/
mv scripts/test_multi_horizon.py tests/integration/
mv scripts/test_normalized_training.py tests/integration/
mv scripts/test_optimization.py tests/integration/
mv scripts/test_phase1_features.py tests/integration/
mv scripts/test_phase2_features.py tests/integration/
mv scripts/test_regime_moe.py tests/integration/
mv scripts/train_simple_test.py tests/integration/
```

### Phase 3: Move Feature Test
```bash
# Move from scripts/ to tests/unit/
mv scripts/test_default_features.py tests/unit/
```

### Phase 4: Keep smoke_test.py in scripts/
```bash
# Already in correct location (scripts/)
# No action needed
```

## Benefits

### Before (Current State)
- ❌ 7 test files in root directory
- ❌ 17 test files in scripts/
- ❌ Unclear test organization
- ❌ pytest discovery doesn't find scattered tests
- ❌ Difficult to run specific test categories

### After (Proposed State)
- ✅ Clean root directory (0 test files)
- ✅ All tests in tests/ directory
- ✅ Clear organization by test type
- ✅ pytest can discover all tests
- ✅ Easy to run by category:
  - `pytest tests/unit/` - Unit tests
  - `pytest tests/integration/` - Integration tests
  - `pytest tests/exploratory/` - Exploratory tests

## Pytest Usage

After cleanup:

```bash
# Run all tests
pytest tests/

# Run by category
pytest tests/unit/              # Unit tests
pytest tests/integration/       # Integration tests
pytest tests/exploratory/       # Exploratory tests

# Run specific test file
pytest tests/integration/test_phase1_features.py

# Run with markers
pytest -m integration          # Integration tests only
pytest -m "not slow"           # Skip slow tests

# Run smoke test (executable script)
python scripts/smoke_test.py --max-epochs 1
```

## Import Path Fixes

Some tests may need import path fixes after moving:

### Before (in root or scripts/)
```python
import sys
sys.path.append('/home/ubuntu/gogooku3-standalone')
from src.gogooku3.components import ...
```

### After (in tests/)
```python
# No sys.path manipulation needed if using proper package structure
from src.gogooku3.components import ...
```

Or use relative imports:
```python
from gogooku3.components import ...
```

## Verification

After cleanup, verify:

```bash
# 1. Check root directory is clean
ls -la test*.py
# Expected: No test*.py files

# 2. Check scripts/ has no test files (except in testing/)
find scripts/ -maxdepth 1 -name "test*.py"
# Expected: No files

# 3. Check all tests can be discovered
pytest --collect-only tests/
# Expected: All test files listed

# 4. Check imports work
pytest tests/unit/ --collect-only
pytest tests/integration/ --collect-only
pytest tests/exploratory/ --collect-only
```

## Risks and Mitigation

### Risk 1: Import paths may break
**Mitigation**: Test imports after moving, fix sys.path if needed

### Risk 2: Some tests may depend on being in specific directory
**Mitigation**: Update relative paths in tests to use project root

### Risk 3: CI/CD may reference specific test paths
**Mitigation**: Update pytest commands in CI/CD config

## Status: Awaiting Approval

This is a proposed plan. Execute with:
```bash
bash scripts/maintenance/cleanup_test_files.sh
```

Or execute manually following Phase 1-4 above.
