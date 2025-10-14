# GitHub Issues to Create

## Issue 1: Fix Hydra argument parsing in train_atft.py
**Title:** Fix Hydra CLI argument parsing error in train_atft.py

**Description:**
The train_atft.py script fails to parse Hydra override arguments correctly, showing "unrecognized arguments" error when passing configuration overrides.

**Error:**
```
train_atft.py: error: unrecognized arguments: data.source.train_files=[output/ml_dataset_latest_full.parquet] data.batch.batch_size=32
```

**Tasks:**
- [ ] Investigate Hydra initialization in train_atft.py
- [ ] Fix argument parsing to properly handle configuration overrides
- [ ] Test with various override patterns
- [ ] Update documentation with correct usage examples

**Priority:** High
**Labels:** bug, training-pipeline

---

## Issue 2: Fix UnifiedFeatureConverter sequence creation
**Title:** UnifiedFeatureConverter fails to create valid sequences from small datasets

**Description:**
When using small sample sizes (e.g., 1000 rows), UnifiedFeatureConverter.convert_to_atft_format() fails with "No valid sequences created from dataset" error.

**Error:**
```
ERROR - ❌ Conversion failed: No valid sequences created from dataset
```

**Tasks:**
- [ ] Debug sequence creation logic in UnifiedFeatureConverter
- [ ] Handle edge cases for small datasets
- [ ] Add minimum sequence length validation
- [ ] Implement better error messages for debugging
- [ ] Add unit tests for various dataset sizes

**Priority:** High
**Labels:** bug, data-processing

---

## Issue 3: Add missing flow and market features to ML dataset
**Title:** ML dataset missing several enrichment columns (flow_days_since_flow, mkt_vol, etc.)

**Description:**
The current ML dataset pipeline produces usable output but is missing several advanced features that were previously available:
- flow_days_since_flow
- flow_imp_flow
- mkt_vol
- mkt_rsi
- mkt_macd
- mkt_bb_position
- beta_lag

**Tasks:**
- [ ] Identify which pipeline step should generate these features
- [ ] Restore or re-implement the missing feature generators
- [ ] Validate feature quality and coverage
- [ ] Update ml_dataset_builder.py if needed
- [ ] Add feature validation tests

**Priority:** Medium
**Labels:** enhancement, feature-engineering

---

## Issue 4: Create comprehensive integration tests
**Title:** Add end-to-end integration tests for training pipeline

**Description:**
We need comprehensive integration tests to catch module import issues and configuration problems before they reach production.

**Tasks:**
- [ ] Create test for all module imports (ProductionDatasetV2, ProductionDataModuleV2, DayBatchSampler, etc.)
- [ ] Add smoke test with minimal data (100-1000 rows)
- [ ] Test Hydra configuration overrides
- [ ] Test both USE_OPTIMIZED_LOADER=1 and =0 paths
- [ ] Add CI/CD pipeline integration test
- [ ] Create test fixtures with small parquet files

**Priority:** Medium
**Labels:** testing, infrastructure

---

## Issue 5: Improve documentation for module architecture
**Title:** Document the bridge module architecture and migration path

**Description:**
The current codebase has a complex bridge/compatibility layer between legacy scripts and modern gogooku3 package. This needs proper documentation.

**Tasks:**
- [ ] Document the module structure and import paths
- [ ] Create migration guide from legacy to modern imports
- [ ] Add docstrings to bridge modules explaining their purpose
- [ ] Update CLAUDE.md with troubleshooting guide
- [ ] Create architecture diagram showing module relationships

**Priority:** Low
**Labels:** documentation

---

## Issue 6: Consolidate duplicate implementations
**Title:** Remove redundant code and consolidate implementations

**Description:**
There are multiple implementations of similar functionality:
- ProductionDatasetOptimized and ProductionDatasetV2 (should be unified)
- Multiple feature converter implementations
- Duplicate validation logic

**Tasks:**
- [ ] Identify all duplicate implementations
- [ ] Create unified implementation plan
- [ ] Deprecate old implementations with proper warnings
- [ ] Update all references to use consolidated versions
- [ ] Remove deprecated code after transition period

**Priority:** Low
**Labels:** refactoring, technical-debt
