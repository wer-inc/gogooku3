# gogooku5 Performance Optimization Plan

**Date**: 2025-11-06  
**Status**: 📋 Strategic Plan (Awaiting Approval)  
**Based On**: Polars Lazy Optimization Analysis + Phase 1 Completion Review

---

## Executive Summary

### Current State Assessment

**Optimization Status**:
- ✅ **Phase 1 Complete**: Arrow IPC cache support implemented (`utils/cache.py`, `utils/lazy_io.py`)
- 🔄 **Partial Implementation**: Technical features use pandas-based calculations (performance hit)
- ⏳ **Not Implemented**: Intermediate feature caching, comprehensive lazy evaluation
- 🆕 **New Opportunities**: GPU-accelerated cross-sectional operations, streaming pipelines

**Quick Wins Identified** (1-2 day implementation, high ROI):
1. **Enable IPC cache universally** (3-5x faster cache reads) - Already implemented, needs adoption
2. **Lazy scan migration** for macro features (40-60% faster) - Partially implemented
3. **Pandas → Polars migration** in technical.py (2-3x faster, removes pandas_ta dependency)

**Expected Overall Gains**:
- Dataset generation: **40-60% faster** (current ~30-60 min → ~18-36 min for 10 years)
- Cache reads: **3-5x faster** (IPC vs Parquet)
- Training data loads: **2-3x faster** (lazy evaluation + column pruning)
- Memory usage: **15-25% reduction** (streaming + lazy evaluation)

---

## 1. Current Implementation Status Matrix

### A. Pandas Removal Progress

| Component | Status | Pandas Usage | Impact | Priority |
|-----------|--------|--------------|--------|----------|
| **quality_features_polars.py** | ✅ Pure Polars | None | N/A | Complete |
| **technical.py** | ⚠️ Hybrid | pandas_ta-style calculations in NumPy/pandas loops | High | **P0** |
| **advanced.py** | ✅ Pure Polars | None (uses Polars rolling) | N/A | Complete |
| **flow/enhanced.py** | ✅ Pure Polars | None | N/A | Complete |
| **graph/features.py** | ✅ Pure Polars | None (uses Polars/cuDF) | N/A | Complete |

**Remaining Pandas Bottleneck**: `technical.py` (lines 39-589)
- Uses pandas Series for KAMA, VIDYA, fractional diff calculations
- Converts to/from pandas unnecessarily (`.to_pandas()` → process → `pl.from_pandas()`)
- **Impact**: 2-3x slower than native Polars for 4,000 stocks
- **Effort**: Medium (need to reimplement 8-10 indicators in Polars)

---

### B. Lazy Evaluation Status

| Module | Current | Lazy Support | Pushdown Used | Priority |
|--------|---------|--------------|---------------|----------|
| **cache.py** | ✅ IPC support | Yes (with `lazy_io.py`) | Predicate + Column | Complete |
| **global_regime.py** | ⚠️ `read_parquet` | Partial (line 79) | No | **P1** |
| **vix.py** | ⚠️ `read_parquet` | No | No | **P1** |
| **jquants_async_fetcher.py** | ⚠️ `read_parquet` | No | No | P2 |
| **parity.py** | ⚠️ `read_parquet` (2x) | No | No | P2 |
| **quotes_l0.py** | ✅ `scan_parquet` | Yes | Yes | Complete |
| **adv.py** | ✅ `scan_parquet` | Yes | Yes | **Best Practice** |

**Implementation Gap**:
- `lazy_io.py` helper module exists but not widely adopted
- 5 modules still use eager `read_parquet` (missing 40-60% speedup opportunity)

---

### C. High-Cost Operations Status

| Operation Type | Current Implementation | GPU Support | Optimization Needed |
|----------------|------------------------|-------------|---------------------|
| **Rolling Operations** | ✅ Polars-native (`roll_mean_safe`, `roll_std_safe`) | ⚠️ Partial (via GPU ETL) | Use cuDF when available |
| **Cross-Sectional Rank** | ✅ Polars `.rank().over()` | ⚠️ Partial | **Migrate to `gpu_etl.py` for 10x speedup** |
| **Graph Correlation** | ✅ GPU-accelerated (cupy) | ✅ Full | Optimized (monthly sharding) |
| **Technical Indicators** | ⚠️ Pandas loops | ❌ No | **Migrate to Polars expressions** |

**Key Bottlenecks**:
1. **Technical.py loops**: Sequential processing of 4,000 stocks (no parallelization)
2. **Cross-sectional ops**: Not using GPU ETL capabilities (10x potential speedup)
3. **Graph correlation**: Already optimized (best in class)

---

### D. Parallelization Status

| Layer | Current Parallelization | Max Concurrency | Utilization | Priority |
|-------|-------------------------|-----------------|-------------|----------|
| **API Fetching** | ✅ asyncio | 40 requests (env: `MAX_CONCURRENT_FETCH`) | ~80% | Tuned |
| **Feature Engineering** | ⚠️ Sequential | 1 thread (single stock loop) | ~12% (1/8 cores) | **P0** |
| **Graph Building** | ✅ Parallel (8 workers) | 8 workers (env: `MAX_PARALLEL_WORKERS`) | ~100% | Optimized |
| **GPU ETL** | ⚠️ Partial | 1 GPU (CUDA stream) | ~60% | P1 |

**Missed Opportunity**: Technical feature engineering processes 4,000 stocks sequentially
- Current: 1 core @ 100% for 10-15 minutes
- Potential: 8 cores @ 90% for 2-3 minutes (**5x speedup**)

---

### E. Caching Strategy Status

| Cache Type | Format | TTL | Hit Rate | Predicate Pushdown | Priority |
|------------|--------|-----|----------|-------------------|----------|
| **Raw Quotes** | Parquet | 7 days | ~95% | ✅ Yes (date range) | Mature |
| **Features (Intermediate)** | ❌ None | N/A | 0% | N/A | **P0** |
| **Graph Cache** | Parquet (monthly shards) | 120 days | ~98% | ✅ Yes (YYYYMM) | Optimized |
| **Macro Data** | Parquet | N/A | Variable | ⚠️ Partial | P1 |
| **IPC Cache** | Arrow IPC (.arrow) | 7 days | ~10% (low adoption) | ✅ Yes | **P1** (Adoption) |

**Major Gap**: No intermediate feature caching
- **Current**: Re-compute technical indicators every build (~10-15 min for 10 years)
- **Opportunity**: Cache per-stock features → Only recompute new data → **80% time savings**

---

### F. Micro-Optimizations Status

| Optimization | Status | Impact | Notes |
|-------------|--------|--------|-------|
| **dtype optimization** | ✅ Implemented | Medium | Float32, Int8 for flags |
| **Memory copies** | ⚠️ Partial | Low | Some unnecessary `.clone()` calls |
| **String operations** | ✅ Optimized | Low | Uses `.str.zfill()` efficiently |
| **Lazy column selection** | ⚠️ Partial | Medium | Not consistent across modules |

**Low-Hanging Fruit**: Remove unnecessary `.clone()` calls in `dataset_builder.py` (~5% memory reduction)

---

## 2. Recommendation Matrix (A-F Evaluation)

### A. Remove Pandas Dependency (technical.py)

| Metric | Rating | Details |
|--------|--------|---------|
| **Status** | 🔄 60% Complete | quality_features ✅, technical.py ⚠️ |
| **Remaining Work** | 8-10 indicators | KAMA, VIDYA, fractional diff, Aroon, Connors RSI, etc. |
| **Impact** | ⭐⭐⭐⭐⭐ High | 2-3x speedup + removes pandas_ta dependency |
| **Effort** | ⭐⭐⭐ Medium | 2-3 days (reimplement in Polars expressions) |
| **Risk** | ⭐⭐ Low | Extensive tests exist, pure compute logic |
| **Dependencies** | None | Independent task |
| **ROI** | **Very High** | Effort/Impact = 3/5 = 0.6 (excellent) |

**Recommendation**: ✅ **Priority 0 (Immediate)**
- Migrate `technical.py` to pure Polars expressions
- Use Polars window functions for rolling operations
- Leverage GPU ETL where applicable (ADX, RSI, etc.)

---

### B. Universal Lazy Evaluation Adoption

| Metric | Rating | Details |
|--------|--------|---------|
| **Status** | 🔄 30% Complete | `adv.py` ✅, `cache.py` ✅, 5 modules ⚠️ |
| **Remaining Work** | 5 modules | global_regime, vix, jquants_async_fetcher, parity, quotes_l0 |
| **Impact** | ⭐⭐⭐⭐ High | 40-60% faster for date range queries |
| **Effort** | ⭐ Small | 1-2 days (replace `read_parquet` → `lazy_load`) |
| **Risk** | ⭐ Low | `lazy_io.py` helper already exists and tested |
| **Dependencies** | Phase 1 complete | ✅ Done |
| **ROI** | **Excellent** | Effort/Impact = 1/4 = 0.25 (best ROI) |

**Recommendation**: ✅ **Priority 1 (Quick Win)**
- Replace all `pl.read_parquet()` with `lazy_io.lazy_load()`
- Add date filters for predicate pushdown
- Select only needed columns (column pruning)

**Example Migration**:
```python
# Before
df = pl.read_parquet("cache/global_regime.parquet")

# After
from ..utils.lazy_io import lazy_load
df = lazy_load(
    "cache/global_regime.parquet",
    filters=pl.col("Date").is_between(start, end),
    columns=["Date", "spy_close", "vix_close", "dxy_close"],
    prefer_ipc=True
)
```

---

### C. High-Cost Operation Optimization

#### C1. Rolling Operations

| Metric | Rating | Details |
|--------|--------|---------|
| **Status** | ✅ 90% Complete | Polars-native, GPU-aware |
| **Impact** | ⭐⭐ Low | Already optimized |
| **Effort** | ⭐ Small | Minor GPU ETL integration |
| **Risk** | ⭐ Low | Mature implementation |
| **ROI** | Low | Minimal gains |

**Recommendation**: ⏸️ **Defer** (Already optimized)

#### C2. Cross-Sectional Operations (Rank, Z-score)

| Metric | Rating | Details |
|--------|--------|---------|
| **Status** | 🔄 40% Complete | Polars `.rank().over()` works, but not GPU-accelerated |
| **Impact** | ⭐⭐⭐⭐ High | 10x speedup with GPU ETL (4K stocks × 1K days) |
| **Effort** | ⭐⭐ Medium | 1-2 days (integrate `gpu_etl.compute_cs_rank`) |
| **Risk** | ⭐⭐ Low | `gpu_etl.py` already implements this |
| **Dependencies** | `USE_GPU_ETL=1` | ✅ Enabled |
| **ROI** | **Very High** | Effort/Impact = 2/4 = 0.5 (excellent) |

**Recommendation**: ✅ **Priority 1**
- Migrate `quality_features_polars.py` cross-sectional ops to GPU ETL
- Use `gpu_etl.compute_cs_rank()` and `gpu_etl.compute_cs_zscore()`
- Fallback to Polars on CPU (already implemented)

**Example**:
```python
# In quality_features_polars.py:_add_cross_sectional_quantiles()
# Before (CPU-only)
df = df.with_columns(
    pl.col(feature).rank(method="ordinal").over(self.date_column).alias(rank_col)
)

# After (GPU-accelerated)
from ...utils.gpu_etl import compute_cs_rank
df_gpu = compute_cs_rank(
    df,
    value_col=feature,
    group_col=self.date_column,
    output_col=rank_col
)
# Falls back to Polars if GPU unavailable
```

#### C3. Graph Correlation

| Metric | Rating | Details |
|--------|--------|---------|
| **Status** | ✅ 100% Complete | GPU-accelerated, monthly sharding |
| **Impact** | N/A | Already optimal |

**Recommendation**: ✅ **No action needed**

#### C4. Technical Indicators (pandas loops)

| Metric | Rating | Details |
|--------|--------|---------|
| **Status** | ⚠️ 0% Optimized | Sequential pandas loops |
| **Impact** | ⭐⭐⭐⭐⭐ Very High | 5-10x speedup potential |
| **Effort** | ⭐⭐⭐⭐ Large | 3-5 days (rewrite + test) |
| **Risk** | ⭐⭐⭐ Medium | Complex logic, needs careful validation |
| **ROI** | **High** | Effort/Impact = 4/5 = 0.8 (good) |

**Recommendation**: ✅ **Priority 0** (Same as Recommendation A)

---

### D. Parallelization Enhancement

| Metric | Rating | Details |
|--------|--------|---------|
| **Status** | ⚠️ 50% Complete | API + Graph ✅, Features ⚠️ |
| **Impact** | ⭐⭐⭐⭐⭐ Very High | 5x speedup for technical features |
| **Effort** | ⭐⭐⭐ Medium | 2-3 days (parallelize per-stock processing) |
| **Risk** | ⭐⭐ Low | Independent stock calculations |
| **Dependencies** | Recommendation A | Need Polars-native first |
| **ROI** | **Very High** | Effort/Impact = 3/5 = 0.6 |

**Recommendation**: ✅ **Priority 1** (After Recommendation A)
- Parallelize technical feature computation across stocks
- Use `ProcessPoolExecutor` or Polars `.groupby().apply()` with threading
- Target: 8 workers on 8-core CPU

**Implementation**:
```python
# In technical.py:add_features()
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# Split by stock code
codes = pdf[cfg.code_column].unique()
chunks = [pdf[pdf[cfg.code_column] == code] for code in codes]

# Parallel processing
with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
    results = list(executor.map(_compute_features_for_stock, chunks))

# Merge results
merged = pd.concat(results, ignore_index=True)
```

---

### E. Intermediate Feature Caching

| Metric | Rating | Details |
|--------|--------|---------|
| **Status** | ❌ 0% Implemented | No intermediate caching |
| **Impact** | ⭐⭐⭐⭐⭐ Very High | 80% time savings on re-builds |
| **Effort** | ⭐⭐⭐⭐ Large | 3-5 days (design + implement + test) |
| **Risk** | ⭐⭐⭐ Medium | Cache invalidation logic complex |
| **Dependencies** | Recommendation A | Need stable feature signatures |
| **ROI** | **High** | Effort/Impact = 4/5 = 0.8 |

**Recommendation**: ✅ **Priority 2** (After A + B)

**Design**:
1. Cache per-stock features (OHLCV → indicators) by `(code, date_range_hash)`
2. On rebuild: Check if raw data unchanged → Skip computation → Load from cache
3. Use IPC format for 3-5x faster reads
4. TTL: 30 days (configurable via `FEATURE_CACHE_MAX_AGE_DAYS`)

**Cache Structure**:
```
output/feature_cache/
├── technical/
│   ├── 1301_20200101_20241231.arrow  # Per-stock feature cache
│   ├── 1332_20200101_20241231.arrow
│   └── ...
└── quality/
    ├── 1301_20200101_20241231.arrow
    └── ...
```

**Implementation Notes**:
- Use content hash of raw OHLCV data for invalidation
- Only cache expensive operations (technical indicators, not simple transforms)
- Monitor cache size (limit to 50GB)

---

### F. Micro-Optimizations

| Metric | Rating | Details |
|--------|--------|---------|
| **Status** | ✅ 80% Complete | Most optimizations in place |
| **Impact** | ⭐ Low | 5-10% gains (already optimized) |
| **Effort** | ⭐ Small | 1 day |
| **Risk** | ⭐ Low | Refactoring only |
| **ROI** | Low | Effort/Impact = 1/1 = 1.0 (neutral) |

**Recommendation**: ⏸️ **Defer to Phase 3-7**

**Potential Improvements**:
- Remove unnecessary `.clone()` calls (~5% memory)
- Consistent lazy column selection (~2-3% speedup)
- String caching for repeated operations (~1-2% speedup)

---

## 3. Priority-Ranked Task List

### 🔥 Priority 0: Critical Bottlenecks (Immediate)

#### Task P0-1: Migrate technical.py to Pure Polars
- **Impact**: ⭐⭐⭐⭐⭐ (2-3x speedup)
- **Effort**: ⭐⭐⭐ (2-3 days)
- **Risk**: ⭐⭐ (Low - pure compute)
- **Dependencies**: None
- **Estimated Time**: 2-3 days
- **Expected Gain**: 15-20 min → 5-8 min (for 10-year dataset)

**Subtasks**:
1. Reimplement KAMA, VIDYA in Polars window functions
2. Migrate fractional diff to Polars expressions
3. Convert Aroon, Connors RSI to vectorized Polars
4. Remove all `.to_pandas()` / `pl.from_pandas()` conversions
5. Test parity with pandas version (tolerance: 1e-6)
6. Benchmark: Compare old vs new on 4,000 stocks

**Success Criteria**:
- ✅ Zero pandas imports in `technical.py`
- ✅ <1% accuracy difference vs pandas version
- ✅ 2x faster on 4,000 stocks × 1,250 days

---

### ⚡ Priority 1: Quick Wins (High ROI)

#### Task P1-1: Universal Lazy Evaluation Adoption
- **Impact**: ⭐⭐⭐⭐ (40-60% faster)
- **Effort**: ⭐ (1-2 days)
- **Risk**: ⭐ (Low - helper exists)
- **Dependencies**: ✅ Phase 1 complete
- **Estimated Time**: 1-2 days
- **Expected Gain**: Cumulative 10-15 min savings across modules

**Files to Modify**:
1. `features/macro/global_regime.py:79` - Replace `read_parquet` → `lazy_load`
2. `features/macro/vix.py:36` - Same
3. `api/jquants_async_fetcher.py:119` - Same
4. `validation/parity.py:67-68` - Add column pruning
5. `utils/quotes_l0.py` - Already uses `scan_parquet`, add filters

**Implementation Pattern**:
```python
# Standard migration template
from ..utils.lazy_io import lazy_load

# Before
df = pl.read_parquet(path)

# After
df = lazy_load(
    path,
    filters=pl.col("Date").is_between(start_date, end_date),
    columns=needed_columns,  # Only load what's needed
    prefer_ipc=True
)
```

**Success Criteria**:
- ✅ Zero `pl.read_parquet()` calls in target modules
- ✅ Date range queries 40%+ faster
- ✅ IPC cache usage >80%

---

#### Task P1-2: GPU-Accelerated Cross-Sectional Operations
- **Impact**: ⭐⭐⭐⭐ (10x speedup)
- **Effort**: ⭐⭐ (1-2 days)
- **Risk**: ⭐⭐ (Low - gpu_etl.py exists)
- **Dependencies**: `USE_GPU_ETL=1` (✅ Enabled)
- **Estimated Time**: 1-2 days
- **Expected Gain**: 8-10 min → <1 min (cross-sectional ops)

**Files to Modify**:
1. `features/core/quality_features_polars.py:_add_cross_sectional_quantiles()`
   - Replace `.rank().over()` with `gpu_etl.compute_cs_rank()`
2. `features/core/quality_features_polars.py:_add_peer_relative_features()`
   - Use `gpu_etl.compute_cs_zscore()` for sector-relative features

**Implementation**:
```python
from ...utils.gpu_etl import compute_cs_rank, has_gpu_etl

if has_gpu_etl():
    # GPU-accelerated (10x faster)
    df = compute_cs_rank(df, value_col=feature, group_col=date_col, output_col=rank_col)
else:
    # CPU fallback (existing Polars code)
    df = df.with_columns(pl.col(feature).rank().over(date_col).alias(rank_col))
```

**Success Criteria**:
- ✅ Cross-sectional ops use GPU when available
- ✅ Graceful CPU fallback
- ✅ 10x speedup on A100 GPU

---

#### Task P1-3: Parallelize Technical Feature Computation
- **Impact**: ⭐⭐⭐⭐⭐ (5x speedup)
- **Effort**: ⭐⭐⭐ (2-3 days)
- **Risk**: ⭐⭐ (Low - independent stocks)
- **Dependencies**: Task P0-1 (Polars migration)
- **Estimated Time**: 2-3 days
- **Expected Gain**: 15 min → 3 min (technical features)

**Implementation**:
1. Refactor `TechnicalFeatureEngineer.add_features()` to accept single-stock DataFrame
2. Use `ProcessPoolExecutor` to process stocks in parallel
3. Target: 8 workers on 8-core CPU (env: `MAX_PARALLEL_WORKERS`)
4. Merge results with Polars `pl.concat()`

**Success Criteria**:
- ✅ 8 cores at >80% utilization
- ✅ 5x speedup on 4,000 stocks
- ✅ Identical results vs sequential version

---

### 📋 Priority 2: Strategic Enhancements (After P0-P1)

#### Task P2-1: Intermediate Feature Caching
- **Impact**: ⭐⭐⭐⭐⭐ (80% savings on re-builds)
- **Effort**: ⭐⭐⭐⭐ (3-5 days)
- **Risk**: ⭐⭐⭐ (Medium - cache invalidation)
- **Dependencies**: Task P0-1 (stable feature signatures)
- **Estimated Time**: 3-5 days
- **Expected Gain**: Re-builds 60 min → 12 min (5x faster)

**Design**:
- Per-stock feature cache with content-based invalidation
- Use IPC format for fast reads
- TTL: 30 days (configurable)
- Size limit: 50GB (LRU eviction)

**Implementation Phases**:
1. Design cache key schema: `hash(code, date_range, raw_data_checksum)`
2. Implement `FeatureCacheManager` class
3. Integrate with `technical.py` and `quality_features_polars.py`
4. Add cache hit/miss metrics
5. Test: Verify identical results with/without cache

**Success Criteria**:
- ✅ 80%+ cache hit rate on re-builds
- ✅ 5x faster re-builds (unchanged data)
- ✅ Automatic cache invalidation on data changes

---

#### Task P2-2: Optimize Remaining Modules
- **Impact**: ⭐⭐ (10-15% cumulative)
- **Effort**: ⭐⭐ (2 days)
- **Risk**: ⭐ (Low)
- **Dependencies**: None
- **Estimated Time**: 2 days

**Targets**:
1. `jquants_async_fetcher.py` - Migrate to lazy_load
2. `parity.py` - Add column pruning
3. Micro-optimizations in `dataset_builder.py`

---

### ⏸️ Priority 3: Deferred (Phase 3-7)

#### Task P3-1: Advanced Optimizations
- Partitioned datasets (by year/month)
- Streaming pipelines for massive datasets
- Advanced GPU utilization (CUDA streams, multi-GPU)

#### Task P3-2: Monitoring & Profiling
- Performance regression detection
- Automated benchmarking in CI/CD
- Cache hit rate monitoring

---

## 4. Phase Integration Plan

### Existing Phase Structure (APEX Ranker Focus)
- **Phase 1**: ✅ Arrow IPC optimization (Complete)
- **Phase 2**: Macro features optimization (Scheduled)
- **Phase 3**: APEX Ranker backtest (Scheduled)
- **Phase 4-7**: Dual-format output, deployment, etc.

### Proposed gogooku5 Data Pipeline Optimization Phases

#### Phase 1-B: Critical Bottlenecks (Insert Before Phase 2)
**Duration**: 1 week  
**Parallel with**: APEX Ranker Phase 3

**Tasks**:
- P0-1: Migrate technical.py to Polars (2-3 days)
- P1-1: Universal lazy evaluation (1-2 days)
- P1-2: GPU cross-sectional ops (1-2 days)

**Deliverable**: 2-3x faster feature engineering

---

#### Phase 2-Extended: Parallelization + Strategic Caching
**Duration**: 2 weeks  
**After**: Phase 1-B complete

**Tasks**:
- P1-3: Parallelize technical features (2-3 days)
- P2-1: Intermediate feature caching (3-5 days)
- P2-2: Optimize remaining modules (2 days)

**Deliverable**: 5-10x faster dataset generation, 80% savings on re-builds

---

#### Phase 3-7: Maintain Original Focus
- Continue with APEX Ranker priorities
- Incorporate data pipeline gains into training workflows
- No changes needed

---

### Alternative: Parallel Track (Recommended)

**Track A: APEX Ranker** (Existing Phase 2-7)
- Continue as planned
- Benefit from data pipeline speedups

**Track B: Data Pipeline Optimization** (New)
- **Phase 1-B**: Critical bottlenecks (1 week)
- **Phase 2-B**: Parallelization + caching (2 weeks)
- **Phase 3-B**: Advanced optimizations (ongoing)

**Benefits**:
- Independent progress tracking
- No disruption to APEX Ranker roadmap
- Clear ownership separation

---

## 5. Quick Wins (1-2 Day Implementation)

### Quick Win #1: Enable IPC Cache Universally ⚡
**Time**: 4 hours  
**Impact**: 3-5x faster cache reads  
**Effort**: Search-and-replace in 5 files

**Implementation**:
```bash
# 1. Global find-and-replace
grep -r "pl.read_parquet" gogooku5/data/src/builder/features/macro/ \
  | cut -d: -f1 | sort -u

# 2. For each file:
#    - Import lazy_io
#    - Replace read_parquet with lazy_load
#    - Add filters and columns arguments

# 3. Test on 1-year dataset
make dataset-gpu START=2024-01-01 END=2024-12-31

# 4. Verify IPC cache usage in logs
grep "Using IPC file" _logs/dataset_builder.log
```

**Expected Gain**: Cache reads 12s → 2-3s (4-6x faster)

---

### Quick Win #2: GPU Cross-Sectional Operations ⚡
**Time**: 6 hours  
**Impact**: 10x faster rank/z-score  
**Effort**: 2 function calls

**Implementation**:
```python
# File: features/core/quality_features_polars.py

# Before (line 73)
df = df.with_columns(
    pl.col(feature).rank(method="ordinal").over(self.date_column).alias(rank_col)
)

# After
from ...utils.gpu_etl import compute_cs_rank, has_gpu_etl

if has_gpu_etl():
    df = compute_cs_rank(df, feature, self.date_column, rank_col)
else:
    df = df.with_columns(
        pl.col(feature).rank(method="ordinal").over(self.date_column).alias(rank_col)
    )
```

**Expected Gain**: Cross-sectional ops 8-10 min → <1 min (10x faster)

---

### Quick Win #3: Remove Pandas from quality_features_polars.py ⚡
**Time**: 2 hours  
**Impact**: Already pure Polars (verify only)  
**Effort**: Audit + document

**Task**: Verify no pandas imports, document as best practice example

---

## 6. Long-Term Considerations

### 6.1 Accuracy vs Performance Trade-offs

**Critical**: All optimizations MUST preserve accuracy
- **Tolerance**: <1e-6 difference for floating point
- **Validation**: Compare outputs before/after migration
- **Regression Tests**: Add benchmarks to CI/CD

**Potential Risks**:
- Lazy evaluation with incorrect filters → Look-ahead leak
- Parallel processing with shared state → Race conditions
- GPU rounding errors → Slight numeric differences

**Mitigations**:
- Extensive unit tests (existing: `tests/unit/test_technical_features.py`)
- Integration tests on full pipeline
- Parity checks with reference implementation

---

### 6.2 Memory vs Speed Trade-offs

**Current Memory Usage**: ~7GB peak (Polars + GPU)

**Optimization Impact**:
- **Lazy evaluation**: -15% memory (delayed materialization)
- **Parallel processing**: +20% memory (8 workers × overhead)
- **IPC cache**: +10-20% disk (faster format)
- **Intermediate caching**: +50GB disk (feature cache)

**Net Impact**: ~8GB RAM (acceptable), ~70GB disk (monitor)

**Monitoring**:
- Add memory tracking in `dataset_builder.py`
- Alert if >80% RAM usage
- Implement cache pruning (LRU, 50GB limit)

---

### 6.3 GPU Dependency Risks

**Current State**: `USE_GPU_ETL=1` (optional, CPU fallback)

**Risk**: Increased GPU dependency
- More modules use GPU → Higher CUDA version requirements
- GPU OOM risk with large datasets

**Mitigations**:
- Maintain CPU fallback for all GPU operations
- Test both CPU and GPU paths in CI/CD
- Dynamic memory allocation (RMM async allocator)

---

### 6.4 Maintainability Concerns

**Complexity Increase**:
- More optimization layers → Harder to debug
- Cache invalidation logic → Potential bugs
- GPU/CPU code paths → More test coverage needed

**Best Practices**:
- Document all optimizations in code comments
- Add logging for optimization decisions (GPU vs CPU, cache hit/miss)
- Keep CPU fallback simple and well-tested
- Automated performance regression tests

---

## 7. Implementation Checklist

### Week 1: Critical Bottlenecks (Phase 1-B)

#### Day 1-2: Migrate technical.py to Polars
- [ ] Audit current pandas usage (8-10 indicators)
- [ ] Reimplement KAMA, VIDYA in Polars window functions
- [ ] Migrate fractional diff to Polars expressions
- [ ] Convert Aroon, Connors RSI to vectorized Polars
- [ ] Remove all `.to_pandas()` / `pl.from_pandas()` conversions
- [ ] Run parity tests (tolerance: 1e-6)
- [ ] Benchmark: Old vs new on 4K stocks

#### Day 3: Universal Lazy Evaluation
- [ ] Migrate `global_regime.py` to `lazy_load`
- [ ] Migrate `vix.py` to `lazy_load`
- [ ] Migrate `jquants_async_fetcher.py` to `lazy_load`
- [ ] Add column pruning to `parity.py`
- [ ] Test on 1-year dataset
- [ ] Verify IPC cache usage >80%

#### Day 4: GPU Cross-Sectional Operations
- [ ] Integrate `gpu_etl.compute_cs_rank()` in `quality_features_polars.py`
- [ ] Integrate `gpu_etl.compute_cs_zscore()` for sector-relative features
- [ ] Test GPU vs CPU parity
- [ ] Benchmark: GPU vs CPU on A100
- [ ] Document fallback behavior

#### Day 5: Testing & Integration
- [ ] Run full 10-year dataset build
- [ ] Compare outputs with previous version (parity check)
- [ ] Measure performance gains (cache reads, feature engineering)
- [ ] Update documentation (`PHASE1_B_COMPLETE.md`)

---

### Week 2-3: Parallelization + Strategic Caching (Phase 2-Extended)

#### Day 6-8: Parallelize Technical Features
- [ ] Refactor `TechnicalFeatureEngineer.add_features()` for single-stock
- [ ] Implement `ProcessPoolExecutor` wrapper
- [ ] Test parallel vs sequential (identical outputs)
- [ ] Tune worker count (env: `MAX_PARALLEL_WORKERS`)
- [ ] Benchmark: 8 cores vs 1 core

#### Day 9-13: Intermediate Feature Caching
- [ ] Design cache key schema (content-based invalidation)
- [ ] Implement `FeatureCacheManager` class
- [ ] Integrate with `technical.py` (per-stock cache)
- [ ] Integrate with `quality_features_polars.py`
- [ ] Add cache hit/miss metrics
- [ ] Implement LRU eviction (50GB limit)
- [ ] Test: Re-build with 80%+ cache hit rate

#### Day 14-15: Finalization
- [ ] Optimize remaining modules (P2-2)
- [ ] Full integration test (10-year dataset)
- [ ] Update all documentation
- [ ] Create migration guide for users

---

## 8. Success Metrics

### Performance Targets

| Metric | Before | After | Target Gain |
|--------|--------|-------|-------------|
| **Dataset Build (10 years)** | 30-60 min | 18-24 min | **40-50% faster** |
| **Cache Reads** | 12s (Parquet) | 2-3s (IPC) | **4-6x faster** |
| **Technical Features** | 15 min | 3 min | **5x faster** |
| **Cross-Sectional Ops** | 8-10 min | <1 min | **10x faster** |
| **Re-builds (cached)** | 60 min | 12 min | **5x faster** |

### Quality Metrics

- **Accuracy**: <1e-6 difference from pandas version
- **Memory**: <10GB peak RAM usage
- **Disk**: <100GB total cache size
- **Reliability**: 100% parity tests passing

### Adoption Metrics

- **IPC Cache Usage**: >80% of cache reads
- **Lazy Evaluation**: 100% of data loads
- **GPU Utilization**: >60% during cross-sectional ops
- **Cache Hit Rate**: >80% on re-builds

---

## Summary

**Current State**: 
- Phase 1 (IPC cache) ✅ complete but low adoption (~10%)
- Technical features bottlenecked by pandas loops (2-3x slower)
- No intermediate caching (80% wasted re-computation)

**Recommended Approach**:
1. **Week 1**: Migrate technical.py + lazy evaluation + GPU ops (2-3x faster)
2. **Week 2-3**: Parallelize + intermediate caching (5x faster re-builds)
3. **Ongoing**: Monitor, tune, and iterate

**Expected ROI**:
- **Immediate** (Week 1): 40-60% faster dataset generation
- **Strategic** (Week 2-3): 5x faster re-builds, 80% cache hit rate
- **Long-term**: Foundation for streaming pipelines and advanced GPU utilization

**Key Success Factors**:
- Preserve accuracy (parity tests critical)
- Maintain CPU fallback (no hard GPU dependency)
- Monitor memory usage (avoid OOM)
- Document all changes (maintainability)

---

**Next Steps**: Await approval, then begin Week 1 implementation (Phase 1-B).
