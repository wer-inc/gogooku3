# Model Input Dimensions Configuration Guide

**Purpose**: Clarify the correct usage of `model.input_dims.*` parameters to avoid confusion between feature counts and sequence lengths.

---

## 📊 Current Dataset Specification

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Total features** | 373 | Number of feature columns in dataset |
| **Sequence length** | 60 | Number of days in time-series window |
| **Horizons** | [1, 5, 10, 20] | Prediction horizons in days |

---

## ⚙️ Configuration Parameters

### Key Parameters

```yaml
model.input_dims:
  total_features: 373        # Total number of features
  historical_features: ???   # Features used in time-series (NOT sequence length!)
  basic_features: ???        # Features used as static/cross-sectional
```

### ⚠️ Common Mistake

```python
# ❌ WRONG - Confusing with sequence_length
model.input_dims.historical_features = 60  # This is NOT sequence length!

# ✅ CORRECT - Number of features used in time-series
model.input_dims.historical_features = 373  # All features as time-series
```

**Sequence length** is configured separately:
```yaml
model.tft.temporal.max_sequence_length: 60  # ← This is the 60-day window
```

---

## 📋 Configuration Patterns

### Pattern 1: Safe Mode (Default)

**Use case**: Stable training without time-series complexity

```yaml
model.input_dims:
  total_features: 373
  historical_features: 0      # Time-series DISABLED
  basic_features: 373          # All features as static
```

**Characteristics**:
- ✅ Most stable (no RNN/attention complexity)
- ✅ Fast training
- ⚠️ Loses temporal patterns
- ⚠️ Lower capacity

**Automatic in Safe Mode**:
```python
# scripts/integrated_ml_training_pipeline.py:690-706
if FORCE_SINGLE_PROCESS == "1":
    overrides.append("model.input_dims.historical_features=0")
    overrides.append("model.input_dims.basic_features=373")
```

---

### Pattern 2: Full Time-Series (Recommended)

**Use case**: Leverage temporal patterns with TFT/LSTM

```yaml
model.input_dims:
  total_features: 373
  historical_features: 373    # All features as time-series
  basic_features: 0            # No static features
```

**Characteristics**:
- ✅ Full model capacity
- ✅ Learns temporal patterns
- ✅ Better performance (if stable)
- ⚠️ Requires more memory
- ⚠️ Slower training

**How to enable**:
```bash
# Override Safe Mode default
python scripts/integrated_ml_training_pipeline.py \
  --max-epochs 60 \
  model.input_dims.historical_features=373 \
  model.input_dims.basic_features=0
```

---

### Pattern 3: Hybrid (Experimental)

**Use case**: Mix time-series and static features

```yaml
model.input_dims:
  total_features: 373
  historical_features: 200    # Subset as time-series
  basic_features: 173          # Remainder as static
```

**Characteristics**:
- 🔬 Experimental
- 💡 Can separate feature types (e.g., price/volume as time-series, fundamentals as static)
- ⚠️ Requires careful feature selection

**Example**:
```python
# Time-series features: price, volume, technical indicators
historical_features = [
    'close', 'volume', 'rsi_14d', 'macd_12_26',
    # ... (200 features)
]

# Static features: fundamentals, sector, market cap
basic_features = [
    'per', 'pbr', 'roe', 'market_cap', 'sector_code',
    # ... (173 features)
]
```

---

## 🚀 Recommended Workflow

### Step 1: Start with Safe Mode
```bash
# Initial training (no time-series)
FORCE_SINGLE_PROCESS=1 \
  python scripts/integrated_ml_training_pipeline.py \
    --max-epochs 10 \
    --batch-size 1024
# → Automatically sets historical_features=0
```

### Step 2: Verify stability
- Check: No deadlock, no OOM
- Check: Val Sharpe > 0.0
- Check: No NaN in predictions

### Step 3: Enable time-series
```bash
# Full time-series mode
FORCE_SINGLE_PROCESS=1 \
  python scripts/integrated_ml_training_pipeline.py \
    --max-epochs 10 \
    --batch-size 1024 \
    model.input_dims.historical_features=373 \
    model.input_dims.basic_features=0
```

### Step 4: Verify NaN
```bash
# Check logs for NaN warnings
grep -i "nan\|inf" _logs/training/phase1_*.log

# If NaN detected → rollback to Pattern 1
# If stable → continue to Phase 2
```

---

## 🔍 Debugging Guide

### Symptom: NaN in predictions

**Possible causes**:
1. `historical_features` set too high → gradient explosion
2. Normalization statistics corrupted (check cache)
3. Sequence length mismatch

**Solutions**:
```bash
# 1. Clear cache
./scripts/clean_atft_cache.sh --force

# 2. Reduce historical_features
model.input_dims.historical_features=200  # Instead of 373

# 3. Check normalization
python scripts/detect_data_leakage.py  # Now includes stats check
```

### Symptom: Low performance with time-series

**Possible causes**:
1. Features not suitable for time-series (e.g., constant values)
2. Insufficient training epochs
3. Learning rate too high

**Solutions**:
```bash
# 1. Analyze feature quality
python scripts/analyze_baseline_features.py

# 2. Longer training
--max-epochs 60  # Instead of 10

# 3. Lower LR
--lr 1e-4  # Instead of 2e-4
```

---

## 📝 Quick Reference

| Scenario | historical_features | basic_features | Notes |
|----------|---------------------|----------------|-------|
| Safe Mode (default) | 0 | 373 | Most stable |
| Full time-series | 373 | 0 | Best performance (if stable) |
| Hybrid (experimental) | 100-300 | 73-273 | Requires tuning |
| Debugging NaN | 0 → 100 → 200 → 373 | 373 → 273 → 173 → 0 | Gradual increase |

---

## ✅ Validation Checklist

Before training:
- [ ] Understand `historical_features` ≠ sequence_length
- [ ] Choose Pattern 1/2/3 based on stability requirements
- [ ] Clear cache if changing configuration: `./scripts/clean_atft_cache.sh --force`
- [ ] Verify settings in logs: `grep "historical_features" _logs/training/*.log`

After training:
- [ ] Check for NaN: `grep -i nan _logs/training/*.log`
- [ ] Verify performance meets expectations
- [ ] Document configuration in `EXPERIMENT_STATUS.md`

---

**Last updated**: 2025-10-24
**Related**: [TRAINING_COMMANDS.md](TRAINING_COMMANDS.md), [EXPERIMENT_STATUS.md](../EXPERIMENT_STATUS.md)
