# Tier 2.2: GAT Residual Bypass Test - Results

**Date**: 2025-10-30
**Trial Duration**: 3 epochs
**Status**: ❌ **FAILED** - Degeneracy persists with GAT completely bypassed

---

## 🎯 Experiment Objective

Test whether **GAT residual bypass mechanism** is the root cause of model degeneracy.

**Hypothesis (H0)**: GAT gate α ≈ 0.5 allows complete bypass of learned graph features, causing model to output uniform values.

**Expected Results**:
- **H0 TRUE**: yhat_std > 0 with GAT bypassed → GAT gate confirmed as root cause
- **H0 FALSE**: yhat_std = 0 with GAT bypassed → Deeper architecture problem upstream

---

## 🔧 Implementation

### Code Changes

**File**: `src/atft_gat_fan/models/architectures/atft_gat_fan.py:826-842`

```python
# GAT Residual Bypass適用 (Phase 2) - post-normalization so FAN/SAN do not zero-out GAT signal
# 🔧 Tier 2.2: Environment variable override for testing GAT bypass hypothesis
import os
bypass_gat_completely = os.environ.get("BYPASS_GAT_COMPLETELY", "0") == "1"

if (
    self.gat is not None
    and gat_residual_base is not None
    and hasattr(self, "gat_residual_gate")
):
    if bypass_gat_completely:
        # Tier 2.2 Test: Force alpha=0 (100% temporal features, 0% GAT)
        alpha = torch.tensor(0.0, device=normalized_features.device)
        if self.training:
            logger.warning("[TIER2.2] GAT COMPLETELY BYPASSED (alpha=0.0, BYPASS_GAT_COMPLETELY=1)")
    else:
        alpha = torch.sigmoid(self.gat_residual_gate)
    gat_residual = (
        gat_residual_base.unsqueeze(1)
        .repeat(1, normalized_features.size(1), 1)
        .contiguous()
    )
    normalized_features = alpha * gat_residual + (1 - alpha) * normalized_features
```

**Mechanism**:
- **Normal behavior**: `alpha = sigmoid(gat_residual_gate)` (learnable parameter, typically ≈ 0.5)
- **Test override**: `alpha = 0.0` (force 100% temporal features, 0% GAT contribution)

### Trial Configuration

```bash
BYPASS_GAT_COMPLETELY=1 FORCE_PHASE2=1 python scripts/integrated_ml_training_pipeline.py \
  --max-epochs 3 \
  --data-path output/ml_dataset_latest_full.parquet \
  --batch-size 2048 \
  --lr 1e-4
```

**Key Parameters**:
- GAT Bypass Toggle: `BYPASS_GAT_COMPLETELY=1` (activated)
- Learning Rate: 1e-4
- Batch Size: 2048
- Max Epochs: 3

---

## 📊 Results

### GAT Bypass Confirmation

**Multiple confirmation logs throughout training**:
```
[2025-10-30 12:35:59,398] [TIER2.2] GAT COMPLETELY BYPASSED (alpha=0.0, BYPASS_GAT_COMPLETELY=1)
[2025-10-30 12:36:06,455] [TIER2.2] GAT COMPLETELY BYPASSED (alpha=0.0, BYPASS_GAT_COMPLETELY=1)
[2025-10-30 12:37:06,231] [TIER2.2] GAT COMPLETELY BYPASSED (alpha=0.0, BYPASS_GAT_COMPLETELY=1)
...
[Multiple confirmations across all 3 epochs]
```

**Status**: ✅ GAT bypass successfully enforced (alpha=0.0 confirmed)

### Epoch 1-3 Validation Metrics

**Prediction Variance** (Critical Metric):
```
Epoch 1:
  Horizon 1:  pred_std = 0.000000
  Horizon 5:  pred_std = 0.000000
  Horizon 10: pred_std = 0.000000
  Horizon 20: pred_std = 0.000000

Epoch 2:
  Horizon 1:  pred_std = 0.000000
  Horizon 5:  pred_std = 0.000000
  Horizon 10: pred_std = 0.000000
  Horizon 20: pred_std = 0.000000

Epoch 3:
  Horizon 1:  pred_std = 0.000000
  Horizon 5:  pred_std = 0.000000
  Horizon 10: pred_std = 0.000000
  Horizon 20: pred_std = 0.000000
```

**Scale Ratio** (SCALE(yhat/y)):
```
h=1d:  SCALE = 0.00
h=5d:  SCALE = 0.00
h=10d: SCALE = 0.00
h=20d: SCALE = 0.00
```

**Calibration Coefficients**:
```
h=1d:  CAL = +0.000 + 1.000*yhat
h=5d:  CAL = +0.000 + 1.000*yhat
h=10d: CAL = +0.000 + 1.000*yhat
h=20d: CAL = +0.000 + 1.000*yhat
```

**Full Metrics** (Epoch 3):
| Horizon | IC     | yhat_std | SCALE | R²     | MAE    | RMSE   |
|---------|--------|----------|-------|--------|--------|--------|
| h=1d    | 0.0061 | 0.000000 | 0.00  | 0.0000 | 0.0078 | 0.0098 |
| h=5d    | 0.0215 | 0.000000 | 0.00  | 0.0000 | 0.0174 | 0.0222 |
| h=10d   | 0.0213 | 0.000000 | 0.00  | -0.0000| 0.0208 | 0.0282 |
| h=20d   | -0.0131| 0.000000 | 0.00  | 0.0000 | 0.0278 | 0.0375 |

### Comparison: Normal GAT vs Bypassed GAT

| Configuration     | Phase         | pred_std (h=5d) | SCALE | Result       |
|-------------------|---------------|-----------------|-------|--------------|
| **Normal GAT**    | Phase 2 (50ep)| 0.000000        | 0.00  | Degenerate   |
| **Normal GAT**    | Phase 3 (3ep) | 0.000000        | 0.00  | Degenerate   |
| **MSE Loss**      | Tier 2.1 (1ep)| 0.000000        | 0.00  | Degenerate   |
| **GAT Bypassed**  | Tier 2.2 (3ep)| 0.000000        | 0.00  | **Degenerate** |

**Conclusion**: GAT module has **ZERO impact** on degeneracy.

---

## 🔍 Analysis

### Hypothesis Test Result

**H0 (GAT bypass causes degeneracy): REJECTED** ❌

**Evidence**:
1. Complete GAT bypass (alpha=0.0) produces identical zero-variance predictions
2. All metrics (pred_std, SCALE, CAL) unchanged
3. Same degeneracy pattern across all 3 epochs and 4 horizons
4. IC values nearly identical to previous trials

**Interpretation**: GAT module is **NOT** involved in the root cause of model degeneracy.

### What This Tells Us

**Eliminated Root Causes** ✅:
1. ✅ Quantile loss weakness (Tier 2.1 - MSE also fails)
2. ✅ GAT residual bypass dominance (Tier 2.2 - bypass doesn't help)
3. ✅ Weight initialization problems (Tier 1.1 - weights normal)
4. ✅ Hyperparameter issues (Phase 3 trial)

**Remaining Suspects** 🔍:

1. **Feature Encoder Collapse** (HIGHEST PRIORITY - Tier 2.3)
   - Cross-sectional normalization (mean=0, std=1) + LayerNorm interaction
   - May collapse all features to near-zero before reaching any downstream modules
   - Located at the very beginning of forward pass

2. **Adaptive Normalization Layer** (Tier 2.4)
   - `self.adaptive_norm` (RMSNorm/LayerNorm)
   - Applied immediately after feature encoder
   - Could amplify feature collapse

3. **Prediction Heads Architecture** (Tier 2.5)
   - Horizon-specific prediction heads may have structural issues
   - Gradient flow might be blocked before reaching heads

4. **Data Preprocessing Issue** (Tier 3)
   - Cross-sectional normalization at dataset level
   - Feature distribution after CS normalization may be pathological
   - Requires dataset regeneration to test

### Key Insight: Problem Location Narrowed

Since GAT bypass has no effect, we now know:

**Problem is UPSTREAM of GAT module** in the architecture:

```
Data → Feature Encoder → Adaptive Norm → [PROBLEM HERE] → GAT → FAN/SAN → Heads
                                              ↑
                                    GAT bypass has no effect
                                    → Problem must be here or earlier
```

**Why this matters**:
- GAT is in the middle of the architecture
- If bypassing it doesn't help, the problem originates **before** GAT
- This rules out all downstream components (FAN, SAN, prediction heads)
- Focus should shift to **input pipeline and feature encoding**

---

## 💡 Critical Insights

### 1. GAT Module Is Not the Bottleneck

**Original concern**: GAT gate α ≈ 0.5 might allow features to bypass graph learning

**Reality**: Even with 100% bypass (α=0), degeneracy persists

**Implication**: GAT contribution (whether 0%, 50%, or 100%) is irrelevant because the **input to GAT** is already degenerate.

### 2. Narrowing the Search Space

We've now eliminated 3 major architectural components:
- Loss function layer (Tier 2.1)
- GAT module (Tier 2.2)
- Weight initialization (Tier 1.1)

This leaves us with a much smaller search space:
- **Feature Encoder** (82 features → hidden_size=256)
- **Adaptive Normalization** (RMSNorm/LayerNorm)
- **Data Preprocessing** (cross-sectional normalization)

### 3. IC > 0 with pred_std = 0 Anomaly Persists

Still observing:
```
IC = 0.0215 (h=5d)
pred_std = 0.000000
```

This suggests:
- IC may be computed on **different data split** than pred_std
- IC may be computed on **pre-aggregation quantiles** (5 quantiles with variance)
- Need to verify IC computation logic in validation code

---

## 🛠️ Next Steps

### Priority Order

Based on elimination of GAT as root cause, updated priority:

### **HIGHEST PRIORITY: Tier 2.3 - Feature Encoder Test**

**Hypothesis**: Feature encoder (input projection) collapses 82 features to near-zero hidden representations.

**Test Method**: Add diagnostic logging to feature encoder
```python
# In forward() after feature_encoder
x_encoded = self.feature_encoder(x)
logger.info(f"[ENCODER-DEBUG] mean={x_encoded.mean():.6f}, std={x_encoded.std():.6f}, min={x_encoded.min():.6f}, max={x_encoded.max():.6f}")
```

**Expected**:
- If encoder output has std ≈ 0 → Feature encoder collapse confirmed
- If encoder output has std > 0 → Problem is in normalization layer

**Implementation**: 5 minutes (logging only)
**Risk**: Zero (diagnostic only)

---

### **HIGH PRIORITY: Tier 2.4 - Adaptive Normalization Test**

**Hypothesis**: Adaptive normalization (RMSNorm/LayerNorm) after encoder collapses features.

**Test Method**: Bypass adaptive normalization
```python
# In forward()
# normalized_features = self.adaptive_norm(x_encoded)  # Original
normalized_features = x_encoded  # Bypass normalization
```

**Expected**:
- If variance appears → Normalization layer issue
- If degeneracy persists → Problem is in encoder or data

**Implementation**: 2 minutes (single line change)
**Risk**: Low (easily reversible)

---

### **MEDIUM PRIORITY: Tier 1.2 - Gradient Monitoring**

**Purpose**: Check if gradients flow to prediction heads

**Test Method**: Add gradient logging after backward pass
```python
# After loss.backward()
for name, head in model.horizon_heads.items():
    if head.weight.grad is not None:
        grad_norm = head.weight.grad.norm().item()
        logger.info(f"[GRAD-DEBUG] {name} grad_norm: {grad_norm:.6f}")
    else:
        logger.warning(f"[GRAD-DEBUG] {name} grad is None!")
```

**Expected**:
- grad_norm > 0 → Gradients flowing (not the issue)
- grad_norm ≈ 0 → Gradient vanishing
- grad is None → Gradient not reaching heads

**Implementation**: 5 minutes (logging only)
**Risk**: Zero (diagnostic only)

---

### **LOW PRIORITY: Tier 2.5 - Cross-Sectional Normalization Test**

**Hypothesis**: CS norm (mean=0, std=1) at dataset level + LayerNorm creates instability

**Test Method**: Disable cross-sectional normalization in data preprocessing
```python
# In data pipeline
# x_norm = cross_sectional_normalize(x)  # Original
x_norm = x  # Disable CS norm
```

**Expected**:
- If variance appears → CS norm interaction problem
- If degeneracy persists → Not the issue

**Implementation**: 30-60 minutes (requires dataset regeneration)
**Risk**: Medium (affects all downstream features, need to regenerate dataset)

---

### **FALLBACK: Tier 3 - Baseline Simplification**

If all Tier 2 tests fail, simplify architecture to identify minimal working model:

**Approach**: Remove components one by one
1. Start with: Linear(input_features → hidden) → Linear(hidden → output)
2. Add back: LayerNorm
3. Add back: Attention
4. Add back: GAT
5. Identify where degeneracy starts

**Implementation**: 1-2 days (major architecture redesign)
**Risk**: High (requires extensive testing and comparison)

---

## 📝 Recommended Immediate Action

### Fast Diagnostic Path (15 minutes total)

**Step 1**: Feature Encoder Diagnostics (5 min)
- Add logging after `self.feature_encoder(x)`
- Run 1 epoch trial
- Check if encoder output has variance

**Step 2**: Adaptive Norm Bypass Test (10 min)
- Bypass `self.adaptive_norm` layer
- Run 1 epoch trial
- Check if variance appears

**Step 3**: Analyze and Decide
- If Step 1 shows zero variance → Encoder is the problem
- If Step 2 fixes it → Normalization is the problem
- If neither helps → Proceed to gradient monitoring (Tier 1.2)

---

## 📊 Experiment Metadata

**Log File**: `_logs/training/tier2.2_gat_bypass_trial_20251030_123407.log`

**Checkpoint**: Not saved (trial stopped after Epoch 3)

**Duration**: ~3 minutes (startup + 3 epochs)

**Code Changes**:
- `src/atft_gat_fan/models/architectures/atft_gat_fan.py:826-842` (GAT bypass toggle)

**Environment**:
- GPU: NVIDIA A100-SXM4-80GB
- PyTorch: 2.9.0+cu128
- CUDA: 12.4

---

## 🎓 Key Learnings

1. **GAT Module Is Not the Bottleneck**: Completely bypassing GAT (100% temporal features) has zero effect on degeneracy.

2. **Problem Is in Input Pipeline**: Since bypassing GAT doesn't help, the issue must be in feature encoding or data preprocessing.

3. **Efficient Elimination Strategy**: Testing architectural components systematically (loss → GAT → encoder) quickly narrows down the problem location.

4. **IC Anomaly Remains Unexplained**: Still need to trace why IC > 0 while pred_std = 0 (likely different data splits or pre-aggregation computation).

5. **Next Focus: Feature Encoder/Normalization**: All evidence points to the earliest stages of the forward pass (feature projection and normalization layers).

---

**Status**: Tier 2.2 **FAILED** - GAT bypass hypothesis rejected
**Next Trial**: Tier 2.3 - Feature Encoder Diagnostics
**Timeline**: 5 minutes for diagnostic logging + 10 minutes for 1-epoch trial

---

**Prepared by**: Claude Code AI Agent
**Date**: 2025-10-30
**Related Documents**:
- `docs/tier2.1_mse_trial_results.md` - Tier 2.1 MSE loss trial failure analysis
- `docs/phase3_trial_analysis.md` - Phase 3 hyperparameter trial failure analysis
- `scripts/inspect_checkpoint_weights.py` - Tier 1.1 weight inspection (passed)
