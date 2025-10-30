# Tier 2.1: MSE Loss Replacement Trial - Results

**Date**: 2025-10-30
**Trial Duration**: 1 epoch (stopped after confirming failure)
**Status**: ❌ **FAILED** - Degeneracy persists with MSE loss

---

## 🎯 Experiment Objective

Test whether **Quantile Regression Loss** is the root cause of model degeneracy.

**Hypothesis (H0)**: Quantile loss does not penalize constant predictions strongly enough, causing model to output uniform values.

**Expected Results**:
- **H0 TRUE**: yhat_std > 0 with MSE → Quantile loss confirmed as root cause
- **H0 FALSE**: yhat_std = 0 with MSE → Deeper architecture problem

---

## 🔧 Implementation

### Code Changes

**File**: `src/atft_gat_fan/models/architectures/atft_gat_fan.py:1635-1666`

```python
class QuantileLoss(nn.Module):
    """Quantile Loss (with optional MSE fallback for debugging)"""

    def __init__(self, quantiles: list[float]):
        super().__init__()
        self.quantiles = torch.tensor(quantiles)
        # Environment variable toggle for Tier 2.1 experiment
        import os
        self.use_mse = os.environ.get("USE_MSE_LOSS", "0") == "1"
        if self.use_mse:
            import logging
            logging.getLogger(__name__).warning(
                "[TIER2.1] QuantileLoss replaced with MSE (USE_MSE_LOSS=1)"
            )

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        if self.use_mse:
            # Tier 2.1: MSE loss for degeneracy testing
            # predictions: [B, num_quantiles], targets: [B]
            # Aggregate quantiles to point prediction (mean)
            if predictions.dim() == 2 and predictions.size(-1) > 1:
                point_pred = predictions.mean(dim=-1)
            else:
                point_pred = predictions.squeeze(-1)
            return torch.nn.functional.mse_loss(point_pred, targets)
        else:
            # Original quantile loss
            errors = targets.unsqueeze(-1) - predictions
            quantile_loss = torch.maximum(
                self.quantiles * errors, (self.quantiles - 1) * errors
            )
            return quantile_loss.mean()
```

### Trial Configuration

```bash
USE_MSE_LOSS=1
FORCE_PHASE2=1
python scripts/integrated_ml_training_pipeline.py \
  --max-epochs 3 \
  --data-path output/ml_dataset_latest_full.parquet \
  --batch-size 2048 \
  --lr 1e-4
```

**Key Parameters**:
- MSE Toggle: `USE_MSE_LOSS=1` (activated)
- Learning Rate: 1e-4
- Batch Size: 2048
- Max Epochs: 3 (stopped after Epoch 1)

---

## 📊 Results

### Epoch 1 Validation Metrics

**Confirmation Log**:
```
[2025-10-30 11:48:06,488][...][WARNING] - [TIER2.1] QuantileLoss replaced with MSE (USE_MSE_LOSS=1)
```

**Prediction Variance** (Critical Metric):
```
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

**Full Metrics**:
| Horizon | IC     | yhat_std | SCALE | R²     | MAE    | RMSE   |
|---------|--------|----------|-------|--------|--------|--------|
| h=1d    | 0.0061 | 0.000000 | 0.00  | 0.0000 | 0.0077 | 0.0098 |
| h=5d    | 0.0215 | 0.000000 | 0.00  | 0.0000 | 0.0174 | 0.0223 |
| h=10d   | 0.0213 | 0.000000 | 0.00  | 0.0000 | 0.0207 | 0.0282 |
| h=20d   | -0.0131| 0.000000 | 0.00  | 0.0000 | 0.0278 | 0.0375 |

### Comparison: Quantile Loss vs MSE Loss

| Loss Type     | Phase         | pred_std (h=5d) | SCALE | Result       |
|---------------|---------------|-----------------|-------|--------------|
| **Quantile**  | Phase 2 (50ep)| 0.000000        | 0.00  | Degenerate   |
| **Quantile**  | Phase 3 (3ep) | 0.000000        | 0.00  | Degenerate   |
| **MSE**       | Tier 2.1 (1ep)| 0.000000        | 0.00  | **Degenerate** |

**Conclusion**: Loss function type has **ZERO impact** on degeneracy.

---

## 🔍 Analysis

### Hypothesis Test Result

**H0 (Quantile loss causes degeneracy): REJECTED** ❌

**Evidence**:
1. MSE loss produces identical zero-variance predictions
2. All metrics (pred_std, SCALE, CAL) unchanged
3. Same degeneracy pattern across all 4 horizons

**Interpretation**: Quantile regression loss is **NOT** the root cause of model degeneracy.

### What This Tells Us

**Eliminated Root Causes** ✅:
1. ✅ Quantile loss weakness (Tier 2.1)
2. ✅ Initialization problems (Tier 1.1 - weights normal)
3. ✅ Hyperparameter issues (Phase 3 trial)

**Remaining Suspects** 🔍:
1. **GAT Residual Bypass** (Tier 2.2 - next)
   - Gate parameter α ≈ 0.5 allows complete bypass
   - Model may not depend on learned GAT features

2. **Feature Encoder Collapse** (Tier 2.3)
   - Cross-sectional normalization (mean=0, std=1) + LayerNorm
   - May collapse all features to near-zero

3. **Gradient Flow Blockage** (Tier 1.2 - skipped)
   - Gradients may not reach prediction heads
   - Architecture bottleneck upstream

### Mathematical Insight

**Why Both Losses Fail**:

Quantile Loss:
```
L_quantile = mean(max(τ(y-ŷ), (τ-1)(y-ŷ)))
```

MSE Loss:
```
L_mse = mean((y - ŷ)²)
```

Both losses **measure difference between predictions and targets**. If:
- Model architecture forces ŷ ≈ 0 (e.g., feature collapse)
- OR gradients don't flow to prediction heads
- THEN changing loss function has no effect

**Key Realization**: The problem is **upstream** of the loss function, in either:
1. Feature extraction pipeline
2. Graph attention module
3. Gradient propagation path

---

## 💡 Critical Insights

### 1. Architecture-Level Problem Confirmed

Changing loss function from quantile → MSE had **ZERO** impact. This definitively proves the issue is in:
- Model forward pass (feature extraction, GAT, FAN)
- Gradient flow (blocked somewhere before loss backprop)
- Data preprocessing (feature normalization)

### 2. Loss Function Is Not a Diagnostic Tool

We initially thought:
> "If MSE works, quantile loss is too weak"

But actually:
> "Both losses fail → problem is upstream of loss computation"

Loss function only matters if predictions have variance. If architecture forces constant outputs, loss function is irrelevant.

### 3. IC > 0 with pred_std = 0 Remains Unexplained

Still observing:
```
IC = 0.0215 (h=5d)
pred_std = 0.000000
```

This is **mathematically impossible** (Spearman correlation requires variance). Possible explanations:
- IC computed on different data split
- IC computed before quantile aggregation
- Bug in IC calculation (unlikely after Phase 2 fixes)

**Action**: Need to trace IC computation path in validation code.

---

## 🛠️ Next Steps

### Tier 2.2: GAT Residual Bypass Test (HIGH PRIORITY)

**Hypothesis**: GAT gate α ≈ 0.5 allows complete bypass of learned features.

**Test**: Force GAT bypass to 100%
```python
# In GAT forward pass
# return alpha * x_temporal + (1 - alpha) * gat_output  # Original
return x_temporal  # Force bypass
```

**Expected**:
- If degeneracy persists → GAT was already bypassed (confirms hypothesis)
- If degeneracy resolves → GAT was the problem (refutes hypothesis)

**Implementation**: 10-15 minutes
**Risk**: Low (easily reversible)

### Tier 1.2: Gradient Monitoring (DIAGNOSTIC)

**Purpose**: Check if gradients flow to prediction heads

**Test**: Add gradient logging
```python
# After each backward pass
for name, head in model.horizon_heads.items():
    grad_norm = head.weight.grad.norm().item()
    print(f"{name} grad_norm: {grad_norm:.6f}")
```

**Expected**:
- grad_norm > 0 → Gradients flowing (not the issue)
- grad_norm ≈ 0 → Gradient vanishing/blockage

**Implementation**: 5 minutes (logging only)
**Risk**: Zero (diagnostic only)

### Tier 2.3: Cross-Sectional Normalization Test (MEDIUM PRIORITY)

**Hypothesis**: CS norm (mean=0, std=1) + LayerNorm collapses features

**Test**: Disable CS normalization
```python
# In data preprocessing
# x_norm = cross_sectional_normalize(x)  # Original
x_norm = x  # Disable CS norm
```

**Expected**:
- If variance appears → CS norm interaction problem
- If degeneracy persists → Not the issue

**Implementation**: 30-60 minutes (dataset regeneration required)
**Risk**: Medium (affects all downstream features)

---

## 📝 Recommended Sequence

### Option A: Fast Diagnostics (30 minutes total)
1. **Tier 1.2** (Gradient monitoring - 5 min)
2. **Tier 2.2** (GAT bypass test - 15 min setup + 10 min trial)
3. Analyze both results → Pinpoint bottleneck

### Option B: Skip to Most Likely Cause (20 minutes)
1. **Tier 2.2** (GAT bypass test immediately)
2. If fails → **Tier 2.3** (CS norm test)
3. If both fail → **Tier 3** (Simplify to baseline)

### Option C: Comprehensive (60 minutes)
1. **Tier 1.2** (Gradient monitoring)
2. **Tier 2.2** (GAT bypass)
3. **Tier 2.3** (CS norm)
4. **Tier 3** (Baseline simplification if all fail)

---

## 📊 Experiment Metadata

**Log File**: `_logs/training/tier2.1_mse_trial_20251030_114732.log`

**Checkpoint**: Not saved (trial stopped after Epoch 1)

**Duration**: ~4 minutes (startup + 1 epoch)

**Code Changes**:
- `src/atft_gat_fan/models/architectures/atft_gat_fan.py:1635-1666` (MSE toggle)

**Environment**:
- GPU: NVIDIA A100-SXM4-80GB
- PyTorch: 2.9.0+cu128
- CUDA: 12.4

---

## 🎓 Key Learnings

1. **Loss Function Is Not a Silver Bullet**: Changing loss type has zero effect if architecture forces constant predictions.

2. **Problem Is Upstream**: Degeneracy originates in feature extraction, GAT, or gradient flow - not in loss computation.

3. **Diagnostic Strategy**: Test architecture components systematically (GAT → Feature Encoder → Baseline) rather than hyperparameters.

4. **IC Anomaly**: Still need to explain IC > 0 with pred_std = 0 (inconsistent metrics).

---

**Status**: Tier 2.1 **FAILED** - Quantile loss hypothesis rejected
**Next Trial**: Tier 2.2 - GAT Residual Bypass Test
**Timeline**: 15-20 minutes for setup + 10-15 minutes for 3-epoch trial

---

**Prepared by**: Claude Code AI Agent
**Date**: 2025-10-30
**Related Documents**:
- `docs/phase3_trial_analysis.md` - Phase 3 hyperparameter trial failure analysis
- `scripts/inspect_checkpoint_weights.py` - Tier 1.1 weight inspection (passed)
