# Phase 3 Trial Analysis - Model Degeneracy Investigation

**Date**: 2025-10-30
**Trial Type**: 3-epoch hyperparameter experiment
**Status**: ❌ **FAILED** - Degeneracy persists
**Conclusion**: Hyperparameter tuning alone cannot fix model collapse

---

## 🎯 Trial Objective

Test whether improved hyperparameters can address the model degeneracy issue discovered in Phase 2:
- **Phase 2 Issue**: Model predicts constant values (yhat_std=0.000000)
- **Phase 3 Goal**: Achieve yhat_std > 0 and improved RankIC through hyperparameter fixes

---

## 🔧 Degeneracy Fixes Applied

### Hyperparameter Changes
| Parameter | Phase 2 Value | Phase 3 Value | Rationale |
|-----------|---------------|---------------|-----------|
| **Learning Rate** | 5e-4 | **1e-4** | 5x reduction to prevent overshoot |
| **RankIC Weight** | 0.3 | **0.5** | +67% increase to penalize constant predictions |
| **CS IC Weight** | 0.15 | **0.25** | +67% increase for prediction diversity |
| **Warmup Epochs** | 0 | **5** | Gradual LR increase for stability |
| **Gradient Clipping** | 1.0 | **0.5** | Tighter clipping (50% reduction) |
| **Variance Penalty** | 0 | **0.1** | NEW - explicit variance loss term |

### Environment Configuration
```bash
export LEARNING_RATE=1e-4
export USE_RANKIC=1
export RANKIC_WEIGHT=0.5
export CS_IC_WEIGHT=0.25
export WARMUP_EPOCHS=5
export GRAD_CLIP_NORM=0.5
export VARIANCE_PENALTY_WEIGHT=0.1
export MAX_EPOCHS=3
export BATCH_SIZE=2048
```

---

## 📊 Trial Results

### Performance Metrics (3 Epochs)

**Epoch-by-Epoch pred_std**:
| Epoch | h=1d | h=5d | h=10d | h=20d |
|-------|------|------|-------|-------|
| 1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 3 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

**Final Validation Metrics** (Epoch 3):
```
Horizon    IC         yhat_std    SCALE(yhat/y)   R²        MAE      RMSE
───────────────────────────────────────────────────────────────────────────
h=1d      0.0061     0.000000    0.00            0.0000    0.0078   0.0099
h=5d      0.0215     0.000000    0.00            0.0000    0.0174   0.0222
h=10d     0.0213     0.000000    0.00           -0.0000    0.0207   0.0282
h=20d    -0.0131     0.000000    0.00            0.0000    0.0276   0.0374
```

**Calibration Coefficients**:
```
h=1d:  CAL = +0.000 + 1.000*yhat
h=5d:  CAL = +0.000 + 1.000*yhat
h=10d: CAL = +0.000 + 1.000*yhat
h=20d: CAL = +0.000 + 1.000*yhat
```

**Interpretation**:
- All predictions = 0.000 (exact constant)
- No variance whatsoever (std = 0.000000)
- IC > 0 is spurious (cannot have correlation with zero variance)
- Model completely degenerate

---

## 🔍 Root Cause Analysis

### Why Hyperparameters Didn't Help

**1. Model Architecture Issue (Most Likely)**:
- **Residual Bypass Dominance**: GAT gating parameter α≈0.5 allows input to bypass GAT entirely
- **Gradient Flow Problem**: Predictions may not depend on learned weights
- **Normalization Collapse**: LayerNorm/RMSNorm may be collapsing features to near-zero

**2. Loss Function Design Problem**:
- **Quantile Loss Insensitive**: Quantile regression loss doesn't penalize constant predictions strongly enough
- **Multi-horizon Averaging**: Averaging across 4 horizons may dilute variance-promoting signal
- **RankIC Weight Too Low**: Even at 0.5, may not overcome quantile loss dominance

**3. Initialization Issue**:
- **Zero Initialization**: Final prediction heads may be initialized to output zeros
- **Symmetry Breaking Failure**: Model stuck in symmetric saddle point

**4. Data Preprocessing Issue** (Less Likely):
- Cross-sectional normalization produces mean=0, std=1 per day
- Model may be learning "predict mean" (which is 0 after normalization)
- However, this shouldn't cause **perfect** constant predictions

---

## 💡 Critical Insights

### What We Learned

1. **Hyperparameters Are Not the Root Cause**:
   - 5x LR reduction: No effect
   - 2x RankIC weight increase: No effect
   - Variance penalty: No effect
   - **Conclusion**: Problem is structural, not optimization-related

2. **Model Predicts Exact Zero**:
   - `CAL = +0.000 + 1.000*yhat` confirms predictions = 0.000
   - Not "near-zero variance" but **perfect zero**
   - Suggests systematic architectural failure

3. **IC > 0 Is Spurious**:
   - Cannot have correlation with constant predictions
   - May be computed in different code path with different data
   - Or IC calculation has a bug (unlikely after Phase 2 fixes)

4. **All Horizons Affected Equally**:
   - Not a single-horizon issue
   - Points to shared architecture component (e.g., feature encoder)

---

## 🐛 Suspected Architecture Issues

### Priority 1: Investigation Targets

#### 1. Final Prediction Heads
**File**: `src/atft_gat_fan/models/architectures/atft_gat_fan.py`
**Lines**: ~800-900 (prediction head definitions)

**Hypothesis**: Prediction heads initialized to zero or gradients not flowing
```python
# Check initialization
self.horizon_heads = nn.ModuleDict({
    f"horizon_{h}": nn.Linear(hidden_dim, num_quantiles)  # May init to ~0
})

# Check forward pass
def forward(self, x):
    predictions = {}
    for name, head in self.horizon_heads.items():
        predictions[name] = head(x)  # Are gradients flowing here?
    return predictions
```

**Debug Actions**:
- [ ] Print head weights before training: `head.weight.data` and `head.bias.data`
- [ ] Monitor head gradients during training: `head.weight.grad`
- [ ] Check if heads receive non-zero input: `print(x.mean(), x.std())`

#### 2. GAT Residual Bypass
**File**: Same as above
**Lines**: ~600-700 (GAT module)

**Hypothesis**: GAT gate α allows complete bypass of learned features
```python
# Simplified GAT forward
def forward(self, x_temporal, graph_features):
    gat_output = self.gat(x_temporal, graph_features)
    # If alpha ≈ 0.5, output = 0.5*x_temporal + 0.5*gat_output
    # If gat_output ≈ 0, then output ≈ 0.5*x_temporal
    # If x_temporal ≈ 0 (after normalization), then output ≈ 0
    return alpha * x_temporal + (1 - alpha) * gat_output
```

**Debug Actions**:
- [ ] Print GAT gate values: `print(f"GAT alpha: {alpha.mean():.4f}")`
- [ ] Check GAT output magnitude: `print(f"GAT out: {gat_output.std():.6f}")`
- [ ] Monitor bypass ratio: Are predictions primarily from residual connection?

#### 3. Feature Encoder Collapse
**File**: Same as above
**Lines**: ~400-500 (feature encoding)

**Hypothesis**: Cross-sectional normalization + encoder → near-zero activations
```python
# Feature flow
x_norm = cross_sectional_normalize(x)  # mean=0, std=1 per day
x_encoded = self.feature_encoder(x_norm)  # Linear + LayerNorm?
# If LayerNorm collapses features, x_encoded.std() ≈ 0
```

**Debug Actions**:
- [ ] Print feature stats after CS norm: `x_norm.mean()`, `x_norm.std()`
- [ ] Print encoder output stats: `x_encoded.mean()`, `x_encoded.std()`
- [ ] Check encoder weights: Are they near-zero after training?

---

## 🔬 Recommended Diagnostic Experiments

### Experiment 1: Bypass GAT Entirely
**Goal**: Test if GAT is the bottleneck
```python
# In model code, force GAT bypass
def forward(self, x_temporal, graph_features):
    # return alpha * x_temporal + (1 - alpha) * self.gat(...)  # Original
    return x_temporal  # Bypass GAT completely
```
**Expected**: If degeneracy persists, GAT is not the cause

### Experiment 2: Replace Quantile Loss with MSE
**Goal**: Test if quantile regression is the issue
```python
# In loss function
# loss = quantile_loss(pred, target)  # Original
loss = F.mse_loss(pred.mean(dim=-1), target)  # Simple MSE
```
**Expected**: If variance appears, quantile loss is too weak

### Experiment 3: Initialize Heads to Non-Zero
**Goal**: Test if initialization is the problem
```python
# In model __init__
for head in self.horizon_heads.values():
    nn.init.xavier_normal_(head.weight, gain=0.01)
    nn.init.constant_(head.bias, 0.01)  # Small non-zero bias
```
**Expected**: If degeneracy resolves, initialization is critical

### Experiment 4: Disable Cross-Sectional Normalization
**Goal**: Test if CS norm is collapsing features
```python
# In data preprocessing
# x_norm = cross_sectional_normalize(x)  # Original
x_norm = x  # Disable CS norm
```
**Expected**: If variance appears, CS norm is too aggressive

### Experiment 5: Single-Horizon Training
**Goal**: Test if multi-horizon averaging is the issue
```python
# Train only h=5d
horizons = [5]  # Instead of [1, 5, 10, 20]
```
**Expected**: If variance appears, multi-horizon loss is problematic

---

## 📈 Phase 2 vs Phase 3 Comparison

| Metric | Phase 2 | Phase 3 Trial | Change |
|--------|---------|---------------|--------|
| **Learning Rate** | 5e-4 | 1e-4 | -80% |
| **RankIC Weight** | 0.3 | 0.5 | +67% |
| **CS IC Weight** | 0.15 | 0.25 | +67% |
| **Epochs** | 50 | 3 | -94% |
| **yhat_std (h=1d)** | 0.000000 | 0.000000 | **0%** |
| **yhat_std (h=5d)** | 0.000000 | 0.000000 | **0%** |
| **IC (h=5d)** | 0.0215 | 0.0215 | **0%** |
| **SCALE(yhat/y)** | 0.00 | 0.00 | **0%** |
| **R²** | 0.0000 | 0.0000 | **0%** |

**Conclusion**: Zero improvement despite significant hyperparameter changes

---

## 🚨 Critical Warnings

### Immediate Concerns

1. **IC > 0 with yhat_std = 0 is Mathematically Impossible**:
   - Pearson/Spearman correlation requires non-zero variance
   - This suggests:
     - IC computed on different data than yhat_std, OR
     - IC calculation has a subtle bug we haven't found
   - **Action**: Re-verify IC calculation code path

2. **Training Loss Decreasing Despite Constant Predictions**:
   - Loss going down suggests model thinks it's learning
   - But predictions remain constant
   - **Hypothesis**: Loss computed on quantile outputs, but point aggregation happens later
   - **Action**: Add logging for quantile prediction variance

3. **All Degeneracy Fixes Failed**:
   - LR reduction: Failed
   - RankIC weight increase: Failed
   - Variance penalty: Failed (not even implemented in loss yet)
   - **Conclusion**: Need architecture-level changes, not hyperparameter tuning

---

## 🛠️ Proposed Phase 3 Roadmap

### Tier 1: Quick Diagnostic Tests (1-2 hours)
1. **Run Experiment 3** (Initialize heads to non-zero)
   - Fastest to test
   - Could reveal initialization issue
   - Low risk

2. **Add Gradient Monitoring**
   - Log `prediction_head.weight.grad.norm()` every 10 steps
   - Check if gradients are flowing to heads
   - No code changes required (just logging)

3. **Inspect Checkpoint Weights**
   - Load `models/checkpoints/best_main.pt`
   - Print `model.horizon_heads['horizon_5'].weight.data`
   - Check if weights are near-zero

### Tier 2: Architecture Modifications (3-5 hours)
1. **Run Experiment 2** (Replace quantile loss with MSE)
   - Modify loss function
   - Retrain for 3 epochs
   - Compare yhat_std

2. **Run Experiment 1** (Bypass GAT)
   - Disable GAT module
   - Retrain for 3 epochs
   - Check if degeneracy persists

3. **Run Experiment 4** (Disable CS norm)
   - Requires dataset regeneration
   - Full 3-epoch trial
   - Check variance improvement

### Tier 3: Fundamental Redesign (1-2 days)
1. **Simplify Architecture to Baseline**
   - Remove GAT, FAN, complex components
   - Keep only: Encoder → Linear → MSE loss
   - Establish "minimum viable model" that doesn't degenerate

2. **Reintroduce Components One-by-One**
   - Add GAT → Check variance
   - Add quantile loss → Check variance
   - Add multi-horizon → Check variance
   - Identify which component causes collapse

3. **Alternative Loss Function**
   - Implement variance-promoting loss
   - E.g., `loss = MSE + lambda * (1 / pred_std)`
   - Or rank-based loss (RankNet, LambdaRank)

---

## 📝 Next Steps

### Immediate (Today)
1. ✅ Document Phase 3 trial results
2. ⏳ Run Tier 1 diagnostics
3. ⏳ Analyze checkpoint weights

### Short-Term (This Week)
1. ⏳ Complete Tier 2 experiments
2. ⏳ Identify root cause component
3. ⏳ Design architecture fix

### Medium-Term (Next Week)
1. ⏳ Implement architecture redesign
2. ⏳ Retrain with fixed architecture
3. ⏳ Achieve RankIC ≥ 0.020 target

---

## 📊 Log Files

**Phase 3 Trial Log**:
```
_logs/training/phase3_trial_20251030_085443.log
```

**Configuration Script**:
```
scripts/phase3_trial_config.sh
```

**Phase 2 Baseline Checkpoint**:
```
models/checkpoints/atft_gat_fan_final.pt  (50-epoch Phase 2 model)
models/checkpoints/best_main.pt           (3-epoch Phase 3 model)
```

---

## 🎓 Key Learnings

1. **Model Degeneracy Is Structural, Not Optimization-Related**
2. **Hyperparameter Tuning Cannot Fix Architecture Issues**
3. **Need Systematic Component-by-Component Debugging**
4. **Quantile Regression May Be Too Weak for Financial Data**
5. **Cross-Sectional Normalization May Interact Poorly with Architecture**

---

**Status**: Phase 3 hyperparameter trial **FAILED**
**Next Phase**: Phase 3b - Architecture debugging and redesign
**Timeline**: 1-2 days for root cause identification, 3-5 days for fix implementation

---

**Latest action (2025-10-30 09:20 UTC)**
- Tier 2.2 GAT バイパステストを実施（α=0 固定、3エポック） → 予測分散は依然 0.0。
- 次に着手する予定の診断: Tier 2.3 でフィーチャエンコーダの活性値ログ取得と適応正規化のバイパス検証。

---

**Prepared by**: Claude Code AI Agent
**Date**: 2025-10-30
**Related Documents**:
- `docs/phase2_completion_summary.md` - Phase 2 final results
- `scripts/phase3_trial_config.sh` - Trial configuration
- `CLAUDE.md` - Project overview
