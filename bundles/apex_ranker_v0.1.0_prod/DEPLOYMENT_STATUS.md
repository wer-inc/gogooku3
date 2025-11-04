# APEX Ranker v0.1.0 Production Deployment Status

**Date**: 2025-11-04
**Bundle**: `apex_ranker_v0.1.0_prod.tar.gz` (12 MB)
**Git Tag**: `apex-ranker-v0.1.0-prod`

---

## ✅ Phase B: Lightweight Validation (COMPLETE)

### Validation Results

**Bundle Integrity** (SHA256 verified):
- ✅ Model: `639c119640bc55ad8c8fc1e890a1c91bcf4e92b2406f752356a24971d7b6703d`
- ✅ Config: `ec3f74e3f8e264ff76ed16a33e095b49f2ef98112afddcc07daeafe7e8e50be2`
- ✅ Size: 12 MB (correct)

**Smoke Test Execution** (2025-09-20 to 2025-10-03):
```
Model Init: in_features=89, patch_multiplier=auto, add_csz=False
Effective Dim: 178 (89×2, CS-Z in checkpoint)
Rebalances: 2 (monthly)
Supply Stability: candidate_kept=53 on both dates
Transaction Costs: ¥25,119 (25 bps)
Execution: Clean completion, no errors
```

### Critical Discovery: patch_multiplier Configuration

**Problem Identified**:
- Model checkpoint was trained WITH CS-Z (178-dimensional embeddings)
- Initial attempt to add `patch_multiplier: 1` forced CS-Z OFF
- This created dimension mismatch: checkpoint expects 178ch, config forced 89ch

**Root Cause**:
```yaml
# INCORRECT (initial attempt):
model:
  patch_multiplier: 1  # Forces 89 channels (CS-Z OFF)
  # Result: RuntimeError - size mismatch [178, 1, 16] vs [89, 1, 16]

# CORRECT (current):
model:
  # NO patch_multiplier line - defaults to 'auto'
  # Allows auto-detection of 178 channels from checkpoint
```

**Resolution**:
- Reverted config to original (no explicit `patch_multiplier`)
- Model loader auto-detects 178ch from checkpoint metadata
- Sets `add_csz=False` internally (features already doubled in checkpoint)
- Loads weights correctly without dimension mismatch

### Key Lesson

**The "356ch bug prevention" is already built into PyTorch's checkpoint loading via `patch_multiplier=auto`.**

**New Operational Rule**:
- **Default**: Leave `patch_multiplier` unset (auto-detection)
- **Only specify** `patch_multiplier: 1` when checkpoint is verified to use raw 89ch
- **Never force** without verifying checkpoint training configuration

---

## 📊 Production Performance (BASELINE Configuration)

**Configuration**: EI=OFF, A.3=OFF, A.4=OFF

**Backtest Results** (2024-01-01 to 2025-10-31, 22 months):
- **Sharpe Ratio**: 1.439
- **Total Return**: +44.85%
- **Annualized Return**: +23.52%
- **Max Drawdown**: 16.40%
- **Win Rate**: 56.9%
- **Transaction Costs**: ¥370,429 (3.70% of capital)
- **Rebalances**: 22 (monthly)

**Supply Stability**:
- `candidate_kept=53` on all 22 rebalance dates
- Autosupply working correctly (k_min=53, 1.5× target_top_k=35)
- 100% fallback rate is EXPECTED and correct

**Why A.3 and A.4 are OFF**:
- A.3 (hysteresis): -19.5% Sharpe degradation
- A.4 (risk neutralization): Zero impact, adds complexity
- EI (enhanced inference): Zero impact in testing

---

## 🎯 Next Steps: Phase A Deployment (Optional)

### Step 5. API Server Deployment

**Startup Command**:
```bash
export APEX_MODEL="apex_ranker_v0.1.0_prod/models/apex_ranker_v0_enhanced.pt"
export APEX_CONFIG="apex_ranker_v0.1.0_prod/configs/v0_base_89_cleanADV.yaml"
export PANEL_CACHE_DIR="/var/lib/apex-ranker/panel_cache"
export USE_CACHE=1

mkdir -p "$PANEL_CACHE_DIR" && chmod 775 "$PANEL_CACHE_DIR"

python apex-ranker/apex_ranker/api/server.py \
  --model "$APEX_MODEL" \
  --config "$APEX_CONFIG" \
  --cache-dir "$PANEL_CACHE_DIR" \
  --port 8000 &
```

**Health Checks**:
```bash
# Health endpoint
curl -s http://localhost:8000/health

# Inference test
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"horizon": 20, "top_k": 35}' | jq

# Prometheus metrics
curl -s http://localhost:8000/metrics | head
```

### Step 6. Panel Cache Persistence

**Configuration**:
- Set `PANEL_CACHE_DIR=/var/lib/apex-ranker/panel_cache` (persistent storage)
- Survives server restarts
- First inference: ~2 minutes (panel build)
- Subsequent inferences: <1 second (cache hit)

**Expected Logs**:
```
[Inference] Building panel cache...
[Inference] Panel cache saved to: /var/lib/apex-ranker/panel_cache/...
```

### Step 7. Monitoring & Alerts

**4 Critical Metrics** (exposed via `/metrics`):

| Metric | Normal Range | Alert Threshold | Severity |
|--------|--------------|-----------------|----------|
| `apex_selected_count` | 53 | <53 | CRITICAL |
| `apex_effective_dim` | 178 | !=178 | CRITICAL |
| `apex_inference_latency_ms` | <1000 | >1000 | WARNING |
| `apex_daily_cost_bps` | <20 | >20 | WARNING |

**Prometheus Alert Rules**:
```promql
# Supply stability breach
max_over_time(apex_selected_count[1d]) < 53

# Dimension mismatch (CS-Z incompatibility)
apex_effective_dim != 178

# Latency degradation
histogram_quantile(0.95, apex_inference_latency_ms) > 1000

# Cost spike
avg_over_time(apex_daily_cost_bps[7d]) > 20
```

---

## 🛡️ Prevention Measures (Anti-Regression)

### 1. MANIFEST.lock (Bundle Metadata)

**Add to bundle root**:
```json
{
  "bundle_version": "v0.1.0-prod",
  "created": "2025-11-04T00:00:00Z",
  "checkpoint": {
    "file": "models/apex_ranker_v0_enhanced.pt",
    "sha256": "639c119640bc55ad8c8fc1e890a1c91bcf4e92b2406f752356a24971d7b6703d",
    "base_features": 89,
    "effective_dim": 178,
    "csz_mode": "trained_with_csz",
    "patch_multiplier": "auto"
  },
  "config": {
    "file": "configs/v0_base_89_cleanADV.yaml",
    "sha256": "ec3f74e3f8e264ff76ed16a33e095b49f2ef98112afddcc07daeafe7e8e50be2",
    "patch_multiplier": "auto (unset)"
  },
  "expected_behavior": {
    "model_init_log": "[Model Init] in_features=89, patch_multiplier=auto, add_csz=False",
    "candidate_kept": 53,
    "rebalance_freq": "monthly"
  }
}
```

### 2. Startup Validation

**Add to server.py and backtest scripts**:
```python
def validate_bundle_compatibility(model_path, config_path, manifest_path):
    """4-way verification: ckpt, config, dataset, manifest"""
    # 1. Load checkpoint metadata
    ckpt = torch.load(model_path, map_location='cpu')
    ckpt_shape = ckpt['state_dict']['encoder.patch_embed.conv.weight'].shape
    effective_dim = ckpt_shape[0]  # Should be 178

    # 2. Load config
    config = yaml.safe_load(open(config_path))
    patch_mult = config.get('model', {}).get('patch_multiplier', 'auto')

    # 3. Load manifest
    manifest = json.load(open(manifest_path))
    expected_dim = manifest['checkpoint']['effective_dim']

    # 4. Verify consistency
    assert effective_dim == expected_dim, \
        f"Dimension mismatch: ckpt={effective_dim}, manifest={expected_dim}"
    assert patch_mult == 'auto' or patch_mult is None, \
        f"Config must use auto detection, got: {patch_mult}"

    print(f"✅ Bundle validation passed: effective_dim={effective_dim}, csz=auto")
```

### 3. Logging Improvements

**Add to inference initialization**:
```python
logger.info("="*80)
logger.info("RESOLVED SETTINGS")
logger.info(f"  base_features={config.data.feature_count}")
logger.info(f"  csz_mode=auto  # ckpt={model.effective_dim}ch")
logger.info(f"  patch_multiplier=auto")
logger.info(f"  effective_dim={model.effective_dim}")
logger.info("="*80)
```

### 4. Monitoring Integration

**Add metric to `/metrics` endpoint**:
```python
# In server.py
from prometheus_client import Gauge

effective_dim_gauge = Gauge('apex_effective_dim',
                             'Model effective dimension (should be 178)')

# Update on startup
effective_dim_gauge.set(model.effective_dim)
```

---

## 📋 Day-0 Runbook

### Pre-Flight Checklist

```bash
# 1. Dry-run validation (3-point handshake)
python apex-ranker/scripts/backtest_smoke_test.py \
  --model "$APEX_MODEL" \
  --config "$APEX_CONFIG" \
  --start-date 2025-09-01 --end-date 2025-09-05 \
  --rebalance-freq monthly --horizon 20 --top-k 35 \
  --features-mode fill-zero \
  --dry-run

# Expected output:
# RESOLVED SETTINGS:
#   features=89, csz=auto (ckpt=178ch), effective_dim=178

# 2. Production smoke test (5 days, 2 runs for reproducibility)
python apex-ranker/scripts/backtest_smoke_test.py \
  --model "$APEX_MODEL" \
  --config "$APEX_CONFIG" \
  --data output/ml_dataset_latest_clean_with_adv.parquet \
  --start-date 2025-09-01 --end-date 2025-09-05 \
  --rebalance-freq monthly --horizon 20 --top-k 35 \
  --features-mode fill-zero \
  --output /tmp/smoke_run1.json

# Run again and compare top-5 stocks (must match exactly)
python apex-ranker/scripts/backtest_smoke_test.py \
  --model "$APEX_MODEL" \
  --config "$APEX_CONFIG" \
  --data output/ml_dataset_latest_clean_with_adv.parquet \
  --start-date 2025-09-01 --end-date 2025-09-05 \
  --rebalance-freq monthly --horizon 20 --top-k 35 \
  --features-mode fill-zero \
  --output /tmp/smoke_run2.json

diff <(jq -S '.daily_history[0].top_stocks' /tmp/smoke_run1.json) \
     <(jq -S '.daily_history[0].top_stocks' /tmp/smoke_run2.json)
# Expected: No differences (100% reproducible)

# 3. Monthly rebalance rehearsal (already validated in Phase B)
# Date: 2025-09-20 to 2025-10-03, expecting 2 rebalances
# Status: ✅ Passed (see Phase B results above)
```

### Systemd Service Configuration

**/etc/systemd/system/apex-ranker.service**:
```ini
[Unit]
Description=APEX-Ranker Production API
After=network-online.target

[Service]
Environment=APEX_MODEL=/opt/apex/apex_ranker_v0.1.0_prod/models/apex_ranker_v0_enhanced.pt
Environment=APEX_CONFIG=/opt/apex/apex_ranker_v0.1.0_prod/configs/v0_base_89_cleanADV.yaml
Environment=PANEL_CACHE_DIR=/var/lib/apex-ranker/panel_cache
Environment=USE_CACHE=1
WorkingDirectory=/opt/apex
ExecStartPre=/usr/bin/python /opt/apex/scripts/validate_bundle.py
ExecStart=/usr/bin/python apex-ranker/apex_ranker/api/server.py \
  --model ${APEX_MODEL} \
  --config ${APEX_CONFIG} \
  --cache-dir ${PANEL_CACHE_DIR} \
  --port 8000
Restart=always
RestartSec=10
User=apex
Group=apex

[Install]
WantedBy=multi-user.target
```

**Enable and start**:
```bash
sudo systemctl daemon-reload
sudo systemctl enable apex-ranker.service
sudo systemctl start apex-ranker.service
sudo systemctl status apex-ranker.service
```

### Monthly Rebalancing Schedule

**Configuration**: 1st business day of month, 09:00 JST

**Implementation**:
- Internal calendar in API server calculates next business day
- Excludes Japanese holidays (already implemented)
- Executes `/predict` endpoint automatically
- Logs rebalance execution to audit trail

---

## 🔍 Troubleshooting Guide

### Issue: Dimension Mismatch Error

**Symptom**:
```
RuntimeError: size mismatch for encoder.patch_embed.conv.weight:
  copying a param with shape torch.Size([178, 1, 16]) from checkpoint,
  the shape in current model is torch.Size([89, 1, 16])
```

**Diagnosis**:
- Config has explicit `patch_multiplier: 1` (forces CS-Z OFF)
- Checkpoint was trained with CS-Z ON (178 channels)

**Fix**:
1. Remove `patch_multiplier` line from config (use auto-detection)
2. Verify config hash matches bundle: `ec3f74e3f8e264ff76ed16a33e095b49f2ef98112afddcc07daeafe7e8e50be2`
3. Re-run smoke test

### Issue: Low Supply (candidate_kept < 53)

**Symptom**:
```
[Backtest] 2025-XX-XX: candidate_kept=48 sign=1
```

**Diagnosis**:
- Selection gate filtering too aggressively
- Possible data quality issue on that date

**Fix**:
1. Check `k_ratio` in config (should be 0.60)
2. Check `k_min` in config (should be 53)
3. Verify dataset has sufficient stocks on that date
4. Review selection gate threshold logs

### Issue: High Transaction Costs

**Symptom**:
```
Avg daily cost: >20 bps
```

**Diagnosis**:
- Too frequent rebalancing (should be monthly)
- High portfolio turnover

**Fix**:
1. Verify `rebalance_freq: monthly` in execution
2. Consider reducing `top_k` from 35 to 30
3. Implement cost-aware optimization (Phase 4)

---

## 📈 Production Benchmarks

**Expected Performance** (monthly rebalancing, top-35, h20):
- Sharpe Ratio: 1.4-1.5
- Max Drawdown: 15-20%
- Win Rate: 55-60%
- Transaction Costs: <5% of capital per year
- Supply Stability: 100% (candidate_kept=53 always)

**If metrics deviate >20% from benchmarks**:
1. Check dataset freshness (last modified date)
2. Verify model checkpoint integrity (SHA256)
3. Review recent market regime changes
4. Consider retraining (if data drift detected)

---

## ✅ Deployment Checklist

- [x] Phase B: Lightweight validation passed
- [x] Bundle integrity verified (SHA256)
- [x] Smoke test executed successfully (2 rebalances)
- [x] Supply stability confirmed (candidate_kept=53)
- [x] Dimension compatibility resolved (patch_multiplier=auto)
- [ ] Phase A: API server deployment
- [ ] Panel cache persistence configured
- [ ] Prometheus monitoring setup
- [ ] Systemd service enabled
- [ ] Production smoke test (5 days, 2 runs)
- [ ] Monthly rebalance rehearsal
- [ ] MANIFEST.lock created
- [ ] validate_bundle.py implemented

---

## 📚 References

- Bundle: `/workspace/gogooku3/bundles/apex_ranker_v0.1.0_prod.tar.gz`
- Runbook: `/workspace/gogooku3/bundles/apex_ranker_v0.1.0_prod/docs/RUNBOOK.md`
- P0 Testing: `P0_3_COMPLETION_REPORT.md`
- Git Tag: `apex-ranker-v0.1.0-prod`

---

**Status**: Ready for Phase A deployment (optional)
**Next Action**: Execute Steps 5-7 or remain in manual execution mode
