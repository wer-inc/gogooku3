# APEX Ranker v0.1.0 - Production Deployment Guide

**Status**: Ready for Deployment
**BASELINE Configuration**: Top-35 / Monthly / h20 / EI=OFF, A.3=OFF, A.4=OFF
**Performance**: Sharpe 1.439, Return 44.85%, MaxDD 16.40%
**Last Updated**: 2025-11-04

---

## Executive Summary

This guide provides step-by-step procedures to deploy APEX Ranker v0.1.0 in production **without breaking anything** (壊さず本番稼働する).

**Key Principle**: All validation checks MUST pass before proceeding to production.

---

## Day-0: Pre-Deployment Validation (本日中)

Run these checks **on the same day** before deployment to ensure system readiness.

### 0.1 Bundle Health Check (4-Way Verification)

**Purpose**: Verify compatibility between checkpoint, config, dataset, and manifest.

```bash
# Run validation script
python scripts/validate_bundle.py \
  --bundle bundles/apex_ranker_v0.1.0_prod \
  --dataset output/ml_dataset_latest_clean_with_adv.parquet
```

**Expected Output**:
```
✅ All required files present
✅ Loaded manifest: v0.1.0-prod
✅ Loaded checkpoint: effective_dim=178
✅ Loaded config: patch_multiplier=auto (unset)
✅ Auto-detection mode: will load 178ch from checkpoint
✅ PASSED (with warnings)
```

**Critical Checks**:
- ✅ Checkpoint dimension: 178 (CS-Z trained)
- ✅ Config `patch_multiplier`: **UNSET** (auto-detection)
- ✅ No dimension mismatch errors

**GO/NO-GO Decision**:
- ❌ **NO-GO** if any CRITICAL errors appear
- ✅ **GO** if only warnings (e.g., feature count mismatch 395 vs 89 is acceptable)

---

### 0.2 Smoke Test (5-Day + Reproducibility)

**Purpose**: Verify model loads correctly and produces deterministic results.

```bash
# Run 1: Initial smoke test (crosses month boundary)
python apex-ranker/scripts/backtest_smoke_test.py \
  --model bundles/apex_ranker_v0.1.0_prod/models/apex_ranker_v0_enhanced.pt \
  --config bundles/apex_ranker_v0.1.0_prod/configs/v0_base_89_cleanADV.yaml \
  --data output/ml_dataset_latest_clean_with_adv.parquet \
  --start-date 2025-09-29 --end-date 2025-10-03 \
  --rebalance-freq monthly --horizon 20 --top-k 35 \
  --features-mode fill-zero \
  --output /tmp/smoke_run1.json \
  2>&1 | tee /tmp/smoke_run1.log

# Run 2: Reproducibility verification (MUST match Run 1)
python apex-ranker/scripts/backtest_smoke_test.py \
  --model bundles/apex_ranker_v0.1.0_prod/models/apex_ranker_v0_enhanced.pt \
  --config bundles/apex_ranker_v0.1.0_prod/configs/v0_base_89_cleanADV.yaml \
  --data output/ml_dataset_latest_clean_with_adv.parquet \
  --start-date 2025-09-29 --end-date 2025-10-03 \
  --rebalance-freq monthly --horizon 20 --top-k 35 \
  --features-mode fill-zero \
  --output /tmp/smoke_run2.json \
  2>&1 | tee /tmp/smoke_run2.log

# Verify reproducibility
echo "=== Comparing Key Metrics ===" && \
grep -E "(Total return|Sharpe ratio|Total trades|Total transaction costs):" /tmp/smoke_run1.log > /tmp/metrics1.txt && \
grep -E "(Total return|Sharpe ratio|Total trades|Total transaction costs):" /tmp/smoke_run2.log > /tmp/metrics2.txt && \
diff /tmp/metrics1.txt /tmp/metrics2.txt && \
echo "✅ REPRODUCIBILITY VERIFIED" || echo "❌ REPRODUCIBILITY FAILED"
```

**Expected Outcome**:
- Both runs complete without errors
- All metrics **100% identical** (total trades, return, Sharpe, costs)
- Minor JSON portfolio order differences are acceptable
- Floating-point precision differences (<1e-10) are acceptable

**GO/NO-GO Decision**:
- ❌ **NO-GO** if metrics differ by >0.01% or if either run crashes
- ✅ **GO** if both runs complete successfully with identical metrics

---

### 0.3 Monthly Rebalance Rehearsal (Dry-Run)

**Purpose**: Simulate production rebalancing schedule for latest available month.

```bash
# Get latest complete month range (e.g., October 2025)
LATEST_MONTH_START="2025-10-01"
LATEST_MONTH_END="2025-10-31"

# Run monthly rehearsal
python apex-ranker/scripts/backtest_smoke_test.py \
  --model bundles/apex_ranker_v0.1.0_prod/models/apex_ranker_v0_enhanced.pt \
  --config bundles/apex_ranker_v0.1.0_prod/configs/v0_base_89_cleanADV.yaml \
  --data output/ml_dataset_latest_clean_with_adv.parquet \
  --start-date ${LATEST_MONTH_START} \
  --end-date ${LATEST_MONTH_END} \
  --rebalance-freq monthly \
  --horizon 20 \
  --top-k 35 \
  --features-mode fill-zero \
  --output /tmp/monthly_rehearsal.json \
  2>&1 | tee /tmp/monthly_rehearsal.log

# Verify output
echo "=== Monthly Rehearsal Results ===" && \
grep -E "(Rebalances executed|Total trades|Total return):" /tmp/monthly_rehearsal.log
```

**Expected Output**:
```
Rebalances executed: 1
Total trades: ~70 (varies by month)
Total return: <varies>
```

**Critical Checks**:
- ✅ Exactly 1 rebalance executed (1st business day of month)
- ✅ Portfolio contains exactly 35 stocks
- ✅ No runtime errors or warnings
- ✅ Transaction costs reported correctly

**GO/NO-GO Decision**:
- ❌ **NO-GO** if rebalance count ≠ 1 or if portfolio size ≠ 35
- ✅ **GO** if all checks pass

---

## Day-1: Production Deployment (稼働開始)

### 1.1 Environment Setup

**Create production directories**:

```bash
# Create cache directory with proper permissions
sudo mkdir -p /var/lib/apex-ranker/panel_cache
sudo chown $(whoami):$(whoami) /var/lib/apex-ranker/panel_cache
sudo chmod 755 /var/lib/apex-ranker/panel_cache

# Create log directory
sudo mkdir -p /var/log/apex-ranker
sudo chown $(whoami):$(whoami) /var/log/apex-ranker
sudo chmod 755 /var/log/apex-ranker

# Verify directories
ls -ld /var/lib/apex-ranker/panel_cache
ls -ld /var/log/apex-ranker
```

**Expected Output**:
```
drwxr-xr-x 2 <user> <user> 4096 ... /var/lib/apex-ranker/panel_cache
drwxr-xr-x 2 <user> <user> 4096 ... /var/log/apex-ranker
```

---

### 1.2 API Server Startup

**Create systemd service** (recommended for production):

```bash
# Create service file
sudo tee /etc/systemd/system/apex-ranker.service > /dev/null <<'EOF'
[Unit]
Description=APEX Ranker Production API
After=network.target

[Service]
Type=simple
User=<your-user>
WorkingDirectory=/workspace/gogooku3
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
Environment="PANEL_CACHE_DIR=/var/lib/apex-ranker/panel_cache"
ExecStart=/usr/bin/python3 apex-ranker/apex_ranker/api/server.py \
  --model bundles/apex_ranker_v0.1.0_prod/models/apex_ranker_v0_enhanced.pt \
  --config bundles/apex_ranker_v0.1.0_prod/configs/v0_base_89_cleanADV.yaml \
  --data output/ml_dataset_latest_clean_with_adv.parquet \
  --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=10s
StandardOutput=append:/var/log/apex-ranker/server.log
StandardError=append:/var/log/apex-ranker/server.log

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable apex-ranker.service
sudo systemctl start apex-ranker.service

# Check status
sudo systemctl status apex-ranker.service
```

**Alternative: Manual startup** (for testing):

```bash
# Set cache directory
export PANEL_CACHE_DIR=/var/lib/apex-ranker/panel_cache

# Start API server
python apex-ranker/apex_ranker/api/server.py \
  --model bundles/apex_ranker_v0.1.0_prod/models/apex_ranker_v0_enhanced.pt \
  --config bundles/apex_ranker_v0.1.0_prod/configs/v0_base_89_cleanADV.yaml \
  --data output/ml_dataset_latest_clean_with_adv.parquet \
  --host 0.0.0.0 --port 8000 \
  2>&1 | tee /var/log/apex-ranker/server.log &

# Note PID
echo "API Server PID: $!"
```

**Startup Verification**:

```bash
# Wait 10 seconds for server to start
sleep 10

# Test health endpoint
curl http://localhost:8000/health

# Expected output:
# {"status":"ok","model":"loaded","timestamp":"2025-11-04T..."}
```

---

### 1.3 Production API Usage

**Get Monthly Top-35 Predictions** (1st business day of each month):

```bash
# Example: Get predictions for next month's rebalance
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2025-12-01",
    "top_k": 35,
    "horizon": 20
  }' | jq '.'
```

**Expected Response**:
```json
{
  "date": "2025-12-01",
  "top_k": 35,
  "horizon": 20,
  "predictions": [
    {"code": "69370", "score": 0.8523, "rank": 1},
    {"code": "24040", "score": 0.8201, "rank": 2},
    ...
  ],
  "metadata": {
    "model_version": "v0.1.0",
    "config": "BASELINE (Top-35, Monthly, h20)",
    "timestamp": "2025-11-04T..."
  }
}
```

---

### 1.4 Monitoring Setup

**Critical Thresholds** (Alert if exceeded):

| Metric | Threshold | Action |
|--------|-----------|--------|
| **API Response Time** | >5 seconds | Investigate cache/dataset loading |
| **Panel Cache Size** | >10 GB | Prune old cache files |
| **Prediction Consistency** | >5% change vs previous day | Manual review required |
| **Portfolio Size** | ≠35 stocks | STOP - Configuration error |

**Setup monitoring script**:

```bash
# Create monitoring cron job (runs every hour)
crontab -e

# Add this line:
0 * * * * /workspace/gogooku3/apex-ranker/scripts/monitor_production.sh >> /var/log/apex-ranker/monitor.log 2>&1
```

**Monitor script** (`apex-ranker/scripts/monitor_production.sh`):

```bash
#!/bin/bash
# Production monitoring script

TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
LOG_FILE="/var/log/apex-ranker/monitor.log"

echo "[$TIMESTAMP] Running production health check"

# Check 1: API health
HEALTH=$(curl -s http://localhost:8000/health | jq -r '.status')
if [ "$HEALTH" != "ok" ]; then
  echo "❌ ALERT: API health check failed (status: $HEALTH)"
  # Send alert (email/Slack/etc)
fi

# Check 2: Cache size
CACHE_SIZE=$(du -sh /var/lib/apex-ranker/panel_cache | awk '{print $1}')
echo "✅ Panel cache size: $CACHE_SIZE"

# Check 3: Recent logs for errors
ERRORS=$(tail -n 100 /var/log/apex-ranker/server.log | grep -c "ERROR")
if [ "$ERRORS" -gt 0 ]; then
  echo "⚠️  WARNING: $ERRORS errors in last 100 log lines"
fi

echo "[$TIMESTAMP] Health check complete"
echo "---"
```

---

## Operational Guards

### Critical Rules (NEVER violate these)

1. **MANIFEST.lock Enforcement**:
   - ALWAYS validate bundle before deployment (Day-0.1)
   - NEVER modify `patch_multiplier` in config (must stay unset)
   - NEVER load different checkpoint without re-validation

2. **Configuration Freeze**:
   - BASELINE: EI=OFF, A.3=OFF, A.4=OFF
   - Top-K: 35 (NEVER change without full backtest)
   - Rebalance: Monthly, 1st business day only
   - Horizon: 20 days

3. **Cache Management**:
   - Panel cache stored at `/var/lib/apex-ranker/panel_cache`
   - Max size: 10 GB (prune if exceeded)
   - Cache rebuilds automatically on dataset changes

4. **Code Quality**:
   - NEVER commit with `__pycache__` directories
   - Run `find . -type d -name __pycache__ -exec rm -rf {} +` before commits
   - Pre-commit hooks must pass

5. **Linter Exceptions**:
   - Maintain `.ruff.toml` settings (no changes without review)
   - Accept existing warning suppression patterns
   - Document any new exceptions in commit messages

---

## Rollback Procedure

If production deployment encounters issues:

```bash
# Stop API server
sudo systemctl stop apex-ranker.service

# Or if running manually:
kill <API_PID>

# Restore previous bundle (if available)
python scripts/validate_bundle.py \
  --bundle bundles/apex_ranker_v0.0.1_prod \
  --dataset output/ml_dataset_latest_clean_with_adv.parquet

# Restart with previous version
sudo systemctl start apex-ranker.service
```

---

## Daily Operations Checklist

**Every Trading Day @ 15:30 JST** (after market close):

1. ✅ Update dataset with latest market data
   ```bash
   make dataset-bg START=<recent-date> END=<today>
   ```

2. ✅ Verify dataset integrity
   ```bash
   python scripts/validate_dataset.py output/ml_dataset_latest_clean_with_adv.parquet
   ```

3. ✅ Check API logs for errors
   ```bash
   tail -n 50 /var/log/apex-ranker/server.log | grep -E "ERROR|WARNING"
   ```

**1st Business Day of Month @ 09:00 JST**:

1. ✅ Generate monthly Top-35 predictions
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"date": "<YYYY-MM-DD>", "top_k": 35, "horizon": 20}' \
     > predictions_$(date +%Y%m%d).json
   ```

2. ✅ Log predictions for audit trail
   ```bash
   python apex-ranker/scripts/monitor_predictions.py log \
     --predictions predictions_$(date +%Y%m%d).json \
     --log-dir /var/log/apex-ranker/predictions \
     --model-version v0.1.0
   ```

3. ✅ Generate monthly summary report
   ```bash
   python apex-ranker/scripts/monitor_predictions.py summary \
     --log-dir /var/log/apex-ranker/predictions
   ```

---

## FAQ

**Q: What if smoke test shows different results between runs?**
A: This is a CRITICAL issue. DO NOT deploy. Investigate:
- Check dataset for corruption
- Verify model file integrity (checksum)
- Review recent code changes

**Q: Can I change Top-K from 35 to 50?**
A: Only after running full 22-month backtest and passing GO/NO-GO criteria (Sharpe +3% OR Return +3%, AND MaxDD <+2pp). Our Top-25 experiment showed -10.8% Sharpe degradation, so Top-35 is optimal.

**Q: How do I update the dataset?**
A: Run `make dataset-bg` daily after market close (15:30 JST). Incremental updates are automatic.

**Q: What if panel cache grows too large?**
A: Prune cache older than 30 days:
```bash
find /var/lib/apex-ranker/panel_cache -type f -mtime +30 -delete
```

**Q: Can I enable Enhanced Inference (EI) features?**
A: NO. All EI enhancements (A.3 Hysteresis, A.4 Risk Neutralization) degraded performance in backtests. BASELINE configuration is optimal.

---

## Support & Escalation

**Deployment Issues**:
1. Check logs: `/var/log/apex-ranker/server.log`
2. Review Day-0 validation outputs
3. Consult MANIFEST.lock for compatibility rules

**Performance Degradation**:
1. Compare current predictions vs historical averages
2. Run monthly rehearsal to verify system health
3. Check for dataset staleness (last update >2 days)

**Critical Alerts**:
- API downtime >5 minutes → Restart service
- Prediction consistency >5% change → Manual review
- Portfolio size ≠35 → STOP operations immediately

---

## Appendix: Bundle Contents

```
bundles/apex_ranker_v0.1.0_prod/
├── MANIFEST.lock                          # 4-way verification metadata
├── models/
│   └── apex_ranker_v0_enhanced.pt         # 178ch CS-Z trained model
├── configs/
│   └── v0_base_89_cleanADV.yaml           # BASELINE config (patch_multiplier=auto)
└── docs/
    ├── BASELINE_DECISION.md               # Rationale for BASELINE choice
    └── BACKTEST_RESULTS_2024_2025.md      # Full performance analysis
```

**MANIFEST.lock Critical Fields**:

```json
{
  "checkpoint": {
    "base_features": 89,
    "effective_dim": 178,
    "csz_mode": "trained_with_csz",
    "patch_multiplier": "auto"
  },
  "operational_rules": {
    "patch_multiplier_policy": "NEVER set explicit value",
    "config_freeze": "EI=OFF, A.3=OFF, A.4=OFF",
    "rebalance_schedule": "1st business day of month, 09:00 JST"
  }
}
```

---

**Document Version**: 1.0
**Last Review**: 2025-11-04
**Next Review**: 2025-12-01 (after first month of production)
