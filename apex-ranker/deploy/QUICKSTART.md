# APEX Ranker Production Deployment - Quick Start

**Time to production: 15 minutes**

---

## Prerequisites (Already Validated ✅)

- Day-0 validation checks passed
- Bundle: `bundles/apex_ranker_v0.1.0_prod`
- Dataset: `output/ml_dataset_latest_clean_with_adv.parquet`
- Configuration: BASELINE (Top-35/Monthly/H20/EI OFF)

---

## Installation (5 minutes)

### Step 1: Create directories

```bash
sudo mkdir -p /var/lib/apex-ranker/panel_cache
sudo mkdir -p /var/log/apex-ranker
sudo mkdir -p /var/log/apex-ranker/predictions
sudo chown -R $(whoami):$(whoami) /var/lib/apex-ranker /var/log/apex-ranker
```

### Step 2: Install systemd service

```bash
# Copy service file
sudo cp apex-ranker/deploy/apex-ranker.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable service (auto-start on boot)
sudo systemctl enable apex-ranker.service

# Start service
sudo systemctl start apex-ranker.service

# Check status
sudo systemctl status apex-ranker.service
```

**Expected output**:
```
● apex-ranker.service - APEX Ranker Production API v0.1.0
   Loaded: loaded (/etc/systemd/system/apex-ranker.service; enabled)
   Active: active (running) since ...
```

### Step 3: Verify API health

```bash
# Wait 30 seconds for startup
sleep 30

# Test health endpoint
curl http://localhost:8000/health

# Expected:
# {"status":"ok","model":"loaded","timestamp":"2025-11-04T..."}
```

If health check fails, review logs:
```bash
tail -f /var/log/apex-ranker/server.log
```

---

## Monitoring Setup (5 minutes)

### Option A: Prometheus (Recommended)

```bash
# 1. Copy alerting rules
sudo cp apex-ranker/deploy/prometheus_rules.yml /etc/prometheus/rules/

# 2. Edit /etc/prometheus/prometheus.yml
# Add under rule_files:
#   - /etc/prometheus/rules/prometheus_rules.yml

# 3. Reload Prometheus
sudo systemctl reload prometheus

# 4. Verify rules loaded
curl http://localhost:9090/api/v1/rules | jq '.data.groups[].name'
# Expected: apex_ranker_critical, apex_ranker_warnings, apex_ranker_info
```

### Option B: Cron-based monitoring (Simple)

```bash
# Add monitoring script to crontab
crontab -e

# Add these lines:
# Health checks every hour
0 * * * * curl -s http://localhost:8000/health >> /var/log/apex-ranker/health_checks.log

# Premarket checks (8:30 AM JST)
30 8 * * * /workspace/gogooku3/apex-ranker/scripts/day1_operations.sh premarket >> /var/log/apex-ranker/operations.log 2>&1

# Order generation (9:00 AM JST, 1st business day only)
0 9 1 * * /workspace/gogooku3/apex-ranker/scripts/day1_operations.sh order >> /var/log/apex-ranker/operations.log 2>&1

# EOD monitoring (4:00 PM JST)
0 16 * * * /workspace/gogooku3/apex-ranker/scripts/day1_operations.sh eod >> /var/log/apex-ranker/operations.log 2>&1
```

---

## Day-1 Operations (5 minutes)

### First business day of month @ 8:30 AM JST

Run premarket checks:

```bash
apex-ranker/scripts/day1_operations.sh premarket
```

**Expected output**:
```
[2025-11-04 08:30:00] Checking API health...
[2025-11-04 08:30:01] ✅ API health: OK
[2025-11-04 08:30:01] Requesting predictions for 2025-12-01...
[2025-11-04 08:30:05] ✅ Predictions received: 70 stocks
[2025-11-04 08:30:05] Validating supply constraints...
[2025-11-04 08:30:05] ✅ Supply validation passed: selected_count=53
[2025-11-04 08:30:05] Checking sector concentration...
[2025-11-04 08:30:05] Estimating transaction costs...
[2025-11-04 08:30:05] ✅ Premarket checks complete
```

### First business day of month @ 9:00 AM JST

Generate rebalance orders:

```bash
apex-ranker/scripts/day1_operations.sh order
```

**Expected output**:
```
[2025-11-04 09:00:00] Generating rebalance orders...
[2025-11-04 09:00:02] ✅ Orders generated: /var/log/apex-ranker/predictions/orders_2025-12-01.csv
[2025-11-04 09:00:02] Total orders: 35
```

Review orders file:
```bash
cat /var/log/apex-ranker/predictions/orders_$(date +%Y-%m-%d).csv
```

### Every day @ 4:00 PM JST

Run EOD monitoring:

```bash
apex-ranker/scripts/day1_operations.sh eod
```

---

## Quick Tests

### Test 1: Manual prediction request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2025-12-01",
    "top_k": 35,
    "horizon": 20
  }' | jq '.'
```

### Test 2: Verify service restart

```bash
# Stop service
sudo systemctl stop apex-ranker.service

# Wait 5 seconds
sleep 5

# Verify service stopped
curl http://localhost:8000/health
# Expected: Connection refused

# Restart service
sudo systemctl start apex-ranker.service

# Wait 30 seconds
sleep 30

# Verify service running
curl http://localhost:8000/health
# Expected: {"status":"ok",...}
```

---

## Critical Alerts (Monitor these!)

| Alert | Severity | Threshold | Action |
|-------|----------|-----------|--------|
| `ApexRankerDimensionMismatch` | CRITICAL | effective_dim ≠ 178 | STOP production, validate bundle |
| `ApexRankerSupplyShortage` | CRITICAL | selected_count < 53 | Check data quality |
| `ApexRankerAPIDown` | CRITICAL | Health check fails >2min | Review logs, restart service |
| `ApexRankerPortfolioSizeMismatch` | CRITICAL | portfolio_size ≠ 35 | STOP trading |
| `ApexRankerHighTransactionCosts` | WARNING | costs > 20 bps | Review turnover |
| `ApexRankerSharpeDegradation` | WARNING | 30d Sharpe < 1.0 | Run monthly rehearsal |

---

## Rollback Procedure

If production encounters issues:

```bash
# 1. Stop service
sudo systemctl stop apex-ranker.service

# 2. Review recent logs
tail -n 100 /var/log/apex-ranker/server.log

# 3. If bundle corruption suspected, re-validate
python scripts/validate_bundle.py \
  --bundle bundles/apex_ranker_v0.1.0_prod \
  --dataset output/ml_dataset_latest_clean_with_adv.parquet

# 4. If validation fails, restore from backup
# (ensure you have bundle backups!)
cp -r bundles/apex_ranker_v0.1.0_prod.backup bundles/apex_ranker_v0.1.0_prod

# 5. Restart service
sudo systemctl start apex-ranker.service
```

---

## Maintenance

### Weekly (every Sunday @ 3:00 AM JST)

Clean old cache files:

```bash
# Add to crontab
0 3 * * 0 find /var/lib/apex-ranker/panel_cache -type f -mtime +30 -delete
```

### Monthly (1st Sunday)

Rotate logs:

```bash
# Add to /etc/logrotate.d/apex-ranker
/var/log/apex-ranker/*.log {
    monthly
    rotate 12
    compress
    delaycompress
    notifempty
    create 0644 root root
}
```

### Before each rebalance

Update dataset with latest data:

```bash
# Run dataset update (takes 30-60 minutes)
cd /workspace/gogooku3
make dataset-bg START=2025-11-01 END=2025-11-30

# Verify dataset updated
ls -lh output/ml_dataset_latest_clean_with_adv.parquet
```

---

## Troubleshooting

### API won't start

```bash
# Check logs
sudo journalctl -u apex-ranker.service -n 50

# Common issues:
# 1. Missing dataset file
ls output/ml_dataset_latest_clean_with_adv.parquet

# 2. Missing bundle
ls bundles/apex_ranker_v0.1.0_prod/models/apex_ranker_v0_enhanced.pt

# 3. Port 8000 already in use
sudo lsof -i :8000
```

### Predictions seem inconsistent

```bash
# Run monthly rehearsal to verify system health
python apex-ranker/scripts/backtest_smoke_test.py \
  --model bundles/apex_ranker_v0.1.0_prod/models/apex_ranker_v0_enhanced.pt \
  --config bundles/apex_ranker_v0.1.0_prod/configs/v0_base_89_cleanADV.yaml \
  --data output/ml_dataset_latest_clean_with_adv.parquet \
  --start-date 2025-10-01 --end-date 2025-10-31 \
  --rebalance-freq monthly --horizon 20 --top-k 35 \
  --features-mode fill-zero \
  --output /tmp/monthly_rehearsal.json
```

### High memory usage

```bash
# Check panel cache size
du -sh /var/lib/apex-ranker/panel_cache

# If >10GB, prune old files
find /var/lib/apex-ranker/panel_cache -type f -mtime +7 -delete
```

---

## Support

- **Deployment Guide**: `apex-ranker/PRODUCTION_DEPLOYMENT_GUIDE.md`
- **Configuration**: `bundles/apex_ranker_v0.1.0_prod/MANIFEST.lock`
- **Logs**: `/var/log/apex-ranker/server.log`
- **Operations**: `/var/log/apex-ranker/operations.log`

---

**Production checklist complete** ✅

Your APEX Ranker v0.1.0 BASELINE is now running in production!

**Monthly rebalancing**: 1st business day @ 09:00 JST
**Target portfolio**: 35 stocks
**Expected Sharpe**: 1.439 (based on 2024-2025 backtest)
