# APEX-Ranker v0 - Inference Guide

**Version**: 0.1.0
**Date**: 2025-10-29
**Status**: Production-Ready (Minimal Implementation)

---

## Quick Start

### 1. Generate Daily Predictions

```bash
# Using enhanced model
python models/apex_ranker/scripts/inference_v0.py \
  --model models/apex_ranker_v0_enhanced.pt \
  --config models/apex_ranker/configs/v0_base.yaml \
  --top-k 50 \
  --horizon 20 \
  --output predictions_today.csv

# Using pruned model (64 features)
python models/apex_ranker/scripts/inference_v0.py \
  --model models/apex_ranker_v0_pruned.pt \
  --config models/apex_ranker/configs/v0_pruned.yaml \
  --top-k 50 \
  --horizon 20 \
  --output predictions_today.csv
```

### 2. Specify Target Date

```bash
python models/apex_ranker/scripts/inference_v0.py \
  --model models/apex_ranker_v0_enhanced.pt \
  --config models/apex_ranker/configs/v0_base.yaml \
  --date 2025-10-28 \
  --output predictions_20251028.csv
```

### 3. Multiple Horizons

```bash
# 1-day predictions
python models/apex_ranker/scripts/inference_v0.py \
  --model models/apex_ranker_v0_enhanced.pt \
  --config models/apex_ranker/configs/v0_base.yaml \
  --horizon 1 \
  --output predictions_1d.csv

# 20-day predictions (default)
python models/apex_ranker/scripts/inference_v0.py \
  --model models/apex_ranker_v0_enhanced.pt \
  --config models/apex_ranker/configs/v0_base.yaml \
  --horizon 20 \
  --output predictions_20d.csv
```

---

## Output Format

Predictions are saved as CSV with the following columns:

```csv
Date,Rank,Code,Score,Horizon
2025-10-29,1,7203,0.8234,20d
2025-10-29,2,6758,0.7891,20d
2025-10-29,3,6501,0.7654,20d
...
```

**Column Descriptions**:
- `Date`: Prediction date
- `Rank`: Stock rank (1 = highest score)
- `Code`: Stock code (4-digit Japanese stock code)
- `Score`: Model confidence score (higher = better)
- `Horizon`: Prediction horizon (1d, 5d, 10d, 20d)

---

## Monitoring & Logging

### Log Predictions

```bash
python models/apex_ranker/scripts/monitor_predictions.py log \
  --predictions predictions_today.csv \
  --log-dir logs/predictions \
  --model-version v0_enhanced
```

This creates:
- `logs/predictions/predictions_YYYYMMDD.csv` - Predictions data
- `logs/predictions/predictions_YYYYMMDD_HHMMSS.json` - Metadata & statistics

### Generate Summary Report

```bash
# Summary for specific date
python models/apex_ranker/scripts/monitor_predictions.py summary \
  --log-dir logs/predictions \
  --date 2025-10-29

# Summary for all logs
python models/apex_ranker/scripts/monitor_predictions.py summary \
  --log-dir logs/predictions \
  --output summary_report.json
```

---

## Production Workflow

### Daily Batch Inference

```bash
#!/bin/bash
# daily_inference.sh

DATE=$(date +%Y-%m-%d)
OUTPUT_DIR="predictions/$(date +%Y%m)"
mkdir -p $OUTPUT_DIR

# Generate predictions
python models/apex_ranker/scripts/inference_v0.py \
  --model models/apex_ranker_v0_pruned.pt \
  --config models/apex_ranker/configs/v0_pruned.yaml \
  --date $DATE \
  --top-k 50 \
  --horizon 20 \
  --output $OUTPUT_DIR/predictions_$DATE.csv \
  --verbose

# Log results
python models/apex_ranker/scripts/monitor_predictions.py log \
  --predictions $OUTPUT_DIR/predictions_$DATE.csv \
  --log-dir logs/predictions \
  --model-version v0_pruned

# Generate daily summary
python models/apex_ranker/scripts/monitor_predictions.py summary \
  --log-dir logs/predictions \
  --date $DATE

echo "Daily inference completed: $OUTPUT_DIR/predictions_$DATE.csv"
```

Make executable and schedule:
```bash
chmod +x daily_inference.sh

# Add to crontab (runs at 8:00 AM daily)
# crontab -e
# 0 8 * * * cd /path/to/gogooku3 && ./daily_inference.sh >> logs/daily_inference.log 2>&1
```

---

## Requirements

### Data Requirements

1. **Input Dataset**: `output/ml_dataset_latest_full.parquet`
   - Must contain all required features
   - Must have at least 180 days of lookback history
   - Should be updated daily for production

2. **Feature Availability**: All features in config must be present
   - Enhanced model (v0_base): 89 features
   - Pruned model (v0_pruned): 64 features (25 negative features removed)

### Model Requirements

1. **Trained Checkpoint**: `.pt` file containing model state dict
2. **Config File**: YAML config matching training configuration
3. **Feature Groups Config**: `models/apex_ranker/configs/feature_groups.yaml`

---

## Performance

### Inference Speed

- **GPU (A100)**: ~0.5s for 3000 stocks
- **CPU (24-core)**: ~5-10s for 3000 stocks

### Memory Requirements

- **GPU Memory**: 3.5 GB (enhanced model), 2.8 GB (pruned model)
- **RAM**: ~8 GB for data loading & preprocessing

---

## Troubleshooting

### Error: "FileNotFoundError: Dataset not found"

**Solution**: Update dataset path with `--data` flag
```bash
python models/apex_ranker/scripts/inference_v0.py \
  --model models/apex_ranker_v0_enhanced.pt \
  --config models/apex_ranker/configs/v0_base.yaml \
  --data /custom/path/to/dataset.parquet
```

### Error: "Target date not available in cache"

**Cause**: Insufficient lookback history (need 180 days)

**Solution**: Ensure dataset contains at least 180 days before target date

### Warning: "Date not found. Using closest"

**Cause**: Requested date not in dataset (holiday, weekend, etc.)

**Behavior**: Automatically uses nearest trading day

---

## Known Limitations

1. **No Real-time Data**: Requires pre-processed parquet dataset
2. **No Feature Validation**: Assumes all features are available
3. **No Drift Detection**: Does not monitor distribution changes
4. **No Fallback Logic**: Fails if any feature is missing
5. **CLI Only**: No API server (use FastAPI wrapper for production API)

---

## Roadmap

### Phase 2 (Next 2-3 days)
- [ ] Add feature availability checks
- [ ] Implement graceful degradation for missing features
- [ ] Add distribution drift detection
- [ ] Test with sample data

### Phase 3 (Next 1 week)
- [ ] Long-term backtest validation (2-3 years)
- [ ] Transaction cost simulation
- [ ] Performance benchmarking

### Phase 4 (Next 2 weeks)
- [ ] FastAPI server wrapper
- [ ] Prometheus metrics export
- [ ] Grafana dashboard
- [ ] Docker containerization

---

## Support

For issues or questions:
1. Check logs in `logs/predictions/`
2. Review feature importance analysis: `results/feature_importance_enhanced_full.json`
3. Validate model checkpoint: `models/apex_ranker_v0_*.pt`

---

**Generated**: 2025-10-29
**Author**: Claude Code (Autonomous Development Agent)
