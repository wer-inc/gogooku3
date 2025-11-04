#!/bin/bash
# APEX-Ranker AB Experiments - 6 configurations
# Testing A.3 (Hysteresis) and A.4 (Risk Neutralization) combinations

set -e

MODEL="models/apex_ranker_v0_enhanced.pt"
CONFIG="apex-ranker/configs/v0_base_89_cleanADV.yaml"
DATA="output/ml_dataset_latest_clean_with_adv.parquet"
START="2024-01-01"
END="2025-10-31"
REBAL="monthly"
HORIZON=20
TOPK=35

OUTDIR="results/p0_ab_final"
mkdir -p "$OUTDIR"

echo "========================================="
echo "APEX-Ranker AB Experiments"
echo "Model: $MODEL"
echo "Data: $DATA"
echo "Period: $START to $END"
echo "========================================="
echo ""

# 1. BASE (no A.3, no A.4)
echo "[1/6] Running BASE (no enhancements)..."
python apex-ranker/scripts/backtest_smoke_test.py \
  --model "$MODEL" \
  --config "$CONFIG" \
  --data "$DATA" \
  --start-date "$START" \
  --end-date "$END" \
  --rebalance-freq "$REBAL" \
  --horizon "$HORIZON" \
  --top-k "$TOPK" \
  --output "$OUTDIR/BASE.json" \
  > "$OUTDIR/BASE.log" 2>&1
echo "✅ BASE complete"
echo ""

# 2. A.3 only (Hysteresis)
echo "[2/6] Running A.3 only (Hysteresis)..."
python apex-ranker/scripts/backtest_smoke_test.py \
  --model "$MODEL" \
  --config "$CONFIG" \
  --data "$DATA" \
  --start-date "$START" \
  --end-date "$END" \
  --rebalance-freq "$REBAL" \
  --horizon "$HORIZON" \
  --top-k "$TOPK" \
  --use-enhanced-inference \
  --ei-hysteresis-entry-k 35 \
  --ei-hysteresis-exit-k 60 \
  --output "$OUTDIR/A3_only.json" \
  > "$OUTDIR/A3_only.log" 2>&1
echo "✅ A.3 only complete"
echo ""

# 3. A.4 only (gamma=0.3)
echo "[3/6] Running A.4 only (gamma=0.3)..."
python apex-ranker/scripts/backtest_smoke_test.py \
  --model "$MODEL" \
  --config "$CONFIG" \
  --data "$DATA" \
  --start-date "$START" \
  --end-date "$END" \
  --rebalance-freq "$REBAL" \
  --horizon "$HORIZON" \
  --top-k "$TOPK" \
  --use-enhanced-inference \
  --ei-neutralize-risk \
  --ei-risk-factors "Sector33Code,volatility_60d" \
  --ei-neutralize-gamma 0.3 \
  --ei-ridge-alpha 10.0 \
  --output "$OUTDIR/A4_g030.json" \
  > "$OUTDIR/A4_g030.log" 2>&1
echo "✅ A.4 (gamma=0.3) complete"
echo ""

# 4. A.3 + A.4 (gamma=0.2, conservative)
echo "[4/6] Running A.3+A.4 (gamma=0.2, conservative)..."
python apex-ranker/scripts/backtest_smoke_test.py \
  --model "$MODEL" \
  --config "$CONFIG" \
  --data "$DATA" \
  --start-date "$START" \
  --end-date "$END" \
  --rebalance-freq "$REBAL" \
  --horizon "$HORIZON" \
  --top-k "$TOPK" \
  --use-enhanced-inference \
  --ei-hysteresis-entry-k 35 \
  --ei-hysteresis-exit-k 60 \
  --ei-neutralize-risk \
  --ei-risk-factors "Sector33Code,volatility_60d" \
  --ei-neutralize-gamma 0.2 \
  --ei-ridge-alpha 10.0 \
  --output "$OUTDIR/A3A4_g020.json" \
  > "$OUTDIR/A3A4_g020.log" 2>&1
echo "✅ A.3+A.4 (gamma=0.2) complete"
echo ""

# 5. A.3 + A.4 (gamma=0.3, balanced)
echo "[5/6] Running A.3+A.4 (gamma=0.3, balanced)..."
python apex-ranker/scripts/backtest_smoke_test.py \
  --model "$MODEL" \
  --config "$CONFIG" \
  --data "$DATA" \
  --start-date "$START" \
  --end-date "$END" \
  --rebalance-freq "$REBAL" \
  --horizon "$HORIZON" \
  --top-k "$TOPK" \
  --use-enhanced-inference \
  --ei-hysteresis-entry-k 35 \
  --ei-hysteresis-exit-k 60 \
  --ei-neutralize-risk \
  --ei-risk-factors "Sector33Code,volatility_60d" \
  --ei-neutralize-gamma 0.3 \
  --ei-ridge-alpha 10.0 \
  --output "$OUTDIR/A3A4_g030.json" \
  > "$OUTDIR/A3A4_g030.log" 2>&1
echo "✅ A.3+A.4 (gamma=0.3) complete"
echo ""

# 6. A.3 + A.4 (gamma=0.5, aggressive)
echo "[6/6] Running A.3+A.4 (gamma=0.5, aggressive)..."
python apex-ranker/scripts/backtest_smoke_test.py \
  --model "$MODEL" \
  --config "$CONFIG" \
  --data "$DATA" \
  --start-date "$START" \
  --end-date "$END" \
  --rebalance-freq "$REBAL" \
  --horizon "$HORIZON" \
  --top-k "$TOPK" \
  --use-enhanced-inference \
  --ei-hysteresis-entry-k 35 \
  --ei-hysteresis-exit-k 60 \
  --ei-neutralize-risk \
  --ei-risk-factors "Sector33Code,volatility_60d" \
  --ei-neutralize-gamma 0.5 \
  --ei-ridge-alpha 10.0 \
  --output "$OUTDIR/A3A4_g050.json" \
  > "$OUTDIR/A3A4_g050.log" 2>&1
echo "✅ A.3+A.4 (gamma=0.5) complete"
echo ""

echo "========================================="
echo "✅ ALL 6 EXPERIMENTS COMPLETE"
echo "Results saved to: $OUTDIR/"
echo "========================================="
echo ""
echo "Next: Run comparison analysis:"
echo "  python scripts/analyze_apex_ab_results.py"
