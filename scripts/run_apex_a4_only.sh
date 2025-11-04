#!/bin/bash
# A.4-only experiments: Risk Neutralization with gamma=0.2/0.3/0.5
# A.3 (hysteresis) is DISABLED by setting entry_k = exit_k = 35
# This isolates A.4 effects for clean measurement

set -e

MODEL="models/apex_ranker_v0_enhanced.pt"
CONFIG="apex-ranker/configs/v0_base_89_cleanADV.yaml"
DATA="output/ml_dataset_latest_clean_with_adv.parquet"
START="2024-01-01"
END="2025-10-31"
TOP_K=35
HORIZON=20
REBAL="monthly"

# Risk neutralization factors (comma-separated)
# NOTE: sector33_code is lowercase in dataset (not Sector33Code)
FACTORS="sector33_code,volatility_60d"

echo "========================================"
echo "APEX-Ranker A.4-only Experiments"
echo "Model: $MODEL"
echo "Data: $DATA"
echo "Period: $START to $END"
echo "Factors: $FACTORS"
echo "========================================"
echo ""

# [1/3] A.4 only (gamma=0.2) - Conservative
echo "[1/3] Running A.4 (gamma=0.2) - Conservative..."
python apex-ranker/scripts/backtest_smoke_test.py \
  --model "$MODEL" \
  --config "$CONFIG" \
  --data "$DATA" \
  --start-date "$START" \
  --end-date "$END" \
  --top-k "$TOP_K" \
  --horizon "$HORIZON" \
  --rebalance-freq "$REBAL" \
  --use-enhanced-inference \
  --ei-neutralize-risk \
  --ei-risk-factors $FACTORS \
  --ei-neutralize-gamma 0.2 \
  --ei-ridge-alpha 10.0 \
  --ei-hysteresis-entry-k 35 \
  --ei-hysteresis-exit-k 35 \
  --output "results/p0_ab_final/A4only_g020.json" \
  > "results/p0_ab_final/A4only_g020.log" 2>&1
echo "✅ A.4 (gamma=0.2) complete"
echo ""

# [2/3] A.4 only (gamma=0.3) - Balanced
echo "[2/3] Running A.4 (gamma=0.3) - Balanced..."
python apex-ranker/scripts/backtest_smoke_test.py \
  --model "$MODEL" \
  --config "$CONFIG" \
  --data "$DATA" \
  --start-date "$START" \
  --end-date "$END" \
  --top-k "$TOP_K" \
  --horizon "$HORIZON" \
  --rebalance-freq "$REBAL" \
  --use-enhanced-inference \
  --ei-neutralize-risk \
  --ei-risk-factors $FACTORS \
  --ei-neutralize-gamma 0.3 \
  --ei-ridge-alpha 10.0 \
  --ei-hysteresis-entry-k 35 \
  --ei-hysteresis-exit-k 35 \
  --output "results/p0_ab_final/A4only_g030.json" \
  > "results/p0_ab_final/A4only_g030.log" 2>&1
echo "✅ A.4 (gamma=0.3) complete"
echo ""

# [3/3] A.4 only (gamma=0.5) - Aggressive
echo "[3/3] Running A.4 (gamma=0.5) - Aggressive..."
python apex-ranker/scripts/backtest_smoke_test.py \
  --model "$MODEL" \
  --config "$CONFIG" \
  --data "$DATA" \
  --start-date "$START" \
  --end-date "$END" \
  --top-k "$TOP_K" \
  --horizon "$HORIZON" \
  --rebalance-freq "$REBAL" \
  --use-enhanced-inference \
  --ei-neutralize-risk \
  --ei-risk-factors $FACTORS \
  --ei-neutralize-gamma 0.5 \
  --ei-ridge-alpha 10.0 \
  --ei-hysteresis-entry-k 35 \
  --ei-hysteresis-exit-k 35 \
  --output "results/p0_ab_final/A4only_g050.json" \
  > "results/p0_ab_final/A4only_g050.log" 2>&1
echo "✅ A.4 (gamma=0.5) complete"
echo ""

echo "========================================"
echo "All A.4 experiments completed!"
echo "Results saved to results/p0_ab_final/"
echo "========================================"
