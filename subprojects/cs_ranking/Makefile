PY := python

.PHONY: help run quick

help:
	@echo "Cross-sectional ranking helpers"
	@echo "  make run   DATA=output/ml_dataset_latest_full.parquet EPOCHS=120 BATCH=2048 LR=2e-4"
	@echo "  make quick DATA=output/ml_dataset_latest_full.parquet (3 epochs sanity)"

DATA ?= output/ml_dataset_latest_full.parquet
EPOCHS ?= 120
BATCH ?= 2048
LR ?= 2e-4

run:
	./run_ranking.sh --data $(DATA) --epochs $(EPOCHS) --batch-size $(BATCH) --lr $(LR)

quick:
	USE_RANKIC=1 RANKIC_WEIGHT=0.5 CS_IC_WEIGHT=0.3 SHARPE_WEIGHT=0.1 \
	USE_PAIRWISE_RANK=1 PAIRWISE_RANK_WEIGHT=0.5 PAIRWISE_SAMPLE_RATIO=0.25 \
	USE_DAY_BATCH=1 ALLOW_UNSAFE_DATALOADER=1 NUM_WORKERS=8 PERSISTENT_WORKERS=1 PREFETCH_FACTOR=4 PIN_MEMORY=1 \
	$(PY) ../../scripts/train.py --data-path $(DATA) --epochs 3 --batch-size $(BATCH) --lr $(LR) --mode optimized --no-background
