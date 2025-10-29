# APEX-Ranker v0 (minimum viable implementation)

This directory hosts the standalone implementation of the APEX-Ranker
training stack described in `docs/VV/README.md`.  The current focus is
the **v0 baseline**:

- PatchTST encoder (time-patch transformer)
- Multi-horizon ranking head (1d / 5d / 10d / 20d)
- Composite ranking loss (ListNet + RankNet + optional MSE)
- Day-wise batching with cross-sectional Z-score normalisation
- GPU-first training loop (AMP if available)

The implementation is intentionally modular so additional phases
(quantile head, adaptive graph attention, macro slots, …) can be added
without refactoring the v0 code paths.

## Layout

```
apex-ranker/
├── apex_ranker/
│   ├── data/            # Feature selection, normalisation, panel dataset
│   ├── losses/          # ListNet / RankNet / composite losses
│   ├── models/          # PatchTST encoder + APEXRankerV0 model wrapper
│   └── utils/           # Config helpers, evaluation metrics
├── configs/
│   ├── feature_groups.yaml  # Core50 / Plus30 feature bundles
│   └── v0_base.yaml         # Training hyper-parameters and data paths
└── scripts/
    └── train_v0.py      # CLI for end-to-end training
```

## Quick start

1. Ensure the latest dataset artifacts exist:

   - `output/ml_dataset_latest_full.parquet`
   - `dataset_features_detail.json`

2. Review `configs/v0_base.yaml` and adjust:

   - `data.parquet_path` if the dataset is stored elsewhere
   - `train.val_days` / `train.epochs` for the desired splits
   - `feature_groups` to enable `core50` or extend with `plus30` (default) for added momentum factors

3. (Optional) re-create the python environment and install deps.
4. (Optional) run a readiness check:

   ```bash
   python apex-ranker/scripts/check_training_ready.py --config apex-ranker/configs/v0_base.yaml
   ```

   The script validates column coverage, targets, and lookback depth before running a full training job.  
   *Note:* As of the current dataset only `is_flow_valid` / `is_valid_ma` carry signal; `is_stmt_valid` / `is_sec_cs_valid` were removed from the active mask list and will be logged as zero-coverage if present.

5. (Optional) quick sanity run (CPU/GPU agnostic):

   ```bash
   python apex-ranker/scripts/train_v0.py \
     --config apex-ranker/configs/v0_base.yaml \
     --max-train-days 3 \
     --max-epochs 1 \
     --max-train-steps 2
   ```

   These flags throttle training for smoke-tests. Omit them (or increase) for real experiments.

6. Launch full training:

   ```bash
   python apex-ranker/scripts/train_v0.py --config apex-ranker/configs/v0_base.yaml
   ```

   The script performs:

   - feature selection + cross-sectional Z-score augmentation
   - day-wise panel construction (lookback = 180 by default)
   - GPU-enabled training with composite ranking loss
   - simple validation metrics (RankIC / Precision@K per horizon)

### Mask handling

Some mask columns (e.g. `is_stmt_valid`) may have zero positive coverage in certain datasets.  
During dataset loading they are automatically dropped with a warning so that eligible days are kept for training.
## Extending the feature set

`configs/feature_groups.yaml` defines groups of column names that can be
referenced from any training config.  The supplied groups follow the
“core50 → plus30 → optional” recommendation from the design document.
Add or override groups as required for ablation experiments.

## Known limitations (to be addressed in later phases)

- No distributed (DDP) support yet – single-GPU training only.
- Quantile / risk heads and the adaptive graph modules are placeholders.
- Hyper-parameter search (Optuna) is not wired in this skeleton.

Despite these gaps, the current code path is sufficient to train and
evaluate the v0 baseline end-to-end on the latest dataset.
