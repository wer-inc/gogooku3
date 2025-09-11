#!/bin/bash

# Export environment variables for Student-t distribution
export ENABLE_STUDENT_T=1
export USE_T_NLL=1
export NLL_WEIGHT=0.02

# Run training directly with minimal config
python scripts/train_atft.py \
    data.source.data_dir=output/atft_data/train \
    train.batch.train_batch_size=512 \
    train.trainer.max_epochs=1 \
    train.trainer.precision=bf16-mixed \
    train.trainer.check_val_every_n_epoch=1 \
    train.trainer.enable_progress_bar=true