#!/usr/bin/env python3
"""
Quick Run Patch for train_atft.py
Adds MAX_STEPS_PER_EPOCH and VAL_INTERVAL_STEPS support via environment variables.

Usage:
  1. Import this at the top of train_atft.py: from quick_run_patch import check_early_stop, should_validate
  2. In train_epoch(), add early stop check in batch loop
  3. In main training loop, add periodic validation
"""
import logging
import os

logger = logging.getLogger(__name__)

# Global step counter for cross-epoch tracking
_global_step = 0


def get_global_step() -> int:
    """Get current global step count."""
    return _global_step


def increment_global_step() -> int:
    """Increment and return global step count."""
    global _global_step
    _global_step += 1
    return _global_step


def check_early_stop(batch_idx: int, epoch: int) -> bool:
    """
    Check if training should stop early based on environment variables.

    Args:
        batch_idx: Current batch index within epoch
        epoch: Current epoch number

    Returns:
        True if training should stop, False otherwise
    """
    # Per-epoch step limit
    max_steps_per_epoch = os.getenv("MAX_STEPS_PER_EPOCH")
    if max_steps_per_epoch:
        try:
            max_val = int(max_steps_per_epoch)
            if batch_idx >= max_val:
                logger.info(
                    f"[QuickRun] Reached MAX_STEPS_PER_EPOCH={max_val} at batch {batch_idx}, epoch {epoch}"
                )
                return True
        except ValueError:
            pass

    # Global step limit
    max_total_steps = os.getenv("MAX_TOTAL_STEPS")
    if max_total_steps:
        try:
            max_val = int(max_total_steps)
            if _global_step >= max_val:
                logger.info(
                    f"[QuickRun] Reached MAX_TOTAL_STEPS={max_val} at global_step {_global_step}"
                )
                return True
        except ValueError:
            pass

    return False


def should_validate(batch_idx: int, epoch: int) -> bool:
    """
    Check if validation should run at this step.

    Args:
        batch_idx: Current batch index within epoch
        epoch: Current epoch number

    Returns:
        True if validation should run, False otherwise
    """
    val_interval = os.getenv("VAL_INTERVAL_STEPS")
    if not val_interval:
        return False

    try:
        interval = int(val_interval)
        # Validate at the specified interval
        if _global_step > 0 and (_global_step % interval) == 0:
            logger.info(
                f"[QuickRun] Triggering validation at global_step {_global_step} (interval={interval})"
            )
            return True
    except ValueError:
        pass

    return False
