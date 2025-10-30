#!/usr/bin/env python3
"""
Production rollout helper
- Runs multi-seed training with recommended toggles
- Keeps checkpoints isolated per seed via CKPT_TAG and RUN_DIR
- Optionally runs Walk-Forward + Embargo evaluation per seed
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run(cmd: list[str], env: dict | None = None, cwd: str | None = None) -> int:
    print("$", " ".join(cmd))
    proc = subprocess.Popen(cmd, env=env, cwd=cwd)
    proc.wait()
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="ATFT-GAT-FAN production rollout")
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated seeds (e.g., 42,43,44)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=os.getenv("DATA_PATH", "data/raw/large_scale/ml_dataset_full.parquet"),
        help="Path to parquet dataset for training/eval",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="WF+Embargo evaluation splits",
    )
    parser.add_argument(
        "--embargo-days",
        type=int,
        default=20,
        help="Embargo days for WF evaluation",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run WF+Embargo evaluation after each training",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="runs/rollout",
        help="Output base directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.getenv("DEVICE", "cuda"),
        help="Device for evaluation (cuda/cpu)",
    )
    args = parser.parse_args()

    # Resolve rollout directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(args.out) / f"{ts}"
    base_dir.mkdir(parents=True, exist_ok=True)

    # Common toggles (recommended high-perf)
    common_env = {
        # Mixed precision
        "USE_AMP": os.getenv("USE_AMP", "1"),
        "AMP_DTYPE": os.getenv("AMP_DTYPE", "bf16"),
        # Heads and losses
        "ENABLE_QUANTILES": os.getenv("ENABLE_QUANTILES", "1"),
        "ENABLE_STUDENT_T": os.getenv("ENABLE_STUDENT_T", "1"),
        "USE_DIR_AUX": os.getenv("USE_DIR_AUX", "1"),
        "USE_CS_IC": os.getenv("USE_CS_IC", "1"),
        # SWA + snapshot
        "USE_SWA": os.getenv("USE_SWA", "1"),
        "SNAPSHOT_ENS": os.getenv("SNAPSHOT_ENS", "1"),
        "SNAPSHOT_NUM": os.getenv("SNAPSHOT_NUM", "4"),
        # W&B default on (already default in script, but keep explicit)
        "WANDB_ENABLED": os.getenv("WANDB_ENABLED", "1"),
        # Data path passthrough
        "DATA_PATH": args.data_path,
    }

    # If a single parquet file is provided, steer training to single-file aware path
    # by disabling the placeholder optimized loader and enabling CV split logic.
    if str(args.data_path).lower().endswith(".parquet"):
        # Force the non-optimized loader (optimized stub is a no-op in this repo)
        common_env.setdefault("USE_OPTIMIZED_LOADER", "0")
        # Ensure we go through the CV codepath in train_atft.py which honors DATA_PATH
        common_env.setdefault("CV_FOLDS", "2")

    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    summary = {
        "timestamp": ts,
        "data_path": args.data_path,
        "seeds": seeds,
        "results": [],
    }

    for s in seeds:
        seed = int(s)
        tag = f"s{seed}"
        run_dir = base_dir / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Environment for this run
        env = os.environ.copy()
        env.update(common_env)
        env.update(
            {
                "SEED": str(seed),
                "CKPT_TAG": tag,
                # Split outputs/logs
                "RUN_DIR": str(run_dir / "run"),
                # Name the W&B run clearly
                "WANDB_RUN_NAME": f"train_{tag}_{ts}",
            }
        )

        # Train
        ret = run([sys.executable, "scripts/train_atft.py"], env=env)
        if ret != 0:
            print(f"[seed={seed}] training failed with code {ret}")
            return ret

        # Locate best checkpoint
        ckpt = Path(f"models/checkpoints/atft_gat_fan_best_{tag}.pt")
        if ckpt.exists():
            # Copy into seed folder for isolation
            dst = run_dir / ckpt.name
            try:
                shutil.copy2(ckpt, dst)
            except Exception:
                pass
            # Try to read best val_loss from checkpoint
            best_val_loss = None
            try:
                import torch  # lazy import

                obj = torch.load(ckpt, map_location="cpu")
                if isinstance(obj, dict) and "val_loss" in obj:
                    best_val_loss = float(obj["val_loss"])  # type: ignore
            except Exception:
                best_val_loss = None
        else:
            print(f"[seed={seed}] best checkpoint not found: {ckpt}")
            best_val_loss = None

        # Optional WF evaluation
        eval_result = None
        if args.eval and ckpt.exists():
            eval_out = run_dir / "wf_eval"
            eval_out.mkdir(parents=True, exist_ok=True)
            ret = run(
                [
                    sys.executable,
                    "scripts/evaluate_with_wf.py",
                    "--model-path",
                    str(ckpt),
                    "--data-path",
                    args.data_path,
                    "--n-splits",
                    str(args.n_splits),
                    "--embargo-days",
                    str(args.embargo_days),
                    "--output-dir",
                    str(eval_out),
                    "--device",
                    args.device,
                ]
            )
            if ret != 0:
                print(f"[seed={seed}] WF evaluation failed with code {ret}")
            else:
                # Pick latest summary json in folder
                try:
                    latest = max(eval_out.glob("wf_summary_*.json"), key=lambda p: p.stat().st_mtime)
                    with open(latest) as f:
                        eval_result = json.load(f)
                except Exception:
                    eval_result = None

        summary["results"].append(
            {
                "seed": seed,
                "tag": tag,
                "checkpoint": str(ckpt.resolve()),
                "best_val_loss": best_val_loss,
                "wf_eval": eval_result,
            }
        )

    # Save rollout summary
    summary_path = base_dir / "rollout_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n✅ Rollout summary saved to {summary_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
