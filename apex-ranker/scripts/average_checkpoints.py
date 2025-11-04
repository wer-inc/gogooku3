#!/usr/bin/env python3
"""Average multiple PyTorch checkpoints element-wise.

This helper is intended for blending EMA snapshots (e.g., epochs 3/6/10)
after training ``train_v0.py`` with ``--ema-snapshot-epochs``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average multiple Torch checkpoints into a single state dict."
    )
    parser.add_argument(
        "checkpoints",
        nargs="+",
        help="Paths to checkpoint files produced by train_v0.py",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for averaged checkpoint (.pt)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for loading checkpoints (default: cpu)",
    )
    return parser.parse_args()


def load_state_dict(path: Path, device: str) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint {path} is not a state dict.")
    return payload


def average_state_dicts(
    checkpoints: list[Path],
    *,
    device: str,
) -> dict[str, torch.Tensor]:
    accumulator: dict[str, torch.Tensor] = {}
    for idx, ckpt_path in enumerate(checkpoints, start=1):
        state = load_state_dict(ckpt_path, device)
        if not accumulator:
            accumulator = {k: v.clone().to(device) for k, v in state.items()}
        else:
            for key, tensor in state.items():
                if key not in accumulator:
                    raise KeyError(f"Key '{key}' missing from accumulator when loading {ckpt_path}")
                accumulator[key] += tensor.to(device)
    if not accumulator:
        raise ValueError("No checkpoints provided.")
    scale = 1.0 / len(checkpoints)
    for key, tensor in accumulator.items():
        accumulator[key] = tensor.mul(scale)
    return accumulator


def main() -> None:
    args = parse_args()
    checkpoint_paths = [Path(p) for p in args.checkpoints]
    for path in checkpoint_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    averaged = average_state_dicts(checkpoint_paths, device=args.device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(averaged, output_path)
    print(
        f"[Average] Saved blended checkpoint ({len(checkpoint_paths)} inputs) to {output_path}"
    )


if __name__ == "__main__":
    main()
