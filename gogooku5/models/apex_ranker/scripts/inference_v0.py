"""Placeholder inference entrypoint for APEX-Ranker."""
from __future__ import annotations

import argparse
from typing import NoReturn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with APEX-Ranker")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the dataset parquet file.",
    )
    parser.add_argument(
        "--checkpoint",
        default="artifacts/apex_ranker.ckpt",
        help="Model checkpoint to load.",
    )
    parser.add_argument(
        "--output",
        default="predictions/apex_ranker.parquet",
        help="Where to write predictions.",
    )
    return parser.parse_args()


def main() -> NoReturn:
    args = parse_args()
    raise SystemExit(
        "APEX-Ranker inference pipeline not implemented yet. "
        f"Received dataset={args.dataset}, checkpoint={args.checkpoint}, output={args.output}."
    )


if __name__ == "__main__":
    main()
