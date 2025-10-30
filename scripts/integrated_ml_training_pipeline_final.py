#!/usr/bin/env python3
"""
Thin alias for the integrated training pipeline.

Backwards-compatible entrypoint matching docs that reference
`scripts/integrated_ml_training_pipeline_final.py`.
"""

import runpy
from pathlib import Path


def main() -> None:
    target = Path(__file__).with_name("integrated_ml_training_pipeline.py")
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()

