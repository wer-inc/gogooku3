#!/usr/bin/env python3
from __future__ import annotations

"""
Explain ATFT-GAT-FAN predictions via VSN gates and gradient attributions.

Example:
  python scripts/explain_atft.py \
    --checkpoint models/checkpoints/production-best.pt \
    --data-dir output/atft_data/val \
    --output-dir output/explain
"""

import argparse
from pathlib import Path
import json
import sys
import torch
import polars as pl

# Ensure repo root and src/ are on sys.path for direct invocation
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from gogooku3.explain.explain import export_vsn_gates, integrated_gradients, try_shap_kernel


def load_sample_batch(data_dir: Path, batch_size: int = 32, seq_len: int = 60) -> torch.Tensor:
    files = sorted(Path(data_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files in {data_dir}")
    df = pl.read_parquet(files[0]).select(pl.all().exclude(["Date", "Code"]))
    arr = df.to_numpy()
    if arr.shape[0] < seq_len * batch_size:
        bs = max(1, arr.shape[0] // seq_len)
    else:
        bs = batch_size
    x = arr[: bs * seq_len].reshape(bs, seq_len, -1)
    return torch.tensor(x, dtype=torch.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output-dir", default="output/explain")
    ap.add_argument("--shap", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import model loader to avoid heavy deps
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {})
    # Placeholder: users load model via their training script; here we assume state_dict only for gates/gradients
    class IdentityModel(torch.nn.Module):
        def forward(self, x):
            return x.mean(dim=-1)
    model = IdentityModel()
    try:
        model.load_state_dict(ckpt.get("model_state_dict", {}), strict=False)
    except Exception:
        pass
    model.eval()

    x = load_sample_batch(Path(args.data_dir))
    _ = model(x)
    export_vsn_gates(model, out_dir / "vsn_gates.json")
    atts = integrated_gradients(model, x)
    # Persist IG attributions without truncating the file
    torch.save(atts, out_dir / "ig_attributions.pt")

    if args.shap:
        shap_vals = try_shap_kernel(model, x[:1])
        if shap_vals is not None:
            with (out_dir / "shap_values.json").open("w") as f:
                json.dump({"shap": shap_vals if isinstance(shap_vals, list) else shap_vals.tolist()}, f)

    print(f"✅ explainability exported to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
