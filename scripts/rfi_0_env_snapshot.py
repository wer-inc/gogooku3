"""RFI-0: Enhanced environment snapshot"""
import json
import os
import platform
import sys


def main():
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "env": {k: v for k, v in os.environ.items() if k in [
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_MAX_THREADS", "PYARROW_NUM_THREADS", "POLARS_MAX_THREADS",
            "RAYON_NUM_THREADS", "MALLOC_ARENA_MAX", "FORCE_SINGLE_PROCESS",
            "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF", "TOKENIZERS_PARALLELISM"
        ]},
    }

    # Try torch (may segfault)
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["torch_cuda_arch"] = torch.cuda.get_arch_list()
            info["torch_backends"] = {
                "cudnn": torch.backends.cudnn.version(),
                "mps": hasattr(torch.backends, "mps"),
            }
        else:
            info["torch_cuda_arch"] = []
            info["torch_backends"] = {"cudnn": None, "mps": False}
    except Exception as e:
        info["torch"] = f"ERROR: {e}"

    # Try torch_geometric
    try:
        import torch_geometric
        info["torch_geometric"] = torch_geometric.__version__
    except Exception:
        info["torch_geometric"] = None

    # Try pyarrow
    try:
        import pyarrow
        info["pyarrow"] = pyarrow.__version__
    except Exception:
        info["pyarrow"] = None

    # Try polars
    try:
        import polars
        info["polars"] = polars.__version__
    except Exception:
        info["polars"] = None

    output_path = "output/reports/diag_bundle/env_snapshot_enhanced.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(info, f, indent=2)

    print(json.dumps(info, indent=2))
    print(f"\nSaved: {output_path}")

if __name__ == "__main__":
    main()
