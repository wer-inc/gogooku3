"""
I/Oと数値ライブラリのスレッド暴走を抑止して DataLoader を安定化。
必ず "import torch" より前に import すること。
"""
import os

SAFE_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_MAX_THREADS": "1",
    "PYARROW_NUM_THREADS": "4",   # Arrowは軽く並列可
    "POLARS_MAX_THREADS": "4",
    "RAYON_NUM_THREADS": "4",     # 一部Rust実装が使う
    "MALLOC_ARENA_MAX": "2",
}

if os.getenv("FORCE_SINGLE_PROCESS", "0") != "1":
    for k, v in SAFE_DEFAULTS.items():
        os.environ.setdefault(k, v)

def configure_torch_threads():
    import torch
    # 物理コアが多い環境での過剰スレッドを抑止
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

def set_spawn_start_method():
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # 既に設定済み
