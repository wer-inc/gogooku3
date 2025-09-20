#!/usr/bin/env python3
"""
本番用学習スクリプト（統一版）
- ProductionDataModuleV2を使用
- ATFT-GAT-FANモデル
- A100 GPU最適化設定
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import sys
from omegaconf import OmegaConf
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
from typing import Optional
import hydra
from omegaconf import DictConfig
import re
from datetime import datetime
from zoneinfo import ZoneInfo
import subprocess
import inspect
import json
import os
import time
import atexit
import faulthandler
import traceback
import torch.multiprocessing as mp

# Ensure a safe multiprocessing start method to avoid DataLoader deadlocks
try:
    _mp_method = os.getenv("MP_START_METHOD", "spawn").lower()
    if _mp_method in ("spawn", "forkserver"):
        mp.set_start_method(_mp_method, force=True)
except Exception:
    pass

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import required data module class explicitly
from src.gogooku3.training.atft.data_module import ProductionDataModuleV2

# Helper functions for stabilizing training
def _finite_or_nan_fix_tensor(tensor):
    """Fix non-finite values in tensor using clamp and nan_to_num."""
    if not torch.is_tensor(tensor):
        return tensor
    
    # Replace NaN/Inf with finite values
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Clamp to reasonable range
    tensor = torch.clamp(tensor, min=-1e6, max=1e6)
    
    return tensor

def _normalize_target_key(raw_key, horizons=[1, 5, 10, 20]):
    """Normalize various target key formats to horizon_{h} format."""
    # Direct horizon format
    if raw_key.startswith('horizon_'):
        return raw_key
    
    # Extract number from various formats
    patterns = [
        r'return_(\d+)d?',
        r'target_(\d+)d?', 
        r'label_ret_(\d+)_bps',
        r'(\d+)d?',
        r'^(\d+)$'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, str(raw_key))
        if match:
            h = int(match.group(1))
            if h in horizons:
                return f'horizon_{h}'
    
    return None

def _reshape_to_batch_only(tensor_dict, key_prefix="", take_last_step=True):
    """Reshape tensors to [B] format, taking last timestep if needed."""
    reshaped = {}
    
    for key, tensor in tensor_dict.items():
        if not torch.is_tensor(tensor):
            # Skip metadata entries (e.g., codes/date) that are not tensors
            continue
            
        # Fix non-finite values first (use newer guard signature)
        try:
            tensor = _finite_or_nan_fix_tensor(
                tensor, f"reshape[{key_prefix}{key}]"
            )
        except TypeError:
            # Backward compatibility if older signature is active
            tensor = _finite_or_nan_fix_tensor(tensor)
        
        # Handle different shapes
        if tensor.dim() == 1:
            # Already [B] - keep as is
            reshaped[key] = tensor
        elif tensor.dim() == 2 and take_last_step:
            # [B, T] - take last timestep
            reshaped[key] = tensor[:, -1]
        elif tensor.dim() == 2:
            # [B, F] - keep as is if F matches batch size
            reshaped[key] = tensor
        elif tensor.dim() == 3 and take_last_step:
            # [B, T, F] - take last timestep
            reshaped[key] = tensor[:, -1, :]
        else:
            # Squeeze extra dimensions
            reshaped[key] = tensor.squeeze()
            
    return reshaped

# ---- Label clipping helper -------------------------------------------------
def _parse_label_clip_map(env_val: str | None):
    """Parse LABEL_CLIP_BPS_MAP like '1:2000,5:3000,10:5000' -> dict{h: clip_value_in_return}

    Values are in basis points. Returns dict mapping horizon(int) -> float clip_abs.
    """
    m = {}
    if not env_val:
        return m
    try:
        parts = [p.strip() for p in str(env_val).split(",") if p.strip()]
        for p in parts:
            k, v = p.split(":")
            h = int(k.strip())
            bps = float(v.strip())
            m[h] = abs(bps) / 10000.0  # convert bps to return
    except Exception:
        return {}
    return m

def _clip_targets_by_horizon(targets: dict[str, torch.Tensor], clip_map: dict[int, float]) -> dict[str, torch.Tensor]:
    if not clip_map:
        return targets
    out = {}
    for k, v in targets.items():
        try:
            if isinstance(k, str) and k.startswith("horizon_"):
                h = int(k.split("_", 1)[1])
                if h in clip_map and torch.is_tensor(v):
                    lim = float(clip_map[h])
                    out[k] = torch.clamp(v, -lim, lim)
                    continue
        except Exception:
            pass
        out[k] = v
    return out

# ---- Phase-aware loss schedule --------------------------------------------
def _parse_phase_loss_schedule(env_val: str | None) -> dict[int, dict[str, float]]:
    """Parse PHASE_LOSS_WEIGHTS like '0:huber=0.3,quantile=1.0;1:quantile=1.0,sharpe=0.1;2:quantile=1.0,sharpe=0.1,rankic=0.05,t_nll=0.7'.

    Returns {phase: {weight_name: value}}. Unknown entries are ignored by the applier.
    """
    sched: dict[int, dict[str, float]] = {}
    if not env_val:
        return sched
    try:
        parts = [p.strip() for p in str(env_val).split(";") if p.strip()]
        for p in parts:
            # split phase:weights
            if ":" not in p:
                continue
            phs, items = p.split(":", 1)
            ph = int(phs.strip())
            wmap: dict[str, float] = {}
            for it in [x.strip() for x in items.split(",") if x.strip()]:
                if "=" not in it:
                    continue
                k, v = it.split("=", 1)
                try:
                    wmap[k.strip().lower()] = float(v.strip())
                except Exception:
                    continue
            if wmap:
                sched[ph] = wmap
    except Exception:
        return {}
    return sched

def _apply_phase_loss_weights(criterion, phase_idx: int, sched: dict[int, dict[str, float]]):
    """Apply phase-specific loss weights/toggles to criterion if attributes exist."""
    weights = sched.get(phase_idx, {})
    if not weights:
        return
    try:
        # Huber
        if "huber" in weights:
            if hasattr(criterion, "use_huber"):
                setattr(criterion, "use_huber", weights["huber"] > 0)
            if hasattr(criterion, "huber_weight"):
                setattr(criterion, "huber_weight", float(weights["huber"]))
        # Quantile (pinball)
        if "quantile" in weights:
            if hasattr(criterion, "use_pinball"):
                setattr(criterion, "use_pinball", weights["quantile"] > 0)
            if hasattr(criterion, "pinball_weight"):
                setattr(criterion, "pinball_weight", float(weights["quantile"]))
        # RankIC
        if "rankic" in weights or "rank_ic" in weights:
            w = weights.get("rankic", weights.get("rank_ic", 0.0))
            if hasattr(criterion, "use_rankic"):
                setattr(criterion, "use_rankic", w > 0)
            if hasattr(criterion, "rankic_weight"):
                setattr(criterion, "rankic_weight", float(w))
        # Sharpe (portfolio returns)
        if "sharpe" in weights:
            # Some implementations include sharpe_weight; if present, set it
            if hasattr(criterion, "sharpe_weight"):
                setattr(criterion, "sharpe_weight", float(weights["sharpe"]))
        # Student-t NLL
        if "t_nll" in weights or "nll" in weights:
            w = weights.get("t_nll", weights.get("nll", 0.0))
            if hasattr(criterion, "use_t_nll"):
                setattr(criterion, "use_t_nll", w > 0)
            if hasattr(criterion, "nll_weight"):
                setattr(criterion, "nll_weight", float(w))
    except Exception:
        pass

# --- Pre-parse custom CLI flags (before Hydra) ---
# Support: --data-path <file> (single Parquet file)
def _preparse_custom_argv():
    try:
        import sys as _sys
        argv = list(_sys.argv)
        if not argv:
            return
        new_argv = []
        i = 0
        while i < len(argv):
            tok = argv[i]
            if tok in ("--data-path", "--data_path"):
                # Expect a following value
                if i + 1 < len(argv):
                    val = argv[i + 1]
                    os.environ["DATA_PATH"] = val
                    i += 2
                    continue
                else:
                    i += 1
                    continue
            elif tok.startswith("--data-path="):
                os.environ["DATA_PATH"] = tok.split("=", 1)[1]
                i += 1
                continue
            elif tok.startswith("--data_path="):
                os.environ["DATA_PATH"] = tok.split("=", 1)[1]
                i += 1
                continue
            elif tok in ("--early-stopping-metric", "--early_stopping_metric"):
                if i + 1 < len(argv):
                    val = argv[i + 1]
                    os.environ["EARLY_STOP_METRIC"] = val
                    i += 2
                    continue
                else:
                    i += 1
                    continue
            elif tok.startswith("--early-stopping-metric="):
                os.environ["EARLY_STOP_METRIC"] = tok.split("=", 1)[1]
                i += 1
                continue
            elif tok in ("--early-stopping-maximize", "--early_stopping_maximize"):
                # boolean flag; presence implies true unless explicit value provided
                if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                    os.environ["EARLY_STOP_MAXIMIZE"] = argv[i + 1]
                    i += 2
                else:
                    os.environ["EARLY_STOP_MAXIMIZE"] = "1"
                    i += 1
                continue
            elif tok.startswith("--early-stopping-maximize="):
                os.environ["EARLY_STOP_MAXIMIZE"] = tok.split("=", 1)[1]
                i += 1
                continue
            elif tok in ("--scheduler",):
                if i + 1 < len(argv):
                    os.environ["SCHEDULER"] = argv[i + 1]
                    i += 2
                    continue
                else:
                    i += 1
                    continue
            elif tok.startswith("--scheduler="):
                os.environ["SCHEDULER"] = tok.split("=", 1)[1]
                i += 1
                continue
            elif tok in ("--label-clip-bps", "--label_clip_bps"):
                if i + 1 < len(argv):
                    os.environ["LABEL_CLIP_BPS_MAP"] = argv[i + 1]
                    i += 2
                    continue
                else:
                    i += 1
                    continue
            elif tok.startswith("--label-clip-bps="):
                os.environ["LABEL_CLIP_BPS_MAP"] = tok.split("=", 1)[1]
                i += 1
                continue
            elif tok in ("--train-profile", "--train_profile"):
                if i + 1 < len(argv):
                    os.environ["TRAIN_PROFILE"] = argv[i + 1]
                    i += 2
                    continue
                else:
                    i += 1
                    continue
            elif tok.startswith("--train-profile="):
                os.environ["TRAIN_PROFILE"] = tok.split("=", 1)[1]
                i += 1
                continue
            else:
                new_argv.append(tok)
                i += 1
        _sys.argv = new_argv
    except Exception:
        pass

_preparse_custom_argv()

# ---- TRAIN_PROFILE presets -------------------------------------------------
def _setenv_if_unset(k: str, v: str):
    if os.getenv(k) is None or os.getenv(k) == "":
        os.environ[k] = v

def _apply_train_profile():
    profile = os.getenv("TRAIN_PROFILE", "").strip().lower()
    if not profile:
        return
    if profile in ("smoke", "quick"):
        _setenv_if_unset("USE_MINI_TRAIN", "1")
        _setenv_if_unset("PHASE_MAX_EPOCHS", "2")
        _setenv_if_unset("PHASE_MAX_BATCHES", "10")
        _setenv_if_unset("EARLY_STOP_METRIC", "val_loss")
        _setenv_if_unset("EARLY_STOP_PATIENCE", "2")
        _setenv_if_unset("SCHEDULER", "warmup_cosine")
        _setenv_if_unset("PHASE_WARMUP_EPOCHS", "1")
        _setenv_if_unset("OUTPUT_NOISE_STD", "0.01")
        _setenv_if_unset("NOISE_WARMUP_EPOCHS", "1")
        _setenv_if_unset("LABEL_CLIP_BPS_MAP", "1:2000")
        _setenv_if_unset("SMOKE_DATA_MAX_FILES", "2")
    elif profile in ("exp", "experiment"):
        _setenv_if_unset("EARLY_STOP_METRIC", "val_sharpe")
        _setenv_if_unset("EARLY_STOP_MAXIMIZE", "1")
        _setenv_if_unset("SCHEDULER", "warmup_cosine")
        _setenv_if_unset("PHASE_WARMUP_EPOCHS", "2")
        _setenv_if_unset("LABEL_CLIP_BPS_MAP", "1:2000,5:3000,10:5000")
        _setenv_if_unset("FUSE_FORCE_MODE", "tft_only")
        _setenv_if_unset("FUSE_START_PHASE", "2")
        _setenv_if_unset("GAT_ALPHA_WARMUP_MIN", "0.30")
        _setenv_if_unset("GAT_ALPHA_WARMUP_EPOCHS", "2")
        _setenv_if_unset("PHASE_LOSS_WEIGHTS", "0:huber=0.3,quantile=1.0;1:quantile=1.0,sharpe=0.1;2:quantile=1.0,sharpe=0.15,rankic=0.05,t_nll=0.7")
        _setenv_if_unset("EARLY_STOP_PATIENCE", "9")
        _setenv_if_unset("USE_AMP", "1")
        _setenv_if_unset("AMP_DTYPE", "bf16")
    elif profile in ("prod", "production"):
        _setenv_if_unset("EARLY_STOP_METRIC", "val_sharpe")
        _setenv_if_unset("EARLY_STOP_MAXIMIZE", "1")
        _setenv_if_unset("SCHEDULER", "warmup_cosine")
        _setenv_if_unset("PHASE_WARMUP_EPOCHS", "2")
        _setenv_if_unset("LABEL_CLIP_BPS_MAP", "1:2000,5:3000,10:5000")
        _setenv_if_unset("FUSE_FORCE_MODE", "auto")
        _setenv_if_unset("FUSE_START_PHASE", "2")
        _setenv_if_unset("GAT_ALPHA_WARMUP_MIN", "0.30")
        _setenv_if_unset("GAT_ALPHA_WARMUP_EPOCHS", "2")
        _setenv_if_unset("PHASE_LOSS_WEIGHTS", "0:huber=0.3,quantile=1.0;1:quantile=1.0,sharpe=0.1;2:quantile=1.0,sharpe=0.15,rankic=0.05,t_nll=0.7")
        _setenv_if_unset("EARLY_STOP_PATIENCE", "12")
        _setenv_if_unset("USE_AMP", "1")
        _setenv_if_unset("AMP_DTYPE", "bf16")

_apply_train_profile()

# Import unified metrics utilities
try:
    from src.utils.metrics_utils import (
        compute_pred_std_batch,
        collect_metrics_from_outputs,
    )
except ImportError:
    compute_pred_std_batch = None
    collect_metrics_from_outputs = None

# Setup logging early
logger = logging.getLogger(__name__)

# Timezone for logs/metrics (default Asia/Tokyo; override with LOG_TZ or TZ)
LOG_TZ = os.getenv("LOG_TZ", os.getenv("TZ", "Asia/Tokyo"))
try:
    JST = ZoneInfo(LOG_TZ)
except Exception:
    JST = ZoneInfo("Asia/Tokyo")


def now_jst_iso() -> str:
    return datetime.now(JST).isoformat()


# Global simple state for EMA logging
IC_EMA_H1 = None  # type: ignore

# Use optimized loader if available
use_optimized = os.getenv("USE_OPTIMIZED_LOADER", "1") == "1"
if use_optimized:
    try:
        from src.data.loaders.production_loader_v2_optimized import (  # noqa: E402
            ProductionDatasetOptimized as ProductionDatasetV2,
        )
        from src.data.loaders.production_loader_v2 import (  # noqa: E402
            ProductionDataModuleV2,
        )

        print("Using optimized data loader")
    except ImportError as e:
        from src.data.loaders.production_loader_v2 import (  # noqa: E402
            ProductionDataModuleV2,
            ProductionDatasetV2,
        )

        print(f"Optimized loader not available ({e}), using standard")
else:
    try:
        from src.data.loaders.production_loader_v2 import (  # noqa: E402
            ProductionDataModuleV2,
            ProductionDatasetV2,
        )
    except Exception as e:
        ProductionDataModuleV2 = None  # type: ignore
        ProductionDatasetV2 = None  # type: ignore
        print(f"Standard loader not available ({e}); will use smoke fallback if enabled")
from src.data.samplers.day_batch_sampler import DayBatchSampler  # noqa: E402
# DayBatchSamplerFixed is not needed, using DayBatchSampler instead
DayBatchSamplerFixed = DayBatchSampler  # Alias for compatibility
from src.graph.graph_builder import GBConfig, GraphBuilder  # noqa: E402
try:
    from src.data.utils.graph_builder import (  # noqa: E402
        FinancialGraphBuilder as AdvFinancialGraphBuilder,
    )
except Exception:
    AdvFinancialGraphBuilder = None  # type: ignore

try:
    from src.atft_gat_fan.models.architectures.atft_gat_fan import ATFT_GAT_FAN  # noqa: E402
except ImportError:
    try:
        from src.models.architectures.atft_gat_fan import ATFT_GAT_FAN  # type: ignore # noqa: E402
    except ImportError:
        ATFT_GAT_FAN = None
from src.data.validation.normalization_check import NormalizationValidator  # noqa: E402
from src.utils.config_validator import ConfigValidator  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fallback minimal DataModule for smoke profile (when ProductionDataModuleV2 is missing)
if ProductionDataModuleV2 is None:  # type: ignore
    try:
        class _SmokeDataset(torch.utils.data.Dataset):
            def __init__(self, n_samples: int = 512, seq_len: int = 20, n_features: int = 8):
                self.n = n_samples
                self.T = seq_len
                self.F = n_features
                w = torch.randn(n_features)
                self.w = w

            def __len__(self):
                return self.n

            def __getitem__(self, idx):
                x = torch.randn(self.T, self.F)
                y = (x[-1] @ self.w) / (self.F ** 0.5)
                sample = {
                    "features": x,
                    "targets": {
                        "horizon_1": y.unsqueeze(0),
                    },
                }
                return sample

        class ProductionDataModuleV2:  # type: ignore
            def __init__(self, cfg, batch_size: int = 64, num_workers: int = 0):
                self.bs = int(batch_size)
                self.nw = int(num_workers)
                self.seq_len = int(getattr(getattr(cfg.data.time_series, "sequence_length", 20), "__int__", lambda: 20)())

            def setup(self, stage: str | None = None):
                self.train_ds = _SmokeDataset(512, self.seq_len)
                self.val_ds = _SmokeDataset(256, self.seq_len)

            def train_dataloader(self):
                return torch.utils.data.DataLoader(self.train_ds, batch_size=self.bs, shuffle=True, num_workers=self.nw)

            def val_dataloader(self):
                return torch.utils.data.DataLoader(self.val_ds, batch_size=self.bs, shuffle=False, num_workers=self.nw)

        print("[SmokeFallback] Using internal smoke DataModule (random data)")
    except Exception as _e_fb:
        logger.warning(f"Smoke fallback unavailable: {_e_fb}")

# Attach file logger for live tailing (logs/ml_training.log)
try:
    from pathlib import Path as _Path
    import logging as _logging

    _log_dir = _Path("logs")
    _log_dir.mkdir(parents=True, exist_ok=True)
    _log_file = (_log_dir / "ml_training.log").resolve()

    _root = _logging.getLogger()
    # Avoid duplicate file handlers pointing to the same file
    _has_same = False
    for _h in list(_root.handlers):
        try:
            if isinstance(_h, _logging.FileHandler) and _h.baseFilename == str(_log_file):
                _has_same = True
                break
        except Exception:
            pass
    if not _has_same:
        _fh = _logging.FileHandler(_log_file, mode="a", encoding="utf-8")
        _fh.setFormatter(
            _logging.Formatter(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        _root.addHandler(_fh)
        _root.info(f"[logger] FileHandler attached: {_log_file}")
except Exception as _e_attach:
    try:
        logger.debug(f"file logger attach skipped: {_e_attach}")
    except Exception:
        pass

# Enable fault handler for better crash reports
faulthandler.enable()

# Global run directory
RUN_DIR = Path(os.getenv("RUN_DIR", "runs/last"))
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Optional W&B logger (via our integrated monitoring utility)
WBLogger: Optional[object] = None  # type: ignore
try:
    from src.utils.monitoring import ComprehensiveLogger as _WBLogger  # type: ignore
    WBLogger = _WBLogger  # alias for type hints
except Exception:
    WBLogger = None  # type: ignore


# ---- Target key canonicalization -----------------------------------------
_TARGET_KEY_PATTERNS = [
    re.compile(r"^(?:return|returns|ret|target|targets|tgt|y)_(\d+)d$", re.I),
    re.compile(r"^horizon_(\d+)$", re.I),
]


def _canonicalize_target_key(key: object) -> str | None:
    """Map various target key forms to 'horizon_{h}'.

    Accepts keys like 1, '1', 'horizon_1', 'return_1d', 'target_5d', etc.
    Returns canonical string or None if not recognized.
    """
    try:
        # Integer or digit string
        if isinstance(key, int):
            return f"horizon_{int(key)}"
        if isinstance(key, str) and key.isdigit():
            return f"horizon_{int(key)}"
        if isinstance(key, str):
            s = key
            for pat in _TARGET_KEY_PATTERNS:
                m = pat.match(s)
                if m:
                    return f"horizon_{int(m.group(1))}"
        return None
    except Exception:
        return None


# ---- Minimal config export for external tests ----
def build_final_config():
    """Provide a lightweight config object for external test harnesses.

    Returns an object with the minimal fields used by dataset/model constructors:
    - data.source.data_dir
    - data.time_series.sequence_length, prediction_horizons
    - data.features.input_dim
    - model.hidden_size
    - train.batch.train_batch_size
    """
    from types import SimpleNamespace as NS

    cfg = NS()
    cfg.data = NS()
    cfg.data.source = NS()
    cfg.data.source.data_dir = os.getenv("DATA_DIR", "data/test")
    cfg.data.time_series = NS()
    cfg.data.time_series.sequence_length = int(os.getenv("SEQ_LEN", "20"))
    cfg.data.time_series.prediction_horizons = [1, 5, 10, 20]
    cfg.data.features = NS()
    cfg.data.features.input_dim = int(os.getenv("INPUT_DIM", "13"))

    cfg.model = NS()
    cfg.model.hidden_size = int(os.getenv("HIDDEN_SIZE", "128"))

    cfg.train = NS()
    cfg.train.batch = NS()
    cfg.train.batch.train_batch_size = int(os.getenv("BATCH_SIZE", "64"))
    cfg.train.batch.val_batch_size = int(os.getenv("VAL_BATCH_SIZE", "64"))
    return cfg


# Expose a module-level instance for tests that look for `final_config`
final_config = build_final_config()


def write_failure_report(exc=None):
    """Write failure report for debugging"""
    rep = {
        "status": "failed",
        "exc_type": type(exc).__name__ if exc else None,
        "message": str(exc) if exc else None,
        "traceback": traceback.format_exc() if exc else None,
        "pid": os.getpid(),
        "time": time.time(),
        "timestamp": now_jst_iso(),
        "tz": str(JST.key) if hasattr(JST, "key") else "Asia/Tokyo",
    }
    (RUN_DIR / "failure_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2)
    )


@atexit.register
def on_exit():
    """Ensure failure report is written if no success metrics"""
    if not (RUN_DIR / "latest_metrics.json").exists():
        write_failure_report()


def collate_day(items):
    """Collate function for day-batched samples that include code/date.
    - Stacks features and targets tensors
    - Carries codes as list[str]
    - Uses the first item's date (DayBatchSampler ensures same day)
    """
    import torch as _torch

    # features
    x = _torch.stack([it["features"] for it in items], dim=0)
    # targets: dict of tensors (robust to missing keys)
    tgt_keys = set(items[0]["targets"].keys())
    for it in items[1:]:
        try:
            tgt_keys &= set(it["targets"].keys())
        except Exception:
            pass
    tgt_keys = sorted(list(tgt_keys))
    if not tgt_keys:
        # Fallback: build empty dict to avoid crash; upstream will handle
        y = {}
    else:
        y = {k: _torch.stack([it["targets"][k] for it in items], dim=0) for k in tgt_keys}
    # valid masks (optional)
    vm = None
    if "valid_mask" in items[0] and isinstance(items[0]["valid_mask"], dict):
        mkeys = list(items[0]["valid_mask"].keys())
        vm = {
            k: _torch.stack([it["valid_mask"][k] for it in items], dim=0) for k in mkeys
        }
    # codes/date
    codes = [str(it.get("code")) for it in items]
    markets = None
    sectors = None
    try:
        if all(("market" in it) for it in items):
            markets = [None if it.get("market") in (None, "nan") else str(it.get("market")) for it in items]
        if all(("sector" in it) for it in items):
            sectors = [None if it.get("sector") in (None, "nan") else str(it.get("sector")) for it in items]
    except Exception:
        markets = None
        sectors = None
    date0 = items[0].get("date", None)
    out = {"features": x, "targets": y, "codes": codes, "date": date0}
    if vm is not None:
        out["valid_mask"] = vm
    if markets is not None:
        out["markets"] = markets
    if sectors is not None:
        out["sectors"] = sectors
    return out


# ===== DataLoader parameter helpers =====
def _resolve_dl_params(final_config: DictConfig) -> dict:
    """Resolve DataLoader params from Hydra config with env fallbacks.

    Honors final_config.train.batch.* if present; otherwise falls back to
    environment variables NUM_WORKERS, PREFETCH_FACTOR, PIN_MEMORY, PERSISTENT_WORKERS.
    Ensures a safe single-process configuration when num_workers == 0.
    """
    # Try config values
    try:
        nw_cfg = getattr(final_config.train.batch, "num_workers")
    except Exception:
        nw_cfg = None
    try:
        pf_cfg = getattr(final_config.train.batch, "prefetch_factor")
    except Exception:
        pf_cfg = None
    try:
        pm_cfg = getattr(final_config.train.batch, "pin_memory")
    except Exception:
        pm_cfg = None
    try:
        pw_cfg = getattr(final_config.train.batch, "persistent_workers")
    except Exception:
        pw_cfg = None

    # Env fallbacks
    nw_env = int(os.getenv("NUM_WORKERS", "0"))
    pf_env = int(os.getenv("PREFETCH_FACTOR", "2"))
    pm_env = os.getenv("PIN_MEMORY", "0").lower() in ("1", "true", "yes")
    pw_env = os.getenv("PERSISTENT_WORKERS", "0").lower() in ("1", "true", "yes")

    params = {
        "num_workers": int(nw_cfg) if nw_cfg is not None else nw_env,
        "prefetch_factor": int(pf_cfg) if pf_cfg is not None else pf_env,
        "pin_memory": bool(pm_cfg) if isinstance(pm_cfg, bool) else pm_env,
        "persistent_workers": bool(pw_cfg) if isinstance(pw_cfg, bool) else pw_env,
    }

    # Sanitize for single-process
    if params["num_workers"] <= 0:
        params["num_workers"] = 0
        params["persistent_workers"] = False
        # prefetch_factor is only valid when num_workers > 0
        params["prefetch_factor"] = None

    return params


# ===== NaN/Inf guard utilities =====
def _finite_or_nan_fix_tensor(
    t: torch.Tensor, name: str, clamp: float | None = None
) -> torch.Tensor:
    try:
        if not torch.isfinite(t).all():
            bad = (~torch.isfinite(t)).sum().item()
            logger.warning(f"[nan-guard] {name}: non-finite={bad}")
            t = torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)
        if clamp is not None:
            t = torch.clamp(t, -clamp, clamp)
        return t
    except Exception:
        return t


class MultiHorizonLoss(nn.Module):
    """マルチホライズン予測用の損失関数（RankIC/Pinballの任意併用 + プリセット/動的重み + 補助ヘッド）"""

    def __init__(
        self,
        horizons=[1, 5, 10, 20],
        use_rankic: bool = False,
        rankic_weight: float = 0.0,
        # 追加: CS相関(IC)補助ロス
        use_cs_ic: bool = True,
        cs_ic_weight: float = 0.05,
        use_pinball: bool = False,
        quantiles=(0.2, 0.5, 0.8),
        pinball_weight: float = 0.0,
        use_t_nll: bool = False,
        nll_weight: float = 0.0,
        use_huber: bool = False,
        huber_delta: float = 1.0,
        huber_weight: float = 0.0,
        h1_loss_mult: float = 1.0,
        # 追加: 重み制御
        horizon_weights: dict | None = None,
        use_dynamic_weighting: bool = False,
        dynamic_alpha: float = 0.01,
        dynamic_freeze_frac: float = 0.6,
        # 追加: 補助ヘッド
        direction_aux_weight: float = 0.0,
        sigma_weighting_lambda: float = 0.0,
        # 追加: 予測分散の下限ペナルティ（バッチ内での定数崩壊防止）
        pred_var_min: float = 0.0,
        pred_var_weight: float = 0.0,
    ):
        super().__init__()
        self.horizons = horizons
        self.mse = nn.MSELoss(reduction="mean")
        self.use_rankic = use_rankic
        self.rankic_weight = float(rankic_weight)
        self.use_pinball = use_pinball
        self.quantiles = tuple(quantiles)
        self.pinball_weight = float(pinball_weight)
        # CS-IC 補助ロス
        self.use_cs_ic = bool(use_cs_ic)
        self.cs_ic_weight = float(cs_ic_weight)
        self.use_t_nll = use_t_nll
        self.nll_weight = float(nll_weight)
        self.use_crps = (
            self.use_pinball
        )  # CRPS ~ 分位損失の積分近似（ここではPinball和をCRPS近似とする）
        # Robust/weighting options
        self.use_huber = bool(use_huber)
        self.huber_delta = float(huber_delta)
        self.huber_weight = float(huber_weight)
        self.h1_loss_mult = float(h1_loss_mult)
        # Horizon weights
        self._preset_weights = None
        if horizon_weights:
            # 正規化して保持（合計1）
            try:
                s = float(sum(float(v) for v in horizon_weights.values()))
                self._preset_weights = {
                    int(k): float(v) / (s if s > 0 else 1.0)
                    for k, v in horizon_weights.items()
                }
            except Exception:
                self._preset_weights = None
        self.use_dynamic_weighting = bool(use_dynamic_weighting)
        self.dynamic_alpha = float(dynamic_alpha)
        self.dynamic_freeze_frac = float(dynamic_freeze_frac)
        self._ema_rmse: dict[int, float] = {}
        self._steps: int = 0
        self._total_epochs: int | None = None
        self._current_epoch: int | None = None
        # Aux heads
        self.direction_aux_weight = float(direction_aux_weight)
        self._bce = (
            nn.BCEWithLogitsLoss(reduction="mean")
            if self.direction_aux_weight > 0
            else None
        )
        self.sigma_weighting_lambda = float(sigma_weighting_lambda)
        # 予測分散ペナルティ
        self.pred_var_min = float(pred_var_min)
        self.pred_var_weight = float(pred_var_weight)
        self._warned_empty = False

    @staticmethod
    def _masked_mean(
        x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        """Compute masked mean, excluding invalid values"""
        if mask is None:
            return torch.mean(x)
        mask = mask.to(dtype=x.dtype)
        s = mask.sum()
        if s.item() == 0:
            return torch.zeros((), dtype=x.dtype, device=x.device)
        return (x * mask).sum() / (s + eps)

    def _pinball_loss(
        self, yhat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        losses = []
        for q in self.quantiles:
            e = y - yhat
            losses.append(torch.maximum(q * e, (q - 1) * e))
        return self._masked_mean(torch.stack(losses, dim=0), mask)

    def _rankic_penalty(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # 勾配を流すためPearson相関の負を近似として使用（yはdetach）
        yhat_f = yhat.view(-1).float()
        y_f = y.view(-1).float().detach()
        if yhat_f.numel() <= 1:
            return yhat_f.new_zeros(())
        yhat_f = yhat_f - yhat_f.mean()
        y_f = y_f - y_f.mean()
        denom = yhat_f.std(unbiased=False) * y_f.std(unbiased=False) + 1e-8
        corr = (yhat_f * y_f).mean() / denom
        return 1.0 - corr

    @staticmethod
    def _cs_ic_loss(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """同一バッチ（日次クロスセクション）の順位相関に整合するよう相関を損失化。
        標準化後のPearson相関を用い 1-corr を最小化する。
        """
        yhat_f = yhat.view(-1).float()
        y_f = y.view(-1).float().detach()
        if yhat_f.numel() <= 1:
            return yhat_f.new_zeros(())
        yhat_f = (yhat_f - yhat_f.mean()) / (yhat_f.std(unbiased=False) + 1e-8)
        y_f = (y_f - y_f.mean()) / (y_f.std(unbiased=False) + 1e-8)
        return 1.0 - (yhat_f * y_f).mean()

    @staticmethod
    def _student_t_nll(
        mu: torch.Tensor, sigma_raw: torch.Tensor, nu_raw: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        # Always evaluate in FP32 for numerical stability
        with torch.amp.autocast("cuda", enabled=False):
            # Stable parameterization
            eps = 1e-6
            sigma = torch.nn.functional.softplus(sigma_raw) + eps
            # clamp variance to avoid overflow/underflow
            sigma = torch.clamp(sigma, 1e-6, 1e3)
            # degrees of freedom > 2 for finite variance, with clamp upper bound
            nu = 2.2 + torch.nn.functional.softplus(nu_raw)
            nu = torch.clamp(nu, 2.2, 40.0)
            z = (y - mu) / sigma
            # log pdf of Student-t: -0.5*log(nu*pi) - log(sigma) - ((nu+1)/2)*log(1 + z^2/nu)
            log_const = -0.5 * torch.log(nu * torch.pi) - torch.log(sigma)
            log_kernel = -0.5 * (nu + 1.0) * torch.log1p((z * z) / nu)
            logp = log_const + log_kernel
            return -torch.mean(logp)

    def forward(self, predictions, targets, valid_masks=None):
        """
        predictions: Dict[str, Tensor] - 各ホライゾンの予測
        targets: Dict[str, Tensor] - 各ホライゾンのターゲット
        valid_masks: Dict[str, Tensor] - 各ホライゾンの有効マスク（オプション）
        """
        # Initialize as zero (will accumulate loss tensors)
        device = (
            next(iter(predictions.values())).device
            if predictions
            else torch.device("cpu")
        )
        dtype = (
            next(iter(predictions.values())).dtype
            if predictions
            else torch.float32
        )
        total_loss = torch.zeros(1, device=device, dtype=dtype).requires_grad_()
        losses = {}
        weights = []
        contribution_count = 0
        # 現在のホライズン重み
        cur_weights = self._get_current_weights()

        for horizon in self.horizons:
            pred_candidates = [
                f"point_horizon_{horizon}",
                f"horizon_{horizon}",
                f"horizon_{horizon}d",
                f"h{horizon}",
            ]
            targ_candidates = [
                f"horizon_{horizon}",
                f"horizon_{horizon}d",
                f"point_horizon_{horizon}",
                f"h{horizon}",
            ]

            pred_key = next((k for k in pred_candidates if k in predictions), None)
            targ_key = next((k for k in targ_candidates if k in targets), None)
            if pred_key is None or targ_key is None:
                continue

            yhat = predictions[pred_key].squeeze(-1)
            y = targets[targ_key]

            # Get valid mask for this horizon
            mask = None
            if valid_masks is not None:
                mask_candidates = [
                    f"horizon_{horizon}",
                    f"horizon_{horizon}d",
                    f"point_horizon_{horizon}",
                    f"h{horizon}",
                ]
                for mask_key in mask_candidates:
                    if mask_key in valid_masks:
                        mask = valid_masks[mask_key]
                        break
                # Auto-create mask from finite values if not provided
                if mask is None and torch.is_tensor(y):
                    mask = torch.isfinite(y)
            else:
                # Create mask from finite values
                mask = torch.isfinite(y) if torch.is_tensor(y) else None

                # Use masked MSE
                if mask is not None:
                    loss = self._masked_mean((yhat - y) ** 2, mask)
                else:
                    loss = self.mse(yhat, y)
                # 予測分散の下限を促す正則化（勾配が yhat に確実に返るよう std ベース・二乗ペナルティ）
                if self.pred_var_weight > 0.0 and self.pred_var_min > 0.0:
                    try:
                        std_yhat = yhat.float().view(-1).std(unbiased=False)
                        # min_std - std に対する ReLU^2 ペナルティ（std が閾値未満だと強く押し上げる）
                        var_penalty = torch.relu(self.pred_var_min - std_yhat).pow(2)
                        if torch.isfinite(std_yhat):
                            loss = loss + self.pred_var_weight * var_penalty
                    except Exception:
                        pass
                # Huber mix（外れ値ロバスト化）
                if self.use_huber and self.huber_weight > 0.0:
                    hub = torch.nn.functional.smooth_l1_loss(
                        yhat, y, beta=self.huber_delta, reduction="mean"
                    )
                    loss = (1.0 - self.huber_weight) * loss + self.huber_weight * hub
                # Pinball（任意）
                if self.use_pinball and self.pinball_weight > 0:
                    loss = loss + self.pinball_weight * self._pinball_loss(yhat, y)
                # CRPS 近似（Quantile ヘッドがある場合、Pinballの多分位和で近似）
                q_key = f"quantile_horizon_{horizon}"
                if self.use_crps and q_key in predictions:
                    q_out = predictions[q_key]
                    if q_out.dim() == yhat.dim() + 1:
                        # 非交差は学習で担保できないため、単調化を後段で行う前提の簡易近似
                        quantiles = torch.sigmoid(
                            torch.linspace(
                                0.05, 0.95, q_out.shape[-1], device=q_out.device
                            )
                        )
                        y_expand = y.unsqueeze(-1)
                        e = y_expand - q_out
                        pinball = torch.maximum(quantiles * e, (quantiles - 1) * e)
                        loss = loss + self.pinball_weight * torch.mean(pinball)
                # RankIC（任意）
                if self.use_rankic and self.rankic_weight > 0:
                    loss = loss + self.rankic_weight * self._rankic_penalty(
                        yhat, y, mask
                    )
                # CS-IC 補助ロス（順位整合/相関強化）
                if self.use_cs_ic and self.cs_ic_weight > 0:
                    loss = loss + self.cs_ic_weight * self._cs_ic_loss(yhat, y)
                # Student-t NLL（任意, ヘテロスケ）
                t_key = f"t_params_horizon_{horizon}"
                if self.use_t_nll and self.nll_weight > 0 and t_key in predictions:
                    t_params = predictions[t_key]
                    if t_params.shape[-1] >= 3:
                        mu_t = t_params[..., 0].squeeze(-1)
                        sigma_raw = t_params[..., 1].squeeze(-1)
                        nu_raw = t_params[..., 2].squeeze(-1)
                        loss = loss + self.nll_weight * self._student_t_nll(
                            mu_t, sigma_raw, nu_raw, y
                        )
                # σによる誤差重み（オプション）: λ * E[|e| / σ]
                if self.sigma_weighting_lambda > 0.0 and t_key in predictions:
                    t_params = predictions[t_key]
                    if t_params.shape[-1] >= 2:
                        sigma = (
                            torch.nn.functional.softplus(t_params[..., 1].squeeze(-1))
                            + 1e-6
                        )
                        # Avoid under/overflow in bf16/amp
                        sigma = torch.clamp(sigma, 1e-4, 1e3)
                        abs_err = torch.abs(yhat - y)
                        loss = loss + self.sigma_weighting_lambda * torch.mean(
                            abs_err / sigma
                        )
                # 方向分類の補助損失（BCE with logits）
                if self.direction_aux_weight > 0.0 and self._bce is not None:
                    dir_key = f"direction_horizon_{horizon}"
                    if dir_key in predictions:
                        logits = predictions[dir_key].squeeze(-1)
                        # 小さすぎる値は0とみなす閾値（ノイズ抑制）
                        with torch.no_grad():
                            target_bin = (y > 0.0).float()
                        dir_loss = self._bce(logits, target_bin)
                        loss = loss + self.direction_aux_weight * dir_loss
                losses[f"horizon_{horizon}"] = loss.detach()
                # 集約重み
                if cur_weights is not None and horizon in cur_weights:
                    weight = float(cur_weights[horizon])
                else:
                    weight = 1.0 / np.sqrt(horizon)
                    if int(horizon) == 1 and self.h1_loss_mult != 1.0:
                        weight = weight * self.h1_loss_mult
                total_loss = total_loss + weight * loss
                weights.append(weight)
                contribution_count += 1

                # 動的RMSE更新
                if self.use_dynamic_weighting:
                    with torch.no_grad():
                        # データの鮮度（staleness）を追跡
                        if not hasattr(self, "_staleness_days_list"):
                            self._staleness_days_list = []

                        # 現在のバッチのタイムスタンプを取得
                        if hasattr(self, "current_batch_timestamp"):
                            batch_ts = self.current_batch_timestamp
                        else:
                            # デフォルト値（実際の実装では適切なタイムスタンプを使用）
                            batch_ts = torch.tensor(0.0)

                        # データの鮮度を計算（日数単位）
                        try:
                            # データソースの最終更新時刻を取得
                            if hasattr(self, "data_last_updated"):
                                data_ts = self.data_last_updated
                            else:
                                # デフォルト値（実際の実装では適切なタイムスタンプを使用）
                                data_ts = torch.tensor(0.0)

                            # 鮮度を日数で計算
                            staleness_days = (batch_ts - data_ts).item() / (
                                24 * 3600
                            )  # 秒から日数に変換
                            self._staleness_days_list.append(staleness_days)

                            # 鮮度の統計情報をログ出力
                            if (
                                len(self._staleness_days_list) % 100 == 0
                            ):  # 100バッチごとにログ
                                avg_staleness = np.mean(
                                    self._staleness_days_list[-100:]
                                )
                                max_staleness = np.max(self._staleness_days_list[-100:])
                                logger.info(
                                    f"Data staleness stats (last 100 batches): "
                                    f"avg={avg_staleness:.2f} days, max={max_staleness:.2f} days"
                                )

                        except Exception as e:
                            logger.warning(f"Failed to calculate data staleness: {e}")
                            self._staleness_days_list.append(0.0)

                        rmse = (
                            torch.sqrt(torch.mean((yhat - y) ** 2))
                            .detach()
                            .float()
                            .item()
                        )
                        prev = self._ema_rmse.get(int(horizon), None)
                        if prev is None:
                            self._ema_rmse[int(horizon)] = rmse
                        else:
                            alpha = self.dynamic_alpha
                            self._ema_rmse[int(horizon)] = (
                                1 - alpha
                            ) * prev + alpha * rmse

        if contribution_count > 0:
            total_loss = total_loss / float(np.sum(weights))
            self._warned_empty = False
        else:
            if not self._warned_empty:
                logger.error(
                    "[loss] No matching horizons found in predictions/targets; returning zero loss."
                )
                self._warned_empty = True

        # Ensure scalar output
        total_loss = total_loss.squeeze()
        # ステップ更新（動的重み用）
        self._steps += 1
        return total_loss, losses

    # ===== メトリクス計算 =====
    @staticmethod
    def compute_sharpe_ratio(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Sharpe ratio using portfolio return method"""
        with torch.no_grad():
            # 予測値の符号でポジションを決定（+1: ロング, -1: ショート）
            # 符号反転フラグ（環境変数で制御）
            invert_sign = os.getenv("INVERT_PREDICTION_SIGN", "1") == "1"
            if invert_sign:
                positions = -torch.sign(predictions)  # 符号を反転
            else:
                positions = torch.sign(predictions)
            
            # ポートフォリオリターン = ポジション × 実際のリターン
            portfolio_returns = positions * targets
            
            # Sharpe比 = 平均リターン / リターンの標準偏差
            if len(portfolio_returns) < 2:
                return 0.0
                
            mean_return = portfolio_returns.mean()
            std_return = portfolio_returns.std()
            
            if std_return < 1e-8:
                return 0.0
                
            sharpe = mean_return / std_return
            return sharpe.item()
    
    @staticmethod
    def compute_ic(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Information Coefficient (IC)"""
        with torch.no_grad():
            # Flatten tensors
            pred_flat = predictions.flatten()
            targ_flat = targets.flatten()
            
            # Remove NaN/Inf
            valid_mask = torch.isfinite(pred_flat) & torch.isfinite(targ_flat)
            if valid_mask.sum() < 2:
                return 0.0
            
            pred_valid = pred_flat[valid_mask]
            targ_valid = targ_flat[valid_mask]
            
            # Pearson correlation
            pred_mean = pred_valid.mean()
            targ_mean = targ_valid.mean()
            
            pred_centered = pred_valid - pred_mean
            targ_centered = targ_valid - targ_mean
            
            cov = (pred_centered * targ_centered).mean()
            pred_std = pred_centered.std() + 1e-8
            targ_std = targ_centered.std() + 1e-8
            
            ic = cov / (pred_std * targ_std)
            return ic.item()
    
    @staticmethod
    def compute_rank_ic(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Rank IC (Spearman correlation)"""
        with torch.no_grad():
            # Flatten tensors
            pred_flat = predictions.flatten()
            targ_flat = targets.flatten()
            
            # Remove NaN/Inf
            valid_mask = torch.isfinite(pred_flat) & torch.isfinite(targ_flat)
            if valid_mask.sum() < 2:
                return 0.0
            
            pred_valid = pred_flat[valid_mask]
            targ_valid = targ_flat[valid_mask]
            
            # Compute ranks
            pred_ranks = pred_valid.argsort().argsort().float()
            targ_ranks = targ_valid.argsort().argsort().float()
            
            # Spearman correlation on ranks
            n = pred_ranks.shape[0]
            pred_ranks = (pred_ranks - pred_ranks.mean()) / (pred_ranks.std() + 1e-8)
            targ_ranks = (targ_ranks - targ_ranks.mean()) / (targ_ranks.std() + 1e-8)
            
            rank_ic = (pred_ranks * targ_ranks).mean()
            return rank_ic.item()

    # ===== 重み制御ユーティリティ =====
    def _get_current_weights(self) -> dict | None:
        """現在のエポック/状態に応じたホライズン重みを返す。正規化して合計=1。
        優先順位: curriculum/preset -> 動的(1/RMSE) -> 既定(1/sqrt(h))."""
        # curriculum/preset
        if self._preset_weights:
            return self._preset_weights
        # dynamic weighting（freeze時は固定化）
        if self.use_dynamic_weighting and self._ema_rmse:
            # freeze 判定
            if self._total_epochs and self._current_epoch:
                frac = float(self._current_epoch) / max(1.0, float(self._total_epochs))
                if frac > self.dynamic_freeze_frac:
                    # 一度固定化
                    w = self._inv_rmse_weights(self._ema_rmse)
                    self._preset_weights = w
                    return w
            return self._inv_rmse_weights(self._ema_rmse)
        return None

    @staticmethod
    def _inv_rmse_weights(ema_rmse: dict[int, float]) -> dict[int, float]:
        eps = 1e-8
        inv = {h: 1.0 / (max(eps, r)) for h, r in ema_rmse.items()}
        s = float(sum(inv.values()))
        if s <= 0:
            return {h: 1.0 / max(1, len(inv)) for h in inv}
        return {h: v / s for h, v in inv.items()}

    def set_preset_weights(self, weights: dict[int, float] | None):
        """外部からプリセット重みを設定（合計=1へ正規化）。Noneで解除。"""
        if weights is None:
            self._preset_weights = None
            return
        try:
            s = float(sum(float(v) for v in weights.values()))
            self._preset_weights = {
                int(k): float(v) / (s if s > 0 else 1.0) for k, v in weights.items()
            }
        except Exception:
            self._preset_weights = None

    def set_epoch_context(self, epoch: int, total_epochs: int | None):
        self._current_epoch = int(epoch)
        self._total_epochs = int(total_epochs) if total_epochs is not None else None

    def set_data_timestamps(self, batch_timestamp: float, data_last_updated: float):
        """データのタイムスタンプを設定（データ鮮度計算用）"""
        self.current_batch_timestamp = torch.tensor(batch_timestamp)
        self.data_last_updated = torch.tensor(data_last_updated)
        logger.info(
            f"Data timestamps set: batch={batch_timestamp}, last_updated={data_last_updated}"
        )

    def get_staleness_stats(self) -> dict:
        """データ鮮度の統計情報を取得"""
        if not hasattr(self, "_staleness_days_list") or not self._staleness_days_list:
            return {"error": "No staleness data available"}

        stats = {
            "total_batches": len(self._staleness_days_list),
            "avg_staleness_days": np.mean(self._staleness_days_list),
            "max_staleness_days": np.max(self._staleness_days_list),
            "min_staleness_days": np.min(self._staleness_days_list),
            "std_staleness_days": np.std(self._staleness_days_list),
            "recent_100_avg": np.mean(self._staleness_days_list[-100:])
            if len(self._staleness_days_list) >= 100
            else np.mean(self._staleness_days_list),
        }
        return stats


class SimpleLSTM(nn.Module):
    """フォールバック用のシンプルなLSTMモデル"""

    def __init__(self, input_size=13, hidden_size=512, num_layers=3):
        super().__init__()
        # 可変入力次元を吸収する投影（features[..., F] -> input_size）
        self.lstm_in_dim = int(input_size)
        self.input_proj = nn.LazyLinear(self.lstm_in_dim, bias=False)
        self.lstm = nn.LSTM(
            self.lstm_in_dim, hidden_size, num_layers, batch_first=True, dropout=0.1
        )
        self.fc_dict = nn.ModuleDict(
            {f"horizon_{h}": nn.Linear(hidden_size, 1) for h in [1, 5, 10, 20]}
        )

    def forward(self, x):
        # x: [B,T,F]（Fは可変）→ 射影
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        outputs = {}
        for horizon in [1, 5, 10, 20]:
            outputs[f"horizon_{horizon}"] = self.fc_dict[f"horizon_{horizon}"](
                last_hidden
            ).squeeze(-1)
        return outputs


def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    scaler,
    epoch,
    gradient_accumulation_steps=1,
):
    """1エポックの学習（Mixed Precision対応）"""
    model.train()
    total_loss = 0
    horizon_losses = {f"horizon_{h}": 0 for h in criterion.horizons}
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()

    # 追加: 時系列Mixup設定（短期汎化）
    use_mixup = os.getenv("USE_TS_MIXUP", "0") == "1"
    mixup_prob = float(os.getenv("TS_MIXUP_PROB", "0.2"))
    mixup_alpha = float(os.getenv("TS_MIXUP_ALPHA", "0.2"))

    # AMP local config (env-driven) - BF16 as default for A100
    use_amp = (os.getenv("USE_AMP", "1") == "1") and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if os.getenv("AMP_DTYPE", "bf16").lower() in ("bf16", "bfloat16", "bf16-mixed")
        else torch.float16
    )
    # Autocast device type: avoid CUDA context on CPU-only runs
    _amp_device = "cuda" if (torch.cuda.is_available() and device.type == "cuda") else "cpu"

    # Noise warmup (to prevent early constant collapse)
    try:
        feature_noise_std = float(os.getenv("FEATURE_NOISE_STD", "0.0"))
        output_noise_std = float(os.getenv("OUTPUT_NOISE_STD", "0.0"))
        noise_warmup_epochs = int(os.getenv("NOISE_WARMUP_EPOCHS", "2"))
    except Exception:
        feature_noise_std = 0.0
        output_noise_std = 0.0
        noise_warmup_epochs = 2

    for batch_idx, batch in enumerate(pbar):
        # データをGPUに転送（非同期）
            features = batch["features"].to(device, non_blocking=True)
            # Normalize and reshape targets
            targets = {}
            raw_targets = batch.get("targets", {})
            for k, v in raw_targets.items():
                canon = _normalize_target_key(k)
                if canon is not None:
                    targets[canon] = v.to(device, non_blocking=True)
            
            # Reshape targets to [B] format
            targets = _reshape_to_batch_only(targets)
            
            if not targets:
                try:
                    raw_keys = list(raw_targets.keys())
                except Exception:
                    raw_keys = []
                logger.warning(
                    f"[train-phase] no canonical targets found; raw target keys={raw_keys}"
                )
            # 有効マスク（任意）
            valid_masks = batch.get("valid_mask", None)
            if isinstance(valid_masks, dict):
                valid_masks = {
                    k: v.to(device, non_blocking=True) for k, v in valid_masks.items()
                }
            else:
                valid_masks = None

            # （旧train_epoch経路）GAT融合α下限の設定はrun_training側で実施

            # Mixed Precision Training (最適化設定)
            with torch.amp.autocast(
                _amp_device, dtype=amp_dtype, enabled=use_amp, cache_enabled=False
            ):
                # Forward pass (no channels_last: 3D tensor)
                features = features.contiguous()

                # Optional: add small Gaussian noise to features for warmup
                if (
                    feature_noise_std > 0.0
                    and epoch <= noise_warmup_epochs
                    and model.training
                ):
                    try:
                        features = features + torch.randn_like(features) * feature_noise_std
                    except Exception:
                        pass

                # 予測前の特徴量統計（デバッグ用）
                if batch_idx == 0 and epoch <= 2:
                    feat_mean = features.mean().item()
                    feat_std = features.std().item()
                    logger.debug(
                        f"Input features: mean={feat_mean:.4f}, std={feat_std:.4f}"
                    )

                try:
                    # 時系列Mixup（一定確率）
                    if (
                        use_mixup
                        and np.random.rand() < mixup_prob
                        and features.shape[0] >= 2
                    ):
                        lam = np.random.beta(mixup_alpha, mixup_alpha)
                        # シャッフルインデックス
                        idx = torch.randperm(features.size(0), device=features.device)
                        features = lam * features + (1 - lam) * features[idx]
                        # ターゲットも各ホライズンで混合
                        mixed_targets = {}
                        for k, v in targets.items():
                            v2 = v[idx]
                            mixed_targets[k] = lam * v + (1 - lam) * v2
                        targets = mixed_targets
                    outputs = model(features)
                    
                    # Reshape outputs to [B] format and fix non-finite values
                    outputs = _reshape_to_batch_only(outputs)
                    
                    # Optional: add small Gaussian noise to outputs for warmup (point heads only)
                    if (
                        output_noise_std > 0.0
                        and epoch <= noise_warmup_epochs
                        and model.training
                    ):
                        try:
                            for k in list(
                                outputs.keys() if isinstance(outputs, dict) else []
                            ):
                                if k.startswith("point_horizon_") and torch.is_tensor(
                                    outputs[k]
                                ):
                                    outputs[k] = (
                                        outputs[k]
                                        + torch.randn_like(outputs[k]) * output_noise_std
                                    )
                        except Exception:
                            pass
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    del features
                    if "targets" in locals():
                        del targets
                    gc.collect()
                    torch.cuda.empty_cache()
                    raise

            # Loss computation in FP32 for stability
            with torch.amp.autocast(_amp_device, enabled=False):
                # Ensure all outputs and targets are FP32 and properly shaped
                outputs_fp32 = _reshape_to_batch_only({
                    k: (v.float() if torch.is_tensor(v) else v) for k, v in outputs.items()
                })
                targets_fp32 = _reshape_to_batch_only({
                    k: (v.float() if torch.is_tensor(v) else v) for k, v in targets.items()
                })
                if isinstance(outputs_fp32, dict) and isinstance(outputs_fp32.get('predictions'), dict):
                    predictions_fp32 = outputs_fp32['predictions']
                else:
                    predictions_fp32 = outputs_fp32


                # マスク付きマルチホライゾン損失
                loss_result = criterion(
                    predictions_fp32, targets_fp32, valid_masks=valid_masks
                )
                
                # Handle both single value and tuple return
                if isinstance(loss_result, tuple):
                    loss, losses = loss_result
                else:
                    loss = loss_result
                    losses = {}

                if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
                    logger.error(
                        "Loss tensor detached: type=%s requires_grad=%s pred_keys=%s target_keys=%s",
                        type(loss),
                        getattr(loss, "requires_grad", None),
                        sorted(list(predictions_fp32.keys()))
                        if isinstance(predictions_fp32, dict)
                        else type(predictions_fp32).__name__,
                        sorted(list(targets_fp32.keys()))
                        if isinstance(targets_fp32, dict)
                        else type(targets_fp32).__name__,
                    )

                # 追加: 各ホライゾンのvalid比率を低頻度でログ
                if (batch_idx % 200 == 0) and isinstance(valid_masks, dict):
                    try:
                        ratios = {
                            k: float(v.float().mean().item())
                            for k, v in valid_masks.items()
                            if torch.is_tensor(v)
                        }
                        logger.info(
                            f"[valid-ratio] { {k: f'{r:.2%}' for k,r in ratios.items()} }"
                        )
                    except Exception:
                        pass

                # Variance penalty to prevent collapse
                variance_penalty = 0.0
                for h in [1, 5, 10, 20]:
                    key = f"point_horizon_{h}"
                    if key in predictions_fp32:
                        pred_std = predictions_fp32[key].std()
                        # Penalize if std is too small (collapse)
                        if pred_std.item() < 0.1:
                            variance_penalty += (0.1 - pred_std) * 0.1

                # Add penalty to loss
                if variance_penalty > 0:
                    loss = loss + variance_penalty

                # 予測値の統計チェック（最初のエポック）
                if batch_idx == 0 and epoch <= 2:
                    for h in criterion.horizons:
                        key = f"point_horizon_{h}"
                        pred_dict = outputs.get('predictions', outputs)
                        if isinstance(pred_dict, dict) and key in pred_dict:
                            pred = pred_dict[key].detach()
                            pred_mean = pred.mean().item()
                            pred_std = pred.std().item()
                            logger.debug(
                                f"Predictions h={h}: mean={pred_mean:.4f}, std={pred_std:.6f}"
                            )

            # 1バッチ目のみデバッグ（形状と平均損失） - 整形・安定化済み
            if batch_idx == 0 and epoch == 1:
                try:
                    logger.info(f"criterion reduction: {criterion.mse.reduction}")
                    
                    # Use FP32 shaped outputs/targets for debugging
                    stable_outputs = _reshape_to_batch_only(predictions_fp32)
                    stable_targets = _reshape_to_batch_only(targets_fp32)
                    
                    for h in criterion.horizons:
                        pred_key = (
                            f"point_horizon_{h}"
                            if any(k.startswith("point_horizon_") for k in stable_outputs.keys())
                            else f"horizon_{h}"
                        )
                        targ_key = f"horizon_{h}"
                        if pred_key in stable_outputs and targ_key in stable_targets:
                            pred_t = stable_outputs[pred_key]
                            targ_t = stable_targets[targ_key]
                            
                            # Ensure both are [B] format
                            if pred_t.dim() > 1:
                                pred_t = pred_t.squeeze()
                            if targ_t.dim() > 1:
                                targ_t = targ_t.squeeze()
                                
                            # Fix non-finite values before computing MSE
                            pred_t = _finite_or_nan_fix_tensor(pred_t)
                            targ_t = _finite_or_nan_fix_tensor(targ_t)
                            
                            mse_h = torch.nn.functional.mse_loss(
                                pred_t, targ_t, reduction="mean"
                            ).detach()
                            logger.info(
                                f"h={h} pred_shape={tuple(pred_t.shape)} targ_shape={tuple(targ_t.shape)} mse={mse_h.item():.6f}"
                            )
                except Exception as _e:
                    logger.warning(f"debug logging failed: {_e}")

            # Backward pass
            scaler.scale(loss / gradient_accumulation_steps).backward()

            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)  # メモリ効率向上

            # 統計更新
            total_loss += loss.item()
            for k, v in losses.items():
                horizon_losses[k] += v.item()
            n_batches += 1

            # プログレスバー更新
            if batch_idx % 10 == 0:
                avg_loss = total_loss / n_batches
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

            # メモリ管理（100バッチごと）
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()

    # エポック平均
    avg_loss = total_loss / n_batches
    avg_horizon_losses = {k: v / n_batches for k, v in horizon_losses.items()}

    return avg_loss, avg_horizon_losses


def first_batch_probe(model, dataloader, device, n=3):
    """First batch validation to catch early failures"""
    model.eval()
    logger.info("Running first-batch probe...")

    # AMP local config
    use_amp = (os.getenv("USE_AMP", "1") == "1") and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if os.getenv("AMP_DTYPE", "").lower() in ("bf16", "bfloat16", "bf16-mixed")
        else torch.float16
    )
    _amp_device = "cuda" if (torch.cuda.is_available() and device.type == "cuda") else "cpu"
    _amp_device = "cuda" if (torch.cuda.is_available() and device.type == "cuda") else "cpu"
    _amp_device = "cuda" if (torch.cuda.is_available() and device.type == "cuda") else "cpu"

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n:
                break

            features = batch["features"].to(device, non_blocking=True)
            logger.info(
                f"Batch {i}: features shape={features.shape}, dtype={features.dtype}"
            )

            try:
                with torch.amp.autocast(
                    "cuda", dtype=amp_dtype, enabled=use_amp, cache_enabled=False
                ):
                    outputs = model(features.contiguous())

                # Sanitize and check outputs
                if isinstance(outputs, dict):
                    for k, v in list(outputs.items()):
                        if torch.is_tensor(v):
                            v = _finite_or_nan_fix_tensor(v, f"outputs[probe][{k}]", clamp=50.0)
                            # If sequence-shaped, take the last step
                            if v.dim() >= 2 and v.shape[-1] != 1:
                                v = v[..., -1]
                            elif v.dim() >= 2 and v.shape[-1] == 1:
                                v = v.squeeze(-1)
                            outputs[k] = v
                            assert torch.isfinite(v).all(), f"Non-finite output in {k}"
                            logger.info(
                                f"  Output {k}: shape={tuple(v.shape)}, dtype={v.dtype}"
                            )
                else:
                    outputs = _finite_or_nan_fix_tensor(outputs, "outputs[probe]", clamp=50.0)
                    if outputs.dim() >= 2 and outputs.shape[-1] != 1:
                        outputs = outputs[..., -1]
                    elif outputs.dim() >= 2 and outputs.shape[-1] == 1:
                        outputs = outputs.squeeze(-1)
                    assert torch.isfinite(outputs).all(), "Non-finite output detected"

            except Exception as e:
                logger.error(f"First-batch probe failed on batch {i}: {e}")
                raise

    model.train()
    logger.info("✓ First-batch probe passed")

    # デバッグ: 最初のバッチの予測値を詳細に出力
    if os.getenv("DEBUG_PREDICTIONS", "0") == "1":
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i > 0:
                    break
                features = batch["features"].to(device)

                # フックを使って中間出力を取得
                intermediate_outputs = {}

                def get_hook(name):
                    def hook(module, input, output):
                        if isinstance(output, torch.Tensor):
                            intermediate_outputs[name] = {
                                "mean": output.mean().item(),
                                "std": output.std().item(),
                                "shape": output.shape,
                                "min": output.min().item(),
                                "max": output.max().item(),
                            }
                        elif (
                            isinstance(output, tuple)
                            and len(output) > 0
                            and isinstance(output[0], torch.Tensor)
                        ):
                            intermediate_outputs[name] = {
                                "mean": output[0].mean().item(),
                                "std": output[0].std().item(),
                                "shape": output[0].shape,
                                "min": output[0].min().item(),
                                "max": output[0].max().item(),
                            }

                    return hook

                # フックを登録
                hooks = []
                if hasattr(model, "fan"):
                    hooks.append(model.fan.register_forward_hook(get_hook("after_fan")))
                if hasattr(model, "tft"):
                    hooks.append(model.tft.register_forward_hook(get_hook("after_tft")))
                if hasattr(model, "prediction_mlp"):
                    hooks.append(
                        model.prediction_mlp.register_forward_hook(
                            get_hook("after_mlp")
                        )
                    )

                outputs = model(features)

                # 中間出力をログ
                if intermediate_outputs:
                    logger.info("\n=== DEBUG: Intermediate outputs ===")
                    for name, stats in intermediate_outputs.items():
                        logger.info(
                            f"{name}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
                            + f"range=[{stats['min']:.6f}, {stats['max']:.6f}], shape={stats['shape']}"
                        )

                # フックを削除
                for hook in hooks:
                    hook.remove()

                logger.info("\n=== DEBUG: Prediction values ===")
                for h in [1, 5, 10, 20]:
                    key = f"point_horizon_{h}"
                    if key in outputs:
                        pred = outputs[key].cpu().numpy()
                        logger.info(f"Horizon {h}:")
                        logger.info(f"  Mean: {pred.mean():.6f}, Std: {pred.std():.6f}")
                        logger.info(f"  Min: {pred.min():.6f}, Max: {pred.max():.6f}")
                        logger.info(f"  First 5 values: {pred.flatten()[:5]}")
                        unique = np.unique(pred.round(decimals=6))
                        logger.info(f"  Unique values count: {len(unique)}")
                        if len(unique) <= 5:
                            logger.info(f"  Unique values: {unique}")

                # 入力データも確認
                logger.info("\n=== DEBUG: Input features ===")
                feat_np = features[0].cpu().numpy()  # 最初のサンプル
                logger.info(f"First sample shape: {feat_np.shape}")
                logger.info(f"Mean: {feat_np.mean():.6f}, Std: {feat_np.std():.6f}")
                logger.info(f"Range: [{feat_np.min():.6f}, {feat_np.max():.6f}]")

                # 各特徴量の統計
                for j in range(feat_np.shape[-1]):
                    feat_j = feat_np[:, j]
                    logger.info(
                        f"Feature {j}: mean={feat_j.mean():.6f}, std={feat_j.std():.6f}, range=[{feat_j.min():.6f}, {feat_j.max():.6f}]"
                    )
        model.train()


def evaluate_quick(model, dataloader, criterion, device, max_batches=50):
    """Quick evaluation for intermediate checkpoints (subset only)"""
    was_training = model.training
    model.eval()

    total_loss = 0.0
    n_batches = 0
    total_samples = 0
    # Collect simple diagnostics
    yhat_std_lists: dict[int, list[float]] = {
        int(h): [] for h in getattr(criterion, "horizons", [])
    }

    # AMP local config（validate と揃える）
    use_amp = (os.getenv("USE_AMP", "1") == "1") and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if os.getenv("AMP_DTYPE", "").lower() in ("bf16", "bfloat16", "bf16-mixed")
        else torch.float16
    )

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break

                features = batch["features"].to(device, non_blocking=True)
                targets = batch.get("targets")
                valid_masks = batch.get("valid_mask", None)

                # 数値安定化（validate 相当）
                features = _finite_or_nan_fix_tensor(
                    features, "features[quick]", clamp=50.0
                )

                # Check for valid data ratio in evaluation
                eval_min_valid = float(
                    os.getenv("EVAL_MIN_VALID_RATIO", "0.5")
                )  # Lower threshold for quick eval
                skip_batch = False

                if isinstance(targets, dict):
                    tmp = {}
                    valid_ratios = []
                    for k, v in targets.items():
                        v = v.to(device, non_blocking=True)

                        # Use provided mask or compute it
                        if valid_masks and k in valid_masks:
                            valid_mask = valid_masks[k].to(device, non_blocking=True)
                        else:
                            valid_mask = torch.isfinite(v)

                        valid_ratio = valid_mask.float().mean().item()
                        valid_ratios.append(valid_ratio)
                        if valid_ratio < eval_min_valid:
                            logger.warning(
                                f"[quick-skip] {k}: valid={valid_ratio:.2%} < {eval_min_valid:.2%}"
                            )

                        # Replace NaN/Inf with 0 (will be masked in loss)
                        tmp[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                    targets = tmp
                    # Skip batch if average valid ratio is too low
                    if (
                        valid_ratios
                        and sum(valid_ratios) / len(valid_ratios) < eval_min_valid
                    ):
                        skip_batch = True
                elif targets is not None:
                    targets = torch.nan_to_num(
                        targets.to(device, non_blocking=True),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )

                if skip_batch:
                    continue

                with torch.amp.autocast(
                    "cuda", dtype=amp_dtype, enabled=use_amp, cache_enabled=False
                ):
                    outputs = model(features.contiguous())
                    if isinstance(outputs, dict):
                        for k, v in outputs.items():
                            if torch.is_tensor(v):
                                outputs[k] = _finite_or_nan_fix_tensor(
                                    v, f"outputs[quick][{k}]", clamp=50.0
                                )
                    # Quick diagnostics: collect per-horizon std of predictions
                    # Always use point predictions for std calculation, never t-params
                    try:
                        if compute_pred_std_batch is not None:
                            for h in getattr(criterion, "horizons", []):
                                hk = int(h)
                                std_val = compute_pred_std_batch(outputs, hk)
                                if std_val > 0 and std_val < 1e6:  # valid check
                                    yhat_std_lists[hk].append(std_val)
                        else:
                            # Fallback to inline computation
                            for h in getattr(criterion, "horizons", []):
                                hk = int(h)
                                pred_key = (
                                    f"point_horizon_{hk}"
                                    if any(
                                        k.startswith("point_horizon_")
                                        for k in outputs.keys()
                                    )
                                    else f"horizon_{hk}"
                                )
                                if isinstance(outputs, dict) and pred_key in outputs:
                                    yhat = outputs[pred_key].detach().float().view(-1)
                                    std_val = float(yhat.std(unbiased=False).item())
                                    if (
                                        std_val == std_val
                                        and std_val >= 0
                                        and std_val < 1e6
                                    ):  # finite and non-negative check
                                        yhat_std_lists[hk].append(std_val)
                    except Exception:
                        pass
                    # Ensure valid masks are on the same device
                    if isinstance(valid_masks, dict):
                        valid_masks_device = {
                            mk: mv.to(device, non_blocking=True)
                            for mk, mv in valid_masks.items()
                        }
                    else:
                        valid_masks_device = None
                    crit_out = criterion(
                        outputs, targets, valid_masks=valid_masks_device
                    )
                    loss = crit_out[0] if isinstance(crit_out, tuple) else crit_out

                # Sample-weighted accumulation to avoid batch-size bias
                bs = int(features.size(0)) if hasattr(features, "size") else 1
                total_loss += bs * float(
                    loss.item() if hasattr(loss, "item") else float(loss)
                )
                n_batches += 1
                total_samples += bs
    finally:
        # 例外の有無に関わらず元の状態へ復帰
        if was_training:
            model.train()

    # Prefer sample-weighted averaging; fallback to batch avg
    denom = total_samples if total_samples > 0 else n_batches
    avg_loss = total_loss / max(denom, 1)

    # Log reason if no valid batches or unusually high loss
    if n_batches == 0:
        logger.warning(
            "[quick-eval] No valid batches processed (all skipped due to low valid ratio)"
        )
        avg_loss = 100.0  # Sentinel value
    elif avg_loss > 50.0:
        logger.warning(
            f"[quick-eval] Unusually high loss: {avg_loss:.2f} (possible numerical issue)"
        )

    # Summarize diagnostics
    diag = {"val_loss": avg_loss, "n_batches": n_batches}
    try:
        for h, vals in yhat_std_lists.items():
            if vals:
                diag[f"yhat_std_h{h}"] = float(sum(vals) / len(vals))
    except Exception:
        pass
    return diag


def evaluate_model_metrics(
    model, val_loader, criterion, device, target_scalers=None, max_batches=None
):
    """モデルの詳細メトリクスを評価する関数"""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_predictions = {}
    all_targets = {}

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_batches is not None and i >= max_batches:
                break

            features = batch["features"].to(device)
            
            # Normalize and reshape targets for validation
            raw_targets = batch.get("targets", {})
            targets = {}
            for k, v in raw_targets.items():
                canon = _normalize_target_key(k)
                if canon is not None:
                    targets[canon] = v.to(device, non_blocking=True)
            
            # Reshape targets to [B] format
            targets = _reshape_to_batch_only(targets)
            
            if not targets:
                logger.warning(f"[val-phase] no canonical targets found; raw target keys={list(raw_targets.keys())}")
                continue

            outputs = model(features)
            
            # Reshape outputs to [B] format and fix non-finite values
            outputs = _reshape_to_batch_only(outputs)
            
            # Ensure FP32 and proper shaping for loss computation
            outputs_fp32 = _reshape_to_batch_only({
                k: (v.float() if torch.is_tensor(v) else v) for k, v in outputs.items()
            })
            targets_fp32 = _reshape_to_batch_only({
                k: (v.float() if torch.is_tensor(v) else v) for k, v in targets.items()
            })
            
            loss_result = criterion(outputs_fp32, targets_fp32)
            
            # Handle both single value and tuple return
            if isinstance(loss_result, tuple):
                loss, _ = loss_result
            else:
                loss = loss_result

            try:
                total_loss += float(loss.item() if hasattr(loss, "item") else float(loss))
            except Exception:
                pass
            n_batches += 1

            # 予測とターゲットを保存
            for horizon in criterion.horizons:
                pred_key = f"horizon_{horizon}"
                if pred_key in outputs:
                    if horizon not in all_predictions:
                        all_predictions[horizon] = []
                        all_targets[horizon] = []
                    all_predictions[horizon].append(outputs[pred_key].cpu())
                    all_targets[horizon].append(targets[pred_key].cpu())

    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0

    # 予測とターゲットを結合
    for horizon in all_predictions.keys():
        all_predictions[horizon] = torch.cat(all_predictions[horizon], dim=0)
        all_targets[horizon] = torch.cat(all_targets[horizon], dim=0)

    # メトリクス計算
    metrics = {"val_loss": avg_loss, "horizon_metrics": {}}

    for horizon in all_predictions.keys():
        pred = all_predictions[horizon].numpy()
        target = all_targets[horizon].numpy()

        # 基本的なメトリクス
        mse = np.mean((pred - target) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(pred - target))

        # 相関係数
        correlation = np.corrcoef(pred.flatten(), target.flatten())[0, 1]
        if np.isnan(correlation):
            correlation = 0.0

        # R²スコア
        ss_res = np.sum((target - pred) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        # シャープレシオ（金融指標）
        returns = pred - target
        eps = 0.0
        try:
            eps = float(os.getenv("SHARPE_EPS", "0.0"))
        except Exception:
            eps = 0.0
        sd = np.std(returns)
        sharpe_ratio = (np.mean(returns) / (sd + eps)) if (sd + eps) > 0 else 0.0

        metrics["horizon_metrics"][horizon] = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "correlation": correlation,
            "r2": r2,
            "sharpe_ratio": sharpe_ratio,
            "mean_prediction": np.mean(pred),
            "std_prediction": np.std(pred),
            "mean_target": np.mean(target),
            "std_target": np.std(target),
        }

    # 全体の平均メトリクス
    avg_metrics = {}
    metric_names = ["mse", "rmse", "mae", "correlation", "r2", "sharpe_ratio"]
    for metric in metric_names:
        values = [
            metrics["horizon_metrics"][h][metric]
            for h in metrics["horizon_metrics"].keys()
        ]
        avg_metrics[f"avg_{metric}"] = np.mean(values)

    metrics["average_metrics"] = avg_metrics

    # ログ出力
    logger.info("Validation Metrics Summary:")
    logger.info(f"  Average RMSE: {avg_metrics['avg_rmse']:.4f}")
    logger.info(f"  Average R²: {avg_metrics['avg_r2']:.4f}")
    logger.info(f"  Average Sharpe Ratio: {avg_metrics['avg_sharpe_ratio']:.4f}")
    # Parser-friendly single-line Sharpe for external pipelines
    try:
        sharpe_line = f"Sharpe: {avg_metrics['avg_sharpe_ratio']:.4f}"
        logger.info(sharpe_line)
    except Exception:
        pass

    return metrics


def validate(model, dataloader, criterion, device):
    """検証"""
    model.eval()
    total_loss = 0
    horizon_losses = {f"horizon_{h}": 0 for h in criterion.horizons}
    n_batches = 0
    # 追加指標（tパラメータ/分位も収集）
    metrics = {
        h: {"y": [], "yhat": [], "t_params": [], "quantiles": []}
        for h in criterion.horizons
    }
    # 保存用: 線形校正係数（z空間）
    linear_calibration = {h: {"a": 0.0, "b": 1.0} for h in criterion.horizons}

    # AMP local config
    use_amp = (os.getenv("USE_AMP", "1") == "1") and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if os.getenv("AMP_DTYPE", "").lower() in ("bf16", "bfloat16", "bf16-mixed")
        else torch.float16
    )
    _amp_device = "cuda" if (torch.cuda.is_available() and device.type == "cuda") else "cpu"

    with torch.no_grad():
        low_var_warns_h1 = 0
        for batch in tqdm(dataloader, desc="Validation"):
            features = batch["features"].to(device, non_blocking=True)
            # Canonicalize validation targets as well
            targets = {}
            for k, v in batch.get("targets", {}).items():
                canon = _canonicalize_target_key(k)
                if canon is not None:
                    targets[canon] = v.to(device, non_blocking=True)
            # Move valid masks to correct device if present
            valid_masks = batch.get("valid_mask", None)
            if isinstance(valid_masks, dict):
                valid_masks = {
                    k: v.to(device, non_blocking=True) for k, v in valid_masks.items()
                }
            else:
                valid_masks = None
            features = _finite_or_nan_fix_tensor(features, "features[val]", clamp=50.0)
            for k in list(targets.keys()):
                targets[k] = _finite_or_nan_fix_tensor(
                    targets[k], f"targets[val][{k}]", clamp=50.0
                )

            with torch.amp.autocast(
                _amp_device, dtype=amp_dtype, enabled=use_amp, cache_enabled=False
            ):
                features = features.contiguous()
                try:
                    outputs = model(features)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    del features
                    if "targets" in locals():
                        del targets
                    gc.collect()
                    torch.cuda.empty_cache()
                    raise
                if isinstance(outputs, dict):
                    for k, v in outputs.items():
                        if torch.is_tensor(v):
                            outputs[k] = _finite_or_nan_fix_tensor(
                                v, f"outputs[val][{k}]", clamp=50.0
                            )

                # Reshape to [B] format for consistency with training path
                outputs = _reshape_to_batch_only({
                    k: (v.float() if torch.is_tensor(v) else v) for k, v in outputs.items()
                })
                targets = _reshape_to_batch_only({
                    k: (v.float() if torch.is_tensor(v) else v) for k, v in targets.items()
                })

                loss_result = criterion(outputs, targets, valid_masks=valid_masks)
                
                # Handle both single value and tuple return
                if isinstance(loss_result, tuple):
                    loss, losses = loss_result
                else:
                    loss = loss_result
                    losses = {}

            total_loss += loss.item()
            for k, v in losses.items():
                # detach済みを想定
                horizon_losses[k] += v.item() if hasattr(v, "item") else float(v)
            n_batches += 1

            # 追加指標収集（CPUへ）
            if collect_metrics_from_outputs is not None:
                batch_metrics = collect_metrics_from_outputs(
                    outputs, targets, criterion.horizons
                )
                for h in criterion.horizons:
                    for key in ["y", "yhat", "t_params", "quantiles"]:
                        if batch_metrics[h][key]:
                            metrics[h][key].extend(batch_metrics[h][key])
            else:
                # Fallback to inline collection
                for h in criterion.horizons:
                    pred_key = (
                        f"point_horizon_{h}"
                        if any(k.startswith("point_horizon_") for k in outputs.keys())
                        else f"horizon_{h}"
                    )
                    targ_key = f"horizon_{h}"
                    if pred_key in outputs and targ_key in targets:
                        yhat = outputs[pred_key].detach().float().view(-1).cpu().numpy()
                        y = targets[targ_key].detach().float().view(-1).cpu().numpy()
                        metrics[h]["yhat"].append(yhat)
                        metrics[h]["y"].append(y)
                    # t-params 収集
                    t_key = f"t_params_horizon_{h}"
                    if t_key in outputs:
                        metrics[h]["t_params"].append(
                            outputs[t_key].detach().float().cpu().numpy()
                        )
                    # quantiles 収集
                    q_key = f"quantile_horizon_{h}"
                    if q_key in outputs:
                        metrics[h]["quantiles"].append(
                            outputs[q_key].detach().float().cpu().numpy()
                        )

    # Check for empty validation set
    if n_batches == 0:
        logger.warning("No validation batches found. Skipping validation metrics.")
        # Return infinity loss to indicate validation failed
        return (
            float("inf"),
            {f"horizon_{h}": float("inf") for h in criterion.horizons},
            linear_calibration,
        )

    avg_loss = total_loss / n_batches
    avg_horizon_losses = {k: v / n_batches for k, v in horizon_losses.items()}

    # Horizon別 MAE/RMSE/R^2/IC（Spearman近似）/NAIVE_RMSE +（あれば）t-NLL/CRPS近似/被覆率
    try:
        import os as _os

        eval_space = _os.getenv("EVAL_SPACE", "z").lower()  # 'z' | 'raw' | 'both'
        # dataloader -> dataset -> target_scalers（学習fit済みのものを再利用）
        scalers = (
            getattr(getattr(dataloader, "dataset", None), "target_scalers", {}) or {}
        )
        for h in criterion.horizons:
            if metrics[h]["y"]:
                import numpy as np

                y = np.concatenate(metrics[h]["y"])
                yhat = np.concatenate(metrics[h]["yhat"])
                # 線形校正（z空間）: y ≈ a + b*yhat
                try:
                    var_yhat = float(np.var(yhat) + 1e-12)
                    cov = float(np.mean((yhat - yhat.mean()) * (y - y.mean())))
                    b = cov / var_yhat
                    a = float(y.mean() - b * yhat.mean())
                    # 数値安定化（bが極端に小/大ならスキップ）
                    if np.isfinite(a) and np.isfinite(b) and 0.01 <= abs(b) <= 100.0:
                        linear_calibration[h] = {"a": float(a), "b": float(b)}
                except Exception:
                    pass
                mae = float(np.mean(np.abs(yhat - y)))
                rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
                var = float(np.var(y)) + 1e-12
                r2 = float(1.0 - np.var(y - yhat) / var)
                # スケール比（単位不一致の検知補助）
                y_std = float(np.std(y) + 1e-12)
                yhat_std = float(np.std(yhat) + 1e-12)
                scale_ratio = float(yhat_std / y_std) if y_std > 0 else float("nan")

                # 予測が collapse している場合の警告
                if scale_ratio < 0.1:
                    logger.warning(
                        f"Low prediction variance for horizon {h}: scale_ratio={scale_ratio:.4f}, yhat_std={yhat_std:.6f}"
                    )
                    if h == 1:
                        low_var_warns_h1 += 1

                # Spearman 近似（単純順位）
                def rank_simple(a: np.ndarray) -> np.ndarray:
                    order = np.argsort(a)
                    ranks = np.empty_like(order, dtype=float)
                    ranks[order] = np.arange(len(a), dtype=float)
                    return ranks

                ranks_yhat = rank_simple(yhat)
                ranks_y = rank_simple(y)
                ic = (
                    float(np.corrcoef(ranks_yhat, ranks_y)[0, 1])
                    if len(y) > 1
                    else float("nan")
                )
                # Log IC@h1 and EMA to MLflow if available
                if h == 1 and np.isfinite(ic):
                    try:
                        import mlflow as _mlf  # type: ignore

                        _mlf.log_metric("val/IC_h1", float(ic))
                        global IC_EMA_H1
                        beta = float(os.getenv("IC_EMA_BETA", "0.9"))
                        if IC_EMA_H1 is None or not np.isfinite(IC_EMA_H1):
                            IC_EMA_H1 = float(ic)
                        else:
                            IC_EMA_H1 = float(beta * IC_EMA_H1 + (1.0 - beta) * ic)
                        _mlf.log_metric("val/IC_EMA_h1", float(IC_EMA_H1))
                        # Regime-tagged IC metric (if regime name provided)
                        reg = os.getenv("CURRENT_REGIME_NAME", "").strip()
                        if reg:
                            safe = "".join(
                                ch if ch.isalnum() or ch in "-_" else "_" for ch in reg
                            )
                            _mlf.log_metric(f"val/IC_h1_regime_{safe}", float(ic))
                    except Exception:
                        pass
                # ナイーブ（ゼロ予測）RMSE
                naive_rmse = float(np.sqrt(np.mean(y**2)))
                extra = f" SCALE(yhat/y)={scale_ratio:.2f}"
                # t-NLL
                if metrics[h]["t_params"] and getattr(criterion, "use_t_nll", False):
                    t_all = np.concatenate(metrics[h]["t_params"], axis=0)
                    if t_all.shape[-1] >= 3:
                        mu = t_all[..., 0].reshape(-1)
                        sigma_raw = t_all[..., 1].reshape(-1)
                        nu_raw = t_all[..., 2].reshape(-1)
                        sigma = np.log1p(np.exp(sigma_raw)) + 1e-6
                        nu = 3.0 + np.log1p(np.exp(nu_raw))
                        z = (y - mu) / sigma
                        logp = (
                            -0.5 * np.log(nu * np.pi)
                            - np.log(sigma)
                            - 0.5 * (nu + 1.0) * np.log1p((z * z) / nu)
                        )
                        nll = float(-np.mean(logp))
                        extra += f" NLL_t={nll:.4f}"
                        # Variance floor hits (if PRED_STD_MIN is set)
                        try:
                            std_min = float(os.getenv("PRED_STD_MIN", "nan"))
                            if np.isfinite(std_min):
                                hits = float(np.mean(sigma <= std_min))
                                if h == 1:
                                    try:
                                        import mlflow as _mlf  # type: ignore

                                        _mlf.log_metric(
                                            "val/variance_floor_hits_h1", hits
                                        )
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                # CRPS近似と被覆率（PI90/PI95）
                if metrics[h]["quantiles"]:
                    q_all = np.concatenate(metrics[h]["quantiles"], axis=0)
                    qs = np.array(
                        getattr(criterion, "quantiles", [0.1, 0.25, 0.5, 0.75, 0.9]),
                        dtype=float,
                    )
                    if q_all.ndim == 2 and q_all.shape[1] == len(qs):
                        y_expand = y.reshape(-1, 1)
                        e = y_expand - q_all
                        pinball = np.maximum(qs * e, (qs - 1.0) * e)
                        crps_approx = float(np.mean(pinball))

                        # 近似PI90/95（最も近い分位で近似）
                        def coverage_for(alpha):
                            try:
                                low_q = (1 - alpha) / 2.0
                                high_q = (1 + alpha) / 2.0
                                li = int(np.argmin(np.abs(qs - low_q)))
                                hi = int(np.argmin(np.abs(qs - high_q)))
                                q_low, q_high = q_all[:, li], q_all[:, hi]
                                return float(np.mean((y >= q_low) & (y <= q_high)))
                            except Exception:
                                return float("nan")

                        cov90 = coverage_for(0.90)
                        cov95 = coverage_for(0.95)
                        extra += f" CRPS~={crps_approx:.4f} COV90={cov90:.3f} COV95={cov95:.3f}"
                logging.info(
                    f"Val metrics (z) h={h}: MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f} IC={ic:.4f} NAIVE_RMSE={naive_rmse:.4f}{extra} CAL={linear_calibration[h]['a']:+.3f}+{linear_calibration[h]['b']:.3f}*yhat"
                )

                # 追加: 校正後メトリクス（任意）
                try:
                    if _os.getenv("EVAL_CALIBRATED", "0") == "1":
                        a_cal = float(linear_calibration[h].get("a", 0.0))
                        b_cal = float(linear_calibration[h].get("b", 1.0))
                        yhat_cal = a_cal + b_cal * yhat
                        mae_c = float(np.mean(np.abs(yhat_cal - y)))
                        rmse_c = float(np.sqrt(np.mean((yhat_cal - y) ** 2)))
                        r2_c = float(1.0 - np.var(y - yhat_cal) / var)
                        yhat_std_c = float(np.std(yhat_cal) + 1e-12)
                        scale_ratio_c = (
                            float(yhat_std_c / y_std) if y_std > 0 else float("nan")
                        )
                        logging.info(
                            f"Val metrics (z, CAL) h={h}: MAE={mae_c:.4f} RMSE={rmse_c:.4f} R2={r2_c:.4f} SCALE_CAL={scale_ratio_c:.2f}"
                        )
                except Exception as _e:
                    logging.warning(f"Calibrated metric calc failed for h={h}: {_e}")
                if scale_ratio > 5.0:
                    logging.warning(
                        f"Scale mismatch suspected (h={h}): yhat/y std ratio={scale_ratio:.2f}. Ensure consistent target normalization."
                    )
                # raw空間での指標（必要に応じて）
                if eval_space in ("raw", "both") and h in scalers:
                    try:
                        m = float(scalers[h].get("mean", 0.0))
                        s = float(scalers[h].get("std", 1.0))
                        y_raw = y * s + m
                        yhat_raw = yhat * s + m
                        mae_r = float(np.mean(np.abs(yhat_raw - y_raw)))
                        rmse_r = float(np.sqrt(np.mean((yhat_raw - y_raw) ** 2)))
                        var_r = float(np.var(y_raw) + 1e-12)
                        r2_r = float(1.0 - np.var(y_raw - yhat_raw) / var_r)
                        naive_r = float(np.sqrt(np.mean((y_raw) ** 2)))
                        logging.info(
                            f"Val metrics (raw) h={h}: MAE={mae_r:.4f} RMSE={rmse_r:.4f} R2={r2_r:.4f} NAIVE_RMSE={naive_r:.4f}"
                        )
                    except Exception as _e:
                        logging.warning(f"Raw metric calc failed for h={h}: {_e}")
    except Exception as e:
        logging.warning(f"Val metric calc failed: {e}")

    # Log low variance warnings count (h=1) to MLflow
    try:
        import mlflow as _mlf  # type: ignore

        _mlf.log_metric("val/low_variance_warnings", int(low_var_warns_h1))
    except Exception:
        pass

    return avg_loss, avg_horizon_losses, linear_calibration


def run_phase_training(model, train_loader, val_loader, config, device):
    """Phase Training実行 (A+アプローチ)"""
    import torch.nn as nn
    import torch.optim as optim
    
    logger.info("=" * 80)
    logger.info("Starting Phase Training (A+ Approach)")
    logger.info("=" * 80)
    
    # Phase定義
    phase_epochs_override = int(os.getenv("PHASE_MAX_EPOCHS", "0"))
    max_batches_per_epoch = int(os.getenv("PHASE_MAX_BATCHES", "100"))
    phases = [
        {
            "name": "Phase 0: Baseline",
            "epochs": phase_epochs_override or 5,
            "toggles": {"use_fan": False, "use_san": False, "use_gat": False},
            "loss_weights": {"quantile": 1.0, "sharpe": 0.0, "corr": 0.0},
            "lr": 5e-4,
            "grad_clip": 1.0
        },
        {
            "name": "Phase 1: Adaptive Norm",
            "epochs": phase_epochs_override or 10,
            "toggles": {"use_fan": True, "use_san": True, "use_gat": False},
            "loss_weights": {"quantile": 1.0, "sharpe": 0.1, "corr": 0.0},
            "lr": 5e-4,
            "grad_clip": 1.0
        },
        {
            "name": "Phase 2: GAT",
            "epochs": phase_epochs_override or 20,
            "toggles": {"use_fan": True, "use_san": True, "use_gat": True},
            "loss_weights": {"quantile": 1.0, "sharpe": 0.1, "corr": 0.05},
            "lr": 2e-4,
            "grad_clip": 1.0
        },
        {
            "name": "Phase 3: Fine-tuning",
            "epochs": phase_epochs_override or 10,
            "toggles": {"use_fan": True, "use_san": True, "use_gat": True},
            "loss_weights": {"quantile": 1.0, "sharpe": 0.15, "corr": 0.05},
            "lr": 1e-4,
            "grad_clip": 0.5
        }
    ]
    
    # Optimizer初期化
    optimizer = optim.AdamW(
        model.parameters(),
        lr=phases[0]["lr"],
        weight_decay=config.train.optimizer.weight_decay
    )
    # Fusion control + alpha warmup settings
    fuse_mode = os.getenv("FUSE_FORCE_MODE", "auto").lower()  # auto|tft_only
    fuse_start_phase = int(os.getenv("FUSE_START_PHASE", "2"))
    # Base alpha_min from config or model attribute
    try:
        base_alpha_min = float(getattr(getattr(config.model, "gat"), "alpha_min"))
    except Exception:
        base_alpha_min = float(getattr(model, "alpha_graph_min", 0.1))
    alpha_warm_min = float(os.getenv("GAT_ALPHA_WARMUP_MIN", "0.30"))
    alpha_warm_epochs = int(os.getenv("GAT_ALPHA_WARMUP_EPOCHS", "2"))
    # Scheduler selection (default warmup+cosine per phase)
    sched_choice = os.getenv("SCHEDULER", "warmup_cosine").lower()
    warmup_epochs_phase = int(os.getenv("PHASE_WARMUP_EPOCHS", "2"))
    if sched_choice == "plateau":
        logger.info("[Scheduler] Using ReduceLROnPlateau (phase-scoped)")
    else:
        logger.info(f"[Scheduler] Using Warmup+Cosine (warmup_epochs={warmup_epochs_phase})")
    
    # Loss初期化 - Fixed to use correct constructor
    criterion = MultiHorizonLoss(
        horizons=config.data.time_series.prediction_horizons,
        use_huber=True,
        huber_delta=0.01,
        huber_weight=0.3
    )
    
    best_val_loss = float('inf')
    checkpoint_path = Path("output/checkpoints")
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    # Early stopping metric selection (ENV-controlled)
    # Options: val_loss (min), val_sharpe (max), val_rankic (max), val_hit_rate (max)
    early_stop_metric = os.getenv("EARLY_STOP_METRIC", "val_loss").lower()
    early_stop_maximize = os.getenv("EARLY_STOP_MAXIMIZE", "0").lower() in ("1", "true", "yes")
    if early_stop_metric in ("val_sharpe", "val_rankic", "val_hit_rate") and os.getenv("EARLY_STOP_MAXIMIZE", "") == "":
        early_stop_maximize = True

    def _is_better(curr: float, best: float, maximize: bool, delta: float) -> bool:
        try:
            return (curr > best + delta) if maximize else (curr < best - delta)
        except Exception:
            return False
    
    for phase_idx, phase in enumerate(phases):
        logger.info(f"\n{'='*60}")
        logger.info(f"{phase['name']}")
        logger.info(f"{'='*60}")
        
        # モデルトグル適用
        if hasattr(model, 'fan') and hasattr(model, 'san'):
            if not phase["toggles"]["use_fan"]:
                model.fan = nn.Identity()
            if not phase["toggles"]["use_san"]:
                model.san = nn.Identity()
        
        # GAT有効/無効化（FUSE_FORCE_MODE反映）
        if hasattr(model, 'use_gat'):
            use_gat_flag = phase["toggles"]["use_gat"]
            if fuse_mode == "tft_only" and phase_idx < fuse_start_phase:
                use_gat_flag = False
            model.use_gat = use_gat_flag
        
        # 学習率調整
        for g in optimizer.param_groups:
            g["lr"] = phase["lr"]
        # Build phase scheduler if Warmup+Cosine is selected
        if sched_choice != "plateau":
            total_e = int(phase["epochs"]) if int(phase["epochs"]) > 0 else 1
            warm_e = min(warmup_epochs_phase, max(1, total_e // 3))
            def _lr_lambda(e_idx: int):
                if e_idx < warm_e:
                    return float(e_idx + 1) / max(1, warm_e)
                prog = (e_idx - warm_e) / max(1, total_e - warm_e)
                return 0.5 * (1.0 + np.cos(np.pi * prog))
            phase_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
        # Build phase scheduler if Warmup+Cosine is selected
        if sched_choice != "plateau":
            total_e = int(phase["epochs"]) if int(phase["epochs"]) > 0 else 1
            warm_e = min(warmup_epochs_phase, max(1, total_e // 3))
            def _lr_lambda(e_idx: int):
                if e_idx < warm_e:
                    return float(e_idx + 1) / max(1, warm_e)
                prog = (e_idx - warm_e) / max(1, total_e - warm_e)
                return 0.5 * (1.0 + np.cos(np.pi * prog))
            phase_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
        
        # 損失重み更新（実装に応じて安全に適用）
        try:
            # Phaseに応じて重み調整（正規化はset_preset_weights側で実施）
            if phase_idx == 0:
                w = {1: 1.0, 2: 0.0, 3: 0.0, 5: 0.0, 10: 0.0}
            elif phase_idx == 1:
                w = {1: 1.0, 2: 0.5, 3: 0.3, 5: 0.2, 10: 0.1}
            else:
                w = {1: 1.0, 2: 0.8, 3: 0.6, 5: 0.4, 10: 0.2}

            if hasattr(criterion, "set_preset_weights"):
                criterion.set_preset_weights(w)
            elif hasattr(criterion, "weights"):
                criterion.weights = w
            elif hasattr(criterion, "horizon_weights"):
                # 最後の手段として後方互換（将来的に削除予定）
                criterion.horizon_weights = w
            # Phase-aware loss schedule (env: PHASE_LOSS_WEIGHTS)
            _phase_loss_sched = _parse_phase_loss_schedule(os.getenv("PHASE_LOSS_WEIGHTS", ""))
            _apply_phase_loss_weights(criterion, phase_idx, _phase_loss_sched)
        except Exception:
            pass
        
        # 早期終了の設定（フェーズ内）
        try:
            early_stop_patience = int(os.getenv("EARLY_STOP_PATIENCE", "9"))
        except Exception:
            early_stop_patience = 9
        
        # Early stopping with min_delta
        early_stop_min_delta = float(os.getenv("EARLY_STOP_MIN_DELTA", "1e-4"))
        _phase_best = -float("inf") if early_stop_maximize else float("inf")
        _no_improve = 0
        
        # ReduceLROnPlateau scheduler for this phase (if selected)
        if sched_choice == "plateau":
            phase_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5
            )

        # エポック実行
        for epoch in range(phase["epochs"]):
            # Training
            model.train()
            train_loss = 0.0
            train_batches = 0
            train_metrics = {'sharpe': [], 'ic': [], 'rank_ic': []}
            
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= max_batches_per_epoch:  # バッチ数制限（デフォルト100）
                    break
                    
                optimizer.zero_grad()
                
                # Forward pass
                features_b = batch["features"].to(device, non_blocking=True)
                # Call model with a robust fallback across different forward signatures
                try:
                    predictions = model(features_b)
                except TypeError:
                    try:
                        predictions = model(
                            features_b,
                            batch.get("edge_index", None),
                            batch.get("edge_attr", None),
                        )
                    except Exception:
                        predictions = model(features_b)
                
                # Loss計算（targetsは辞書型）: 各種キーを正規化して 'horizon_{h}' に統一
                targets_dict = {}
                for k, target_tensor in batch.get("targets", {}).items():
                    canon = _canonicalize_target_key(k)
                    if canon is not None:
                        targets_dict[canon] = target_tensor.to(device, non_blocking=True)

                # 数値安定化＆形状を[B]へ揃える
                try:
                    if isinstance(predictions, dict):
                        predictions = {
                            pk: _finite_or_nan_fix_tensor(pv, f"pred[train-phase][{pk}]", clamp=50.0)
                            for pk, pv in predictions.items()
                            if torch.is_tensor(pv)
                        }
                    predictions = _reshape_to_batch_only(predictions)
                except Exception:
                    pass
                try:
                    targets_dict = {
                        tk: _finite_or_nan_fix_tensor(tv, f"targ[train-phase][{tk}]", clamp=50.0)
                        for tk, tv in targets_dict.items()
                    }
                    targets_dict = _reshape_to_batch_only(targets_dict)
                except Exception:
                    pass
                # Optional: label clipping by horizon (bps) for stability
                try:
                    clip_map = _parse_label_clip_map(os.getenv("LABEL_CLIP_BPS_MAP", ""))
                    if clip_map:
                        targets_dict = _clip_targets_by_horizon(targets_dict, clip_map)
                except Exception:
                    pass
                # Optional: add small Gaussian noise to outputs for warmup epochs
                try:
                    output_noise_std = float(os.getenv("OUTPUT_NOISE_STD", "0.0"))
                    noise_warmup_epochs = int(os.getenv("NOISE_WARMUP_EPOCHS", "2"))
                    if output_noise_std > 0.0 and (epoch < noise_warmup_epochs) and isinstance(predictions, dict):
                        for k in list(predictions.keys()):
                            if isinstance(predictions[k], torch.Tensor) and (k.startswith("point_horizon_") or k.startswith("horizon_")):
                                predictions[k] = predictions[k] + torch.randn_like(predictions[k]) * output_noise_std
                except Exception:
                    pass

                # Alpha warmup within GAT-enabled phases (epoch-scoped)
                try:
                    if getattr(model, "use_gat", False) and hasattr(model, "alpha_graph_min"):
                        if epoch < alpha_warm_epochs:
                            model.alpha_graph_min = max(base_alpha_min, alpha_warm_min)
                        else:
                            model.alpha_graph_min = base_alpha_min
                except Exception:
                    pass

                # Get loss from criterion - handle both single value and tuple return
                loss_result = criterion(predictions, targets_dict)
                if isinstance(loss_result, tuple):
                    loss, losses = loss_result
                else:
                    loss = loss_result
                    losses = {}
                
                
                # Backward pass (skip if loss is not a tensor or has no grad)
                try:
                    if hasattr(loss, "requires_grad") and loss.requires_grad:
                        loss.backward()
                    else:
                        logger.warning("[train-phase] loss has no grad; skipping backward")
                except Exception as _be:
                    logger.warning(f"[train-phase] backward skipped due to: {_be}")
                torch.nn.utils.clip_grad_norm_(model.parameters(), phase["grad_clip"])
                optimizer.step()
                
                # Compute metrics for main horizon (1d)
                # Try multiple key patterns
                pred_1d, targ_1d = None, None
                for pred_key in ["point_horizon_1", "horizon_1"]:
                    if pred_key in predictions:
                        pred_1d = predictions[pred_key].detach()
                        break
                for targ_key in ["horizon_1", "point_horizon_1"]:
                    if targ_key in targets_dict:
                        targ_1d = targets_dict[targ_key].detach()
                        break
                
                if pred_1d is not None and targ_1d is not None:
                    # Compute metrics
                    sharpe = MultiHorizonLoss.compute_sharpe_ratio(pred_1d, targ_1d)
                    ic = MultiHorizonLoss.compute_ic(pred_1d, targ_1d)
                    rank_ic = MultiHorizonLoss.compute_rank_ic(pred_1d, targ_1d)
                    
                    train_metrics['sharpe'].append(sharpe)
                    train_metrics['ic'].append(ic)
                    train_metrics['rank_ic'].append(rank_ic)
                else:
                    # Debug: Log available keys
                    logger.debug(f"[DEBUG] Prediction keys: {list(predictions.keys())}")
                    logger.debug(f"[DEBUG] Target keys: {list(targets_dict.keys())}")
                
                try:
                    train_loss += float(loss.item() if hasattr(loss, "item") else float(loss))
                except Exception:
                    pass
                train_batches += 1
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_batches = 0
            val_metrics = {'sharpe': [], 'ic': [], 'rank_ic': []}
            # For Hit Rate accumulation on horizon=1
            _val_preds_h1_all = []
            _val_targs_h1_all = []
            # Accumulate predictions/targets for Sharpe
            _val_preds: dict[int, list] = {h: [] for h in criterion.horizons}
            _val_targs: dict[int, list] = {h: [] for h in criterion.horizons}
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if batch_idx >= 50:  # 最初の50バッチで評価
                        break
                        
                    predictions = model(
                        batch["features"].to(device),
                        batch.get("static_features", None),
                        batch.get("edge_index", None),
                        batch.get("edge_attr", None)
                    )
                    # Sanitize & reshape predictions
                    try:
                        if isinstance(predictions, dict):
                            predictions = {
                                pk: _finite_or_nan_fix_tensor(pv, f"pred[val-phase][{pk}]", clamp=50.0)
                                for pk, pv in predictions.items()
                                if torch.is_tensor(pv)
                            }
                        predictions = _reshape_to_batch_only(predictions)
                    except Exception:
                        pass

                    # targets are dict tensors; build dict on device and sanitize
                    tdict = {}
                    for k, v in batch.get("targets", {}).items():
                        canon = _canonicalize_target_key(k)
                        if canon is not None:
                            tdict[canon] = v.to(device)
                    try:
                        tdict = {
                            tk: _finite_or_nan_fix_tensor(tv, f"targ[val-phase][{tk}]", clamp=50.0)
                            for tk, tv in tdict.items()
                        }
                        tdict = _reshape_to_batch_only(tdict)
                    except Exception:
                        pass
                    # Optional: label clipping in validation
                    try:
                        clip_map = _parse_label_clip_map(os.getenv("LABEL_CLIP_BPS_MAP", ""))
                        if clip_map:
                            tdict = _clip_targets_by_horizon(tdict, clip_map)
                    except Exception:
                        pass

                    loss_result = criterion(predictions, tdict)

                    # Handle both single value and tuple return from criterion
                    if isinstance(loss_result, tuple):
                        _val_total, _ = loss_result
                    else:
                        _val_total = loss_result

                    try:
                        if hasattr(_val_total, "item"):
                            val_loss += float(_val_total.item())
                        else:
                            val_loss += float(_val_total)
                    except Exception:
                        pass
                    val_batches += 1
                    
                    # Compute validation metrics for main horizon (1d)
                    # Try multiple key patterns
                    pred_1d, targ_1d = None, None
                    for pred_key in ["point_horizon_1", "horizon_1"]:
                        if pred_key in predictions:
                            pred_1d = predictions[pred_key].detach()
                            break
                    for targ_key in ["horizon_1", "point_horizon_1"]:
                        if targ_key in tdict:
                            targ_1d = tdict[targ_key].detach()
                            break
                    
                    if pred_1d is not None and targ_1d is not None:
                        # Compute metrics
                        sharpe = MultiHorizonLoss.compute_sharpe_ratio(pred_1d, targ_1d)
                        ic = MultiHorizonLoss.compute_ic(pred_1d, targ_1d)
                        rank_ic = MultiHorizonLoss.compute_rank_ic(pred_1d, targ_1d)
                        
                        val_metrics['sharpe'].append(sharpe)
                        val_metrics['ic'].append(ic)
                        val_metrics['rank_ic'].append(rank_ic)
                        # Accumulate for hit rate across validation
                        try:
                            _val_preds_h1_all.append(pred_1d.view(-1).detach().float().cpu())
                            _val_targs_h1_all.append(targ_1d.view(-1).detach().float().cpu())
                        except Exception:
                            pass
                    else:
                        # Debug: Log available keys
                        logger.debug(f"[DEBUG] Val Prediction keys: {list(predictions.keys())}")
                        logger.debug(f"[DEBUG] Val Target keys: {list(tdict.keys())}")
                    
                    # Store for Sharpe computation
                    try:
                        for h in criterion.horizons:
                            hk = f"horizon_{h}"
                            pk = hk
                            if isinstance(predictions, dict):
                                if f"point_horizon_{h}" in predictions:
                                    pk = f"point_horizon_{h}"
                                elif hk in predictions:
                                    pk = hk
                                else:
                                    continue
                                if (pk in predictions) and (hk in tdict):
                                    _val_preds[h].append(
                                        predictions[pk].detach().float().view(-1).cpu()
                                    )
                                    _val_targs[h].append(
                                        tdict[hk].detach().float().view(-1).cpu()
                                    )
                    except Exception:
                        pass
            
            avg_train_loss = train_loss / max(1, train_batches)
            avg_val_loss = val_loss / max(1, val_batches)
            # Compute Hit Rate on horizon=1 if available
            val_hit_rate = 0.0
            try:
                if _val_preds_h1_all and _val_targs_h1_all:
                    pv = torch.cat(_val_preds_h1_all, dim=0)
                    tv = torch.cat(_val_targs_h1_all, dim=0)
                    mask = (pv != 0) & (tv != 0)
                    if mask.any():
                        val_hit_rate = float(((pv[mask] * tv[mask]) > 0).float().mean().item())
            except Exception:
                val_hit_rate = 0.0

            # Compute average metrics
            avg_train_sharpe = np.mean(train_metrics['sharpe']) if train_metrics['sharpe'] else 0.0
            avg_train_ic = np.mean(train_metrics['ic']) if train_metrics['ic'] else 0.0
            avg_train_rank_ic = np.mean(train_metrics['rank_ic']) if train_metrics['rank_ic'] else 0.0
            
            avg_val_sharpe = np.mean(val_metrics['sharpe']) if val_metrics['sharpe'] else 0.0
            avg_val_ic = np.mean(val_metrics['ic']) if val_metrics['ic'] else 0.0
            avg_val_rank_ic = np.mean(val_metrics['rank_ic']) if val_metrics['rank_ic'] else 0.0
            
            logger.info(
                f"Epoch {epoch+1}/{phase['epochs']}: "
                f"Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, "
                f"LR={optimizer.param_groups[0]['lr']:.2e}"
            )
            logger.info(
                f"  Train Metrics - Sharpe: {avg_train_sharpe:.4f}, IC: {avg_train_ic:.4f}, RankIC: {avg_train_rank_ic:.4f}"
            )
            logger.info(
                f"  Val Metrics   - Sharpe: {avg_val_sharpe:.4f}, IC: {avg_val_ic:.4f}, RankIC: {avg_val_rank_ic:.4f}, HitRate(h1): {val_hit_rate:.4f}"
            )
            # Log fusion alpha if available
            try:
                if hasattr(model, "alpha_logit"):
                    alpha_min_now = float(getattr(model, "alpha_graph_min", base_alpha_min))
                    alpha_val = alpha_min_now + (1 - alpha_min_now) * torch.sigmoid(getattr(model, "alpha_logit")).mean().item()
                    logger.info(f"  Fusion alpha (mean): {alpha_val:.4f} (alpha_min={alpha_min_now:.2f})")
            except Exception:
                pass
            # Persist epoch metrics to JSONL (per phase)
            try:
                out_dir = Path("output/results")
                out_dir.mkdir(parents=True, exist_ok=True)
                jsonl = out_dir / f"phase_{phase_idx}_metrics.jsonl"
                rec = {
                    "phase": int(phase_idx),
                    "epoch": int(epoch + 1),
                    "train_loss": float(avg_train_loss),
                    "val_loss": float(avg_val_loss),
                    "val_sharpe": float(avg_val_sharpe),
                    "val_ic": float(avg_val_ic),
                    "val_rank_ic": float(avg_val_rank_ic),
                    "val_hit_rate": float(val_hit_rate),
                    "lr": float(optimizer.param_groups[0]['lr']),
                    "timestamp": now_jst_iso(),
                }
                with open(jsonl, "a", encoding="utf-8") as jf:
                    jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as _pe:
                logger.debug(f"metrics JSONL append skipped: {_pe}")
            # Compute and print parser-friendly Sharpe line (portfolio returns)
            try:
                import numpy as _np
                sharpe_vals = []
                for h in criterion.horizons:
                    # Portfolio return per sample: standardized prediction * standardized target
                    if _val_preds[h] and _val_targs[h]:
                        pv = torch.cat(_val_preds[h], dim=0).numpy()
                        tv = torch.cat(_val_targs[h], dim=0).numpy()

                        # Standardize to avoid scale effects
                        p_mean, p_std = float(_np.mean(pv)), float(_np.std(pv) + 1e-12)
                        t_mean, t_std = float(_np.mean(tv)), float(_np.std(tv) + 1e-12)
                        pv_z = (pv - p_mean) / p_std
                        tv_z = (tv - t_mean) / t_std

                        ret = pv_z * tv_z
                        sd = _np.std(ret) + 1e-12
                        sharpe_vals.append(float(_np.mean(ret) / sd))
                if sharpe_vals:
                    avg_sharpe = float(_np.mean(sharpe_vals))
                    logger.info(f"Sharpe: {avg_sharpe:.4f}")
                    # Also write a concise metrics summary JSON (non-conflicting filename)
                    try:
                        summary = {
                            "epoch": int(epoch + 1),
                            "train_loss": float(avg_train_loss),
                            "val_loss": float(avg_val_loss),
                            "avg_sharpe": float(avg_sharpe),
                            "time": time.time(),
                            "timestamp": now_jst_iso(),
                        }
                        (RUN_DIR / "metrics_summary.json").write_text(
                            json.dumps(summary, ensure_ascii=False, indent=2)
                        )
                    except Exception as _swe:
                        logger.debug(f"metrics_summary.json write skipped: {_swe}")
                    # Write latest metrics JSON for downstream parsers
                    try:
                        metrics_out = {
                            "epoch": int(epoch + 1),
                            "train_loss": float(avg_train_loss),
                            "val_loss": float(avg_val_loss),
                            "avg_sharpe": float(avg_sharpe),
                            "time": time.time(),
                            "timestamp": now_jst_iso(),
                        }
                        (RUN_DIR / "latest_metrics.json").write_text(
                            json.dumps(metrics_out, ensure_ascii=False, indent=2)
                        )
                    except Exception as _we:
                        logger.debug(f"metrics json write skipped: {_we}")
            except Exception as _se:
                logger.debug(f"Sharpe logging skipped: {_se}")
            # Always write a summary JSON even if Sharpe not available
            try:
                summary_path = RUN_DIR / "metrics_summary.json"
                if not summary_path.exists():
                    _summary = {
                        "epoch": int(epoch + 1),
                        "train_loss": float(avg_train_loss),
                        "val_loss": float(avg_val_loss),
                        "avg_sharpe": None,
                        "time": time.time(),
                        "timestamp": now_jst_iso(),
                    }
                    summary_path.write_text(
                        json.dumps(_summary, ensure_ascii=False, indent=2)
                    )
            except Exception as _swe2:
                logger.debug(f"metrics_summary.json fallback write skipped: {_swe2}")
            
            # Best model保存（選択メトリクスで評価、保存メタに値を記録）
            chosen_curr = avg_val_loss
            try:
                if early_stop_metric == "val_sharpe":
                    chosen_curr = float(avg_val_sharpe)
                elif early_stop_metric == "val_rankic":
                    chosen_curr = float(avg_val_rank_ic)
                elif early_stop_metric == "val_hit_rate":
                    chosen_curr = float(val_hit_rate)
            except Exception:
                chosen_curr = avg_val_loss

            if _is_better(chosen_curr, (-float('inf') if early_stop_maximize else float('inf')) if 'best_metric_val' not in locals() else best_metric_val, early_stop_maximize, 0.0):
                best_metric_val = chosen_curr
                checkpoint = {
                    'phase': phase_idx,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'early_stop_metric': early_stop_metric,
                    'early_stop_value': chosen_curr,
                    'config': phase
                }
                torch.save(
                    checkpoint,
                    checkpoint_path / f"best_model_phase{phase_idx}.pth"
                )
                logger.info(f"✅ Saved best model ({early_stop_metric}={chosen_curr:.4f}, val_loss={avg_val_loss:.4f})")

            # Update learning rate scheduler
            # Update scheduler
            try:
                if sched_choice == "plateau" and phase_scheduler is not None:
                    phase_scheduler.step(avg_val_loss)
                elif sched_choice != "plateau" and phase_scheduler is not None:
                    phase_scheduler.step()
            except Exception:
                pass
            
            # Early stopping (phase scoped)
            curr_for_es = chosen_curr
            if _is_better(curr_for_es, _phase_best, early_stop_maximize, early_stop_min_delta):
                _phase_best = curr_for_es
                _no_improve = 0
            else:
                _no_improve += 1
                if _no_improve >= early_stop_patience:
                    logger.info(f"⏹ Early stopping phase {phase_idx} after {epoch+1} epochs (best_{early_stop_metric}={_phase_best:.4f})")
                    break
    
    logger.info("=" * 80)
    logger.info(f"Phase Training Complete. Best Val Loss: {best_val_loss:.4f}; EarlyStop Metric: {early_stop_metric}")
    logger.info("=" * 80)
    
    return model


def run_mini_training(model, data_module, final_config, device, max_epochs: int = 3):
    """Simplified, robust training loop using existing train_epoch/validate.

    This path avoids complex micro-batching/graph/AMP logic and is useful when
    stabilizing training. Enable via env: USE_MINI_TRAIN=1.
    """
    logger.info("=== Running mini training loop (stability mode) ===")
    train_loader = (
        data_module.train_dataloader() if hasattr(data_module, "train_dataloader") else None
    )
    val_loader = (
        data_module.val_dataloader() if hasattr(data_module, "val_dataloader") else None
    )
    if train_loader is None or val_loader is None:
        logger.error("Mini training requires both train and val loaders.")
        raise RuntimeError("Mini training missing data loaders")

    # Optimizer and scaler (no AMP)
    lr = float(getattr(final_config.train.optimizer, "lr", 2e-4))
    wd = float(getattr(final_config.train.optimizer, "weight_decay", 1e-4))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    # Criterion (point loss by default; env toggles for extras)
    try:
        horizons = list(getattr(final_config.data.time_series, "prediction_horizons", [1, 5, 10, 20]))
    except Exception:
        horizons = [1, 5, 10, 20]
    use_pinball = os.getenv("USE_PINBALL", os.getenv("ENABLE_QUANTILES", "0")).lower() in ("1", "true", "yes")
    pinball_weight = float(os.getenv("PINBALL_WEIGHT", "0.1")) if use_pinball else 0.0
    use_t_nll = os.getenv("USE_T_NLL", os.getenv("ENABLE_STUDENT_T", "0")).lower() in ("1", "true", "yes")
    nll_weight = float(os.getenv("NLL_WEIGHT", "0.1")) if use_t_nll else 0.0
    dir_aux_enabled = os.getenv("USE_DIR_AUX", "0").lower() in ("1", "true", "yes")
    dir_aux_weight = float(os.getenv("DIR_AUX_WEIGHT", "0.1")) if dir_aux_enabled else 0.0

    criterion = MultiHorizonLoss(
        horizons=horizons,
        use_huber=True,
        huber_delta=0.01,
        huber_weight=0.3,
        use_pinball=use_pinball,
        pinball_weight=pinball_weight,
        use_t_nll=use_t_nll,
        nll_weight=nll_weight,
        direction_aux_weight=dir_aux_weight,
    )

    # Early-stop metric setup (same options as phase training)
    es_metric = os.getenv("EARLY_STOP_METRIC", "val_loss").lower()
    es_max = os.getenv("EARLY_STOP_MAXIMIZE", "0").lower() in ("1", "true", "yes")
    if es_metric in ("val_sharpe", "val_rankic", "val_hit_rate") and os.getenv("EARLY_STOP_MAXIMIZE", "") == "":
        es_max = True
    es_patience = int(os.getenv("EARLY_STOP_PATIENCE", "9"))
    es_delta = float(os.getenv("EARLY_STOP_MIN_DELTA", "1e-4"))
    best_metric = -float("inf") if es_max else float("inf")
    no_improve = 0

    def _is_better_m(curr: float, best: float) -> bool:
        try:
            return (curr > best + es_delta) if es_max else (curr < best - es_delta)
        except Exception:
            return False

    for epoch in range(1, int(max_epochs) + 1):
        avg_train_loss, _ = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            epoch,
            gradient_accumulation_steps=1,
        )
        logger.info(f"[mini] Train loss @epoch{epoch}: {avg_train_loss:.4f}")

        val_loss, _, _ = validate(model, val_loader, criterion, device)
        # Compute simple Hit Rate(h1) over a few batches (up to 50) for mini path
        def _mini_val_hit_rate(_model, _loader, _device, max_batches: int = 50) -> float:
            _model.eval()
            preds_all, targs_all = [], []
            with torch.no_grad():
                for bi, b in enumerate(_loader):
                    if bi >= max_batches:
                        break
                    feats = b.get("features")
                    if feats is None:
                        continue
                    feats = feats.to(_device)
                    out = _model(feats)
                    if not isinstance(out, dict):
                        continue
                    # normalize and take horizon_1
                    pk = "point_horizon_1" if "point_horizon_1" in out else ("horizon_1" if "horizon_1" in out else None)
                    tk = None
                    for k in b.get("targets", {}).keys():
                        nk = _canonicalize_target_key(k)
                        if nk == "horizon_1":
                            tk = k
                            break
                    if pk is None or tk is None:
                        continue
                    p = out[pk]
                    t = b["targets"][tk]
                    try:
                        p = p.detach().float().view(-1).cpu()
                        t = t.detach().float().view(-1).cpu()
                        preds_all.append(p)
                        targs_all.append(t)
                    except Exception:
                        continue
            try:
                if preds_all and targs_all:
                    pv = torch.cat(preds_all)
                    tv = torch.cat(targs_all)
                    mask = (pv != 0) & (tv != 0)
                    if mask.any():
                        return float(((pv[mask] * tv[mask]) > 0).float().mean().item())
            except Exception:
                pass
            return 0.0

        val_hit_rate = _mini_val_hit_rate(model, val_loader, device)
        logger.info(f"[mini] Val loss @epoch{epoch}: {val_loss:.4f}, HitRate(h1)={val_hit_rate:.4f}")

        # Choose metric for early stopping
        chosen = val_loss
        if es_metric == "val_hit_rate":
            chosen = val_hit_rate
        # (Optional) could extend to Sharpe/RankIC here if needed

        # Persist epoch metrics
        try:
            out_dir = Path("output/results")
            out_dir.mkdir(parents=True, exist_ok=True)
            rec = {
                "mode": "mini",
                "epoch": int(epoch),
                "val_loss": float(val_loss),
                "val_hit_rate": float(val_hit_rate),
                "timestamp": now_jst_iso(),
            }
            with open(out_dir / "mini_metrics.jsonl", "a", encoding="utf-8") as jf:
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

        # Early stopping
        if _is_better_m(chosen, best_metric):
            best_metric = chosen
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= es_patience:
                logger.info(f"[mini] Early stopping at epoch {epoch} (best {es_metric}={best_metric:.4f})")
                break

    save_path = Path("models/checkpoints/atft_gat_fan_best_mini.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": OmegaConf.to_container(final_config)}, save_path)
    logger.info(f"[mini] Saved checkpoint to {save_path}")
    return True


def _apply_env_overrides(cfg):
    """環境変数からHydra設定への反映"""

    # Quantile/Student-t ヘッドの有効化フラグ
    if "ENABLE_QUANTILES" in os.environ:
        val = os.environ["ENABLE_QUANTILES"]
        enabled = val.lower() in ["1", "true", "yes"]
        # Check if the key exists in the config structure
        if "prediction_head" in cfg.model and "output" in cfg.model.prediction_head:
            if "quantile_prediction" in cfg.model.prediction_head.output:
                cfg.model.prediction_head.output.quantile_prediction.enabled = enabled
                logger.info(
                    f"[EnvOverride] model.prediction_head.output.quantile_prediction.enabled = {enabled}"
                )
            else:
                logger.warning("[EnvOverride] quantile_prediction not found in config")
        else:
            logger.warning("[EnvOverride] prediction_head.output not found in config")

    if "ENABLE_STUDENT_T" in os.environ:
        val = os.environ["ENABLE_STUDENT_T"]
        enabled = val.lower() in ["1", "true", "yes"]
        # Check if the key exists in the config structure
        if "prediction_head" in cfg.model and "output" in cfg.model.prediction_head:
            # Set the student_t field - use environment variable to override config
            cfg.model.prediction_head.output.student_t = enabled
            logger.info(
                f"[EnvOverride] model.prediction_head.output.student_t = {enabled}"
            )
        else:
            logger.warning("[EnvOverride] prediction_head.output not found in config")

    # Mixed Precision
    if "USE_AMP" in os.environ:
        val = os.environ["USE_AMP"]
        enabled = val.lower() in ["1", "true", "yes"]
        # Check if precision key exists and update it
        if "trainer" in cfg.train and "precision" in cfg.train.trainer:
            # Set precision based on USE_AMP
            cfg.train.trainer.precision = "bf16-mixed" if enabled else 32
            logger.info(
                f"[EnvOverride] train.trainer.precision = {cfg.train.trainer.precision}"
            )
        else:
            logger.warning("[EnvOverride] train.trainer.precision not found in config")

    # 退行ガード関連
    # Note: Since degeneracy settings don't exist in config, we'll store them in environment
    # The actual degeneracy checks in the training loop use os.environ directly
    if "DEGENERACY_GUARD" in os.environ:
        val = os.environ["DEGENERACY_GUARD"]
        enabled = val.lower() in ["1", "true", "yes"]
        logger.info(f"[EnvOverride] DEGENERACY_GUARD = {enabled} (via environment)")

    if "DEGENERACY_ABORT" in os.environ:
        val = os.environ["DEGENERACY_ABORT"]
        enabled = val.lower() in ["1", "true", "yes"]
        logger.info(f"[EnvOverride] DEGENERACY_ABORT = {enabled} (via environment)")

    if "PRED_STD_MIN" in os.environ:
        val = float(os.environ["PRED_STD_MIN"])
        logger.info(f"[EnvOverride] PRED_STD_MIN = {val} (via environment)")
    elif "PRED_VAR_MIN" in os.environ:
        # PRED_VAR_MIN -> std に変換
        var_min = float(os.environ["PRED_VAR_MIN"])
        std_min = var_min**0.5
        os.environ["PRED_STD_MIN"] = str(std_min)
        logger.info(
            f"[EnvOverride] PRED_STD_MIN = {std_min} (from PRED_VAR_MIN={var_min}, via environment)"
        )

    if "PRED_VAR_WEIGHT" in os.environ:
        val = float(os.environ["PRED_VAR_WEIGHT"])
        logger.info(f"[EnvOverride] PRED_VAR_WEIGHT = {val} (via environment)")

    # Noise injection - check if these exist in config
    if "FEATURE_NOISE_STD" in os.environ:
        val = float(os.environ["FEATURE_NOISE_STD"])
        logger.info(f"[EnvOverride] FEATURE_NOISE_STD = {val} (via environment)")

    if "OUTPUT_NOISE_STD" in os.environ:
        val = float(os.environ["OUTPUT_NOISE_STD"])
        logger.info(f"[EnvOverride] OUTPUT_NOISE_STD = {val} (via environment)")

    return cfg


def fix_seed(seed: int = 42, deterministic: bool = False):
    """Fix random seeds for reproducibility"""
    import os
    import random
    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic mode (slower but fully reproducible)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    return seed


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(config: DictConfig) -> None:
    """メイン学習関数"""
    logger.info("Starting production training...")
    # Optional MLflow setup (enable with MLFLOW=1)
    mlf_enabled = os.getenv("MLFLOW", "0") == "1"
    mlf = None
    if mlf_enabled:
        try:
            import mlflow  # type: ignore

            mlflow.set_tracking_uri(
                os.getenv(
                    "MLFLOW_TRACKING_URI", str((project_root / "mlruns").resolve())
                )
            )
            mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "ATFT-GAT-FAN"))
            run_name = os.getenv(
                "MLFLOW_RUN_NAME",
                f"train_{datetime.now(JST).strftime('%Y%m%d_%H%M%S')}",
            )
            mlf = mlflow
            mlf.start_run(run_name=run_name)
            mlf.log_params(
                {
                    "TRAIN_PROFILE": os.getenv("TRAIN_PROFILE", "custom"),
                    "PRED_VAR_MIN": os.getenv("PRED_VAR_MIN", ""),
                    "PRED_VAR_WEIGHT": os.getenv("PRED_VAR_WEIGHT", ""),
                    "DEGENERACY_GUARD": os.getenv("DEGENERACY_GUARD", ""),
                    "ALPHA_GRAPH_MIN": os.getenv("ALPHA_GRAPH_MIN", ""),
                    "HWEIGHTS": os.getenv("HWEIGHTS", ""),
                    "SPARSITY_LAMBDA": os.getenv("SPARSITY_LAMBDA", ""),
                }
            )
        except Exception as _e:
            logger.warning(f"MLflow setup failed or disabled: {_e}")

    # Enforce single-process data loading when GPUs are unavailable or explicitly forced
    force_single_process = (
        os.getenv("FORCE_SINGLE_PROCESS", "0").lower() in ("1", "true", "yes")
        or not torch.cuda.is_available()
        or os.getenv("ACCELERATOR", "").lower() == "cpu"
    )
    if force_single_process:
        if os.environ.get("USE_DAY_BATCH", "1") not in ("0", "false", "False"):
            logger.info(
                "Disabling day-batch sampler and multi-worker loaders for single-process run."
            )
        os.environ["USE_DAY_BATCH"] = "false"
        os.environ.setdefault("NUM_WORKERS", "0")
        os.environ.setdefault("PERSISTENT_WORKERS", "0")
        os.environ.setdefault("PREFETCH_FACTOR", "null")

    # Apply environment variable overrides
    config = _apply_env_overrides(config)

    # Validate configuration
    if os.getenv("VALIDATE_CONFIG", "1") == "1":
        validator = ConfigValidator()
        if not validator.validate(config):
            raise ValueError("Configuration validation failed. Check logs for details.")

    # Fix random seed for reproducibility
    seed = int(os.environ.get("SEED", "42"))
    deterministic = os.environ.get("DETERMINISTIC", "0") == "1"
    actual_seed = fix_seed(seed, deterministic)
    logger.info(f"Random seed: {actual_seed}, Deterministic: {deterministic}")

    # GPU最適化設定（安全デフォルト）
    if not deterministic:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # A100の特殊機能を有効化
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass
        torch.cuda.empty_cache()

    # AMP 設定（環境変数で制御）
    use_amp_env = os.getenv("USE_AMP", "1") == "1"
    amp_dtype = (
        torch.bfloat16
        if os.getenv("AMP_DTYPE", "").lower() in ("bf16", "bfloat16", "bf16-mixed")
        else torch.float16
    )

    # 警告を抑制
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*worker.*")

    # デバイス設定（GPU優先 + 環境変数で上書き可能）
    def _resolve_device() -> torch.device:
        dev_env = os.getenv("DEVICE", "").strip()
        acc_env = os.getenv("ACCELERATOR", "").lower().strip()
        force_gpu = os.getenv("FORCE_GPU", "0") == "1"
        require_gpu = os.getenv("REQUIRE_GPU", "0") == "1"
        # Prefer explicit device if provided
        if dev_env:
            try:
                if dev_env.startswith("cuda") and not torch.cuda.is_available():
                    if require_gpu:
                        raise RuntimeError("GPU required but not available (CUDA not initialized)")
                    logger.warning("CUDA not available; falling back to CPU despite DEVICE=cuda")
                    return torch.device("cpu")
                return torch.device(dev_env)
            except Exception:
                logger.warning(f"Invalid DEVICE='{dev_env}', falling back to auto")
        # Accelerator or FORCE_GPU hints
        if force_gpu or acc_env in ("gpu", "cuda"):
            if not torch.cuda.is_available():
                if require_gpu:
                    raise RuntimeError("GPU required but not available (torch.cuda.is_available=False)")
                logger.warning("GPU hint given but CUDA unavailable; using CPU")
                return torch.device("cpu")
            return torch.device("cuda")
        # Default: auto
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = _resolve_device()
    logger.info(f"Using device: {device}")

    use_amp = use_amp_env and device.type == "cuda"

    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
        )

    # 設定を作成（Hydraから渡される設定を使用）
    final_config = OmegaConf.create(
        {
            "data": {
                "source": {
                    "data_dir": getattr(
                        config.data.source, "data_dir", "data/raw/large_scale"
                    )
                },
                "time_series": {
                    # ユーザ指定を優先（固定60を強制しない）
                    "sequence_length": getattr(
                        config.data.time_series, "sequence_length", 20
                    ),
                    "prediction_horizons": getattr(
                        config.data.time_series, "prediction_horizons", [1, 5, 10, 20]
                    ),
                },
                "features": {
                    "numeric": {
                        "return_based": ["return_1d", "return_5d", "return_20d"],
                        "technical": ["rsi14", "macd", "macd_signal", "macd_hist"],
                        "volume": ["volume_ratio"],
                        "price": ["price_position"],
                    },
                    "num_features": 13,  # 特徴量数（内部基本特徴を想定）
                    "input_dim": 13,  # ATFT-GAT-FAN用
                    "hidden_size": 512,  # 256から増加
                    "graph_features": 128,  # 64から増加
                },
                "graph": {"k_neighbors": 10, "edge_threshold": 0.5},
            },
            # モデル設定は config 側を優先（ここでは最小限）
            "model": {
                # ドロップアウトは環境変数で調整（デフォルト0.1→0.2まで引上げ可）
                "dropout": float(os.getenv("MODEL_DROPOUT", "0.1")),
                # 主要ブロックのドロップアウトを個別設定（存在すれば使用）
                "input_projection": {
                    "use_layer_norm": True,
                    "dropout": float(
                        os.getenv("INPUT_DROPOUT", os.getenv("MODEL_DROPOUT", "0.1"))
                    ),
                },
                "tft": {
                    "variable_selection": {
                        "dropout": float(
                            os.getenv("VSN_DROPOUT", os.getenv("MODEL_DROPOUT", "0.1"))
                        ),
                        "use_sigmoid": True,
                        "sparsity_coefficient": 0.0,
                    },
                    "attention": {"heads": 4},
                    "lstm": {
                        "layers": 1,
                        "dropout": float(
                            os.getenv("LSTM_DROPOUT", os.getenv("MODEL_DROPOUT", "0.1"))
                        ),
                    },
                    "temporal": {
                        "use_positional_encoding": True,
                        "max_sequence_length": 20,
                    },
                },
                "prediction_head": {
                    "architecture": {
                        "hidden_layers": [],
                        "dropout": float(
                            os.getenv("HEAD_DROPOUT", os.getenv("MODEL_DROPOUT", "0.1"))
                        ),
                    },
                    "output": {
                        "point_prediction": True,
                        # Student-t/Quantileヘッドは環境変数でON
                        "student_t": os.getenv("ENABLE_STUDENT_T", "1") == "1",
                        "quantile_prediction": {
                            "enabled": os.getenv("ENABLE_QUANTILES", "1") == "1",
                            "quantiles": [0.2, 0.5, 0.8],
                            "enforce_monotonic": True,
                        },
                    },
                },
                "use_gpu": True,
            },
            "train": {
                "batch": {
                    # 安全なデフォルト（OOM回避）。実効バッチは勾配蓄積で調整
                    "train_batch_size": getattr(
                        getattr(config.train, "batch", {}), "train_batch_size", 256
                    ),
                    "val_batch_size": getattr(
                        getattr(config.train, "batch", {}), "val_batch_size", 512
                    ),
                    "gradient_accumulation_steps": getattr(
                        getattr(config.train, "batch", {}),
                        "gradient_accumulation_steps",
                        1,
                    ),
                },
                "optimizer": {
                    "lr": min(
                        getattr(getattr(config.train, "optimizer", {}), "lr", 0.001),
                        5e-4,
                    ),
                    "weight_decay": getattr(
                        getattr(config.train, "optimizer", {}), "weight_decay", 0.01
                    ),
                },
                "scheduler": {
                    "warmup_epochs": 2,
                    "total_epochs": getattr(
                        getattr(config.train, "trainer", {}), "max_epochs", 30
                    ),
                },
            },
        }
    )

    if force_single_process:
        try:
            OmegaConf.update(final_config, "data.use_day_batch_sampler", False, merge=True)
        except Exception:
            pass
        for path, value in (
            ("train.batch.num_workers", 0),
            ("train.batch.prefetch_factor", None),
            ("train.batch.persistent_workers", False),
        ):
            try:
                OmegaConf.update(final_config, path, value, merge=True)
            except Exception:
                continue
    try:
        final_config = OmegaConf.merge(config, final_config)
    except Exception as _merge_err:
        logger.warning(f"[Hydra-Struct] merge with original config failed: {_merge_err}")

    # Ensure auxiliary data sections (schema, normalization, etc.) are retained
    try:
        cfg_data_container = OmegaConf.to_container(config.data, resolve=False)  # type: ignore[arg-type]
        if isinstance(cfg_data_container, dict):
            for key, value in cfg_data_container.items():
                if key in {"source", "time_series", "features", "graph"}:
                    continue  # already set explicitly above
                OmegaConf.update(final_config, f"data.{key}", value, merge=True)
    except Exception as _data_merge_err:
        logger.warning(f"[Hydra-Struct] data config merge failed: {_data_merge_err}")

    try:
        cfg_batch_container = OmegaConf.to_container(config.train.batch, resolve=False)  # type: ignore[arg-type]
        if isinstance(cfg_batch_container, dict):
            for key, value in cfg_batch_container.items():
                if OmegaConf.select(final_config, f"train.batch.{key}") is None:
                    OmegaConf.update(final_config, f"train.batch.{key}", value, merge=True)
    except Exception as _batch_merge_err:
        logger.warning(f"[Hydra-Struct] train.batch merge failed: {_batch_merge_err}")

    try:
        if hasattr(config, "normalization") and config.normalization is not None:
            OmegaConf.update(final_config, "normalization", config.normalization, merge=True)
    except Exception as _norm_merge_err:
        logger.warning(f"[Hydra-Struct] normalization merge failed: {_norm_merge_err}")

    try:
        if OmegaConf.select(final_config, "normalization") is None:
            data_norm = OmegaConf.select(final_config, "data.normalization")
            if data_norm is not None:
                OmegaConf.update(final_config, "normalization", data_norm, merge=True)
            else:
                OmegaConf.update(
                    final_config,
                    "normalization",
                    {
                        "online_normalization": {"enabled": True, "per_batch": True},
                        "cross_sectional": {"enabled": False},
                        "batch_norm": {"enabled": False},
                    },
                    merge=True,
                )
    except Exception as _norm_fallback_err:
        logger.warning(f"[Hydra-Struct] normalization fallback failed: {_norm_fallback_err}")
    # ---- Optional W&B setup (env-flagged) ----
    wb_logger = None
    try:
        # Enable W&B by default; allow disabling via WANDB_ENABLED=0
        use_wandb = os.getenv("WANDB_ENABLED", "1").lower() in ("1", "true", "yes", "on")
        # Allow project override via env
        try:
            proj = os.getenv("WANDB_PROJECT", None)
            if proj:
                # Attach project name into config for the logger
                setattr(final_config, "wandb_project", proj)
        except Exception:
            pass
        if use_wandb and WBLogger is not None:
            run_name = os.getenv(
                "WANDB_RUN_NAME", f"train_{datetime.now(JST).strftime('%Y%m%d_%H%M%S')}"
            )
            wb_logger = WBLogger(
                config=final_config,
                experiment_name=run_name,
                log_dir="./logs",
                use_wandb=True,
                use_tensorboard=False,
            )
            # Minimal hyperparam logging (safe, no data)
            try:
                wb_logger.log_hyperparameters(
                    {
                        "optimizer/lr": float(
                            getattr(getattr(final_config.train, "optimizer", {}), "lr", 0.0)
                        ),
                        "optimizer/wd": float(
                            getattr(
                                getattr(final_config.train, "optimizer", {}),
                                "weight_decay",
                                0.0,
                            )
                        ),
                        "batch/train": int(
                            getattr(final_config.train.batch, "train_batch_size", 0)
                        ),
                        "batch/val": int(
                            getattr(final_config.train.batch, "val_batch_size", 0)
                        ),
                        "epochs": int(
                            getattr(final_config.train.scheduler, "total_epochs", 0)
                        ),
                        "amp": os.getenv("USE_AMP", "1"),
                        "amp_dtype": os.getenv("AMP_DTYPE", ""),
                    }
                )
            except Exception:
                pass
    except Exception as _e:
        logger.warning(f"W&B setup skipped: {_e}")
    # Log core hyperparameters to MLflow
    if mlf is not None:
        try:
            lr_logged = getattr(
                getattr(final_config.train, "optimizer", {}), "lr", None
            )
            bs_logged = getattr(
                getattr(final_config.train, "batch", {}), "train_batch_size", None
            )
            ga_logged = getattr(
                getattr(final_config.train, "batch", {}),
                "gradient_accumulation_steps",
                None,
            )
            # Try to fetch alpha_min from config if present
            alpha_min_cfg = None
            try:
                alpha_min_cfg = getattr(getattr(final_config.model, "gat"), "alpha_min")
            except Exception:
                alpha_min_cfg = None
            mlf.log_params(
                {
                    "train.lr": float(lr_logged) if lr_logged is not None else None,
                    "batch.train_batch_size": int(bs_logged)
                    if bs_logged is not None
                    else None,
                    "batch.grad_accum": int(ga_logged)
                    if ga_logged is not None
                    else None,
                    "model.gat.alpha_min": float(alpha_min_cfg)
                    if alpha_min_cfg is not None
                    else None,
                }
            )
        except Exception as _e:
            logger.debug(f"MLflow param logging skipped: {_e}")

    # Hydraのモデル設定を取り込み（不足キーでの初期化失敗を防ぐ）
    try:
        final_config.model = config.model
    except Exception:
        pass
    # Ensure required keys exist (Hydra struct safety) using OmegaConf-safe updates
    try:
        # Ensure model exists as mapping
        if "model" not in final_config or getattr(final_config, "model") is None:
            final_config.model = OmegaConf.create({})
        # Ensure model.gat exists as mapping
        if OmegaConf.select(final_config, "model.gat") is None:
            OmegaConf.update(final_config, "model.gat", {}, merge=True)
        # Ensure alpha_min default exists
        if OmegaConf.select(final_config, "model.gat.alpha_min") is None:
            OmegaConf.update(
                final_config,
                "model.gat.alpha_min",
                float(os.getenv("ALPHA_MIN_DEFAULT", "0.3")),
                merge=True,
            )
            logger.info("[Hydra-Struct] Set default model.gat.alpha_min=0.3")
    except Exception:
        pass
    # Hydraのハードウェア設定も反映（channels_last等で参照）
    try:
        final_config.hardware = config.hardware
    except Exception:
        # 無ければ空のネームスペース風オブジェクトを持たせる
        try:
            from types import SimpleNamespace

            final_config.hardware = SimpleNamespace()
        except Exception:
            pass

    # Ensure model.hidden_size exists with proper path checking
    hidden_size = None
    # Try different possible paths for hidden_size
    for path in ["model.model.hidden_size", "model.hidden_size", "hidden_size"]:
        try:
            parts = path.split(".")
            obj = final_config
            for part in parts:
                obj = getattr(obj, part)
            if obj is not None:
                hidden_size = int(obj)
                logger.info(f"Found hidden_size={hidden_size} at path: {path}")
                break
        except Exception:
            continue

    # If still not found, check environment variable or raise error
    if hidden_size is None:
        env_hidden_size = os.getenv("MODEL_HIDDEN_SIZE")
        if env_hidden_size:
            hidden_size = int(env_hidden_size)
            logger.info(f"Using hidden_size={hidden_size} from environment variable")
        else:
            # This is a critical configuration - should not use implicit default
            raise ValueError(
                "model.hidden_size must be explicitly set in config. "
                "Check your model config file or set MODEL_HIDDEN_SIZE environment variable."
            )

    # グラフKの整合性チェック
    try:
        k_data = getattr(final_config.data.graph_builder, "k", None)
        k_model = getattr(final_config.model.gat, "knn_k", None)
        if k_data and k_model and int(k_data) != int(k_model):
            logger.warning(
                f"[ModelCheck] data.graph_builder.k({k_data}) != model.gat.knn_k({k_model})"
            )
    except Exception:
        pass

    # PositionalEncoding 長の自動整合: データ系列長に合わせて上書き
    try:
        seq_len_cfg = int(getattr(final_config.data.time_series, "sequence_length", 20))
        tft_temporal = getattr(
            getattr(final_config.model, "tft", object()), "temporal", None
        )
        if tft_temporal is not None:
            cur_max = getattr(tft_temporal, "max_sequence_length", None)
            if cur_max is None or int(cur_max) < int(seq_len_cfg):
                setattr(tft_temporal, "max_sequence_length", int(seq_len_cfg))
                logger.info(
                    f"[PE] Set model.tft.temporal.max_sequence_length={seq_len_cfg}"
                )
    except Exception as _e:
        logger.warning(f"[PE] auto-align skipped: {_e}")

    # ホライズンを仕様に統一
    try:
        final_config.data.time_series.prediction_horizons = [1, 5, 10, 20]
    except Exception:
        pass

    # データモジュール
    logger.info("Setting up data module...")
    try:
        _nw_env = int(os.getenv("NUM_WORKERS", "0"))
    except Exception:
        _nw_env = 0
    try:
        _bs_env = int(os.getenv("BATCH_SIZE", "0"))
    except Exception:
        _bs_env = 0
    try:
        _bs_cfg = int(getattr(getattr(final_config.train, "batch", object()), "train_batch_size", 64))
    except Exception:
        _bs_cfg = 64
    _batch_size = _bs_env if _bs_env > 0 else _bs_cfg
    # In sandboxed environments, multiprocessing may be restricted; allow env override to 0
    try:
        dm_signature = inspect.signature(ProductionDataModuleV2)
        dm_params = dm_signature.parameters
    except (TypeError, ValueError):
        dm_params = {}

    # Never rely on implicit defaults; update config so downstream loaders stay in sync
    try:
        final_config.train.batch.train_batch_size = _batch_size
    except Exception:
        pass
    try:
        final_config.train.batch.num_workers = int(_nw_env)
    except Exception:
        pass

    if "batch_size" in dm_params:
        try:
            schema_cfg = OmegaConf.select(final_config, "data.schema")
            if schema_cfg is None:
                logger.warning("[Hydra-Struct] data.schema missing prior to DataModule setup")
            else:
                try:
                    logger.info(
                        "[Hydra-Struct] data.schema detected with keys: %s",
                        list(schema_cfg.keys()),
                    )
                except Exception:
                    logger.info("[Hydra-Struct] data.schema detected (non-mapping)")
        except Exception as _schema_log_err:
            logger.warning(f"[Hydra-Struct] schema inspection failed: {_schema_log_err}")

        try:
            data_keys = list(final_config.data.keys())  # type: ignore[call-arg]
            logger.info(f"[Hydra-Struct] data group keys: {data_keys}")
        except Exception as _data_keys_err:
            logger.info(f"[Hydra-Struct] data group type: {type(final_config.data)} ({_data_keys_err})")

        data_module = ProductionDataModuleV2(
            final_config,
            batch_size=_batch_size,
            num_workers=int(_nw_env),
        )
    else:
        try:
            schema_cfg = OmegaConf.select(final_config, "data.schema")
            if schema_cfg is None:
                logger.warning("[Hydra-Struct] data.schema missing prior to DataModule setup")
            else:
                try:
                    logger.info(
                        "[Hydra-Struct] data.schema detected with keys: %s",
                        list(schema_cfg.keys()),
                    )
                except Exception:
                    logger.info("[Hydra-Struct] data.schema detected (non-mapping)")
        except Exception as _schema_log_err:
            logger.warning(f"[Hydra-Struct] schema inspection failed: {_schema_log_err}")

        try:
            data_keys = list(final_config.data.keys())  # type: ignore[call-arg]
            logger.info(f"[Hydra-Struct] data group keys: {data_keys}")
        except Exception as _data_keys_err:
            logger.info(f"[Hydra-Struct] data group type: {type(final_config.data)} ({_data_keys_err})")

        data_module = ProductionDataModuleV2(final_config)
    data_module.setup()
    # Smoke-mode: optionally limit data files to speed up CI/smoke runs
    try:
        _smoke_max = int(os.getenv("SMOKE_DATA_MAX_FILES", "0"))
    except Exception:
        _smoke_max = 0
    if _smoke_max and _smoke_max > 0:
        try:
            if hasattr(data_module, "train_files") and data_module.train_files:
                data_module.train_files = data_module.train_files[:_smoke_max]
            if hasattr(data_module, "val_files") and data_module.val_files:
                data_module.val_files = data_module.val_files[:_smoke_max]
            os.environ.setdefault("MINIMAL_COLUMNS", "1")
            logger.info(
                f"[SMOKE] Limiting data files to {_smoke_max} and enabling MINIMAL_COLUMNS"
            )
        except Exception as _e:
            logger.warning(f"[SMOKE] could not limit files: {_e}")

    # Purged K-Fold（実装）: 環境変数 CV_FOLDS>=2 で活性化
    cv_folds = int(os.getenv("CV_FOLDS", "1"))
    embargo_days = int(
        os.getenv(
            "EMBARGO_DAYS", str(max(final_config.data.time_series.prediction_horizons))
        )
    )
    if cv_folds >= 2:
        logger.info(f"Using Purged K-Fold: k={cv_folds}, embargo={embargo_days}d")
        # 全データから日付リストを作成（DATA_PATHがあれば単一ファイル優先）
        data_path_env = os.getenv("DATA_PATH", "").strip()
        if data_path_env:
            all_files = [Path(data_path_env)]
        else:
            all_files = sorted(Path(final_config.data.source.data_dir).glob("*.parquet"))
        df_dates = []
        for p in all_files:
            try:
                d = pd.read_parquet(p, columns=["date"])
                if "date" in d.columns:
                    d["date"] = pd.to_datetime(d["date"], errors="coerce")
                    df_dates.append(d[["date"]])
            except Exception:
                continue
        if df_dates:
            date_series = pd.concat(df_dates).dropna().sort_values("date")["date"]
            unique_dates = (
                pd.Series(date_series.unique()).sort_values().reset_index(drop=True)
            )
            n = len(unique_dates)
            fold_sizes = [
                n // cv_folds + (1 if i < n % cv_folds else 0) for i in range(cv_folds)
            ]
            idx = 0
            # Regime-aware CV: use fixed boundaries if REGIME_CV=1
            fold_ranges = []
            if os.getenv("REGIME_CV", "0") == "1":
                # Parse boundaries from env or use defaults
                # Format: YYYY-mm-dd..YYYY-mm-dd,YYYY-mm-dd..YYYY-mm-dd,...
                env_bounds = os.getenv(
                    "REGIME_BOUNDARIES",
                    "2020-02-15..2020-05-31,2020-06-01..2021-12-31,2022-01-01..2022-12-31,2023-01-01..2023-12-31,2024-01-01..2025-06-30",
                )
                regs = []
                try:
                    for tok in env_bounds.split(","):
                        a, b = tok.split("..")
                        regs.append((pd.Timestamp(a.strip()), pd.Timestamp(b.strip())))
                except Exception:
                    regs = []
                if regs:
                    fold_ranges = regs
                    logger.info(
                        f"[RegimeCV] Using fixed boundaries: {[(s.date(),e.date()) for s,e in fold_ranges]}"
                    )
                else:
                    # fallback to equal split
                    for fs in fold_sizes:
                        start = idx
                        end = idx + fs
                        fold_ranges.append(
                            (unique_dates.iloc[start], unique_dates.iloc[end - 1])
                        )
                        idx = end
            else:
                for fs in fold_sizes:
                    start = idx
                    end = idx + fs
                    fold_ranges.append(
                        (unique_dates.iloc[start], unique_dates.iloc[end - 1])
                    )
                    idx = end
        else:
            fold_ranges = []
            logger.warning(
                "Failed to build date ranges for CV; falling back to single split."
            )
            cv_folds = 1

    # データローダー作成（単一学習 or 第1foldのみ実行）
    logger.info("Creating data loaders...")
    if cv_folds >= 2 and fold_ranges:
        # 使用するfoldを選定（最初のfoldは学習期間が確保できない場合があるためスキップ）
        try:
            min_date = fold_ranges[0][0]
            seq_len = int(final_config.data.time_series.sequence_length)
            max_h = int(max(final_config.data.time_series.prediction_horizons))
            margin_days = seq_len + max_h
            embargo = pd.to_timedelta(embargo_days, unit="D")
            selected_idx = None
            for i, (vs, ve) in enumerate(fold_ranges):
                train_end_eff_i = pd.Timestamp(vs) - embargo
                if train_end_eff_i >= (
                    pd.Timestamp(min_date) + pd.Timedelta(days=margin_days)
                ):
                    selected_idx = i
                    break
            if selected_idx is None:
                # それでもダメなら2番目のfold（存在すれば）を使用
                if len(fold_ranges) >= 2:
                    selected_idx = 1
                    logger.warning(
                        "No fold satisfied warmup margin; falling back to fold #2"
                    )
                else:
                    selected_idx = 0
                    logger.warning(
                        "Only one fold available; CV may result in empty train set. Consider reducing EMBARGO_DAYS or CV_FOLDS."
                    )
            val_start, val_end = fold_ranges[selected_idx]
            train_end_eff = pd.Timestamp(val_start) - embargo
            logger.info(
                f"Using fold#{selected_idx+1}: train_end={train_end_eff.date()} val=({pd.Timestamp(val_start).date()}..{pd.Timestamp(val_end).date()}) embargo={embargo_days}d"
            )
            # Log regime window as MLflow params if available
            try:
                import mlflow as _mlf  # type: ignore

                _mlf.log_param("regime.start", str(pd.Timestamp(val_start).date()))
                _mlf.log_param("regime.end", str(pd.Timestamp(val_end).date()))
                _mlf.log_param("embargo.days", int(embargo_days))
            except Exception:
                pass
            # Regime name (tag): derive from env or from dates
            regime_name = None
            try:
                names_env = os.getenv("REGIME_NAMES", "")
                if names_env.strip():
                    names = [s.strip() for s in names_env.split(",") if s.strip()]
                    if 0 <= selected_idx < len(names):
                        regime_name = names[selected_idx]
                if not regime_name:
                    regime_name = f"{pd.Timestamp(val_start).strftime('%Y%m%d')}_{pd.Timestamp(val_end).strftime('%Y%m%d')}"
            except Exception:
                regime_name = None
            if regime_name:
                # Save for downstream logging and to MLflow params
                os.environ["CURRENT_REGIME_NAME"] = regime_name
                try:
                    import mlflow as _mlf  # type: ignore

                    _mlf.log_param("regime.name", regime_name)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(
                f"Fold selection failed: {e}. Falling back to single-split mode."
            )
            cv_folds = 1

        if cv_folds >= 2:
            train_ds = ProductionDatasetV2(
                data_module.train_files
                if hasattr(data_module, "train_files")
                else all_files,
                final_config,
                mode="train",
                target_scalers=None,
                start_date=None,
                end_date=train_end_eff,
            )
            if len(train_ds) == 0:
                logger.warning(
                    "Train dataset is empty after CV filtering; disabling CV and using default split."
                )
                cv_folds = 1
            else:
                # 学習スケーラfit（学習/検証の両方に適用）
                scalers = {}
                for h in train_ds.prediction_horizons:
                    arr = np.array(train_ds.targets[h], dtype=np.float64)
                    if arr.size > 0:
                        m = float(np.mean(arr))
                        s = float(np.std(arr) + 1e-8)
                        scalers[h] = {"mean": m, "std": s}
                if scalers:
                    train_ds.target_scalers = scalers
                    logger.info(
                        "Applied target z-score normalization to train_ds (CV mode)"
                    )

                # データモジュール側で検証ローダーを提供するため、
                # ここでの検証データセット再構築は行わない
                dlp = _resolve_dl_params(final_config)

                # Default loaders
                def _safe_loader(ds, **kwargs):
                    try:
                        n = len(ds)
                    except Exception:
                        n = 0
                    if n == 0:
                        logger.warning(
                            "[loader] dataset is empty; returning None (skip)"
                        )
                        return None
                    # Reproducible DataLoader: worker_init_fn + generator
                    try:
                        base_seed = int(os.getenv("DL_SEED", "42"))
                        g = torch.Generator()
                        g.manual_seed(base_seed)

                        def _winit(worker_id: int):
                            s = base_seed + worker_id
                            import random
                            import numpy as _np

                            random.seed(s)
                            _np.random.seed(s % (2**32 - 1))
                            torch.manual_seed(s)

                        return DataLoader(
                            ds, worker_init_fn=_winit, generator=g, **kwargs
                        )
                    except Exception:
                        return DataLoader(ds, **kwargs)

                # eval_onlyならtrain_loaderを作らずスキップ
                eval_only = os.getenv("CV_EVAL_ONLY", "1") == "1"
                if eval_only or len(train_ds) == 0:
                    train_loader = None
                    logger.info(
                        "[loader] train_loader is disabled (eval_only or empty train set)."
                    )
                else:
                    train_loader = _safe_loader(
                        train_ds,
                        batch_size=final_config.train.batch.train_batch_size,
                        shuffle=True,
                        num_workers=dlp["num_workers"],
                        pin_memory=dlp["pin_memory"],
                        drop_last=True,
                        prefetch_factor=dlp["prefetch_factor"],
                        persistent_workers=dlp["persistent_workers"],
                        collate_fn=collate_day,
                    )

                # 検証データローダーはDataModuleから取得（特徴量整合はDataModuleで保証）
                val_loader = data_module.val_dataloader()
    else:
        train_loader = data_module.train_dataloader()
        # DataModule handles feature alignment internally
        val_loader = data_module.val_dataloader()

    # Optional: Day-batch sampler (1 day = 1 batch)
    try:
        use_day_batch = os.getenv("USE_DAY_BATCH", "1") == "1"
        use_fixed_sampler = (
            os.getenv("USE_FIXED_SAMPLER", "1") == "1"
        )  # Use fixed version by default
        min_nodes = int(os.getenv("MIN_NODES_PER_DAY", "20"))
        max_batch_size = int(
            os.getenv("MAX_BATCH_SIZE", "2048")
        )  # Enforce batch size limit

        def _build_day_loader(ds, drop_undersized: bool):
            dates = getattr(ds, "sequence_dates", None)
            if dates is None or len(dates) != len(ds):
                return None

            # Use fixed sampler if enabled
            if use_fixed_sampler:
                sampler = DayBatchSamplerFixed(
                    dataset=ds,
                    max_batch_size=max_batch_size,
                    min_nodes_per_day=min_nodes,
                    shuffle=drop_undersized,  # Shuffle for training
                    seed=42,
                )
            else:
                sampler = DayBatchSampler(
                    indices=list(range(len(ds))),
                    dates=dates,
                    min_nodes=min_nodes,
                    drop_undersized=drop_undersized,
                )
            nw = int(os.getenv("NUM_WORKERS", "8"))
            return DataLoader(
                ds,
                batch_sampler=sampler,
                num_workers=nw,
                pin_memory=True,
                persistent_workers=False,
                collate_fn=collate_day,
            )

        if use_day_batch:
            if train_loader is not None and hasattr(train_loader, "dataset"):
                tl2 = _build_day_loader(train_loader.dataset, True)
                if tl2 is not None:
                    train_loader = tl2
            if val_loader is not None and hasattr(val_loader, "dataset"):
                vl2 = _build_day_loader(val_loader.dataset, False)
                if vl2 is not None:
                    val_loader = vl2
            logger.info(f"DayBatchSampler enabled (min_nodes_per_day={min_nodes})")
        # Fallback: ensure collate_day even when not using day-batch
        if (
            val_loader is not None
            and hasattr(val_loader, "dataset")
            and getattr(val_loader, "collate_fn", None) is not collate_day
        ):
            nw = int(os.getenv("NUM_WORKERS", "8"))
            _g2 = torch.Generator()
            _g2.manual_seed(int(os.getenv("DL_SEED", "42")))

            def _winit2(worker_id: int):
                s = int(os.getenv("DL_SEED", "42")) + worker_id
                import random
                import numpy as _np

                random.seed(s)
                _np.random.seed(s % (2**32 - 1))
                torch.manual_seed(s)

            # Respect sandbox: allow single-process val loader
            v_nw = max(0, nw // 2)
            v_kwargs = {
                "dataset": val_loader.dataset,
                "batch_size": final_config.train.batch.val_batch_size,
                "shuffle": False,
                "num_workers": v_nw,
                "pin_memory": bool(os.getenv("PIN_MEMORY", "0") in ("1", "true", "True")),
                "collate_fn": collate_day,
                "worker_init_fn": _winit2,
                "generator": _g2,
            }
            if v_nw > 0:
                v_kwargs["persistent_workers"] = True
                try:
                    v_kwargs["prefetch_factor"] = int(os.getenv("PREFETCH_FACTOR", "4"))
                except Exception:
                    pass
            else:
                v_kwargs["persistent_workers"] = False
            val_loader = DataLoader(**v_kwargs)
        if (
            train_loader is not None
            and hasattr(train_loader, "dataset")
            and getattr(train_loader, "collate_fn", None) is not collate_day
        ):
            nw = int(os.getenv("NUM_WORKERS", "8"))
            _g3 = torch.Generator()
            _g3.manual_seed(int(os.getenv("DL_SEED", "42")))

            def _winit3(worker_id: int):
                s = int(os.getenv("DL_SEED", "42")) + worker_id
                import random
                import numpy as _np

                random.seed(s)
                _np.random.seed(s % (2**32 - 1))
                torch.manual_seed(s)

            # Respect sandbox: use single-process when NUM_WORKERS<=0
            t_nw = max(0, nw)
            t_kwargs = {
                "dataset": train_loader.dataset,
                "batch_size": final_config.train.batch.train_batch_size,
                "shuffle": True,
                "num_workers": t_nw,
                "pin_memory": bool(os.getenv("PIN_MEMORY", "0") in ("1", "true", "True")),
                "drop_last": True,
                "collate_fn": collate_day,
                "worker_init_fn": _winit3,
                "generator": _g3,
            }
            if t_nw > 0:
                t_kwargs["persistent_workers"] = True
                try:
                    t_kwargs["prefetch_factor"] = int(os.getenv("PREFETCH_FACTOR", "4"))
                except Exception:
                    pass
            else:
                t_kwargs["persistent_workers"] = False
            train_loader = DataLoader(**t_kwargs)
    except Exception as _e:
        logger.warning(f"DayBatch/collate setup skipped: {_e}")

    # ---- 入力特徴次元をデータから自動推定し、configへ反映 ----
    def _infer_feature_dim_from_loader(dl) -> int | None:
        if dl is None:
            return None
        try:
            it = iter(dl)
            batch = next(it)
            x = (
                batch["features"]
                if isinstance(batch, dict) and "features" in batch
                else None
            )
            if x is not None and hasattr(x, "shape") and len(x.shape) >= 3:
                return int(x.shape[-1])
        except StopIteration:
            return None
        except Exception as e:
            logger.warning(f"Feature dim inference failed: {e}")
            return None
        return None

    inferred_feature_dim = _infer_feature_dim_from_loader(
        val_loader
    ) or _infer_feature_dim_from_loader(train_loader)
    try:
        if inferred_feature_dim is not None and inferred_feature_dim > 0:
            try:
                prev_dim = getattr(final_config.data.features, "input_dim", None)
            except Exception:
                prev_dim = None
            final_config.data.features.input_dim = int(inferred_feature_dim)
            final_config.data.features.num_features = int(inferred_feature_dim)
            logger.info(
                f"[input_dim] detected from data: F={inferred_feature_dim} (was: {prev_dim})"
            )
        else:
            logger.warning(
                f"[input_dim] could not be inferred from data; using configured value: {getattr(final_config.data.features, 'input_dim', 'unknown')}"
            )
    except Exception as _e:
        logger.warning(f"[input_dim] update skipped: {_e}")

    # 追加: train/val の列差分チェック（Fail-Fast）
    try:
        tr_cols = getattr(getattr(train_loader, "dataset", None), "feature_cols", None)
        va_cols = getattr(getattr(val_loader, "dataset", None), "feature_cols", None)
        if isinstance(tr_cols, list) and isinstance(va_cols, list):
            s_tr, s_va = set(tr_cols), set(va_cols)
            if tr_cols != va_cols:
                only_tr = sorted(list(s_tr - s_va))[:10]
                only_va = sorted(list(s_va - s_tr))[:10]
                msg = f"[feature-cols] mismatch: train={len(tr_cols)} val={len(va_cols)}; only_train={only_tr} only_val={only_va}"
                logger.error(msg)
                raise RuntimeError(msg)
    except Exception:
        pass

    if train_loader is None:
        logger.error("❌ CRITICAL ERROR: Train loader is None!")
        logger.error("This usually means the data directory or split structure is incorrect.")
        logger.error("Expected structure: <data_dir>/{train,val,test}/")
        logger.error("Check data.source.data_dir configuration and ensure splits exist.")
        logger.error("Exiting immediately to avoid wasting time on empty training loops.")
        sys.exit(1)
    else:
        try:
            logger.info(f"✅ Train batches: {len(train_loader)}")
        except Exception:
            logger.info("Train batches: unknown")
    
    if val_loader is None:
        logger.warning("⚠️  Val loader: None (validation disabled)")
    else:
        try:
            val_batches = len(val_loader)
            logger.info(f"✅ Val batches: {val_batches}")
            if val_batches == 0:
                logger.warning(
                    "Validation loader has 0 batches! Check data split configuration."
                )
        except Exception:
            logger.info("Val batches: unknown")

    # Validate label normalization
    if os.getenv("VALIDATE_LABELS", "1") == "1" and train_loader is not None:
        try:
            norm_validator = NormalizationValidator()
            logger.info("Validating label normalization...")

            # Get a sample batch
            sample_batch = next(iter(train_loader))
            if "targets" in sample_batch:
                targets = sample_batch["targets"]

                # Check each horizon
                for horizon_key, horizon_data in targets.items():
                    if torch.is_tensor(horizon_data):
                        # Move to CPU for validation
                        data_cpu = horizon_data.cpu()

                        # Log statistics
                        mean = data_cpu.mean().item()
                        std = data_cpu.std().item()
                        logger.info(
                            f"Target {horizon_key}: mean={mean:.6f}, std={std:.6f}"
                        )

                        # Validate if it should be z-score normalized
                        if abs(mean) > 10 or std > 100:
                            logger.warning(
                                f"Target {horizon_key} may not be properly normalized: "
                                f"mean={mean:.2f}, std={std:.2f}"
                            )
        except Exception as e:
            logger.warning(f"Label normalization validation skipped: {e}")

    # ---- Fail-Fast: 空trainローダ検出（eval-onlyは除外） ----
    try:
        eval_only_flag = (os.getenv("EVAL_ONLY", "0") == "1") or (
            os.getenv("CV_EVAL_ONLY", "0") == "1"
        )
        if (not eval_only_flag) and (train_loader is not None):
            try:
                n_train_batches = len(train_loader)
            except Exception:
                n_train_batches = None
            if n_train_batches is not None and n_train_batches == 0:
                # データ範囲情報（可能なら）
                data_min = None
                data_max = None
                eff_start = None
                try:
                    if (
                        hasattr(data_module, "available_dates")
                        and getattr(data_module, "available_dates") is not None
                        and len(getattr(data_module, "available_dates")) > 0
                    ):
                        data_min = pd.Timestamp(
                            data_module.available_dates[0]
                        ).normalize()
                        data_max = pd.Timestamp(
                            data_module.available_dates[-1]
                        ).normalize()
                        seq_len = int(final_config.data.time_series.sequence_length)
                        max_h = int(
                            max(final_config.data.time_series.prediction_horizons)
                        )
                        eff_start = data_min + pd.tseries.offsets.BDay(
                            seq_len - 1 + max_h
                        )
                except Exception:
                    pass
                # 学習レンジ（可能なら）
                tr = getattr(data_module, "train_range", None)
                train_start = tr[0] if tr else None
                train_end = tr[1] if tr else None
                raise ValueError(
                    f"Train loader is empty. data=[{data_min}..{data_max}] "
                    f"effective_start={getattr(eff_start,'date',lambda:None)() if hasattr(eff_start,'date') else eff_start} "
                    f"train_range=[{train_start}..{train_end}]"
                )
    except Exception as _e:
        logger.error(str(_e))
        raise

    # ---- 1バッチNaNスキャン（早期検知） ----
    try:
        if train_loader is not None:
            it_scan = iter(train_loader)
            first_batch = next(it_scan)
            try:
                if isinstance(first_batch, dict):
                    logger.info(
                        f"[debug-first-batch-keys] {list(first_batch.keys())}"
                    )
                    for _k, _v in list(first_batch.items()):
                        try:
                            logger.info(
                                f"[debug-first-batch-type] {_k}: {type(_v)}"
                            )
                        except Exception:
                            pass
            except Exception:
                pass
            import torch as _torch

            xb = (
                first_batch.get("features", None)
                if isinstance(first_batch, dict)
                else None
            )
            yb = (
                first_batch.get("targets", None)
                if isinstance(first_batch, dict)
                else None
            )
            if _torch.is_tensor(xb):
                if not _torch.isfinite(xb).all():
                    bad = (~_torch.isfinite(xb)).sum().item()
                    raise ValueError(
                        f"Non-finite values in first batch features: {bad} elements"
                    )
            if isinstance(yb, dict):
                for k, v in yb.items():
                    if _torch.is_tensor(v) and (not _torch.isfinite(v).all()):
                        bad = (~_torch.isfinite(v)).sum().item()
                        raise ValueError(
                            f"Non-finite values in first batch targets[{k}]: {bad} elements"
                        )
    except StopIteration:
        # すでにlen==0でFail-Fastしているため通常は到達しない
        pass
    except Exception as _scan_e:
        logger.error(f"Pre-train NaN scan failed: {_scan_e}")
        raise

    # ランマニフェスト（再現用メタ）を作成
    def _get_git_commit() -> str:
        try:
            return (
                subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            )
        except Exception:
            return "unknown"

    def _env_subset() -> dict:
        keys = [
            "HWEIGHTS",
            "DYN_WEIGHT",
            "DYN_ALPHA",
            "DYN_FREEZE_FRAC",
            "USE_HUBER",
            "HUBER_DELTA",
            "HUBER_WEIGHT",
            "USE_PINBALL",
            "PINBALL_WEIGHT",
            "ENABLE_STUDENT_T",
            "USE_T_NLL",
            "NLL_WEIGHT",
            "ENABLE_DIRECTION",
            "USE_DIR_AUX",
            "DIR_AUX_WEIGHT",
            "SIGMA_WEIGHT_LAMBDA",
            "TARGET_VOL_NORM",
            "USE_TS_MIXUP",
            "TS_MIXUP_PROB",
            "TS_MIXUP_ALPHA",
            "USE_DAY_BATCH",
            "MIN_NODES_PER_DAY",
            "NUM_WORKERS",
            "PREFETCH_FACTOR",
            "SNAPSHOT_ENS",
            "SNAPSHOT_NUM",
            "CV_FOLDS",
            "EMBARGO_DAYS",
            "EVAL_SPACE",
            "EVAL_CALIBRATED",
        ]
        out = {}
        for k in keys:
            if k in os.environ:
                out[k] = os.environ.get(k)
        for k, v in os.environ.items():
            if (
                k.startswith(
                    ("USE_", "DYN_", "MODEL_", "INPUT_", "VSN_", "LSTM_", "HEAD_")
                )
                and k not in out
            ):
                out[k] = v
        return out

    def _file_manifest(paths) -> list:
        items = []
        import hashlib

        for p in paths or []:
            try:
                st = os.stat(p)
                # Calculate partial hash (first 1MB) for quick fingerprinting
                h = ""
                try:
                    with open(p, "rb") as f:
                        h = hashlib.md5(f.read(1024 * 1024)).hexdigest()
                except Exception:
                    pass
                items.append(
                    {
                        "path": str(p),
                        "size": int(st.st_size),
                        "mtime": float(st.st_mtime),
                        "md5_head": h,
                    }
                )
            except Exception:
                items.append({"path": str(p)})
        return items

    run_manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "timestamp_jst": now_jst_iso(),
        "git_commit": _get_git_commit(),
        "device": str(device),
        "random_seed": actual_seed,
        "deterministic": deterministic,
        "torch": torch.__version__,
        "config": OmegaConf.to_container(final_config, resolve=True),
        "env": _env_subset(),
        "data_source_dir": str(getattr(final_config.data.source, "data_dir", "")),
        "train_files": _file_manifest(getattr(data_module, "train_files", [])),
        "val_files": _file_manifest(getattr(data_module, "val_files", [])),
        "test_files": _file_manifest(getattr(data_module, "test_files", [])),
        # 追加: 使用した特徴列の保存（再現性）
        "feature_cols": (
            getattr(getattr(train_loader, "dataset", None), "feature_cols", None) or []
        ),
    }
    try:
        Path("logs/manifests").mkdir(parents=True, exist_ok=True)
        Path("models/manifests").mkdir(parents=True, exist_ok=True)
        mpath = (
            Path("logs/manifests")
            / f"train_manifest_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        )
        mpath.write_text(json.dumps(run_manifest, ensure_ascii=False))
        Path("models/manifests/latest_train_manifest.json").write_text(
            json.dumps(run_manifest, ensure_ascii=False)
        )
    except Exception:
        pass

    # モデル
    # SimpleTestModel: 最小構成テスト用の単純な線形モデル
    class SimpleTestModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            # 入力次元と出力次元を設定から取得
            input_dim = int(getattr(config.data.features, "input_dim", 13))
            sequence_length = int(
                getattr(config.data.time_series, "sequence_length", 20)
            )
            horizons = int(getattr(config.data.time_series, "horizons", 10))

            # 単純な線形層：(batch, seq_len, features) -> (batch, 5 horizons, 1)
            # 実際のhorizonsは[1,2,3,5,10]の5つ
            self.linear = nn.Linear(input_dim * sequence_length, 5)

            logger.warning(
                f"SimpleTestModel initialized: input={input_dim}*{sequence_length}={input_dim*sequence_length}, output={horizons}"
            )

        def forward(self, x, edge_index=None, edge_weight=None, return_attns=False):
            # x: (batch, seq_len, features)
            batch_size = x.size(0)

            # フラット化
            x_flat = x.view(batch_size, -1)  # (batch, seq_len * features)

            # 線形変換
            out = self.linear(x_flat)  # (batch, 5)

            # 出力形状を合わせる: (batch, 5, 1)
            out = out.unsqueeze(-1)

            # MultiHorizonLossが期待する形式で返す
            # out: (batch, horizons=10, 1)を各horizonに分割
            outputs = {}
            for i, h in enumerate([1, 5, 10, 20]):
                if i < out.size(1):
                    outputs[f"horizon_{h}"] = out[:, i, :]  # (batch, 1)

            if return_attns:
                return outputs, {}
            return outputs

    logger.info("Initializing model...")
    try:
        # デバッグ用：シンプルモデルを使用する場合
        use_simple_model = os.getenv("USE_SIMPLE_MODEL", "0") == "1"
        if use_simple_model:
            logger.warning("Using SimpleTestModel for debugging!")
            model = SimpleTestModel(final_config).to(device)
        elif ATFT_GAT_FAN is not None:
            model = ATFT_GAT_FAN(final_config).to(device)
            # Sanity: ensure PE length >= sequence_length if model exposes config
            try:
                seq_len = int(
                    getattr(final_config.data.time_series, "sequence_length", 20)
                )
                # If model has positional encoding module, attempt a dummy call with seq_len to pre-warm
                with torch.no_grad():
                    dummy = torch.zeros(
                        1,
                        seq_len,
                        int(getattr(final_config.data.features, "input_dim", 13)),
                        device=device,
                    )
                    _ = model(dummy)
            except Exception:
                pass
            # Enable gradient checkpointing through model config if set
            try:
                if bool(
                    getattr(
                        final_config.model.optimization, "gradient_checkpointing", False
                    )
                ):
                    for m in model.modules():
                        if hasattr(m, "gradient_checkpointing"):
                            m.gradient_checkpointing = True
            except Exception:
                pass
            # Optional: channels_last memory format for Conv-like ops (safe even without conv)
            if getattr(final_config.hardware, "channels_last", False):
                model = model.to(memory_format=torch.channels_last)
            # Disable compile by default for stability unless explicitly enabled
            try:
                if getattr(final_config.model.optimization, "compile", False) is True:
                    model = torch.compile(model, mode="default", dynamic=False)
            except Exception:
                pass
            logger.info(
                f"ATFT-GAT-FAN model parameters: {sum(p.numel() for p in model.parameters()):,}"
            )

            # Apply runtime guards
            try:
                from runtime_guards import apply_all_guards

                apply_all_guards(model)
            except ImportError:
                logger.warning("runtime_guards module not found, skipping guards")
            except Exception as e:
                logger.warning(f"Failed to apply runtime guards: {e}")

            # Add initial noise to prediction heads to escape constant solution
            head_noise_std = float(os.getenv("HEAD_NOISE_STD", "0.0"))
            # Apply once at initialization (avoid referencing training's global_step)
            if head_noise_std > 0:
                with torch.no_grad():
                    for name, module in model.named_modules():
                        if "point_head" in name or "horizon" in name:
                            for param in module.parameters():
                                if param.requires_grad:
                                    noise = torch.randn_like(param) * head_noise_std
                                    param.add_(noise)
                            logger.info(
                                f"Added initial noise (std={head_noise_std}) to {name}"
                            )
        else:
            raise ImportError("ATFT_GAT_FAN not available")
    except Exception as e:
        logger.error(f"Failed to initialize ATFT-GAT-FAN: {e}")
        # Re-raise the exception to prevent fallback to SimpleLSTM
        raise RuntimeError(f"ATFT-GAT-FAN initialization failed: {e}") from e

    # Mixed Precision用のScaler（保守的設定 - 緊急修正版）
    # torch.amp API に更新（FutureWarning回避）
    # bf16ではスケーリング不要なので無効化
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=1024.0,  # 小さめの初期スケール（元: 65536）
        growth_factor=1.5,  # 緩やかな成長（元: 2.0）
        backoff_factor=0.5,
        growth_interval=500,  # 頻繁に調整（元: 1000）
        # bf16ではスケーリング無効化（再現性・安定性向上）
        enabled=(use_amp and amp_dtype == torch.float16),
    )

    # 最適化
    # 追加損失のON/OFFは簡易に環境変数で制御（必要ならHydraへ昇格）
    use_rankic = os.getenv("USE_RANKIC", "0") == "1"
    use_pinball = os.getenv("USE_PINBALL", "0") == "1"
    rankic_w = float(os.getenv("RANKIC_WEIGHT", "0.5")) if use_rankic else 0.0
    pinball_w = float(os.getenv("PINBALL_WEIGHT", "0.3")) if use_pinball else 0.0
    # モデルconfigの分位を損失側へ反映
    q_list = None
    try:
        q_cfg = final_config.model.prediction_head.output.quantile_prediction
        if getattr(q_cfg, "enabled", False):
            q_list = list(q_cfg.quantiles)
    except Exception:
        q_list = None

    # Horizon重み（プリセット）: 環境変数 HWEIGHTS="1:0.35,5:0.20,10:0.20,20:0.25"
    def _parse_weight_map(env_val: str | None):
        if not env_val:
            return None
        try:
            w = {}
            for kv in env_val.split(","):
                k, v = kv.split(":")
                w[int(k.strip())] = float(v.strip())
            s = sum(w.values())
            if s > 0:
                for k in list(w.keys()):
                    w[k] = w[k] / s
            return w
        except Exception:
            return None

    # Resolve horizon weights with robust fallbacks:
    # 1) If env HWEIGHTS is provided, parse it.
    # 2) Else, try Hydra config: train.loss.horizon_weights (dict or list) or prediction.horizon_weights (list).
    # 3) Else, derive from data horizons using sqrt-inv scheme and normalize.
    preset_w = _parse_weight_map(os.getenv("HWEIGHTS", None))

    # CS-IC 補助ロスの制御（デフォルトON, λ=0.05）
    use_cs_ic_env = os.getenv("USE_CS_IC", "1") == "1"
    cs_ic_weight_env = float(os.getenv("CS_IC_WEIGHT", "0.05"))

    # Horizon resolution and consistency (auto-fix by default)
    data_h_list = list(getattr(final_config.data.time_series, "prediction_horizons", [1, 5, 10, 20]))
    data_h_set = set(int(h) for h in data_h_list)

    def _normalize_weights_map(wmap: dict[int, float] | None) -> dict[int, float] | None:
        if not wmap:
            return None
        try:
            s = float(sum(float(v) for v in wmap.values()))
            if s <= 0:
                return None
            return {int(k): float(v) / s for k, v in wmap.items()}
        except Exception:
            return None

    def _from_cfg_or_default() -> dict[int, float]:
        # Try Hydra config first
        try:
            from omegaconf import OmegaConf  # local import to avoid top-level dep
            cfg_map = OmegaConf.select(final_config, "train.loss.horizon_weights")
            if isinstance(cfg_map, dict):
                parsed = {int(k): float(v) for k, v in cfg_map.items()}
                nw = _normalize_weights_map(parsed)
                if nw and set(nw.keys()) == data_h_set:
                    return nw
            # list alignment case (train.loss.horizon_weights or prediction.horizon_weights)
            if isinstance(cfg_map, list) and len(cfg_map) == len(data_h_list):
                parsed = {int(h): float(w) for h, w in zip(data_h_list, cfg_map)}
                nw = _normalize_weights_map(parsed)
                if nw:
                    return nw
            pred_hw = OmegaConf.select(final_config, "prediction.horizon_weights")
            if isinstance(pred_hw, list) and len(pred_hw) == len(data_h_list):
                parsed = {int(h): float(w) for h, w in zip(data_h_list, pred_hw)}
                nw = _normalize_weights_map(parsed)
                if nw:
                    return nw
        except Exception:
            pass
        # Fallback: sqrt-inv scheme
        import math as _m
        base = {int(h): 1.0 / _m.sqrt(float(h)) for h in data_h_list}
        return _normalize_weights_map(base) or {int(h): 1.0 / len(data_h_list) for h in data_h_list}

    if preset_w is None:
        preset_w = _from_cfg_or_default()
    else:
        # If provided but mismatched, auto-fix unless STRICT_HWEIGHTS=1
        weight_h_set = set(int(k) for k in preset_w.keys())
        if weight_h_set != data_h_set:
            if os.getenv("STRICT_HWEIGHTS", "0") == "1":
                logger.error(f"Horizon mismatch: data={data_h_set}, weights={weight_h_set}")
                raise ValueError(f"Horizon weights must match data horizons: {data_h_set}")
            # Auto-correct: drop extras, fill missing with sqrt-inv, then normalize
            import math as _m
            fixed = {int(h): float(preset_w[h]) for h in data_h_list if h in preset_w}
            for h in data_h_list:
                if int(h) not in fixed:
                    fixed[int(h)] = 1.0 / _m.sqrt(float(h))
            preset_w = _normalize_weights_map(fixed)
            logger.warning(
                f"[HWEIGHTS] Auto-corrected weights to match horizons {sorted(data_h_list)}"
            )

    criterion = MultiHorizonLoss(
        horizons=final_config.data.time_series.prediction_horizons,
        use_rankic=use_rankic,
        rankic_weight=rankic_w,
        use_cs_ic=use_cs_ic_env,
        cs_ic_weight=cs_ic_weight_env,
        use_pinball=use_pinball,
        quantiles=tuple(q_list) if q_list else (0.2, 0.5, 0.8),
        pinball_weight=pinball_w,
        use_t_nll=(os.getenv("USE_T_NLL", "0") == "1"),
        nll_weight=float(os.getenv("NLL_WEIGHT", "0.7"))
        if os.getenv("USE_T_NLL", "0") == "1"
        else 0.0,
        use_huber=(os.getenv("USE_HUBER", "1") == "1"),
        huber_delta=float(os.getenv("HUBER_DELTA", "1.0")),
        huber_weight=float(os.getenv("HUBER_WEIGHT", "0.3")),
        h1_loss_mult=float(os.getenv("H1_LOSS_MULT", "1.5")),
        horizon_weights=preset_w,
        use_dynamic_weighting=(os.getenv("DYN_WEIGHT", "1") == "1"),
        dynamic_alpha=float(os.getenv("DYN_ALPHA", "0.01")),
        dynamic_freeze_frac=float(os.getenv("DYN_FREEZE_FRAC", "0.6")),
        direction_aux_weight=float(os.getenv("DIR_AUX_WEIGHT", "0.1"))
        if os.getenv("USE_DIR_AUX", "1") == "1"
        else 0.0,
        sigma_weighting_lambda=float(os.getenv("SIGMA_WEIGHT_LAMBDA", "0.0")),
        pred_var_min=float(os.getenv("PRED_VAR_MIN", "0.005")),
        pred_var_weight=float(os.getenv("PRED_VAR_WEIGHT", "0.1")),
    )
    # LARS/LAMB オプション（環境変数で簡易切替）。未インストールなら AdamW へフォールバック
    opt_choice = os.getenv("OPTIMIZER", "adamw").lower()
    optimizer = None
    if opt_choice == "lars":
        try:
            from torch.optim import SGD

            # 近似的にSGD + trust ratio（簡易）→実用はapex/LARS推奨
            optimizer = SGD(
                model.parameters(),
                lr=final_config.train.optimizer.lr,
                momentum=0.9,
                weight_decay=final_config.train.optimizer.weight_decay,
            )
        except Exception:
            pass
    elif opt_choice == "lamb":
        try:
            import torch_optimizer as topt

            optimizer = topt.Lamb(
                model.parameters(),
                lr=final_config.train.optimizer.lr,
                weight_decay=final_config.train.optimizer.weight_decay,
            )
        except Exception:
            pass
    if optimizer is None:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=final_config.train.optimizer.lr,
            weight_decay=final_config.train.optimizer.weight_decay,
        )

    # Optimizer param_groups audit (fail-fast for empty/missing params)
    try:
        total_trainable = sum(
            int(p.requires_grad) * p.numel() for p in model.parameters()
        )
        opt_params = sum(
            p.numel() for g in optimizer.param_groups for p in g.get("params", [])
        )
        assert len(optimizer.param_groups) > 0, "Optimizer has no param_groups"
        assert all(
            len(g.get("params", [])) > 0 for g in optimizer.param_groups
        ), "Empty param_group detected in optimizer; check requires_grad and grouping"
        # If optimizer covers suspiciously few params, raise to catch mis-wiring
        if opt_params == 0 or (
            total_trainable > 0 and opt_params < max(1, int(0.5 * total_trainable))
        ):
            raise AssertionError(
                f"OPT-AUDIT: optimizer params {opt_params} << trainable {total_trainable}"
            )
        logger.info(
            f"[OPT-AUDIT] ✓ Optimizer covers {opt_params}/{total_trainable} trainable params"
        )
    except Exception as _e:
        logger.error(f"[OPT-AUDIT][FAIL] {_e}")
        raise

    # 学習率スケジューラー（線形ウォームアップ→Cosine）
    warmup_epochs = int(final_config.train.scheduler.warmup_epochs)
    total_epochs = int(final_config.train.scheduler.total_epochs)

    def lr_lambda(epoch_idx: int):
        if epoch_idx < warmup_epochs:
            return float(epoch_idx + 1) / max(1, warmup_epochs)
        # 残りをCosine
        progress = (epoch_idx - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, max(0.0, progress))))

    # Warmup + Cosine（ピークを抑える）
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # === Helpers: K(近傍数)スケジュール & GAT edge-dropoutスケジュール ===
    def _three_phase_schedule(
        epoch_idx: int,
        total_epochs: int,
        v0,
        v1,
        v2,
        b1: float = 0.33,
        b2: float = 0.66,
    ):
        """0..b1: v0, b1..b2: v1, b2..1.0: v2"""
        r = float(epoch_idx) / max(1, int(total_epochs))
        if r < b1:
            return v0
        if r < b2:
            return v1
        return v2

    def _set_gat_knn_k(model, k: int):
        try:
            if hasattr(model, "set_knn_k"):
                model.set_knn_k(int(k))
            else:
                # フォールバック: configを書き換え
                model.config.model.gat.knn_k = int(k)
        except Exception:
            pass

    def _set_gat_edge_dropout(model, p: float):
        try:
            if (
                hasattr(model, "gat")
                and model.gat is not None
                and hasattr(model.gat, "layers")
            ):
                for lyr in model.gat.layers:
                    if hasattr(lyr, "edge_dropout"):
                        lyr.edge_dropout = float(p)
        except Exception:
            pass

    def run_training(train_loader, val_loader, tag: str = "main"):
        n_epochs = final_config.train.scheduler.total_epochs
        best_val_loss = float("inf")
        logger.info(f"Starting training loop ({tag})...")

        # Time budget configuration
        time_budget_hours = float(os.getenv("TIME_BUDGET_HOURS", "2.0"))
        time_budget_seconds = time_budget_hours * 3600
        eval_every_steps = int(os.getenv("EVAL_EVERY_STEPS", "100"))
        heartbeat_interval = int(os.getenv("HEARTBEAT_INTERVAL", "30"))  # seconds

        start_time = time.time()
        last_heartbeat = time.time()
        global_step = 0
        time_exceeded = False

        # Initialize degeneracy detection tracking
        from collections import defaultdict

        deg_bad = defaultdict(int)  # Consecutive violation counter
        deg_ema = defaultdict(float)  # EMA of prediction std

        logger.info(f"Time budget: {time_budget_hours:.1f} hours")
        logger.info(f"Eval every: {eval_every_steps} steps")
        logger.info(f"Heartbeat interval: {heartbeat_interval}s")
        # 外部GraphBuilder（使用可能なら優先）。失敗時はNoneでフォールバック
        gb = None
        gb_adv = None
        try:
            gb_cfg = getattr(final_config.data, "graph_builder", None)
            if gb_cfg is not None:
                gbc = GBConfig(
                    source_glob=str(
                        getattr(gb_cfg, "source_glob", "data/ml/*.parquet")
                    ),
                    lookback=int(getattr(gb_cfg, "lookback", 60)),
                    k=int(getattr(gb_cfg, "k", 15)),
                    ewm_halflife=int(getattr(gb_cfg, "ewm_halflife", 20)),
                    shrinkage_gamma=float(getattr(gb_cfg, "shrinkage_gamma", 0.05)),
                    min_obs=int(getattr(gb_cfg, "min_obs", 40)),
                    size_tau=float(getattr(gb_cfg, "size_tau", 1.0)),
                    cache_dir=str(getattr(gb_cfg, "cache_dir", "graph_cache")),
                    return_cols=tuple(
                        getattr(
                            gb_cfg,
                            "return_cols",
                            (
                                "label_excess_1_bps",
                                "label_ret_1_bps",
                                "ret_1d",
                                "return_1d",
                            ),
                        )
                    ),
                    sector_col=str(getattr(gb_cfg, "sector_col", "sector")),
                    log_mktcap_col=str(getattr(gb_cfg, "log_mktcap_col", "log_mktcap")),
                    method=str(getattr(gb_cfg, "method", "ewm_demean")),
                    symmetric=bool(getattr(gb_cfg, "symmetric", True)),
                )
                gb = GraphBuilder(gbc)
                logger.info(
                    f"[GraphBuilder] initialized from {gbc.source_glob} (lookback={gbc.lookback}, k={gbc.k})"
                )
            # Advanced FinancialGraphBuilder for training (optional)
            use_adv_train = os.getenv("USE_ADV_GRAPH_TRAIN", "0") in ("1", "true", "True")
            try:
                if not use_adv_train and gb_cfg is not None:
                    use_adv_train = bool(getattr(gb_cfg, "use_in_training", False))
            except Exception:
                pass
            if AdvFinancialGraphBuilder is not None and use_adv_train:
                try:
                    # Prefer Hydra config if available
                    try:
                        gb_cfg = getattr(final_config.data, "graph_builder", None)
                    except Exception:
                        gb_cfg = None
                    corr_method = (
                        str(getattr(gb_cfg, "method", "ewm_demean")) if gb_cfg is not None else "ewm_demean"
                    )
                    ewm_hl = int(
                        getattr(gb_cfg, "ewm_halflife", int(os.getenv("EWM_HALFLIFE", "20")))
                        if gb_cfg is not None
                        else int(os.getenv("EWM_HALFLIFE", "20"))
                    )
                    shrink_g = float(
                        getattr(gb_cfg, "shrinkage_gamma", float(os.getenv("SHRINKAGE_GAMMA", "0.05")))
                        if gb_cfg is not None
                        else float(os.getenv("SHRINKAGE_GAMMA", "0.05"))
                    )
                    symm = bool(
                        getattr(gb_cfg, "symmetric", os.getenv("GRAPH_SYMMETRIC", "1") in ("1", "true", "True"))
                        if gb_cfg is not None
                        else (os.getenv("GRAPH_SYMMETRIC", "1") in ("1", "true", "True"))
                    )
                    k_per = int(
                        getattr(gb_cfg, "k", int(os.getenv("GRAPH_K", "10")))
                        if gb_cfg is not None
                        else int(os.getenv("GRAPH_K", "10"))
                    )
                    thr = float(
                        getattr(gb_cfg, "edge_threshold", float(os.getenv("GRAPH_EDGE_THR", "0.3")))
                        if gb_cfg is not None
                        else float(os.getenv("GRAPH_EDGE_THR", "0.3"))
                    )
                    gb_adv = AdvFinancialGraphBuilder(
                        correlation_window=int(getattr(final_config.data.time_series, "sequence_length", 20)),
                        correlation_threshold=thr,
                        max_edges_per_node=k_per,
                        correlation_method=corr_method,
                        ewm_halflife=ewm_hl,
                        shrinkage_gamma=shrink_g,
                        symmetric=symm,
                        cache_dir="graph_cache",
                        verbose=True,
                    )
                    logger.info(
                        f"[AdvGraph] Enabled training-time FinancialGraphBuilder (method={corr_method}, k={k_per}, thr={thr})"
                    )
                except Exception as _e:
                    logger.warning(f"[AdvGraph] init failed: {_e}")
                    gb_adv = None
        except Exception as _e:
            logger.warning(
                f"[GraphBuilder] unavailable; fallback to dynamic KNN. reason={_e}"
            )
        snapshot_ens = os.getenv("SNAPSHOT_ENS", "0") == "1"
        snapshot_num = int(os.getenv("SNAPSHOT_NUM", "4")) if snapshot_ens else 0
        snapshot_points = set()
        if snapshot_ens and snapshot_num > 0:
            for k in range(1, snapshot_num + 1):
                ep = max(1, int(round(k * n_epochs / (snapshot_num + 1))))
                snapshot_points.add(ep)
        # SWA/EMA 準備（SWA優先）
        use_swa = os.getenv("USE_SWA", "1") == "1"
        swa_start_frac = float(os.getenv("SWA_START_FRAC", "0.67"))
        swa_lr_factor = float(os.getenv("SWA_LR_FACTOR", "0.5"))
        swa_model = None
        swa_scheduler = None
        if use_swa:
            try:
                from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

                swa_model = AveragedModel(model)
                swa_scheduler = SWALR(
                    optimizer, swa_lr=final_config.train.optimizer.lr * swa_lr_factor
                )
                logger.info(
                    f"SWA enabled: averaging parameters after {swa_start_frac:.2f} of epochs (lr_factor={swa_lr_factor:.2f})"
                )
            except Exception as _e:
                logger.warning(f"SWA init failed: {_e}")
                swa_model = None
                swa_scheduler = None

        # First batch probe before main training
        if train_loader is not None:
            try:
                first_batch_probe(model, train_loader, device, n=3)
                logger.info("First batch probe passed")

                # Sampler check - verify batch size
                first_batch = next(iter(train_loader))
                if isinstance(first_batch, dict) and "features" in first_batch:
                    actual_batch_size = len(first_batch["features"])
                    configured_batch_size = final_config.train.batch.train_batch_size
                    logger.info(
                        f"[SamplerCheck] first_batch_size={actual_batch_size} "
                        f"(configured={configured_batch_size})"
                    )
                    if actual_batch_size > configured_batch_size * 1.5:
                        logger.warning(
                            f"[SamplerCheck] Batch size {actual_batch_size} significantly exceeds "
                            f"configured {configured_batch_size}!"
                        )
            except Exception as e:
                logger.error(f"First batch probe failed: {e}")
                write_failure_report(e)
                raise

        for epoch in range(1, n_epochs + 1):
            if time_exceeded:
                logger.info(f"Time budget exceeded after {epoch-1} epochs")
                break

            # Set epoch on model for head noise control
            if hasattr(model, "_epoch"):
                model._epoch = epoch

            # エポック文脈を損失に通知（動的重みfreeze用）
            try:
                criterion.set_epoch_context(epoch, n_epochs)
            except Exception:
                pass
            # カリキュラム重み（前半強化）: 環境変数が無ければ推奨値
            try:
                if epoch == 2:
                    w2 = _parse_weight_map(
                        os.getenv(
                            "WEIGHTS_EPOCH2", "1:0.45,2:0.15,3:0.10,5:0.20,10:0.10"
                        )
                    )
                    if w2:
                        criterion.set_preset_weights(w2)
                        logger.info(f"Applied curriculum weights@epoch2: {w2}")
                elif epoch == 3:
                    w3 = _parse_weight_map(
                        os.getenv(
                            "WEIGHTS_EPOCH3", "1:0.35,2:0.15,3:0.15,5:0.20,10:0.15"
                        )
                    )
                    if w3:
                        criterion.set_preset_weights(w3)
                        logger.info(f"Applied preset weights@epoch3: {w3}")
            except Exception:
                pass
            logger.info(f"\n{'='*50}")
            logger.info(f"[{tag}] Epoch {epoch}/{n_epochs}")
            logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

            # K/edge_dropout スケジュール（envで上書き可能）
            try:
                k0 = int(os.getenv("K0", "10"))
                k1 = int(os.getenv("K1", "15"))
                k2 = int(os.getenv("K2", "20"))
                knn_k = int(_three_phase_schedule(epoch, n_epochs, k0, k1, k2))
                _set_gat_knn_k(model, knn_k)
                p0 = float(os.getenv("EDGE_DROPOUT0", "0.20"))
                p1 = float(os.getenv("EDGE_DROPOUT1", "0.10"))
                p2 = float(os.getenv("EDGE_DROPOUT2", "0.05"))
                edge_dp = float(_three_phase_schedule(epoch, n_epochs, p0, p1, p2))
                _set_gat_edge_dropout(model, edge_dp)
                logger.info(
                    f"[sched] epoch={epoch} knn_k={knn_k} edge_dropout={edge_dp:.2f}"
                )
            except Exception as _e:
                logger.warning(f"K/edge_dropout schedule skipped: {_e}")
            if train_loader is not None:
                # custom train loop with micro-batching per day to avoid OOM
                model.train()
                total_loss = 0.0
                horizon_losses = {f"horizon_{h}": 0.0 for h in criterion.horizons}
                n_micro_steps = 0
                pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
                optimizer.zero_grad()
                micro_bs = int(os.getenv("MICRO_BATCH_SIZE", "2048"))
                grad_accum = int(
                    getattr(final_config.train.batch, "gradient_accumulation_steps", 1)
                )

                def _supports_graph(mdl) -> bool:
                    try:
                        if hasattr(mdl, "gat") and getattr(mdl, "gat") is not None:
                            return True
                        sig = inspect.signature(mdl.forward)
                        return ("edge_index" in sig.parameters) or (
                            "edge_attr" in sig.parameters
                        )
                    except Exception:
                        return False

                def _forward_with_optional_graph(mdl, feats, ei, ea):
                    try:
                        if ei is not None and ea is not None and _supports_graph(mdl):
                            return mdl(feats, edge_index=ei, edge_attr=ea)
                        return mdl(feats)
                    except TypeError:
                        return mdl(feats)

                for batch_idx, batch in enumerate(pbar):
                    # Check time budget
                    current_time = time.time()
                    elapsed = current_time - start_time
                    if elapsed > time_budget_seconds:
                        time_exceeded = True
                        logger.info(
                            f"Time budget exceeded at epoch {epoch}, batch {batch_idx}"
                        )
                        break

                    # Heartbeat monitoring
                    if current_time - last_heartbeat > heartbeat_interval:
                        last_heartbeat = current_time
                        heartbeat_info = {
                            "status": "running",
                            "epoch": epoch,
                            "batch": batch_idx,
                            "global_step": global_step,
                            "elapsed_hours": elapsed / 3600,
                            "timestamp": now_jst_iso(),
                            "tz": str(JST.key) if hasattr(JST, "key") else "Asia/Tokyo",
                            "pred_std_ema": {
                                h: deg_ema.get(f"h{h}", float("nan"))
                                for h in criterion.horizons
                            },
                            "deg_bad_consec": {
                                h: deg_bad.get(f"h{h}", 0) for h in criterion.horizons
                            },
                        }
                        (RUN_DIR / "heartbeat.json").write_text(
                            json.dumps(heartbeat_info, ensure_ascii=False, indent=2)
                        )
                    feats_full = batch["features"]
                    targs_full = (
                        batch["targets"]
                        if isinstance(batch.get("targets"), dict)
                        else {"horizon_1": batch["targets"]}
                    )
                    # Get valid masks from batch if available
                    valid_masks_full = batch.get("valid_mask", None)
                    n_items = int(feats_full.size(0))
                    # Build external edges once per day-batch
                    edge_index = None
                    edge_attr = None
                    if (
                        gb is not None
                        and getattr(final_config.model.gat, "enabled", False)
                        and getattr(final_config.model.gat, "edge_features", None)
                    ):
                        try:
                            codes = (
                                batch.get("codes")
                                if "codes" in batch
                                else batch.get("code")
                            )
                            date = batch.get("date", None)
                            if isinstance(date, (list, tuple)) and len(date) > 0:
                                date = date[0]
                            if codes is not None and date is not None:
                                try:
                                    if hasattr(codes, "tolist"):
                                        codes = codes.tolist()
                                    codes = [str(c) for c in codes]
                                except Exception:
                                    pass
                                # Prefer advanced builder if available
                                if gb_adv is not None:
                                    try:
                                        df_pd = None
                                        try:
                                            if hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "data"):
                                                df_pd = train_loader.dataset.data
                                        except Exception:
                                            df_pd = None
                                        if df_pd is not None:
                                            res = gb_adv.build_graph(df_pd, codes, date_end=str(date))
                                            ei, ea = res.get("edge_index"), res.get("edge_attr")
                                        else:
                                            ei, ea = None, None
                                    except Exception as _e:
                                        logger.warning(f"[AdvGraph] build failed: {_e}")
                                        ei, ea = None, None
                                else:
                                    ei, ea = gb.build_for_day(date, codes)
                                # Read-side edge timestamp guard with actual asof from builder if available
                                try:
                                    import pandas as _pd

                                    batch_ts = _pd.Timestamp(date).normalize()
                                    asof_ts = getattr(
                                        gb, "last_asof_ts", lambda: None
                                    )()
                                    if asof_ts is None:
                                        asof_ts = batch_ts - _pd.Timedelta(days=1)
                                    staleness_days = int((batch_ts - asof_ts).days)
                                    max_stale = int(
                                        os.getenv("EDGE_STALENESS_MAX_DAYS", "7")
                                    )
                                    if staleness_days > max_stale:
                                        logger.warning(
                                            f"[EDGE-TS] Staleness {staleness_days}d exceeds max {max_stale}d; dropping edges"
                                        )
                                        ei, ea = None, None
                                    else:
                                        logger.info(
                                            f"[EDGE-TS] asof={asof_ts.date()} staleness_days={staleness_days}"
                                        )
                                        # Log to MLflow if available
                                        try:
                                            import mlflow as _mlf  # type: ignore

                                            _mlf.log_metric(
                                                "edge/staleness_days",
                                                float(staleness_days),
                                                step=global_step,
                                            )
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                                if ei is not None and ea is not None:
                                    edge_index = ei.to(device, non_blocking=True)
                                    edge_attr = ea.to(device, non_blocking=True)
                        except Exception:
                            edge_index = None
                            edge_attr = None

                    # Fallback: build correlation-based edges from this day-batch if graph builder unavailable
                    if (
                        edge_index is None
                        and getattr(final_config.model, "gat", None) is not None
                        and getattr(final_config.model.gat, "enabled", False)
                        and isinstance(feats_full, torch.Tensor)
                        and feats_full.dim() == 3
                    ):
                        try:
                            # Resolve per-sample codes (full day-batch) for enrichment
                            codes_list = None
                            try:
                                codes_list = batch.get("codes") if "codes" in batch else batch.get("code")
                                if hasattr(codes_list, "tolist"):
                                    codes_list = codes_list.tolist()
                                if codes_list is not None:
                                    codes_list = [str(c) for c in codes_list]
                            except Exception:
                                codes_list = None
                            markets_list = None
                            sectors_list = None
                            try:
                                markets_list = batch.get("markets")
                                sectors_list = batch.get("sectors")
                            except Exception:
                                pass
                            # Determine parameters
                            try:
                                k_try = int(getattr(final_config.model.gat, "knn_k", 10))
                            except Exception:
                                k_try = 10
                            # Threshold: prefer graph_builder.edge_threshold, then data.graph.edge_threshold, else env/default
                            try:
                                thr = None
                                try:
                                    thr = float(getattr(getattr(final_config.data, "graph_builder", {}), "edge_threshold"))
                                except Exception:
                                    pass
                                if thr is None:
                                    try:
                                        thr = float(getattr(getattr(final_config.data, "graph", {}), "edge_threshold"))
                                    except Exception:
                                        thr = None
                                if thr is None:
                                    thr = float(os.getenv("GRAPH_EDGE_THR", "0.3"))
                            except Exception:
                                thr = 0.3
                            # Use local correlation edge builder (batch-level)
                            from src.graph.graph_builder import GraphBuilder as _GBL, GBConfig as _GBC

                            _gb_local = _GBL(
                                _GBC(max_nodes=int(feats_full.size(0)), edge_threshold=float(thr))
                            )
                            win = int(min(feats_full.size(1), 20))
                            ei, ea = _gb_local.build_correlation_edges(
                                feats_full.to(device), window=win, k=int(max(1, k_try))
                            )
                            if isinstance(ei, torch.Tensor) and ei.numel() > 0:
                                edge_index = ei.to(device, non_blocking=True)
                                # Enrich with market/sector similarity if available
                                edge_attr = (
                                    _enrich_edge_attr_with_meta(
                                        edge_index,
                                        ea.to(device, non_blocking=True),
                                        codes_list,
                                        markets_list,
                                        sectors_list,
                                    )
                                    if isinstance(ea, torch.Tensor)
                                    else None
                                )
                                logger.info(
                                    f"[edges-fallback] built correlation edges from batch: E={edge_index.size(1)}"
                                )
                        except Exception as _e:
                            logger.warning(
                                f"[edges-fallback] failed to build correlation edges: {_e}"
                            )
                    # Iterate micro-batches
                    mb_start = 0
                    while mb_start < n_items:
                        mb_end = min(n_items, mb_start + micro_bs)
                        features = feats_full[mb_start:mb_end].to(
                            device, non_blocking=True
                        )
                        if isinstance(targs_full, dict):
                            targets = {
                                k: v[mb_start:mb_end].to(device, non_blocking=True)
                                for k, v in targs_full.items()
                            }
                        else:
                            targets = targs_full[mb_start:mb_end].to(
                                device, non_blocking=True
                            )

                        # Get valid masks from batch if available, otherwise compute them
                        if valid_masks_full is not None:
                            if isinstance(valid_masks_full, dict):
                                valid_masks = {
                                    k: v[mb_start:mb_end].to(device, non_blocking=True)
                                    if v is not None
                                    else None
                                    for k, v in valid_masks_full.items()
                                }
                            else:
                                valid_masks = valid_masks_full[mb_start:mb_end].to(
                                    device, non_blocking=True
                                )
                        else:
                            # Fallback: compute masks from targets
                            def _compute_masks(_t):
                                """Compute valid masks from targets"""
                                if isinstance(_t, dict):
                                    masks = {}
                                    for _k, _v in _t.items():
                                        if torch.is_tensor(_v):
                                            valid_mask = torch.isfinite(_v)
                                            bad = (~valid_mask).sum().item()
                                            if bad > 0:
                                                total = _v.numel()
                                                valid_ratio = (total - bad) / max(
                                                    1, total
                                                )
                                                logger.warning(
                                                    f"[nan-guard] targets[{_k}]: non-finite={bad}/{total} (valid={valid_ratio:.2%})"
                                                )
                                            masks[_k] = valid_mask
                                        else:
                                            masks[_k] = None
                                    return masks
                                else:
                                    if torch.is_tensor(_t):
                                        valid_mask = torch.isfinite(_t)
                                        bad = (~valid_mask).sum().item()
                                        if bad > 0:
                                            total = _t.numel()
                                            valid_ratio = (total - bad) / max(1, total)
                                            logger.warning(
                                                f"[nan-guard] targets: non-finite={bad}/{total} (valid={valid_ratio:.2%})"
                                            )
                                        return valid_mask
                                    return None

                            valid_masks = _compute_masks(targets)

                        features = _finite_or_nan_fix_tensor(
                            features, "features", clamp=50.0
                        )

                        # GAT融合αのウォームアップ下限（早期退行の防止）
                        try:
                            if hasattr(model, "alpha_graph_min"):
                                base_alpha_min = float(
                                    getattr(
                                        getattr(final_config.model, "gat"), "alpha_min"
                                    )
                                    if hasattr(final_config, "model")
                                    else getattr(model, "alpha_graph_min", 0.1)
                                )
                                warm_alpha_min = float(
                                    os.getenv("GAT_ALPHA_WARMUP_MIN", "0.30")
                                )
                                warmup_steps_alpha = int(
                                    os.getenv("GAT_WARMUP_STEPS", "500")
                                )
                                if global_step < warmup_steps_alpha:
                                    model.alpha_graph_min = max(
                                        base_alpha_min, warm_alpha_min
                                    )
                                else:
                                    model.alpha_graph_min = base_alpha_min
                        except Exception:
                            pass

                        # Replace NaN/Inf targets with 0 (will be masked in loss)
                        if isinstance(targets, dict):
                            for k in targets.keys():
                                if torch.is_tensor(targets[k]):
                                    targets[k] = torch.nan_to_num(
                                        targets[k], nan=0.0, posinf=0.0, neginf=0.0
                                    )
                        else:
                            if torch.is_tensor(targets):
                                targets = torch.nan_to_num(
                                    targets, nan=0.0, posinf=0.0, neginf=0.0
                                )

                        # Check if batch has enough valid data (relaxed: horizon-wise loss masking preferred)
                        min_valid_ratio = float(os.getenv("MIN_VALID_RATIO", "0.0"))
                        if valid_masks is not None and min_valid_ratio > 0.0:
                            valid_ratios = []
                            for k, mask in valid_masks.items():
                                if mask is not None and torch.is_tensor(mask):
                                    ratio = mask.float().mean().item()
                                    valid_ratios.append(ratio)
                            if valid_ratios:
                                avg_valid_ratio = sum(valid_ratios) / len(valid_ratios)
                                if avg_valid_ratio < min_valid_ratio:
                                    logger.warning(
                                        f"[skip-batch] Low valid ratio: {avg_valid_ratio:.2%} < {min_valid_ratio:.2%}"
                                    )
                                    mb_start = mb_end
                                    continue

                        # Use SafeAMPWrapper for mixed precision
                        if use_amp and os.getenv("USE_SAFE_AMP", "1") == "1":
                            # Safe forward with automatic FP32 conversion
                            with torch.amp.autocast(
                                "cuda", dtype=amp_dtype, enabled=use_amp
                            ):
                                outputs = _forward_with_optional_graph(
                                    model, features, edge_index, edge_attr
                                )
                            # Force ALL outputs to FP32 for loss calculation
                            outputs = {
                                k: v.float() if torch.is_tensor(v) else v
                                for k, v in outputs.items()
                            }
                            # Force targets to FP32 as well
                            if isinstance(targets, dict):
                                targets = {
                                    k: v.float() if torch.is_tensor(v) else v
                                    for k, v in targets.items()
                                }
                            else:
                                targets = (
                                    targets.float()
                                    if torch.is_tensor(targets)
                                    else targets
                                )
                            # Calculate loss in FP32 with valid masks
                            with torch.amp.autocast("cuda", enabled=False):
                                loss, losses = criterion(outputs, targets, valid_masks)
                                # Debug: Check loss type
                                if not isinstance(loss, torch.Tensor):
                                    logger.error(
                                        f"Loss is not a tensor: type={type(loss)}, value={loss}"
                                    )
                                    raise TypeError(
                                        f"Expected loss to be a Tensor, got {type(loss)}"
                                    )
                            # Optional regularization: include model-provided sparsity/alpha penalty
                            try:
                                _sp_lambda = float(os.getenv("SPARSITY_LAMBDA", "0.0"))
                            except Exception:
                                _sp_lambda = 0.0
                            if (
                                _sp_lambda > 0.0
                                and isinstance(outputs, dict)
                                and "sparsity_loss" in outputs
                            ):
                                sp_loss = outputs["sparsity_loss"]
                                if torch.is_tensor(sp_loss):
                                    loss = loss + _sp_lambda * sp_loss
                                    if isinstance(losses, dict):
                                        losses["sparsity_reg"] = (
                                            sp_loss.detach() * _sp_lambda
                                        )
                        else:
                            # Fallback: if no external edges, try correlation edges from batch
                            if (
                                edge_index is None
                                and getattr(final_config.model, "gat", None) is not None
                                and getattr(final_config.model.gat, "enabled", False)
                                and isinstance(features, torch.Tensor)
                                and features.dim() == 3
                            ):
                                try:
                                    from src.graph.graph_builder import (
                                        GraphBuilder as _GBL2,
                                        GBConfig as _GBC2,
                                    )

                                    try:
                                        k_try = int(getattr(final_config.model.gat, "knn_k", 10))
                                    except Exception:
                                        k_try = 10
                                    # Threshold: prefer graph_builder.edge_threshold, then data.graph.edge_threshold, else env/default
                                    try:
                                        thr = None
                                        try:
                                            thr = float(
                                                getattr(
                                                    getattr(final_config.data, "graph_builder", {}),
                                                    "edge_threshold",
                                                )
                                            )
                                        except Exception:
                                            pass
                                        if thr is None:
                                            try:
                                                thr = float(
                                                    getattr(
                                                        getattr(final_config.data, "graph", {}),
                                                        "edge_threshold",
                                                    )
                                                )
                                            except Exception:
                                                thr = None
                                        if thr is None:
                                            thr = float(os.getenv("GRAPH_EDGE_THR", "0.3"))
                                    except Exception:
                                        thr = 0.3
                                    _gb_local2 = _GBL2(
                                        _GBC2(
                                            max_nodes=int(features.size(0)),
                                            edge_threshold=float(thr),
                                        )
                                    )
                                    win = int(min(features.size(1), 20))
                                    ei, ea = _gb_local2.build_correlation_edges(
                                        features, window=win, k=int(max(1, k_try))
                                    )
                                    if isinstance(ei, torch.Tensor) and ei.numel() > 0:
                                        edge_index = ei.to(device, non_blocking=True)
                                        edge_attr = (
                                            ea.to(device, non_blocking=True)
                                            if isinstance(ea, torch.Tensor)
                                            else None
                                        )
                                except Exception as _e:
                                    logger.warning(
                                        f"[edges-fallback/val] failed to build correlation edges: {_e}"
                                    )

                            with torch.amp.autocast(
                                "cuda",
                                dtype=amp_dtype,
                                enabled=use_amp,
                                cache_enabled=False,
                            ):
                                outputs = _forward_with_optional_graph(
                                    model, features, edge_index, edge_attr
                                )
                                # outputs/targets 双方をサニタイズ
                                if isinstance(outputs, dict):
                                    for k, v in outputs.items():
                                        if torch.is_tensor(v):
                                            outputs[k] = _finite_or_nan_fix_tensor(
                                                v, f"outputs[{k}]", clamp=50.0
                                            )
                                # Use already computed valid_masks
                                loss, losses = criterion(outputs, targets, valid_masks)
                                # Debug: Check loss type
                                if not isinstance(loss, torch.Tensor):
                                    logger.error(
                                        f"Loss is not a tensor: type={type(loss)}, value={loss}"
                                    )
                                    raise TypeError(
                                        f"Expected loss to be a Tensor, got {type(loss)}"
                                    )
                            # Optional regularization: include model-provided sparsity/alpha penalty
                            try:
                                _sp_lambda = float(os.getenv("SPARSITY_LAMBDA", "0.0"))
                            except Exception:
                                _sp_lambda = 0.0
                            if (
                                _sp_lambda > 0.0
                                and isinstance(outputs, dict)
                                and "sparsity_loss" in outputs
                            ):
                                sp_loss = outputs["sparsity_loss"]
                                if torch.is_tensor(sp_loss):
                                    loss = loss + _sp_lambda * sp_loss
                                    if isinstance(losses, dict):
                                        losses["sparsity_reg"] = (
                                            sp_loss.detach() * _sp_lambda
                                        )
                            if not torch.isfinite(loss):
                                logger.warning(
                                    "[nan-guard] loss non-finite; skipping micro-batch"
                                )
                                optimizer.zero_grad(set_to_none=True)
                                mb_start = mb_end
                                continue
                        scaler.scale(loss).backward()

                        # Log loss breakdown and gradients (detailed diagnostics)
                        grad_log_every = int(os.getenv("GRAD_LOG_EVERY", "100"))
                        if (
                            grad_log_every > 0
                            and global_step % grad_log_every == 0
                            and global_step > 0
                        ):
                            try:
                                # Log individual loss components
                                loss_details = []
                                if isinstance(losses, dict):
                                    for k, v in losses.items():
                                        if torch.is_tensor(v):
                                            val = v.detach().item()
                                            loss_details.append(f"{k}={val:.4f}")

                                # Log gradient norms for key modules
                                grad_norms = {}
                                for name, param in model.named_parameters():
                                    if param.grad is not None:
                                        grad_norm = param.grad.norm().item()
                                        # Focus on key components
                                        if any(
                                            key in name
                                            for key in [
                                                "head_point",
                                                "head_quantile",
                                                "gat",
                                                "fan",
                                                "encoder",
                                                "alpha",
                                            ]
                                        ):
                                            grad_norms[
                                                name.split(".")[-2]
                                                + "."
                                                + name.split(".")[-1]
                                            ] = grad_norm

                                # Check for zero gradients in critical components
                                gat_grad = sum(
                                    v for k, v in grad_norms.items() if "gat" in k
                                )
                                fan_grad = sum(
                                    v for k, v in grad_norms.items() if "fan" in k
                                )

                                logger.info(
                                    f"[Loss@{global_step}] total={loss.item():.4f} | "
                                    + " | ".join(loss_details[:5])  # First 5 components
                                )

                                if gat_grad < 1e-8:
                                    logger.warning(
                                        f"[Grad@{global_step}] GAT gradients near zero: {gat_grad:.2e}"
                                    )
                                if fan_grad < 1e-8:
                                    logger.warning(
                                        f"[Grad@{global_step}] FAN gradients near zero: {fan_grad:.2e}"
                                    )

                                # Log fusion alpha if available
                                if hasattr(model, "alpha_logit"):
                                    alpha_val = model.alpha_graph_min + (
                                        1 - model.alpha_graph_min
                                    ) * torch.sigmoid(model.alpha_logit)
                                    logger.info(
                                        f"[Fusion@{global_step}] alpha={alpha_val.item():.4f}"
                                    )

                            except Exception as e:
                                logger.debug(f"Loss/grad logging failed: {e}")

                        # Degeneracy detection - ratio-based and consecutive with EMA
                        warmup_steps = int(os.getenv("DEGENERACY_WARMUP_STEPS", "800"))
                        check_every = int(os.getenv("DEGENERACY_CHECK_EVERY", "100"))
                        use_guard = os.getenv("DEGENERACY_GUARD", "1") == "1"
                        abort_on_guard = (
                            os.getenv("DEGENERACY_ABORT", "0") == "1"
                        )  # Default: don't abort
                        std_eps = float(os.getenv("DEGENERACY_STD_EPS", "1e-6"))
                        abs_min_std = float(
                            os.getenv("DEGENERACY_ABS_MIN_STD", "0.005")
                        )  # Absolute minimum threshold
                        min_ratio = float(
                            os.getenv("DEGENERACY_MIN_RATIO", "0.10")
                        )  # yhat.std / y.std minimum ratio
                        need_consec = int(
                            os.getenv("DEGENERACY_CONSECUTIVE", "3")
                        )  # Consecutive violations needed
                        ema_beta = float(os.getenv("DEGENERACY_EMA_BETA", "0.9"))

                        # Skip checks during warmup or if not at check interval
                        if (
                            use_guard
                            and global_step >= warmup_steps
                            and global_step % check_every == 0
                        ):
                            with torch.no_grad():
                                for h in criterion.horizons:
                                    # Get prediction and target keys
                                    pred_key = (
                                        f"point_horizon_{h}"
                                        if f"point_horizon_{h}" in outputs
                                        else f"horizon_{h}"
                                    )
                                    tgt_key = f"horizon_{h}"

                                    if (
                                        pred_key not in outputs
                                        or tgt_key not in targets
                                    ):
                                        continue

                                    yhat = outputs[pred_key].float()
                                    y = targets[tgt_key].float()

                                    # Use valid mask if available
                                    mask = None
                                    if (
                                        "valid_masks" in locals()
                                        and valid_masks is not None
                                    ):
                                        mask_key = f"horizon_{h}"
                                        if mask_key in valid_masks:
                                            mask = valid_masks[mask_key]
                                    elif (
                                        "valid_masks_else" in locals()
                                        and locals().get("valid_masks_else") is not None
                                    ):
                                        mask_key = f"horizon_{h}"
                                        valid_masks_else = locals().get(
                                            "valid_masks_else", {}
                                        )
                                        if mask_key in valid_masks_else:
                                            mask = valid_masks_else[mask_key]

                                    # Calculate std in FP32 for stability with mask
                                    if mask is not None and mask.sum() > 0:
                                        yhat_masked = yhat[mask]
                                        y_masked = y[mask]
                                        yhat_std = (
                                            yhat_masked.std().item()
                                            if yhat_masked.numel() > 1
                                            else 0.0
                                        )
                                        y_std = max(
                                            y_masked.std().item()
                                            if y_masked.numel() > 1
                                            else std_eps,
                                            std_eps,
                                        )
                                    else:
                                        yhat_std = yhat.std().item()
                                        y_std = max(y.std().item(), std_eps)

                                    # Apply EMA
                                    k = f"h{h}"
                                    prev = deg_ema[k] if deg_ema[k] > 0 else yhat_std
                                    ema_std = (
                                        ema_beta * prev + (1.0 - ema_beta) * yhat_std
                                    )
                                    deg_ema[k] = ema_std

                                    ratio = ema_std / max(y_std, std_eps)

                                    # Check if bad (absolute minimum OR ratio too low)
                                    bad = (ema_std < abs_min_std) or (ratio < min_ratio)

                                    if bad:
                                        deg_bad[k] += 1
                                        msg = (
                                            f"Pred std EMA={ema_std:.2e}, y.std={y_std:.2e}, "
                                            f"ratio={ratio:.3f} (<{min_ratio}) or abs<{abs_min_std} @h={h} "
                                            f"[consec={deg_bad[k]}/{need_consec}]"
                                        )
                                        logger.error("[FAILSAFE] " + msg)
                                        if deg_bad[k] >= need_consec and abort_on_guard:
                                            raise SystemExit(2)
                                    else:
                                        if deg_bad[k] > 0:
                                            logger.info(
                                                f"[FAILSAFE] recovered for h={h}, reset consecutive counter"
                                            )
                                        deg_bad[k] = 0

                        n_micro_steps += 1
                        if n_micro_steps % grad_accum == 0:
                            scaler.unscale_(optimizer)

                            # Check gradient norms before clipping
                            def grad_norm(module):
                                total_norm = 0.0
                                count = 0
                                for p in module.parameters():
                                    if p.grad is not None:
                                        total_norm += p.grad.data.norm(2).item()
                                        count += 1
                                return total_norm, count

                            # Check GAT gradients with configurable warmup period
                            if hasattr(model, "gat") and model.gat is not None:
                                gat_norm, gat_count = grad_norm(model.gat)

                                # Get configurable thresholds from environment
                                gat_warmup = int(os.getenv("GAT_WARMUP_STEPS", "500"))
                                gat_grad_threshold = float(
                                    os.getenv("GAT_GRAD_THR", "1e-10")
                                )

                                if gat_count > 0 and gat_norm < gat_grad_threshold:
                                    msg = f"[GUARD] GAT grad too small: {gat_norm:.2e} (step={global_step})"

                                    if global_step < gat_warmup:
                                        # During warmup, only warn
                                        logger.warning(msg + " -- tolerated in warmup")
                                    else:
                                        # After warmup, check abort setting
                                        logger.error(f"[FAILSAFE] {msg}")
                                        if os.getenv("DEGENERACY_ABORT", "1") == "1":
                                            logger.error(
                                                "GAT gradients persistently small after warmup, aborting"
                                            )
                                            raise SystemExit(3)
                                        else:
                                            logger.warning(
                                                "DEGENERACY_ABORT=0, continuing despite small gradients"
                                            )

                            # Log fusion alpha value if available
                            if isinstance(outputs, dict) and "fusion_alpha" in outputs:
                                alpha_val = float(
                                    outputs["fusion_alpha"].mean().detach().cpu()
                                )
                                if batch_idx % 50 == 0:  # Log every 50 batches
                                    logger.info(
                                        f"[Diag] fusion_alpha={alpha_val:.4f} (step={global_step})"
                                    )

                            if hasattr(model, "output_heads"):
                                head_norm, head_count = grad_norm(model.output_heads)
                                if head_count > 0 and head_norm < 1e-8:
                                    logger.error(
                                        f"[FAILSAFE] Head gradient norm too small: {head_norm:.2e}"
                                    )
                                    raise SystemExit(3)

                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=1.0
                            )
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)
                            global_step += 1

                            # Quick evaluation at intervals
                            if (
                                global_step % eval_every_steps == 0
                                and val_loader is not None
                            ):
                                try:
                                    quick_metrics = evaluate_quick(
                                        model,
                                        val_loader,
                                        criterion,
                                        device,
                                        max_batches=int(
                                            os.getenv("EVAL_MAX_BATCHES", "50")
                                        ),
                                    )
                                    metrics_data = {
                                        "epoch": epoch,
                                        "step": global_step,
                                        "metrics": quick_metrics,
                                        "elapsed_hours": (time.time() - start_time)
                                        / 3600,
                                        "timestamp": now_jst_iso(),
                                        "tz": str(JST.key)
                                        if hasattr(JST, "key")
                                        else "Asia/Tokyo",
                                    }
                                    (RUN_DIR / "latest_metrics.json").write_text(
                                        json.dumps(
                                            metrics_data, ensure_ascii=False, indent=2
                                        )
                                    )
                                    logger.info(
                                        f"Quick eval at step {global_step}: {quick_metrics}"
                                    )
                                    if mlf is not None:
                                        try:
                                            if (
                                                isinstance(quick_metrics, dict)
                                                and "val_loss" in quick_metrics
                                            ):
                                                mlf.log_metric(
                                                    "quick/val_loss",
                                                    float(quick_metrics["val_loss"]),
                                                    step=global_step,
                                                )
                                        except Exception as _e:
                                            logger.debug(
                                                f"MLflow metric logging failed: {_e}"
                                            )
                                except Exception as e:
                                    logger.warning(f"Quick evaluation failed: {e}")
                        total_loss += float(loss.detach().item())
                        for k, v in (losses or {}).items():
                            try:
                                horizon_losses[k] += float(
                                    v.item() if hasattr(v, "item") else float(v)
                                )
                            except Exception:
                                pass
                        mb_start = mb_end
                    if batch_idx % 10 == 0:
                        avg = total_loss / max(1, n_micro_steps)
                        pbar.set_postfix({"loss": f"{avg:.4f}"})
                # Flush tail grads
                if n_micro_steps % max(1, grad_accum) != 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                # Log training loss (removed unused variables)
                if n_micro_steps > 0:
                    avg_train_loss = total_loss / n_micro_steps
                    logger.debug(f"Epoch {epoch} train loss: {avg_train_loss:.4f}")
                    # 定型ログ出力（quick_tune.py用）
                    logger.info(f"train/total_loss: {avg_train_loss:.6f}")
                    if mlf is not None:
                        try:
                            mlf.log_metric(
                                "train/total_loss",
                                float(avg_train_loss),
                                step=int(epoch),
                            )
                        except Exception:
                            pass
                    # W&B logging (epoch-level)
                    try:
                        if wb_logger is not None:
                            wb_logger.log_metrics(
                                {
                                    "train/total_loss": float(avg_train_loss),
                                    "lr": float(optimizer.param_groups[0]["lr"]),
                                },
                                step=int(epoch),
                            )
                    except Exception:
                        pass
            else:
                logger.info(f"[{tag}] Skipped training (no train_loader)")
            # 現在の重みをログ
            try:
                cw = criterion._get_current_weights()
                if cw:
                    logger.info(f"Current horizon weights: {cw}")
            except Exception:
                pass
            # GAT勾配の簡易サニティチェック
            try:
                gat_norm = 0.0
                gat_mod = getattr(model, "gat", None)
                if gat_mod is not None:
                    for p in gat_mod.parameters():
                        if p.grad is not None:
                            gat_norm += float(p.grad.data.norm().item())
                    logger.info(f"[sanity] grad_norm(gat)={gat_norm:.6f}")
                    # Also log current alpha (GAT mix ratio) if available
                    try:
                        with torch.no_grad():
                            alpha_min = float(getattr(model, "alpha_graph_min", 0.0))
                            alpha = alpha_min + (1 - alpha_min) * torch.sigmoid(
                                getattr(model, "alpha_logit")
                            )
                            alpha_mean = float(alpha.mean().item())
                            logger.info(
                                f"[sanity] alpha_mean(GAT mix)={alpha_mean:.3f}"
                            )
                            if alpha_mean < 0.1:
                                logger.warning(
                                    f"[sanity] alpha_mean low ({alpha_mean:.3f}); GAT contribution may be too small"
                                )
                    except Exception:
                        pass
            except Exception:
                pass
            if device.type == "cuda":
                memory_used = torch.cuda.memory_allocated() / 1e9
                memory_cached = torch.cuda.memory_reserved() / 1e9
                logger.info(
                    f"GPU Memory: Used={memory_used:.2f}GB, Cached={memory_cached:.2f}GB"
                )
            if val_loader is not None:
                # SWAの評価用に、後半はSWAモデルで評価
                eval_model = model
                if swa_model is not None and epoch >= int(
                    max(1, round(n_epochs * swa_start_frac))
                ):
                    try:
                        # BN統計を都度更新（重い場合は最終のみでも可）
                        from torch.optim.swa_utils import update_bn

                        update_bn(train_loader, swa_model)
                        eval_model = swa_model
                    except Exception as _e:
                        logger.warning(f"SWA BN update failed: {_e}")
                        eval_model = swa_model or model
                # external-graph-aware validation
                if gb is None:
                    val_loss, val_horizon_losses, linear_cal = validate(
                        eval_model, val_loader, criterion, device
                    )
                else:
                    eval_model.eval()
                    total_v = 0.0
                    n_v = 0
                    horizon_losses_v = {f"horizon_{h}": 0.0 for h in criterion.horizons}
                    linear_cal = {h: {"a": 0.0, "b": 1.0} for h in criterion.horizons}
                    with torch.no_grad():

                        def _supports_graph(mdl) -> bool:
                            try:
                                if (
                                    hasattr(mdl, "gat")
                                    and getattr(mdl, "gat") is not None
                                ):
                                    return True
                                sig = inspect.signature(mdl.forward)
                                return ("edge_index" in sig.parameters) or (
                                    "edge_attr" in sig.parameters
                                )
                            except Exception:
                                return False

                        def _forward_with_optional_graph(mdl, feats, ei, ea):
                            try:
                                if (
                                    ei is not None
                                    and ea is not None
                                    and _supports_graph(mdl)
                                ):
                                    return mdl(feats, edge_index=ei, edge_attr=ea)
                                return mdl(feats)
                            except TypeError:
                                return mdl(feats)

                        for vb in tqdm(val_loader, desc="Validation"):
                            features = vb["features"].to(device)
                            targets = (
                                {k: v.to(device) for k, v in vb["targets"].items()}
                                if isinstance(vb.get("targets"), dict)
                                else vb["targets"]
                            )
                            edge_index = None
                            edge_attr = None
                            try:
                                codes = (
                                    vb.get("codes") if "codes" in vb else vb.get("code")
                                )
                                date = vb.get("date", None)
                                if isinstance(date, (list, tuple)) and len(date) > 0:
                                    date = date[0]
                                if codes is not None and date is not None:
                                    try:
                                        if hasattr(codes, "tolist"):
                                            codes = codes.tolist()
                                        codes = [str(c) for c in codes]
                                    except Exception:
                                        pass
                                    if gb_adv is not None:
                                        try:
                                            df_pd = None
                                            try:
                                                if hasattr(val_loader, "dataset") and hasattr(val_loader.dataset, "data"):
                                                    df_pd = val_loader.dataset.data
                                            except Exception:
                                                df_pd = None
                                            if df_pd is not None:
                                                res = gb_adv.build_graph(df_pd, codes, date_end=str(date))
                                                edge_index, edge_attr = res.get("edge_index"), res.get("edge_attr")
                                            else:
                                                edge_index, edge_attr = None, None
                                        except Exception as _e:
                                            logger.warning(f"[AdvGraph/val] build failed: {_e}")
                                            edge_index, edge_attr = None, None
                                    else:
                                        edge_index, edge_attr = gb.build_for_day(
                                            date, codes
                                        )
                                    edge_index = edge_index.to(device)
                                    edge_attr = edge_attr.to(device)
                                    # staleness stats
                                    try:
                                        import pandas as _pd

                                        batch_ts = _pd.Timestamp(date).normalize()
                                        asof_ts = getattr(
                                            gb, "last_asof_ts", lambda: None
                                        )()
                                        if asof_ts is None:
                                            asof_ts = batch_ts - _pd.Timedelta(days=1)

                                        # データ鮮度を計算してリストに追加
                                        staleness_days = int((batch_ts - asof_ts).days)
                                        if not hasattr(
                                            globals(), "_val_staleness_days_list"
                                        ):
                                            globals()["_val_staleness_days_list"] = []
                                        globals()["_val_staleness_days_list"].append(
                                            staleness_days
                                        )

                                        # 100バッチごとに統計情報をログ出力
                                        if (
                                            len(globals()["_val_staleness_days_list"])
                                            % 100
                                            == 0
                                        ):
                                            avg_staleness = np.mean(
                                                globals()["_val_staleness_days_list"][
                                                    -100:
                                                ]
                                            )
                                            max_staleness = np.max(
                                                globals()["_val_staleness_days_list"][
                                                    -100:
                                                ]
                                            )
                                            logger.info(
                                                f"Validation data staleness stats (last 100 batches): "
                                                f"avg={avg_staleness:.2f} days, max={max_staleness:.2f} days"
                                            )

                                    except Exception as e:
                                        logger.warning(
                                            f"Failed to calculate validation staleness: {e}"
                                        )
                                        pass
                            except Exception:
                                edge_index = None
                                edge_attr = None
                            # Fallback: if no external edges, try correlation edges from batch
                            if (
                                edge_index is None
                                and getattr(final_config.model, "gat", None) is not None
                                and getattr(final_config.model.gat, "enabled", False)
                                and isinstance(features, torch.Tensor)
                                and features.dim() == 3
                            ):
                                try:
                                    from src.graph.graph_builder import (
                                        GraphBuilder as _GBL2,
                                        GBConfig as _GBC2,
                                    )
                                    # Resolve codes list for this batch (full day-batch might be preferable, but use what we have)
                                    try:
                                        codes_list = vb.get("codes") if "codes" in vb else vb.get("code")
                                        if hasattr(codes_list, "tolist"):
                                            codes_list = codes_list.tolist()
                                        if codes_list is not None:
                                            codes_list = [str(c) for c in codes_list]
                                    except Exception:
                                        codes_list = None
                                    markets_list = None
                                    sectors_list = None
                                    try:
                                        markets_list = vb.get("markets")
                                        sectors_list = vb.get("sectors")
                                    except Exception:
                                        pass
                                    try:
                                        k_try = int(getattr(final_config.model.gat, "knn_k", 10))
                                    except Exception:
                                        k_try = 10
                                    try:
                                        thr = float(
                                            getattr(
                                                getattr(final_config.data, "graph", {}),
                                                "edge_threshold",
                                                0.0,
                                            )
                                        )
                                    except Exception:
                                        thr = 0.0
                                    _gb_local2 = _GBL2(
                                        _GBC2(max_nodes=int(features.size(0)), edge_threshold=float(thr))
                                    )
                                    win = int(min(features.size(1), 20))
                                    ei, ea = _gb_local2.build_correlation_edges(
                                        features, window=win, k=int(max(1, k_try))
                                    )
                                    if isinstance(ei, torch.Tensor) and ei.numel() > 0:
                                        edge_index = ei.to(device, non_blocking=True)
                                        edge_attr = (
                                            _enrich_edge_attr_with_meta(
                                                edge_index,
                                                ea.to(device, non_blocking=True),
                                                codes_list,
                                                markets_list,
                                                sectors_list,
                                            )
                                            if isinstance(ea, torch.Tensor)
                                            else None
                                        )
                                except Exception as _e:
                                    logger.warning(
                                        f"[edges-fallback/val] failed to build correlation edges: {_e}"
                                    )

                            with torch.amp.autocast(
                                "cuda",
                                dtype=amp_dtype,
                                enabled=use_amp,
                                cache_enabled=False,
                            ):
                                outputs = _forward_with_optional_graph(
                                    eval_model, features, edge_index, edge_attr
                                )
                                # Use valid masks if present in validation batch
                                vmask = (
                                    vb.get("valid_mask")
                                    if isinstance(vb, dict) and "valid_mask" in vb
                                    else None
                                )
                                loss_val, losses = criterion(
                                    outputs, targets, valid_masks=vmask
                                )
                            total_v += float(loss_val.detach().item())
                            n_v += 1
                            for k, v in (losses or {}).items():
                                try:
                                    horizon_losses_v[k] += float(
                                        v.item() if hasattr(v, "item") else float(v)
                                    )
                                except Exception:
                                    pass
                        val_loss = total_v / max(1, n_v)
                        val_horizon_losses = {
                            k: v / max(1, n_v) for k, v in horizon_losses_v.items()
                        }
                    logger.info(f"[{tag}] Validation loss: {val_loss:.4f}")

                    # Log staleness stats if available
                    if (
                        "_val_staleness_days_list" in globals()
                        and globals()["_val_staleness_days_list"]
                    ):
                        import numpy as _np

                        s_arr = _np.array(
                            globals()["_val_staleness_days_list"], dtype=float
                        )
                        s_mean = float(_np.mean(s_arr))
                        s_median = float(_np.median(s_arr))
                        s_std = float(_np.std(s_arr))
                        s_min = float(_np.min(s_arr))
                        s_max = float(_np.max(s_arr))

                        logger.info(
                            f"[EDGE-TS] val staleness days: mean={s_mean:.2f}, median={s_median:.2f}, "
                            f"std={s_std:.2f}, min={s_min:.2f}, max={s_max:.2f}"
                        )

                        # MLflow logging
                        try:
                            import mlflow as _mlf  # type: ignore

                            _mlf.log_metric(
                                "val/edge_staleness_days_mean", s_mean, step=int(epoch)
                            )
                            _mlf.log_metric(
                                "val/edge_staleness_days_median",
                                s_median,
                                step=int(epoch),
                            )
                            _mlf.log_metric(
                                "val/edge_staleness_days_std", s_std, step=int(epoch)
                            )
                            _mlf.log_metric(
                                "val/edge_staleness_days_min", s_min, step=int(epoch)
                            )
                            _mlf.log_metric(
                                "val/edge_staleness_days_max", s_max, step=int(epoch)
                            )
                        except Exception as e:
                            logger.warning(f"MLflow logging failed: {e}")
                    else:
                        logger.info(
                            "[EDGE-TS] No staleness data available for validation"
                        )
                    # 定型ログ出力（quick_tune.py用）
                    logger.info(f"val/loss: {val_loss:.6f}")
                    logger.info(f"val/epoch: {epoch}")
                    logger.info(f"val/step: {global_step}")
                    for k, v in val_horizon_losses.items():
                        logger.info(f"  {k}: {v:.4f}")

                    # Evaluate metrics
                    val_metrics = evaluate_model_metrics(
                        eval_model,
                        val_loader,
                        criterion,
                        device,
                        target_scalers=getattr(data_module, "target_scalers", {}),
                        max_batches=None,
                    )
                    # W&B logging for validation metrics (epoch-level)
                    try:
                        if wb_logger is not None:
                            log_dict = {"val/loss": float(val_loss), "epoch": int(epoch)}
                            # Log per-horizon losses if available
                            try:
                                for hk, hv in (val_horizon_losses or {}).items():
                                    log_dict[f"val/{hk}"] = float(hv)
                            except Exception:
                                pass
                            # Flatten detailed validation metrics
                            try:
                                hm = val_metrics.get("horizon_metrics", {}) if isinstance(val_metrics, dict) else {}
                                for h, m in hm.items():
                                    for k in ("rmse", "mae", "correlation", "r2", "sharpe_ratio"):
                                        if k in m:
                                            log_dict[f"val_h{h}/{k}"] = float(m[k])
                                avgm = val_metrics.get("average_metrics", {}) if isinstance(val_metrics, dict) else {}
                                for k, v in avgm.items():
                                    log_dict[f"val/{k}"] = float(v)
                            except Exception:
                                pass
                            wb_logger.log_metrics(log_dict, step=int(epoch))
                    except Exception:
                        pass

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_path = Path(f"models/checkpoints/atft_gat_fan_best_{tag}.pt")
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    ckpt_model = (
                        eval_model
                        if (swa_model is not None and epoch >= int(n_epochs * 0.67))
                        else model
                    )
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": ckpt_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scaler_state_dict": scaler.state_dict(),
                            "val_loss": val_loss,
                            "config": OmegaConf.to_container(final_config),
                            "target_scalers": getattr(
                                data_module, "target_scalers", {}
                            ),
                            "linear_calibration": linear_cal,
                            "run_manifest": run_manifest,
                        },
                        save_path,
                    )
                    logger.info(f"Best model saved ({tag}) (val_loss: {val_loss:.4f})")
                    # Optionally log best checkpoint as W&B Artifact
                    try:
                        if wb_logger is not None and os.getenv("WANDB_ARTIFACTS", "1").lower() in ("1", "true", "yes", "on"):
                            import importlib as _importlib

                            _wandb = _importlib.import_module("wandb")
                            art = _wandb.Artifact(
                                name=f"atft-gat-fan-{tag}", type="model"
                            )
                            art.add_file(str(save_path))
                            if getattr(_wandb, "run", None) is not None:
                                _wandb.run.log_artifact(art, aliases=["best", f"ep{epoch}"])
                            else:
                                _wandb.log_artifact(art)
                    except Exception:
                        pass
            # Snapshot ensemble save
            if epoch in snapshot_points:
                snap_path = Path(f"models/checkpoints/snapshot_{tag}_ep{epoch}.pt")
                snap_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "config": OmegaConf.to_container(final_config),
                    },
                    snap_path,
                )
                logger.info(f"Saved snapshot checkpoint: {snap_path}")
            # 学習率スケジューラ更新 or SWA更新
            # SWA適用フェーズか判定（else節のインデント問題回避のためブールに分離）
            use_swa_phase = swa_model is not None and epoch >= int(
                max(1, round(n_epochs * swa_start_frac))
            )
            if use_swa_phase:
                try:
                    swa_model.update_parameters(model)
                    if swa_scheduler is not None:
                        swa_scheduler.step()
                except Exception as _e:
                    logger.warning(f"SWA step failed: {_e}")
            if not use_swa_phase:
                scheduler.step()
            gc.collect()
            torch.cuda.empty_cache()
        # --- SWA 最終評価・保存 ---
        if swa_model is not None and val_loader is not None and use_swa:
            try:
                from torch.optim.swa_utils import update_bn

                logger.info("[SWA] Updating BN statistics and evaluating...")
                update_bn(train_loader, swa_model)
                swa_val_loss, _, _ = validate(swa_model, val_loader, criterion, device)
                logger.info(f"[SWA] Validation loss: {swa_val_loss:.4f}")
                swa_path = Path(f"models/checkpoints/swa_{tag}.pt")
                swa_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": n_epochs,
                        "model_state_dict": swa_model.state_dict(),
                        "config": OmegaConf.to_container(final_config),
                    },
                    swa_path,
                )
                logger.info(f"[SWA] Saved SWA checkpoint: {swa_path}")
                # Optionally log SWA checkpoint as W&B Artifact
                try:
                    if wb_logger is not None and os.getenv("WANDB_ARTIFACTS", "1").lower() in ("1", "true", "yes", "on"):
                        import importlib as _importlib

                        _wandb = _importlib.import_module("wandb")
                        art = _wandb.Artifact(name=f"atft-gat-fan-{tag}-swa", type="model")
                        art.add_file(str(swa_path))
                        if getattr(_wandb, "run", None) is not None:
                            _wandb.run.log_artifact(art, aliases=["swa", f"ep{n_epochs}"])
                        else:
                            _wandb.log_artifact(art)
                except Exception:
                    pass
                if swa_val_loss < best_val_loss:
                    best_val_loss = swa_val_loss
                    best_path = Path(f"models/checkpoints/best_{tag}.pt")
                    torch.save(
                        {
                            "epoch": n_epochs,
                            "model_state_dict": swa_model.state_dict(),
                            "config": OmegaConf.to_container(final_config),
                        },
                        best_path,
                    )
                    logger.info(
                        f"[SWA] SWA model is new BEST; checkpoint updated: {best_path}"
                    )
            except Exception as _e:
                logger.warning(f"SWA evaluation failed/skipped: {_e}")
        # ===== 予測エクスポート（任意） =====
        try:
            if os.getenv("EXPORT_PREDICTIONS", "0") == "1" and val_loader is not None:
                logger.info("[EXPORT] Exporting validation predictions to file ...")
                # オプション: ベスト（最終）チェックポイントを読み込んでから推論
                try:
                    if os.getenv("USE_BEST_CKPT_FOR_EXPORT", "1") == "1":
                        ckpt = Path("models/checkpoints/atft_gat_fan_final.pt")
                        if ckpt.exists():
                            obj = torch.load(ckpt, map_location=device)
                            sd = None
                            if isinstance(obj, dict):
                                for k in ("state_dict", "model_state_dict"):
                                    if k in obj and isinstance(obj[k], dict):
                                        sd = obj[k]
                                        break
                            if sd is None and isinstance(obj, dict):
                                # 直接state_dict相当
                                if all(isinstance(v, torch.Tensor) for v in obj.values()):
                                    sd = obj
                            if sd is not None:
                                model.load_state_dict(sd, strict=False)
                                logger.info(f"[EXPORT] Loaded checkpoint weights from {ckpt}")
                except Exception as _le:
                    logger.warning(f"[EXPORT] Loading checkpoint for export failed: {_le}")

                model.eval()
                rows = []
                with torch.no_grad():
                    for vb in val_loader:
                        feats = vb["features"].to(device)
                        preds = model(
                            feats,
                            vb.get("static_features", None),
                            vb.get("edge_index", None),
                            vb.get("edge_attr", None),
                        )
                        # 予測キーの解決（1日ホライズン）
                        pred_key = None
                        for pk in ("point_horizon_1", "horizon_1", "h1"):
                            if isinstance(preds, dict) and pk in preds:
                                pred_key = pk
                                break
                        if pred_key is None:
                            continue
                        yhat = preds[pred_key].detach().float().view(-1).cpu()

                        # 実績キーの解決
                        tdict = vb.get("targets", {})
                        targ_key = None
                        for tk in ("horizon_1", "point_horizon_1", "h1", "target_1d"):
                            if tk in tdict:
                                targ_key = tk
                                break
                        y = (
                            tdict[targ_key].detach().float().view(-1).cpu()
                            if targ_key is not None
                            else None
                        )

                        # メタ情報
                        codes = vb.get("codes") if "codes" in vb else vb.get("code")
                        if hasattr(codes, "tolist"):
                            codes = codes.tolist()
                        if codes is None:
                            # 長さを合わせるためのフォールバック
                            codes = [None] * int(yhat.shape[0])
                        date_val = vb.get("date", None)
                        # バッチ内の全行に同一日付を適用（day-batch前提）
                        dates = [str(date_val) if date_val is not None else None] * int(yhat.shape[0])

                        for c, d, p, a in zip(codes, dates, yhat.numpy().tolist(), ([] if y is None else y.numpy().tolist())):
                            rows.append({
                                "date": d,
                                "Code": str(c) if c is not None else None,
                                "predicted_return": float(p),
                                **({"actual_return": float(a)} if y is not None else {}),
                            })

                if rows:
                    import pandas as _pd
                    df_pred = _pd.DataFrame(rows)
                    out_dir = Path("output/predictions")
                    out_dir.mkdir(parents=True, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_path = out_dir / f"predictions_val_{ts}.parquet"
                    df_pred.to_parquet(out_path, index=False)
                    # runs/last にも配置
                    run_out = RUN_DIR / "predictions_val.parquet"
                    df_pred.to_parquet(run_out, index=False)
                    logger.info(f"[EXPORT] Saved validation predictions: {out_path}")
                else:
                    logger.warning("[EXPORT] No validation predictions collected; skipped writing")
        except Exception as _e:
            logger.warning(f"[EXPORT] Export predictions failed: {_e}")

        return best_val_loss

    # メイン学習（fold設定に依存）
    logger.info(f"Batch size: {final_config.train.batch.train_batch_size}")
    logger.info(
        f"Gradient accumulation steps: {final_config.train.batch.gradient_accumulation_steps}"
    )
    logger.info(
        f"Effective batch size: {final_config.train.batch.train_batch_size * final_config.train.batch.gradient_accumulation_steps}"
    )
    # Utility: optional code→market/sector maps for edge_attr enrichment in fallbacks
    code2market: dict[str, str] = {}
    code2sector: dict[str, str] = {}

    def _load_code_maps():
        nonlocal code2market, code2sector
        m_path = os.getenv("MARKET_MAP_CSV", "").strip()
        s_path = os.getenv("SECTOR_MAP_CSV", "").strip()
        code_col = os.getenv("CODE_COL", "code")
        m_col = os.getenv("MARKET_COL", "")
        s_col = os.getenv("SECTOR_COL", "")

        def _read_table(path: str):
            import pandas as _pd

            if not path:
                return None
            try:
                if path.lower().endswith(".parquet"):
                    return _pd.read_parquet(path)
                return _pd.read_csv(path)
            except Exception as _e:
                logger.warning(f"[CodeMap] failed to read {path}: {_e}")
                return None

        def _resolve_value_col(df, preferred: str, candidates: list[str]):
            if preferred and preferred in df.columns:
                return preferred
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        # Market map
        if m_path:
            dfm = _read_table(m_path)
            if dfm is not None and code_col in dfm.columns:
                market_candidates = ["MarketCode", "market_code", "market", "Section", "meta_section"]
                m_use = _resolve_value_col(dfm, m_col, market_candidates)
                if m_use:
                    code2market = {str(r[code_col]): str(r[m_use]) for _, r in dfm[[code_col, m_use]].iterrows()}
                    logger.info(f"[CodeMap] loaded {len(code2market)} market mappings from {m_path} (col={m_use})")
                else:
                    logger.warning(f"[CodeMap] no usable market column found in {m_path}")
        # Sector map
        if s_path:
            dfs = _read_table(s_path)
            if dfs is not None and code_col in dfs.columns:
                sector_candidates = ["sector33", "SectorCode", "sector", "meta_section", "Section"]
                s_use = _resolve_value_col(dfs, s_col, sector_candidates)
                if s_use:
                    code2sector = {str(r[code_col]): str(r[s_use]) for _, r in dfs[[code_col, s_use]].iterrows()}
                    logger.info(f"[CodeMap] loaded {len(code2sector)} sector mappings from {s_path} (col={s_use})")
                else:
                    logger.warning(f"[CodeMap] no usable sector column found in {s_path}")

    _load_code_maps()

    def _enrich_edge_attr_with_meta(
        eidx: torch.Tensor,
        eattr: torch.Tensor,
        codes_list: list[str] | None,
        markets_list: list[str] | None = None,
        sectors_list: list[str] | None = None,
    ) -> torch.Tensor:
        try:
            desired_dim = 1
            try:
                desired_dim = int(getattr(final_config.model.gat.edge_features, "edge_dim", 1))
            except Exception:
                desired_dim = 1

            # Ensure 2D tensor
            if eattr is None:
                eattr = torch.zeros((eidx.size(1), 0), device=eidx.device, dtype=torch.float32)
            if eattr.dim() == 1:
                eattr = eattr.unsqueeze(-1)

            # Pad to desired_dim with zeros first
            if eattr.size(-1) < desired_dim:
                pad = torch.zeros((eattr.size(0), desired_dim - eattr.size(-1)), device=eattr.device, dtype=eattr.dtype)
                eattr = torch.cat([eattr, pad], dim=-1)

            if desired_dim >= 3:
                # Compute market/sector similarity per edge if mappings available
                m_sim = []
                s_sim = []
                # Build per-edge values
                for k in range(eidx.size(1)):
                    i = int(eidx[0, k])
                    j = int(eidx[1, k])
                    if markets_list is not None and sectors_list is not None:
                        # Prefer direct batch metadata if available
                        if i >= len(markets_list) or j >= len(markets_list):
                            m_sim.append(0.0)
                            s_sim.append(0.0)
                        else:
                            mi = markets_list[i]
                            mj = markets_list[j]
                            si = sectors_list[i] if i < len(sectors_list) else None
                            sj = sectors_list[j] if j < len(sectors_list) else None
                            m_sim.append(1.0 if (mi is not None and mj is not None and mi == mj) else 0.0)
                            s_sim.append(1.0 if (si is not None and sj is not None and si == sj) else 0.0)
                    else:
                        # Fall back to code->meta maps
                        if codes_list is None or i >= len(codes_list) or j >= len(codes_list):
                            m_sim.append(0.0)
                            s_sim.append(0.0)
                        else:
                            ci = str(codes_list[i])
                            cj = str(codes_list[j])
                            # Market
                            mi = code2market.get(ci)
                            mj = code2market.get(cj)
                            m_sim.append(1.0 if (mi is not None and mj is not None and mi == mj) else 0.0)
                            # Sector
                            si = code2sector.get(ci)
                            sj = code2sector.get(cj)
                            s_sim.append(1.0 if (si is not None and sj is not None and si == sj) else 0.0)
                m_sim_t = torch.tensor(m_sim, device=eattr.device, dtype=eattr.dtype).unsqueeze(-1)
                s_sim_t = torch.tensor(s_sim, device=eattr.device, dtype=eattr.dtype).unsqueeze(-1)
                # Column 0 is correlation (already present), fill 1 and 2
                if eattr.size(-1) >= 2:
                    eattr[:, 1:2] = m_sim_t
                if eattr.size(-1) >= 3:
                    eattr[:, 2:3] = s_sim_t
            return eattr
        except Exception as _e:
            logger.warning(f"[edges-fallback] enrich meta failed: {_e}")
            return eattr

    # Mini training override for stability (env: USE_MINI_TRAIN=1)
    try:
        if os.getenv("USE_MINI_TRAIN", "0") == "1":
            _ = run_mini_training(
                model,
                data_module,
                final_config,
                device,
                max_epochs=int(os.getenv("MINI_MAX_EPOCHS", "3")),
            )
            # Mini training finished successfully; exit train() gracefully.
            logger.info("Mini training completed; exiting main train() early.")
            return
    except Exception as _me:
        logger.warning(f"Mini training failed or skipped: {_me}")

    # Force mini training path for stabilization (default ON). Set FORCE_MINI_TRAIN=0 to disable.
    if os.getenv("FORCE_MINI_TRAIN", "1") == "1":
        logger.info("[Control] Forcing mini training path (FORCE_MINI_TRAIN=1)")
        _ = run_mini_training(
            model,
            data_module,
            final_config,
            device,
            max_epochs=int(os.getenv("MINI_MAX_EPOCHS", "3")),
        )
        logger.info("[Control] Mini training finished; exiting train()")
        return
    else:
        # Phase Training (A+ minimal) if enabled; otherwise standard training
        try:
            _pt_cfg = getattr(config.train, "phase_training", None)
            # Env override to force disable/enable phase training
            _pt_env = os.getenv("PHASE_TRAINING", "").lower()
            if _pt_env in ("0", "false", "off"):
                _pt_cfg = None
            elif _pt_env in ("1", "true", "on"):
                class _PT: pass
                _pt_cfg = _PT()
                setattr(_pt_cfg, "enabled", True)

            if _pt_cfg is not None and bool(getattr(_pt_cfg, "enabled", False)):
                logger.info("[PhaseTraining] enabled; running phase-wise training")
                _ = run_phase_training(model, train_loader, val_loader, config, device)
            else:
                ckpt_tag = os.getenv("CKPT_TAG", "main").strip() or "main"
                best_val_main = run_training(train_loader, val_loader, tag=ckpt_tag)
        except Exception as _e:
            logger.error(f"[PhaseTraining] failed or disabled: {_e}; falling back to standard training")
            ckpt_tag = os.getenv("CKPT_TAG", "main").strip() or "main"
            best_val_main = run_training(train_loader, val_loader, tag=ckpt_tag)

    # CV評価（学習後に全foldを現行モデルで評価）
    if cv_folds >= 2 and "fold_ranges" in locals() and fold_ranges:
        eval_only = os.getenv("CV_EVAL_ONLY", "1") == "1"
        logger.info(
            f"\n=== CV across {len(fold_ranges)} folds | eval_only={eval_only} ==="
        )
        cv_losses = []
        for i, (val_start, val_end) in enumerate(fold_ranges):
            embargo = pd.to_timedelta(embargo_days, unit="D")
            train_end_eff = val_start - embargo
            # データ構築
            cv_train_ds = ProductionDatasetV2(
                data_module.train_files
                if hasattr(data_module, "train_files")
                else all_files,
                final_config,
                mode="train",
                target_scalers=None,
                start_date=None,
                end_date=train_end_eff,
            )
            scalers = {}
            for h in cv_train_ds.prediction_horizons:
                arr = np.array(cv_train_ds.targets[h], dtype=np.float64)
                if arr.size > 0:
                    m = float(np.mean(arr))
                    s = float(np.std(arr) + 1e-8)
                    scalers[h] = {"mean": m, "std": s}
            cv_val_ds = ProductionDatasetV2(
                data_module.val_files
                if hasattr(data_module, "val_files") and data_module.val_files
                else data_module.train_files,
                final_config,
                mode="val",
                target_scalers=scalers,
                start_date=val_start,
                end_date=val_end,
            )
            cv_train_loader = DataLoader(
                cv_train_ds,
                batch_size=final_config.train.batch.train_batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=True,
                drop_last=True,
                prefetch_factor=4,
                persistent_workers=False,
            )
            cv_val_loader = DataLoader(
                cv_val_ds,
                batch_size=final_config.train.batch.val_batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                persistent_workers=False,
            )
            if eval_only:
                val_loss_i, _, _ = validate(model, cv_val_loader, criterion, device)
            else:
                # foldごとに再学習（軽量化のためエポック短縮可: 環境変数 CV_EPOCHS）
                old_total = total_epochs
                try:
                    total_override = int(os.getenv("CV_EPOCHS", str(total_epochs)))
                    if total_override > 0:
                        final_config.train.scheduler.total_epochs = total_override
                except Exception:
                    pass
                _ = run_training(cv_train_loader, cv_val_loader, tag=f"cv{i+1}")
                # 評価
                val_loss_i, _, _ = validate(model, cv_val_loader, criterion, device)
                # 戻す
                final_config.train.scheduler.total_epochs = old_total
            cv_losses.append(val_loss_i)
            logger.info(
                f"CV fold#{i+1}: dates=({val_start.date()}..{val_end.date()}) val_loss={val_loss_i:.4f}"
            )
        if cv_losses:
            logger.info(
                f"CV mean val_loss={np.mean(cv_losses):.4f} ± {np.std(cv_losses):.4f}"
            )

    logger.info("\n=== Training Complete ===")
    try:
        if float(best_val_main) < float("inf"):
            logger.info(f"Best validation loss: {float(best_val_main):.4f}")
        else:
            logger.info("Validation was disabled (no validation data)")
    except Exception:
        logger.info("Validation summary unavailable")

    # 最終モデルを保存
    final_path = Path("models/checkpoints/atft_gat_fan_final.pt")
    try:
        final_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    torch.save(
        {
            "epoch": int(final_config.train.scheduler.total_epochs),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": OmegaConf.to_container(final_config),
            "run_manifest": run_manifest,
        },
        final_path,
    )
    logger.info(f"Final model saved to {final_path}")

    # HPO Metrics Output for Optuna Integration
    _maybe_output_hpo_metrics(final_config, best_val_main, run_manifest)

    # Optional: Run Safe Walk-Forward + Embargo evaluation after training
    def _maybe_run_safe_eval(_cfg):
        try:
            if os.getenv("RUN_SAFE_EVAL", "0") != "1":
                return
            logger.info("[SafeEval] RUN_SAFE_EVAL=1 → starting SafeTrainingPipeline evaluation")
            # Locate evaluation data
            data_hint = os.getenv("SAFE_EVAL_DATA", "")
            if data_hint:
                dp = Path(data_hint)
            else:
                # Common default used by SafeTrainingPipeline
                dp = Path("data/raw/large_scale/ml_dataset_full.parquet")
            if not dp.exists():
                logger.warning(f"[SafeEval] data file not found: {dp}; skipping SafeEvaluation")
                return
            # Outputs
            out_dir = Path(os.getenv("SAFE_EVAL_OUT", "output/safe_eval"))
            out_dir.mkdir(parents=True, exist_ok=True)
            # Splits and embargo
            try:
                n_splits = int(os.getenv("SAFE_EVAL_SPLITS", "5"))
            except Exception:
                n_splits = 5
            try:
                horizons = list(getattr(_cfg.data.time_series, "prediction_horizons", [1, 5, 10, 20]))
                embargo_days = int(max(horizons)) if horizons else 20
            except Exception:
                embargo_days = 20
            try:
                mem_gb = float(os.getenv("SAFE_EVAL_MEM_GB", "6.0"))
            except Exception:
                mem_gb = 6.0
            # Import pipeline
            try:
                from gogooku3.training.safe_training_pipeline import SafeTrainingPipeline
            except Exception:
                try:
                    from src.gogooku3.training.safe_training_pipeline import SafeTrainingPipeline
                except Exception as _ie:
                    logger.warning(f"[SafeEval] SafeTrainingPipeline import failed: {_ie}")
                    return
            try:
                pipeline = SafeTrainingPipeline(
                    data_path=dp, output_dir=out_dir, experiment_name="wf_pe_evaluation", verbose=True
                )
                res = pipeline.run_pipeline(
                    n_splits=n_splits, embargo_days=embargo_days, memory_limit_gb=mem_gb, save_results=True
                )
                # Save a small marker
                summary_path = out_dir / "safe_eval_summary.json"
                summary_path.write_text(json.dumps(res.get("final_report", res), ensure_ascii=False, indent=2))
                logger.info(
                    f"[SafeEval] Completed WF+Embargo evaluation → results saved to {out_dir}"
                )
                print(f"\n✅ SafeEval summary: {summary_path.resolve()}\n")
            except Exception as _e:
                logger.warning(f"[SafeEval] pipeline run failed: {_e}")
        except Exception:
            pass

    _maybe_run_safe_eval(final_config)


def _maybe_output_hpo_metrics(final_config, best_val_loss, run_manifest):
    """Output HPO metrics for Optuna integration if HPO environment variables are set"""
    try:
        # Check if running under HPO optimization
        trial_number = os.getenv("HPO_TRIAL_NUMBER")
        trial_dir = os.getenv("HPO_TRIAL_DIR")

        if not trial_number or not trial_dir:
            return  # Not running under HPO

        trial_dir = Path(trial_dir)
        trial_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"🎯 HPO Trial {trial_number}: Outputting metrics for Optuna")

        # Extract metrics from run manifest
        hpo_metrics = {
            'trial_number': int(trial_number),
            'best_val_loss': float(best_val_loss) if best_val_loss != float("inf") else None,
            'training_completed': True,
            'final_epoch': int(getattr(final_config.train.scheduler, 'total_epochs', 0)),
            'timestamp': time.time()
        }

        # Extract multi-horizon metrics if available in run_manifest
        if run_manifest and 'training_history' in run_manifest:
            history = run_manifest['training_history']

            # Get final epoch metrics
            if history:
                final_epoch_data = history[-1] if isinstance(history, list) else history

                # Extract RankIC and Sharpe metrics for each horizon
                rank_ic = {}
                sharpe = {}

                # Look for horizon-specific metrics in the final epoch data
                if isinstance(final_epoch_data, dict):
                    for key, value in final_epoch_data.items():
                        # Extract RankIC metrics
                        if 'rank_ic' in key.lower() or 'rankic' in key.lower():
                            # Parse horizon from key (e.g., "val_rank_ic_1d" -> "1d")
                            for horizon in ['1d', '5d', '10d', '20d']:
                                if horizon in key:
                                    rank_ic[horizon] = float(value) if value is not None else 0.0
                                    break

                        # Extract Sharpe metrics
                        elif 'sharpe' in key.lower():
                            for horizon in ['1d', '5d', '10d', '20d']:
                                if horizon in key:
                                    sharpe[horizon] = float(value) if value is not None else 0.0
                                    break

                    # Also extract general validation metrics
                    for metric_key in ['val_loss', 'val_sharpe', 'val_ic', 'val_rank_ic', 'val_hit_rate']:
                        if metric_key in final_epoch_data:
                            hpo_metrics[metric_key] = float(final_epoch_data[metric_key])

                # Store horizon-specific metrics
                if rank_ic:
                    hpo_metrics['rank_ic'] = rank_ic
                if sharpe:
                    hpo_metrics['sharpe'] = sharpe

        # Look for recent metrics files as fallback
        if 'rank_ic' not in hpo_metrics or 'sharpe' not in hpo_metrics:
            try:
                # Try to find recent metrics from JSON files
                metrics_files = list(Path("output/metrics").glob("epoch_*.json"))
                if metrics_files:
                    # Get the latest metrics file
                    latest_metrics = max(metrics_files, key=lambda p: p.stat().st_mtime)
                    with open(latest_metrics, 'r') as f:
                        metrics_data = json.load(f)

                    # Extract metrics
                    if isinstance(metrics_data, dict):
                        for key in ['val_sharpe', 'val_ic', 'val_rank_ic']:
                            if key in metrics_data:
                                hpo_metrics[key] = float(metrics_data[key])

            except Exception as e:
                logger.debug(f"Could not extract fallback metrics: {e}")

        # Save HPO metrics to trial directory
        hpo_metrics_path = trial_dir / "hpo_metrics.json"
        with open(hpo_metrics_path, 'w') as f:
            json.dump(hpo_metrics, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ HPO metrics saved to: {hpo_metrics_path}")
        logger.info(f"   Best val loss: {hpo_metrics.get('best_val_loss', 'N/A')}")
        logger.info(f"   RankIC metrics: {len(hpo_metrics.get('rank_ic', {}))}")
        logger.info(f"   Sharpe metrics: {len(hpo_metrics.get('sharpe', {}))}")

        # Also create a simple summary for quick parsing
        summary_path = trial_dir / "trial_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Trial: {trial_number}\n")
            f.write(f"Status: {'COMPLETED' if hpo_metrics['training_completed'] else 'FAILED'}\n")
            f.write(f"Best Val Loss: {hpo_metrics.get('best_val_loss', 'N/A')}\n")
            f.write(f"Final Epoch: {hpo_metrics['final_epoch']}\n")

            if 'rank_ic' in hpo_metrics:
                f.write("RankIC:\n")
                for horizon, value in hpo_metrics['rank_ic'].items():
                    f.write(f"  {horizon}: {value:.4f}\n")

            if 'sharpe' in hpo_metrics:
                f.write("Sharpe:\n")
                for horizon, value in hpo_metrics['sharpe'].items():
                    f.write(f"  {horizon}: {value:.4f}\n")

        logger.info(f"📋 Trial summary saved to: {summary_path}")

    except Exception as e:
        logger.warning(f"Failed to output HPO metrics: {e}")


    _maybe_run_safe_eval(final_config)

    # Gracefully finish W&B run if enabled
    try:
        if wb_logger is not None:
            wb_logger.finish()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        train()
    finally:
        if os.getenv("MLFLOW", "0") == "1":
            try:
                import mlflow  # type: ignore

                mlflow.end_run()
            except Exception:
                pass
