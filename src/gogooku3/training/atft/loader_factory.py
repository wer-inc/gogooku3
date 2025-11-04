"""
DataLoader factory with spawn, persistent workers, and thread control.
P0-5: DataLoader安定化の中核モジュール
"""
from __future__ import annotations

import faulthandler
import os
import random
import signal

import numpy as np
import torch
from torch.utils.data import DataLoader


def _worker_init_fn(worker_id: int):
    """DataLoaderワーカー初期化関数

    各ワーカープロセスで実行され、以下を設定：
    - 例外時のスタックトレース有効化
    - シグナルハンドラ設定
    - 乱数シードの分散（再現性確保）
    - ワーカー内スレッド数の抑制
    """
    # 例外時にスタックトレース
    faulthandler.enable()
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    # 乱数の分散（再現性）
    # torch.initial_seed()は64bit整数なので、uint32に収まるよう調整
    base_seed = torch.initial_seed()
    seed = (base_seed + worker_id) % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)

    # ワーカー内のスレッドをさらに抑止
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("PYARROW_NUM_THREADS", "1")  # 各ワーカー内は単スレッド


def _collate_passthrough(batch):
    """
    IterableDataset が '1日=1バッチのdict' を yield する前提。
    DataLoader は list[dict] を渡してくるので、単一要素を返す。

    Args:
        batch: DataLoaderが渡すバッチ（通常はlist[dict]）

    Returns:
        単一のdictバッチ（1日分のデータ）
    """
    if isinstance(batch, list) and len(batch) == 1 and isinstance(batch[0], dict):
        return batch[0]
    # もし将来 map-style に切替えたら、ここで通常の collation に分岐
    return batch


def make_loader(
    dataset,
    num_workers: int | None = None,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    timeout: int = 0
) -> DataLoader:
    """安定化されたDataLoaderを作成

    特徴:
    - spawn multiprocessing context（forkの問題回避）
    - persistent_workers（エポック跨ぎの再生成なし）
    - worker_init_fn（スレッド制御、乱数シード）
    - collate_fn（1日=1バッチの前提に対応）

    Args:
        dataset: IterableDataset（1日=1バッチをyield）
        num_workers: ワーカー数（Noneなら自動設定）
        pin_memory: GPU転送の高速化
        prefetch_factor: プリフェッチバッチ数
        timeout: タイムアウト秒数

    Returns:
        設定済みDataLoader
    """
    if num_workers is None:
        # Safe mode なら 0、通常はコア数の1/4〜8
        if os.getenv("FORCE_SINGLE_PROCESS", "0") == "1":
            num_workers = 0
        else:
            cw = os.cpu_count() or 8
            num_workers = min(8, max(2, cw // 4))

    # multiprocessing_context は num_workers > 0 の時のみ有効
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": None,                 # IterableDataset: dataset側でバッチ化済み
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": (num_workers > 0),
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
        "collate_fn": _collate_passthrough,
        "timeout": timeout,
    }

    if num_workers > 0:
        ctx = torch.multiprocessing.get_context("spawn")
        loader_kwargs["multiprocessing_context"] = ctx
        loader_kwargs["worker_init_fn"] = _worker_init_fn

    return DataLoader(**loader_kwargs)
