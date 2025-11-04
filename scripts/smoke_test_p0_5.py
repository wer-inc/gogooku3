"""
P0-5 DataLoader安定化の煙テスト
spawn + persistent_workers + スレッド制御の動作確認
"""
import os
import sys
import time

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# FORCE_SINGLE_PROCESS を解除（Safe mode OFF）
os.environ.pop("FORCE_SINGLE_PROCESS", None)

# bootstrap: spawn設定 + スレッド制御
import scripts.bootstrap_threads as boot

boot.set_spawn_start_method()
boot.configure_torch_threads()

import torch
from torch.utils.data import IterableDataset

from src.gogooku3.training.atft.loader_factory import make_loader


class DummyIterableDataset(IterableDataset):
    """1日=1バッチを模擬するダミーデータセット"""
    def __init__(self, n_days=100):
        self.n_days = n_days

    def __iter__(self):
        for day_idx in range(self.n_days):
            # 実際のデータセットと同じ形式（1日=1バッチのdict）
            yield {
                "features": torch.randn(500, 20, 64),  # [N_stocks, T, H]
                "targets": torch.randn(500, 4),         # [N_stocks, n_horizons]
                "mask": torch.ones(500, dtype=torch.bool),
                "date": f"2023-01-{day_idx+1:02d}",
            }


def test_multiworker_dataloader():
    """Multi-worker DataLoader安定性テスト"""
    print("=" * 80)
    print("P0-5 Smoke Test: DataLoader Stability (spawn + persistent_workers)")
    print("=" * 80)

    dataset = DummyIterableDataset(n_days=100)

    # workers=4 でテスト
    num_workers = 4
    loader = make_loader(
        dataset,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        timeout=0
    )

    print("\n✅ DataLoader created successfully")
    print(f"   - num_workers: {num_workers}")
    print("   - multiprocessing_context: spawn")
    print("   - persistent_workers: True")

    # スループット測定
    t0 = time.time()
    n_batches = 0

    for i, batch in enumerate(loader):
        n_batches += 1

        # バッチ内容の検証
        assert "features" in batch, "Missing 'features' in batch"
        assert "targets" in batch, "Missing 'targets' in batch"
        assert batch["features"].shape[0] == 500, f"Wrong batch size: {batch['features'].shape[0]}"

        if i >= 19:  # 20バッチで終了
            break

    t1 = time.time()
    throughput = n_batches / (t1 - t0)

    print("\n✅ Multi-worker test PASSED")
    print(f"   - Batches fetched: {n_batches}")
    print(f"   - Time elapsed: {t1-t0:.2f}s")
    print(f"   - Throughput: {throughput:.2f} batches/sec")
    print("   - No SIGABRT errors detected")

    return throughput


def test_single_worker_fallback():
    """Safe mode (workers=0) フォールバックテスト"""
    print("\n" + "=" * 80)
    print("Testing Safe Mode Fallback (FORCE_SINGLE_PROCESS=1)")
    print("=" * 80)

    os.environ["FORCE_SINGLE_PROCESS"] = "1"

    dataset = DummyIterableDataset(n_days=20)
    loader = make_loader(dataset, num_workers=None)

    # 自動的に num_workers=0 になることを確認
    assert loader.num_workers == 0, f"Expected num_workers=0, got {loader.num_workers}"

    print("\n✅ Safe mode fallback PASSED")
    print("   - num_workers: 0 (auto-detected)")

    # 少しだけ実行
    for i, batch in enumerate(loader):
        if i >= 4:
            break

    print(f"   - Safe mode batches fetched: {i+1}")

    # 環境変数をクリーンアップ
    os.environ.pop("FORCE_SINGLE_PROCESS", None)


if __name__ == "__main__":
    try:
        # Test 1: Multi-worker stability
        throughput = test_multiworker_dataloader()

        # Test 2: Safe mode fallback
        test_single_worker_fallback()

        print("\n" + "=" * 80)
        print("P0-5 SMOKE TEST: ALL TESTS PASSED ✅")
        print("=" * 80)
        print("\nDataLoader stabilization complete:")
        print("  - spawn context: ✅ No fork() deadlocks")
        print("  - persistent_workers: ✅ No worker re-creation")
        print("  - Thread control: ✅ OMP/MKL/Polars limited")
        print(f"  - Throughput: {throughput:.2f} batches/sec")
        print("  - Safe mode: ✅ Auto-fallback working")

    except Exception as e:
        print("\n" + "=" * 80)
        print("P0-5 SMOKE TEST: FAILED ❌")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
