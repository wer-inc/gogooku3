"""
P0-3: PyG環境診断スクリプト

torch_geometric segfault問題の診断と推奨対策の提示。

診断項目:
1. PyTorch/CUDA バージョン
2. torch_geometric インストール状態
3. GATv2Conv インポート可否
4. 推奨対策の提示
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*80)
print("P0-3: PyG環境診断")
print("="*80)

# 1. PyTorch/CUDA バージョン
print("\n[1] PyTorch/CUDA バージョン")
try:
    import torch
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            print(f"  GPU 0: {torch.cuda.get_device_name(0)}")
    print(f"  Python: {sys.version.split()[0]}")
except Exception as e:
    print(f"  ❌ PyTorch import failed: {e}")
    sys.exit(1)

# 2. torch_geometric インストール状態
print("\n[2] torch_geometric インストール状態")
try:
    import torch_geometric
    print(f"  ✅ torch_geometric: {torch_geometric.__version__}")

    # 拡張パッケージのチェック
    extensions = []
    try:
        import pyg_lib
        extensions.append(f"pyg_lib-{pyg_lib.__version__}")
    except:
        pass
    try:
        import torch_scatter
        extensions.append(f"torch_scatter-{torch_scatter.__version__}")
    except:
        pass
    try:
        import torch_sparse
        extensions.append(f"torch_sparse-{torch_sparse.__version__}")
    except:
        pass
    try:
        import torch_cluster
        extensions.append(f"torch_cluster-{torch_cluster.__version__}")
    except:
        pass

    if extensions:
        print(f"  Extensions: {', '.join(extensions)}")
    else:
        print("  ⚠️  No PyG extensions installed (performance may be limited)")

except Exception as e:
    print(f"  ❌ torch_geometric not installed: {e}")
    print("\n  推奨インストール方法:")
    print("    pip install torch_geometric")

# 3. GATv2Conv インポート可否
print("\n[3] GATv2Conv インポート可否")
gatv2_available = False
try:
    from torch_geometric.nn import GATv2Conv
    print("  ✅ GATv2Conv import successful")
    gatv2_available = True

    # 簡易動作テスト
    try:
        import torch
        z = torch.randn(10, 32)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()
        edge_attr = torch.randn(2, 3)

        gat = GATv2Conv(32, 16, heads=2, edge_dim=3)
        out = gat(z, edge_index, edge_attr)

        print(f"  ✅ GATv2Conv forward test passed (output shape: {out.shape})")
    except Exception as e:
        print(f"  ⚠️  GATv2Conv forward test failed: {e}")

except Exception as e:
    print(f"  ❌ GATv2Conv import failed: {e}")
    print(f"     Error type: {type(e).__name__}")

# 4. 推奨対策
print("\n" + "="*80)
print("診断結果と推奨対策")
print("="*80)

if gatv2_available:
    print("\n✅ PyG環境は正常です。")
    print("   通常モードで学習を実行できます:")
    print("     make train-quick EPOCHS=3")
else:
    print("\n⚠️  PyG環境に問題があります。以下の対策を推奨します:")

    # 現在のPyTorchバージョンに応じた推奨
    torch_version = torch.__version__.split('+')[0]  # "2.9.0+cu128" -> "2.9.0"
    cuda_version = torch.version.cuda if torch.cuda.is_available() else None

    print("\n【対策A】安全シム（GraphConvShim）で即座に学習開始 [推奨]")
    print("  性能: PyG実装の60-80%程度")
    print("  用途: RFI-5/6データ収集に十分")
    print("  実行方法:")
    print("    USE_GAT_SHIM=1 make train-quick EPOCHS=3")

    print("\n【対策B-1】PyTorch 2.8.0+cu128 に降格（安定） [推奨]")
    print("  理由: data.pyg.org で2.8.0+cu128用ホイールが公開済み")
    print("  手順:")
    print("    pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128")
    print("    pip install torch_geometric")
    print("    pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \\")
    print("      -f https://data.pyg.org/whl/torch-2.8.0+cu128.html")

    if torch_version.startswith("2.9"):
        print("\n【対策B-2】PyTorch 2.9.0+cu128 のままPyGをソースビルド [上級]")
        print("  注意: ビルド時間が長く、エラーが発生しやすい")
        print("  手順:")
        print("    export TORCH_CUDA_ARCH_LIST=\"12.0\"")
        print("    pip install -v --no-binary pyg-lib,torch-scatter,torch-sparse,torch-cluster,torch-spline-conv \\")
        print("      pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv")
        print("    pip install torch_geometric")

    print("\n【現状分析】")
    print(f"  PyTorch: {torch.__version__}")
    if cuda_version:
        print(f"  CUDA: {cuda_version}")
        if cuda_version == "12.8":
            print("  ⚠️  CUDA 12.8は比較的新しく、PyGエコシステムの対応が遅延中")
            print("      参考: https://github.com/pyg-team/pytorch_geometric/issues/10142")
    print("  data.pyg.org 公開ホイール: torch-2.8.0+cu128 まで")
    print("  結論: 対策B-1（2.8.0降格）が最も安定")

print("\n" + "="*80)
print("次のステップ")
print("="*80)
print("1. 【すぐ回す】: USE_GAT_SHIM=1 make train-quick EPOCHS=3")
print("   → RFI-5/6を収集（ゲート統計、グラフ統計、Sharpe/RankIC）")
print("2. 【安定化】: 時間を見てPyTorch 2.8.0+cu128へ降格")
print("   → PyG実装（GATv2Conv）に切替、性能向上")
print("3. 【ログ共有】: 以下のメトリクスをご報告ください")
print("   - gat_gate_mean/std (Phase2 1-3epoch)")
print("   - deg_avg/isolates/corr_stats (任意1バッチ)")
print("   - Sharpe_EMA / RankIC / WQL or CRPS / quantile_crossing_rate")
print("="*80)
