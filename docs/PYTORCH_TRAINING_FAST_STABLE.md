# PyTorch で「速く・安定して・再現可能」に学習するための即効セット

このリポジトリの学習スクリプト（例: `scripts/train_atft.py`）でそのまま使える実践的な設定をまとめました。
設定はすべて環境変数／ワンライナーで切り替えられるので、今日から即反映できます。

---

## 1. 再現性（決定論モード）

シードと決定論モードを揃えるだけで「同じハイパラで結果がズレる」問題をほぼ解消できます。

```bash
DETERMINISTIC=1 SEED=42 python scripts/train_atft.py
```

- `scripts/train_atft.py` 内の `fix_seed()` が `PYTHONHASHSEED` / Python / NumPy / PyTorch / CUDA の RNG を一括で固定します。
- `DETERMINISTIC=1` にすると `torch.use_deterministic_algorithms(True)` を自動適用（`warn_only=True` で非決定論カーネルは警告＆代替カーネルに切替）。
- デフォルトは高速寄り (`DETERMINISTIC=0`) なので、検証ルンでは `DETERMINISTIC=1`、本番高速学習では `0` と使い分けてください。

⚠️ 決定論モードは畳み込みや reduce の高速カーネルが使えなくなる場合があり、数％〜十数％遅くなることがあります。

---

## 2. AMP（自動混合精度）

AMP を ON にすると、同じ GPU メモリでバッチサイズを増やせたり、計算時間が短くなります。

```bash
USE_AMP=1 python scripts/train_atft.py
```

- `USE_AMP`（既定値 1）を切ると FP32 フル精度、1 にすると AMP が有効になります。
- AMP 実行時は `torch.amp.autocast(device_type="cuda", dtype=torch.float16)` ＋ `torch.amp.GradScaler` がすでに組み込まれており、そのまま利用可能です。
- TensorFloat-32（TF32）も自動で有効化されるため、A100 では matmul 系が高速化されます。

---

## 3. DataLoader のボトルネック解消

`NUM_WORKERS` などの環境変数を触るだけでデータ供給性能を調整できます。

| 変数 | 推奨値 | 説明 |
|------|--------|------|
| `NUM_WORKERS` | CPU コア数の 1/2〜同数で探索（例: 8, 12, 16） | 0 の場合は単一プロセス |
| `PIN_MEMORY` | `1` (CUDA 使用時) | H→D 転送を非同期化 |
| `PREFETCH_FACTOR` | `2`〜`4`（`NUM_WORKERS>0` のときのみ有効） | 各ワーカーの先読みミニバッチ数 |
| `PERSISTENT_WORKERS` | `1`（長時間学習） | ワーカー再生成のオーバーヘッドを回避 |

例:

```bash
NUM_WORKERS=12 PIN_MEMORY=1 PREFETCH_FACTOR=3 PERSISTENT_WORKERS=1 \
python scripts/train_atft.py
```

- これらの値は Hydra 設定 (`train.batch.*`) からも指定できます。環境変数で上書きしたい場合は上記を参照してください。
- `ALLOW_UNSAFE_DATALOADER=0` を設定すると強制的にシングルプロセスに落とせるので、安定性優先のジョブではこちらを使います。

---

## 4. すぐ試せるワンライナー

```bash
DETERMINISTIC=1 SEED=42 USE_AMP=1 \
NUM_WORKERS=12 PIN_MEMORY=1 PREFETCH_FACTOR=3 PERSISTENT_WORKERS=1 \
python scripts/train_atft.py
```

上記 1 行で「再現性 ON + AMP + GPU を待たせない DataLoader」が揃います。速度と再現性はトレードオフなので、目的に応じて `DETERMINISTIC` や `NUM_WORKERS` を切り替えてください。

---

## Appendix: 素の PyTorch で使えるテンプレ

`scripts/templates/pytorch_fast_training_template.py` に、上記のベストプラクティスをそのまま貼り付けられるテンプレートを用意しました。ATFT 以外の小規模実験でもそのまま流用できます。
