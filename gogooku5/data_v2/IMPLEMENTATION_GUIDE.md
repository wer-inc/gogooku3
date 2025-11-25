このドキュメントは、`data_v2/REQUIREMENTS.md` の設計を実装に落とすためのフレームとガイドラインです。Polars + DuckDB 前提で、最小セットから段階的に構築できるようにステップを整理しています。

---

# 1. フォルダ構成（最小セット）

```
data_v2/
  config/
    sources.yml        # RAWソース定義（API/ファイルパス・列マッピング）
    chunks.yml         # チャンク定義（chunk_id, start, end, warmup_days）
    features.yml       # 有効な feature_family 一覧
    settings.yml       # 並列度・パス・圧縮など
  etl/
    __init__.py
    config_loader.py
    models/
      chunk.py         # Chunk dataclass
      raw.py           # RawPartition (manifest用)
    utils/
      dates.py         # 営業日シフト、日付レンジ
      io.py            # scan/read/write (Polars/DuckDB)
      logging.py
      parallel.py      # Thread/Process pool helper
    raw/
      manifest.py      # rebuild_raw_manifest, upsert_raw_manifest
      prices.py        # ソース別 ingest (例)
      backfill.py      # バックフィル並列
    core/
      prices.py        # RAW -> CORE 正規化 (例)
      dims.py          # dim_security, dim_calendar
    feature/
      base.py          # FeatureFamily Protocol
      registry.py      # features.yml をロードして有効化
      price_momentum.py# 価格系の例 (stateful)
    dataset/
      build_chunk.py   # familyをjoinしてchunk dataset作成
    quality/
      schema_core.py   # pandera/pydantic スキーマ (必要に応じて)
      health_check.py  # PSI/KS/相関など
  cache/               # dim/feature/state/meta (実行後に生成)
  data/                # raw/core/feature/dataset (実行後に生成)
  Makefile             # Dagsterラッパー/ローカル実行用
```

---

# 2. 実装ステップと優先度

1) **設定/モデルの足場を作る**
   - `config/settings.yml` を読み込む `config_loader.py`
   - `models/chunk.py`, `models/raw.py`

2) **RAW manifest と ingest 最小例（prices＋trading_calendar）**
   - `raw/manifest.py`: `rebuild_raw_manifest()` で `data/raw/<source>/dt=.../part-*.parquet` をスキャンして `raw_manifest.parquet` を作る
   - `raw/prices.py`: 指定日の RAW を取得→軽い検証→`data/raw/prices/dt=.../part-*.parquet` に書く→manifest に upsert
   - `raw/trading_calendar.py` または scripts: from/to なしで全期間一括取得し、fingerprint してキャッシュ（同内容ならスキップ）
   - `raw/backfill.py`: 日付レンジで ThreadPool に投げる

3) **CORE 正規化（pricesだけ）**
   - `core/prices.py`: RAW manifest から path を引き、`security_id`/`trade_date`/`close` など型を確定して `data/core/fact_prices/dt=.../part-*.parquet` に保存
   - `core/dims.py`: dim_security/dim_calendar を作成

4) **Feature framework と1つの family 実装**
   - `feature/base.py` (Protocol) と `feature/registry.py`
   - `feature/price_momentum.py`: `ret_1d/5d/20d`, `vol_20d` などを stateful family として実装（chunk.warmup_start〜chunk.end_date で計算し、chunk.start_date以降を返す）

5) **Dataset ビルド（チャンク単位）**
   - `dataset/build_chunk.py`: feature family を LazyFrame join → target 付与 → `data/dataset/chunk=<chunk_id>/ml_dataset.arrow` に書き出し
   - 余力があれば graph 特徴は後付けジョブに分離

6) **品質チェック（簡易版）**
   - `quality/health_check.py`: `data/tools/data_health_check.py` 相当を呼び出せるようにし、PSI/KS/相関のサマリを JSON/ログに出す

7) **Make/Dagster ラッパー**
   - Make でローカル実行（`make raw-backfill`, `make core-prices`, `make feature-price-momentum`, `make dataset-chunk CHUNK=2020Q1` など）
   - Dagster job は後から追加

---

# 3. 具体コードの雛形

## 3.1 models/chunk.py

```python
from dataclasses import dataclass
from datetime import date
from etl.utils.dates import add_business_days

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    start_date: date
    end_date: date
    warmup_days: int

    @property
    def warmup_start(self) -> date:
        return add_business_days(self.start_date, -self.warmup_days)
```

## 3.2 raw/manifest.py

```python
import polars as pl
from pathlib import Path
from etl.models.raw import RawPartition

RAW_ROOT = Path("data/raw")
MANIFEST_PATH = Path("data/raw_manifest.parquet")

def rebuild_raw_manifest() -> None:
    records = []
    for path in RAW_ROOT.rglob("part-*.parquet"):
        # path: data/raw/<source>/dt=YYYY-MM-DD/part-...
        parts = path.parts
        source = parts[-3]
        dt = parts[-2].split("=")[1]
        records.append(
            {
                "source": source,
                "dt": dt,
                "version": 1,
                "path": str(path),
                "ingested_at": path.stat().st_mtime_ns,
                "n_rows": None,
                "status": "ok",
                "checksum": None,
            }
        )
    if not records:
        return
    pl.DataFrame(records).write_parquet(MANIFEST_PATH)
```

## 3.3 raw/prices.py (簡略)

```python
import polars as pl
from datetime import date, datetime
from pathlib import Path
from etl.raw.manifest import rebuild_raw_manifest
from etl.models.raw import RawPartition

RAW_ROOT = Path("data/raw/prices")

def quick_validate_raw(df: pl.DataFrame) -> bool:
    if df.is_empty():
        return False
    for c in ["Code", "Date", "Close"]:
        if c not in df.columns or df[c].null_count() == df.height:
            return False
    return True

def ingest_prices_for_date(dt: date, df: pl.DataFrame) -> RawPartition:
    dt_str = dt.strftime("%Y-%m-%d")
    out_dir = RAW_ROOT / f"dt={dt_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"part-{ts}.parquet"
    df.write_parquet(path, compression="zstd")
    status = "ok" if quick_validate_raw(df) else "failed"
    part = RawPartition(
        source="prices",
        dt=dt,
        version=1,
        path=str(path),
        ingested_at=datetime.utcnow(),
        n_rows=df.height,
        status=status,
        checksum=None,
    )
    # upsert_manifest(part)  # 簡略: rebuildで代用
    return part
```

## 3.4 core/prices.py (正規化)

```python
import polars as pl
from datetime import date
from pathlib import Path

RAW_MANIFEST = Path("data/raw_manifest.parquet")
CORE_ROOT = Path("data/core/fact_prices")

def core_prices_for_date(dt: date) -> None:
    dt_str = dt.strftime("%Y-%m-%d")
    man = pl.read_parquet(RAW_MANIFEST).filter(
        (pl.col("source") == "prices") & (pl.col("dt") == dt_str)
    )
    if man.is_empty():
        return
    path = man["path"][0]
    lf = pl.scan_parquet(path)
    core = (
        lf.select(
            [
                pl.col("Code").alias("security_code"),
                pl.col("Date").cast(pl.Date).alias("trade_date"),
                pl.col("Close").cast(pl.Float32),
                pl.col("Volume").cast(pl.Int64),
            ]
        )
    )
    out_dir = CORE_ROOT / f"dt={dt_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    core.collect(streaming=True).write_parquet(out_dir / "part.parquet", compression="zstd")
```

## 3.5 feature/base.py

```python
from typing import Protocol, Tuple
import polars as pl
from etl.models.chunk import Chunk

class FeatureFamily(Protocol):
    name: str
    stateful: bool

    def needed_sources(self) -> list[str]:
        ...

    def compute(
        self,
        core_sources: dict[str, pl.LazyFrame],
        chunk: Chunk,
        prev_state: pl.DataFrame | None,
    ) -> Tuple[pl.DataFrame, pl.DataFrame | None]:
        ...
```

## 3.6 feature/price_momentum.py（例）

```python
import polars as pl
from etl.feature.base import FeatureFamily
from etl.models.chunk import Chunk

class PriceMomentum(FeatureFamily):
    name = "price_momentum"
    stateful = True

    def needed_sources(self) -> list[str]:
        return ["fact_prices"]

    def compute(self, core_sources, chunk: Chunk, prev_state):
        prices = (
            core_sources["fact_prices"]
            .filter(pl.col("trade_date").is_between(chunk.warmup_start, chunk.end_date))
            .select(["security_code", "trade_date", "close"])
        )
        lf = (
            prices
            .groupby("security_code")
            .sort("trade_date")
            .with_columns(
                [
                    (pl.col("close") / pl.col("close").shift(1) - 1).alias("ret_1d"),
                    (pl.col("close") / pl.col("close").shift(5) - 1).alias("ret_5d"),
                    (pl.col("close").log().diff().rolling_std(window_size=20)).alias("vol_20d"),
                ]
            )
        )
        df = lf.collect()
        features = df.filter(pl.col("trade_date") >= chunk.start_date)
        return features, None  # stateを持たない簡易版
```

## 3.7 dataset/build_chunk.py

```python
import polars as pl
from etl.feature.registry import get_enabled_families, scan_feature_chunk
from etl.models.chunk import Chunk

def build_dataset_for_chunk(chunk: Chunk):
    families = get_enabled_families()
    lf = None
    for fam in families:
        f = scan_feature_chunk(fam.name, chunk)  # LazyFrameで読み込む
        lf = f if lf is None else lf.join(f, on=["security_code", "trade_date"], how="inner")
    target = build_target(chunk)  # 別途実装
    lf = lf.join(target, on=["security_code", "trade_date"], how="inner")
    df = lf.collect(streaming=True)
    df.write_ipc(f"data/dataset/chunk={chunk.chunk_id}/ml_dataset.arrow", compression="zstd")
```

---

# 4. Polars/DuckDB 併用指針

- **Polars**: パイプラインの本流。`scan_parquet` + lazy で列絞り・pushdownを活用。特徴計算は Polars の groupby/sort/rolling で完結させる。
- **DuckDB**: アドホック検証・大きめの結合/集計で活用（必要に応じて）。Polars DataFrame を DuckDB に渡す/受けるだけでコピーを抑えられる。
- デバッグ時: RAWやCOREを DuckDB に外部テーブルとして貼って SQL で分布確認、問題なければ Polars で本番パイプラインを流す。

---

# 5. 品質・モニタリング

- RAW: 軽い検証のみ（ゼロ行・主要列全欠損を弾く）。
- CORE/FEATURE/DATASET: スキーマチェック（pandera/pydantic可）、`data/tools/data_health_check.py` 相当で相関リーク/PSI/KS をチャンク完了後に実行。
- ログ: 進捗と警告は構造化ログ（JSONやプレーンでも良い）で記録し、あとで簡易ダッシュボードへ。

---

# 6. 並列と設定チューニング

- RAW backfill: ThreadPool（IO bound）で日次単位。ソースごとに worker 数を設定。
- CORE/Feature非stateful/Dataset: ProcessPool＋`POLARS_MAX_THREADS` で CPU に合わせて調整。
- Stateful feature family: チャンク順にシーケンシャル。ただし各チャンク内は Polars のマルチスレッドに任せる。
- 圧縮: Parquet=zstd、IPC=lz4 をデフォルト。
- Warmup days: 最大ルックバック + 安全マージン（120〜180営業日程度）に抑え、過剰な過去スキャンを避ける。

---

# 7. 優先タスクの目安

1. `config_loader.py` と `models` を作る。
2. `raw/manifest.py` と `raw/prices.py`（サンプル）で RAW→manifest の流れを通す。
3. `core/prices.py` で CORE を作る。
4. `feature/base.py` と `feature/price_momentum.py`（サンプル）を追加。
5. `dataset/build_chunk.py` で join → dataset 出力まで一度通す。
6. `quality/health_check.py` で簡易監査を仕込む。
7. Make ターゲットを用意（raw-backfill/core-prices/feature-price-momentum/dataset-chunk）。

---

このガイドラインをベースに、`data_v2/REQUIREMENTS.md` の設計を最小スコープ（prices＋基本特徴）から実装し、徐々にソースと特徴ファミリを広げていく流れを推奨します。実データのスキーマに合わせて各 select/rename 部分を書き換えれば、すぐに動かせる構成です。***
