よし、じゃあ一旦「仕様書レベル」で全部まとめ直します。
**このままリポジトリ構成とコード骨格に落とせる**くらいの粒度で書きます。

---

# 0. ゴール & 前提

### ゴール

* 5年バックフィルが **24h → 数時間レベル** まで落ちること
* OOM を極力避け、**チャンク単位で安全に再実行できる**こと
* RAW を一度取れば、**以後は RAW＋キャッシュで何度でも再計算できる**こと
* 日次運用とバックフィルをきれいに分離

### 技術スタック（確定）

* 言語: **Python**
* ETL: **Polars** (+ DuckDB optional)
* フォーマット: Parquet / Arrow(IPC)
* オーケストレーション: Dagster (+ Make はラッパー)
* 並列処理: `concurrent.futures`（Thread/Process）＋ Polars 内部マルチスレッド

---

# 1. ディレクトリ構成（最終形）

```text
project_root/
  config/
    sources.yml       # RAWソース定義
    chunks.yml        # チャンク定義（2020Q1〜）
    features.yml      # 有効な feature_family 定義
    settings.yml      # 並列度・パスなど

  data/
    raw/
      <source>/dt=YYYY-MM-DD/part-<ts>-<uuid>.parquet
    core/
      fact_prices/dt=YYYY-MM-DD/part-*.parquet
      fact_short_selling/dt=YYYY-MM-DD/part-*.parquet
      ...
      dim_security.parquet
      dim_calendar.parquet
    feature/
      <family>/chunk=<chunk_id>/part-*.arrow
    dataset/
      chunk=<chunk_id>/ml_dataset.arrow

  cache/
    dim/
      dim_security.arrow
      dim_calendar.arrow
    feature/
      <family>/<cache_key>.arrow
    state/
      <family>/chunk=<chunk_id>.arrow
    meta/
      feature_cache_index.parquet   # path, last_accessed_at, size, ttl
      chunk_status.parquet          # chunk_id, status, updated_at, notes

  dagster_project/
    # Dagster jobs/ops/graphs 定義

  etl/
    __init__.py
    config_loader.py
    models/             # Chunk, RawPartition, etc.
    utils/
      dates.py
      io.py
      logging.py
      parallel.py
    raw/
      prices.py
      margin.py
      macro.py
      backfill.py
    core/
      prices.py
      margin.py
      macro.py
      dims.py
    feature/
      base.py
      price_momentum.py
      margin_features.py
      macro_features.py
      ...
    dataset/
      build_chunk.py
      merge_all.py
    quality/
      schema_core.py
      schema_feature.py
      health_check.py

  Makefile
  pyproject.toml
  README.md
```

---

# 2. 共通モデル & 設定

## 2.1 Chunk モデル

```python
# etl/models/chunk.py
from dataclasses import dataclass
from datetime import date
from typing import Literal

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    start_date: date
    end_date: date
    warmup_days: int

    @property
    def warmup_start(self) -> date:
        # 営業日換算する util を使う
        from etl.utils.dates import add_business_days
        return add_business_days(self.start_date, -self.warmup_days)
```

* `config/chunks.yml` からロードして `List[Chunk]` に変換
* `warmup_days` は「最大ルックバック + 安全マージン」(例: 120〜180営業日)

## 2.2 RawPartition モデル & manifest

```python
# etl/models/raw.py
from pydantic import BaseModel
from datetime import date, datetime

class RawPartition(BaseModel):
    source: str           # "prices", "short_selling", ...
    dt: date
    version: int
    path: str
    ingested_at: datetime
    n_rows: int
    status: str           # "ok", "failed", "partial"
    checksum: str | None
```

* `data/raw_manifest.parquet` に対応
* manifest は Polars で読み書きしつつ、レコード単位では Pydantic で型安全に扱う

## 2.3 設定 loader

```python
# etl/config_loader.py
import yaml
from dataclasses import dataclass

@dataclass
class Settings:
    data_root: str
    max_parallel_raw: int
    max_parallel_chunks: int
    polars_max_threads: int
    backfill_mode: bool

def load_settings() -> Settings:
    with open("config/settings.yml") as f:
        y = yaml.safe_load(f)
    return Settings(**y)
```

* `backfill_mode: true` の時だけ並列度を上げる、などをここで制御

---

# 3. RAW レイヤー設計

## 3.1 ファイルレイアウト

```text
data/raw/<source>/dt=YYYY-MM-DD/part-<timestamp>-<uuid>.parquet
```

特徴：

* 日次 partition: `dt=YYYY-MM-DD`
* 同じ日を取り直した場合は `part-*` が増える → manifest の `version` で最新を指す

## 3.2 RAW manifest

* ファイル: `data/raw_manifest.parquet`
* カラム例：

```text
source | dt | version | path | ingested_at | n_rows | status | checksum
```

### 更新フロー（build-raw）

```python
# etl/raw/manifest.py
def rebuild_raw_manifest() -> None:
    # data/raw 以下を scan_parquet でスキャンし、メタだけ収集
    # 既存 manifest とマージし、(source, dt, version) ごとに最新を保持
```

* `make build-raw` or Dagster `job_rebuild_raw_manifest` で実行
* RAW 追加時にも 1レコード append/upsert

## 3.3 軽い検証

```python
import polars as pl

def quick_validate_raw(df: pl.DataFrame, required_cols: list[str]) -> bool:
    if df.height == 0:
        return False
    for c in required_cols:
        if c not in df.columns:
            return False
        if df[c].null_count() == df.height:
            return False
    return True
```

* 行数ゼロ・主要列全欠損だけ弾く
* schema validation は RAW ではやらない（落とさない）

## 3.4 ソースごとの ingestion ジョブ

```python
# etl/raw/prices.py
from datetime import date, datetime
import polars as pl
import anyio

async def fetch_prices_for_date(dt: date) -> pl.DataFrame:
    # httpx or DB client で取得 → Polars DataFrame
    ...

def ingest_prices_for_date(dt: date) -> RawPartition:
    df = anyio.run(fetch_prices_for_date, dt)
    ok = quick_validate_raw(df, ["code", "date", "close"])
    status = "ok" if ok else "failed"

    path = write_raw_parquet("prices", dt, df)
    part = RawPartition(
        source="prices",
        dt=dt,
        version=next_version("prices", dt),
        path=path,
        ingested_at=datetime.utcnow(),
        n_rows=df.height,
        status=status,
        checksum=calc_checksum(df) if ok else None,
    )
    upsert_raw_manifest(part)
    return part
```

## 3.5 5年バックフィル並列処理

```python
# etl/raw/backfill.py
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from etl.raw.prices import ingest_prices_for_date
from etl.utils.dates import trading_days
from etl.config_loader import load_settings

def backfill_prices(start_date: date, end_date: date):
    settings = load_settings()
    dates = trading_days(start_date, end_date)

    def _job(dt: date):
        try:
            return ingest_prices_for_date(dt)
        except Exception as e:
            log_error("prices", dt, e)
            return None

    with ThreadPoolExecutor(max_workers=settings.max_parallel_raw) as ex:
        for f in as_completed([ex.submit(_job, d) for d in dates]):
            _ = f.result()
```

* IO bound なので スレッド並列
* ソースごとに `max_parallel_raw` を調整（レート制限に応じて）

---

# 4. CORE レイヤー（正規化倉庫）

## 4.1 役割

* RAW の雑多な schema を **一度だけ** 正規化
* 以後の全処理（feature/dataset）は CORE から読む
* 型・キー・カラム名を固定して、join をシンプル＆高速に

## 4.2 ファイルレイアウト

```text
data/core/fact_prices/dt=YYYY-MM-DD/part-*.parquet
data/core/fact_short_selling/dt=YYYY-MM-DD/part-*.parquet
data/core/fact_margin/dt=YYYY-MM-DD/part-*.parquet
data/core/macro_daily/dt=YYYY-MM-DD/part-*.parquet
data/core/dim_security.parquet
data/core/dim_calendar.parquet
```

## 4.3 CORE schema の例

`fact_prices`:

```text
security_id: int32
trade_date: date
close: float32
open: float32
high: float32
low: float32
volume: int64
value: float64
...
```

* 可能な限り `int32/float32` に落とす
* `security_id` は `dim_security` より付番（初期は codeでも可）

## 4.4 CORE ビルド処理

```python
# etl/core/prices.py
import polars as pl
from datetime import date

def scan_raw_prices(dt: date) -> pl.LazyFrame:
    path = raw_prices_path(dt)  # manifest から pathを引く
    return pl.scan_parquet(path)

def build_core_prices_for_date(dt: date) -> None:
    raw = scan_raw_prices(dt)  # LazyFrame

    core = (
        raw
        .select([
            pl.col("code").alias("security_code"),
            pl.col("date").cast(pl.Date).alias("trade_date"),
            pl.col("close").cast(pl.Float32),
            pl.col("volume").cast(pl.Int64),
            # ...
        ])
    )

    core.collect(streaming=True).write_parquet(
        core_prices_path(dt),
        compression="zstd",
    )
```

* `scan_parquet` + `select` で **必要な列だけ**使う
* `collect(streaming=True)` でメモリを抑える

## 4.5 CORE 並列化

```python
from concurrent.futures import ProcessPoolExecutor
from etl.utils.dates import trading_days
from etl.config_loader import load_settings

def build_core_prices_range(start_date, end_date):
    settings = load_settings()
    dates = trading_days(start_date, end_date)
    with ProcessPoolExecutor(max_workers=settings.max_parallel_chunks) as ex:
        ex.map(build_core_prices_for_date, dates)
```

* CPU＋I/Oなのでプロセス並列
* `POLARS_MAX_THREADS` を settings から環境変数で設定して、
  `process数 × POLARS_MAX_THREADS ≒ CPUコア数` に合わせる

---

# 5. キャッシュ & state 設計

## 5.1 キャッシュレイアウト

```text
cache/dim/dim_security.arrow
cache/dim/dim_calendar.arrow

cache/feature/<family>/<key>.arrow
cache/state/<family>/chunk=<chunk_id>.arrow

cache/meta/feature_cache_index.parquet
```

### feature cache の key

```python
import hashlib, json
from datetime import date

def manifest_fingerprint(sources: list[str], start: date, end: date) -> dict[str, str]:
    # core_manifest 等から (source, start, end) のハッシュを作る想定
    ...

def feature_cache_key(family_name: str, chunk: Chunk) -> str:
    meta = {
        "family": family_name,
        "chunk": chunk.chunk_id,
        "code_version": current_git_commit(),
        "inputs": manifest_fingerprint(
            needed_sources_for_family(family_name),
            chunk.warmup_start,
            chunk.end_date,
        ),
    }
    s = json.dumps(meta, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()
```

### cache 読み書き utility

```python
def load_feature_cache(family_name: str, chunk: Chunk) -> pl.DataFrame | None:
    key = feature_cache_key(family_name, chunk)
    path = f"cache/feature/{family_name}/{key}.arrow"
    if not os.path.exists(path):
        return None
    touch_cache_index(path)
    return pl.read_ipc(path)

def save_feature_cache(family_name: str, chunk: Chunk, df: pl.DataFrame) -> None:
    key = feature_cache_key(family_name, chunk)
    path = f"cache/feature/{family_name}/{key}.arrow"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.write_ipc(path, compression="lz4")
    update_cache_index(path, df)
```

## 5.2 state 永続化

```python
def state_path(family_name: str, chunk_id: str) -> str:
    return f"cache/state/{family_name}/chunk={chunk_id}.arrow"

def load_prev_state(family_name: str, prev_chunk: Chunk | None) -> pl.DataFrame | None:
    if prev_chunk is None:
        return None
    path = state_path(family_name, prev_chunk.chunk_id)
    if not os.path.exists(path):
        return None
    return pl.read_ipc(path)

def save_state(family_name: str, chunk: Chunk, state: pl.DataFrame) -> None:
    path = state_path(family_name, chunk.chunk_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state.write_ipc(path, compression="lz4")
```

---

# 6. Feature Family フレームワーク

## 6.1 インターフェイス

```python
# etl/feature/base.py
from typing import Protocol
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
    ) -> tuple[pl.DataFrame, pl.DataFrame | None]:
        """
        returns:
          - features: DataFrame (security_id, trade_date, feature_cols...)
          - new_state: DataFrame | None (rolling用のstate)
        """
```

* `stateful=False` の family では `prev_state` / `new_state` を使わない（常に None）

## 6.2 価格モメンタムの例（stateful）

```python
# etl/feature/price_momentum.py
import polars as pl
from etl.feature.base import FeatureFamily
from etl.models.chunk import Chunk

class PriceMomentum(FeatureFamily):
    name = "price_momentum"
    stateful = True

    def needed_sources(self) -> list[str]:
        return ["fact_prices"]

    def compute(
        self,
        core_sources: dict[str, pl.LazyFrame],
        chunk: Chunk,
        prev_state: pl.DataFrame | None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        prices = (
            core_sources["fact_prices"]
            .filter(
                pl.col("trade_date").is_between(chunk.warmup_start, chunk.end_date)
            )
            .select(["security_id", "trade_date", "close"])
        )

        # prev_state には (security_id, last_n_returns...) などを持たせる想定
        df, new_state = compute_momentum_with_state(prices, prev_state)

        features = df.filter(pl.col("trade_date") >= chunk.start_date)

        return features, new_state
```

`compute_momentum_with_state` の中身は Polars only で：

* `groupby("security_id")`
* `sort("trade_date")`
* `rolling_*` or `over` window を活用

Python for ループは使わない方針。

## 6.3 非stateful family の例

```python
class MarginSnapshot(FeatureFamily):
    name = "margin_snapshot"
    stateful = False

    def needed_sources(self) -> list[str]:
        return ["fact_margin"]

    def compute(self, core_sources, chunk, prev_state):
        margin = (
            core_sources["fact_margin"]
            .filter(pl.col("trade_date").is_between(chunk.start_date, chunk.end_date))
            .select([...])
        )
        features = margin  # そのまま or 加工
        return features.collect(streaming=True), None
```

---

# 7. Feature 計算のメインループ & 並列化

## 7.1 1 family × 全チャンク（stateful）

```python
from etl.feature.base import FeatureFamily
from etl.cache.state import load_prev_state, save_state
from etl.cache.feature import load_feature_cache, save_feature_cache

def build_features_for_family_all_chunks(family: FeatureFamily, chunks: list[Chunk]):
    prev_state = None
    prev_chunk = None

    for chunk in chunks:
        # キャッシュヒットならスキップ
        cached = load_feature_cache(family.name, chunk)
        if cached is not None:
            prev_state = load_prev_state(family.name, chunk) or prev_state
            prev_chunk = chunk
            continue

        core_sources = load_core_sources(family.needed_sources(), chunk, family.stateful)

        prev_state = load_prev_state(family.name, prev_chunk) or prev_state

        features, new_state = family.compute(core_sources, chunk, prev_state)
        save_feature_cache(family.name, chunk, features)
        if new_state is not None:
            save_state(family.name, chunk, new_state)

        prev_state = new_state
        prev_chunk = chunk
```

* stateful のため **チャンクは時系列順にシーケンシャル処理**
* 各チャンク内部の heavy 処理は Polars のマルチスレッド

## 7.2 non-stateful family のチャンク並列

```python
from concurrent.futures import ProcessPoolExecutor

def build_non_stateful_family_all_chunks(family: FeatureFamily, chunks: list[Chunk], max_workers: int):
    def _job(ch: Chunk):
        cached = load_feature_cache(family.name, ch)
        if cached is not None:
            return
        cores = load_core_sources(family.needed_sources(), ch, stateful=False)
        feats, _ = family.compute(cores, ch, None)
        save_feature_cache(family.name, ch, feats)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        ex.map(_job, chunks)
```

---

# 8. dataset ビルド

## 8.1 チャンクごとに dataset 作成

```text
data/dataset/chunk=<chunk_id>/ml_dataset.arrow
```

```python
# etl/dataset/build_chunk.py
import polars as pl
from etl.models.chunk import Chunk
from etl.feature.registry import get_enabled_families

def build_dataset_for_chunk(chunk: Chunk):
    families = get_enabled_families()
    lf = None

    for family in families:
        f = scan_feature_family_chunk(family.name, chunk)  # LazyFrame
        lf = f if lf is None else lf.join(
            f, on=["security_id", "trade_date"], how="inner"
        )

    target = build_target(chunk)  # LazyFrame
    lf = lf.join(target, on=["security_id", "trade_date"], how="inner")

    df = lf.collect(streaming=True)
    df = optimize_dtypes_for_ml(df)  # float32/int32/categorical など

    df.write_ipc(dataset_path(chunk), compression="zstd")
```

## 8.2 グラフ特徴の後付け

* グラフ特徴は別ジョブで `dataset` に列追加

```python
# etl/dataset/add_graph_features.py
def add_graph_features(chunks: list[Chunk]):
    for ch in chunks:
        df = pl.read_ipc(dataset_path(ch))
        graph_feats = compute_graph_features(df)
        df2 = df.join(graph_feats, on=["security_id", "trade_date"], how="left")
        df2.write_ipc(dataset_path(ch), compression="zstd")
```

---

# 9. 並列戦略まとめ

## 9.1 RAW

* 日付単位（営業日）で ThreadPool/asyncio 並列
* ソースごとに worker 数を設定 (`settings.max_parallel_raw_prices`, etc.)

## 9.2 CORE

* 日付 or 月単位で ProcessPool 並列
* `POLARS_MAX_THREADS` × `process数` ≒ CPU コア数

## 9.3 FEATURE

* stateful family: **チャンクは順次**, Polars のスレッド並列に任せる
* non-stateful family: チャンクを ProcessPool で並列（max_workers = 2〜4）
* family 間の並列は、まずは使わず（設計が落ち着いてから検討）

## 9.4 DATASET

* チャンク単位で ProcessPool 並列
* チャンク内の join は Polars に任せる

---

# 10. スキーマ検証 & データ品質

## 10.1 レイヤー別ポリシー

* RAW: 軽いチェックのみ（行数ゼロ・主要列全欠損）
* CORE: 型・null率・キー制約（`security_id + trade_date` の重複なし等）
* FEATURE: カラム名・型・NA 禁止列チェック
* DATASET: ターゲットがちゃんと入っているか、外れ値チェックなど

## 10.2 実装イメージ

`pandera` や `pydantic` を利用：

```python
# etl/quality/schema_core.py
import pandera as pa
from pandera.typing import Series

class FactPricesSchema(pa.SchemaModel):
    security_id: Series[int]
    trade_date: Series[pa.DateTime]
    close: Series[float]
    volume: Series[int]

    @pa.check("close")
    def close_non_negative(cls, s: Series[float]) -> Series[bool]:
        return s >= 0
```

### health_check

```python
# etl/quality/health_check.py
def run_health_check_for_chunk(chunk: Chunk):
    df = pl.read_ipc(dataset_path(chunk))
    # PSI / KS, 相関リークチェックなど
    ...
```

Dagster ジョブ:

* `job_data_health_check(chunk_id)` として
  チャンク完了後に実行（失敗してもデータ本体は残す）

---

# 11. GC & 容量管理

## 11.1 消さないもの

* `data/raw/**/*`
* 理想は `data/core/**/*` も残す（ストレージ次第）

## 11.2 消して良いもの

* `cache/feature/*`（TTL 過ぎたもの）
* `cache/state/*`（古いチャンク）
* `data/dataset/chunk=*/ml_dataset.arrow`（再計算可能なら古いもの）

## 11.3 GC ジョブ

`cache/meta/feature_cache_index.parquet` が以下を持つ：

```text
path | family | chunk_id | size_bytes | last_accessed_at | ttl_days
```

GC コマンド:

```bash
make gc-cache-old DAYS=14
```

Python:

```python
def gc_feature_cache(days: int):
    now = datetime.utcnow()
    idx = pl.read_parquet("cache/meta/feature_cache_index.parquet")
    to_del = idx.filter(
        (now - pl.col("last_accessed_at")) > pl.duration(days=days)
    )
    for p in to_del["path"]:
        os.remove(p)
```

---

# 12. Dagster & Make のジョブ設計

## 12.1 Dagster jobs（概念）

* `job_ingest_raw_sources_daily`
* `job_backfill_raw_sources(start_date, end_date)`
* `job_build_core_daily`
* `job_build_core_backfill(start_date, end_date)`
* `job_build_features_for_chunk(chunk_id)`
* `job_build_dataset_for_chunk(chunk_id)`
* `job_data_health_check(chunk_id)`
* `job_gc_cache_old`

## 12.2 Makefile

```Makefile
raw-backfill:
    dagster job run job_backfill_raw_sources --config config/backfill_raw.yml

core-backfill:
    dagster job run job_build_core_backfill --config config/backfill_core.yml

chunk-%:
    dagster job run job_build_features_for_chunk --op-args "chunk_id=$*"

dataset-%:
    dagster job run job_build_dataset_for_chunk --op-args "chunk_id=$*"

health-%:
    dagster job run job_data_health_check --op-args "chunk_id=$*"

gc-cache:
    dagster job run job_gc_cache_old
```

---

# 13. 現行からの移行ステップ（実行順）

1. **ディレクトリ再整理**

   * `data/output/raw` → `data/raw` にコピー＆`dt=` partition 付与
   * 既存 `dim_security.parquet` を `data/core/dim_security.parquet` & `cache/dim/` に移動

2. **raw_manifest の構築**

   * `rebuild_raw_manifest()` を走らせて `data/raw_manifest.parquet` を作る

3. **CORE 層の構築**

   * 既存 RAW から `fact_prices`, `fact_short_selling` 等を作成
   * 初期は少量期間（2020Q1〜2020Q2）でテストし、その後フルバックフィル

4. **Chunk 定義 (`config/chunks.yml`) の作成**

   * 2020Q1〜2025Q4 まで一気に定義
   * `warmup_days` は、今使う最大ルックバック＋αに設定

5. **FeatureFamily 実装の移植**

   * まず `PriceMomentum` のような代表的な family を新フレームワークに移植
   * 2020Q1〜2020Q2 までで動作確認

6. **stateful rolling への切り替え**

   * 「各チャンクごとに337営業日読んでいる」ような処理を廃止
   * 初回チャンクだけ長いウォームアップ＋state 引き継ぎへ

7. **全チャンクバックフィル**

   * 並列度を設定（backfill mode）
   * FEATURE → DATASET を順次構築

8. **日次運用ジョブの定義**

   * `job_ingest_raw_sources_daily`
   * `job_build_core_daily`
   * 最新チャンクの dataset 更新

---

# 14. 実装ロードマップ（現実的な順番）

1. **設定 & モデル層**

   * `Chunk`, `RawPartition`, config loader, path utils など共通基盤

2. **RAW レイヤー再構築**

   * ディレクトリ整理 & manifest
   * RAW 取得の並列化（5年バックフィルの高速化）

3. **CORE 層実装**

   * `fact_prices` だけ作る → 全期間 → 他 fact/macro を追加

4. **Feature フレームワーク**

   * `FeatureFamily` base
   * 1〜2個の family（price_momentum 等）を移植

5. **state 機構**

   * state保存/読み出し
   * 1 family で stateful rolling を実装

6. **dataset ビルド**

   * チャンクごとの dataset builder
   * ターゲット生成 & 型最適化

7. **並列化・チューニング**

   * RAW, CORE, FEATURE の並列度を調整
   * 実際の CPU/RAM 使用を見て `max_parallel_*` と `POLARS_MAX_THREADS` を調整

8. **GC & Health Check**

   * cache GC・health_check ジョブを整備

---

こんな感じで、一通り「抜けなく」設計できると思います。
あとは、

* 実際のソースごとの schema（prices / margin / macro）
* 実際の feature_family 一覧（どんな特徴を持つか）

が分かれば、その部分を **具体的なクラス定義・関数レベル** まで落とし込めます。

もしよければ次のステップとして、

* 代表的なソース（例: prices）の実際のカラム
* 代表的な feature（例: `ret_1d`, `ret_5d`, `vol_20d` など）

を1〜2個だけ教えてもらえれば、そこだけ**ほぼコピペで使える実コード**にしてみます。

---

了解しました。/workspace/gogooku3/gogooku5/data にあるリソースをもとに、RAW取得〜CORE正規化〜Feature計算〜Dataset生成までの全処理を新規実装するための具体的な設計・ファイル構成・コードスケルトンを提案します。しばらくお待ちください。
