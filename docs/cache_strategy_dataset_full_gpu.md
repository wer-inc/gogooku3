# dataset-full-gpu のキャッシュ戦略（詳細）

目的: GPU相関グラフ計算の再計算を抑えつつ、鮮度を維持して`make dataset-full-gpu`の周回を高速化する。

推奨ポリシー（結論）
- 更新頻度: 日次（マーケットクローズ後）。
- ディレクトリ: 月シャーディング（例: `output/graph_cache/202509/…`）。
- パラメータ分離: `w<window>-t<threshold>-k<max_k>` でフォルダ分離。
- 保持期間(TTL): 90–120日（状況で調整）。

背景と理由
- 入力の多く（株価、先物、TOPIX、日次信用等）が日次で更新されるため、週/月では鮮度劣化が現れやすい。
- 既定の相関窓は60営業日。1日進むだけでも隣接相関が更新され、週固定では週内のズレが増える。
- キャッシュは日付単位でヒットするため、同日再実行や近傍再実行の高速化メリットが大きい。

Makefile 連携
- 変数
  - `GRAPH_WINDOW`（既定 60）
  - `GRAPH_THRESHOLD`（既定 0.5）
  - `GRAPH_MAX_K`（既定 4）
  - `CACHE_SHARD`（`END`日付の`%Y%m`）
  - `CACHE_DIR=output/graph_cache/<YYYYMM>/w<win>-t<thr>-k<k>`
  - `CACHE_TTL_DAYS`（既定 120、`cache-prune`で使用）
- 実行
  - `make dataset-full-gpu START=2024-01-01 END=2025-09-30`
  - `make cache-stats` … キャッシュ規模の把握
  - `make cache-prune [CACHE_TTL_DAYS=90]` … 古い`.pkl`削除

キー衝突回避（重要）
- `src/data/utils/graph_builder_gpu.py` のキャッシュキーに以下を含めるよう更新:
  - 窓幅`w`、閾値`t`（×100の整数）、`max_k`、`method`、`update_frequency`、`include_negative`、`symmetric`。
- これにより、異なるグラフ設定間でのキャッシュ誤再利用を防ぐ。

週/⽉単位への切り替え（必要な場合）
- コスト制約やバックフィル集中期間のみの一時的運用として推奨。
- 週次運用例（鮮度低下を許容）
  - `GRAPH_WINDOW=60 GRAPH_THRESHOLD=0.5 GRAPH_MAX_K=4` はそのまま、
  - ジョブスケジュールを週1回へ（例: 金曜クローズ後に実行）。
  - 週内のデイリー実行は同一キャッシュを再利用する（`CACHE_DIR`は同月・同パラメータで共通）。
- 月次運用は実運用の鮮度要件と合いにくいため非推奨（長期スナップショット用途には可）。

運用Tips
- ディスクが厳しい場合は`CACHE_TTL_DAYS`を短く（60–90日）し、`cache-prune`をCI/cronへ登録。
- パラメータを変えた場合は自動的に別ディレクトリに保存され、キャッシュの混線を防げる。
- 既存キャッシュファイルはTTLで自然に減るため手動削除は不要。

スケジューリング例（UTC, 毎営業日23:30）
```
30 23 * * 1-5 cd /home/ubuntu/gogooku3 && \
  START=$(date -u -d "yesterday -5 years +1 day" +%F) \
  END=$(date -u -d "yesterday" +%F) \
  make dataset-full-gpu START=$START END=$END >> _logs/cron_dataset.log 2>&1
```

障害時の切り戻し
- 一時的に`--graph-cache-dir`を空のディレクトリへ切替（`CACHE_DIR=output/graph_cache/tmp-$(date +%s)`）して干渉を排除。
- `make cache-stats` で容量・件数を確認しつつ`cache-prune`を実行。

