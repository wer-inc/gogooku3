# アーキテクチャ: Feature Store（Feast）

本システムは Feast を用いて特徴量のオフライン/オンライン提供を行います。

## コンポーネント
- エンティティ: 銘柄ID（例: `local_code`）
- フィーチャービュー: 日次特徴量、市場特徴量など
- オフラインストア: Parquet/ClickHouse
- オンラインストア: Redis（低レイテンシ配信）

## 利用フロー
1. 特徴量生成（バッチ）→ オフラインストアへ書き込み
2. Feast materialize → オンラインストア（Redis）へ同期
3. 推論時にエンティティキーでオンライン取得

## 運用
- materialization スケジュールをDagsterで管理
- スキーマ変更時は後方互換・フルリビルド方針を明示
- 品質ゲート: 欠損/外れ値/将来情報混入をブロック

## 参考
- Feast Docs: https://docs.feast.dev/
- 関連: `docs/architecture/data-pipeline.md`

