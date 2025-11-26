# APEX Ranker Admin Panel

管理画面 for APEX Ranker ML System

## 技術スタック

| Layer | Technology |
|-------|------------|
| Frontend | Vite + React + TypeScript |
| Routing | React Router |
| Data Fetching | TanStack Query |
| HTTP Client | ky |
| Styling | Tailwind CSS |
| Backend | FastAPI |
| ORM | SQLAlchemy (async) |
| Database | PostgreSQL |

## クイックスタート

### Docker Compose (推奨)

```bash
# 開発環境を起動
make dev

# または
docker compose up --build
```

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### ローカル開発

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend (別ターミナル)
cd frontend
npm install
npm run dev
```

## ディレクトリ構成

```
admin/
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI エントリーポイント
│   │   ├── routers/          # API エンドポイント
│   │   │   ├── health.py     # ヘルスチェック
│   │   │   ├── users.py      # ユーザー管理
│   │   │   ├── models.py     # MLモデル管理
│   │   │   └── predictions.py# 予測機能
│   │   ├── models/           # Pydantic スキーマ
│   │   ├── services/         # ビジネスロジック
│   │   └── db/               # データベース設定
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   │   ├── components/       # UIコンポーネント
│   │   ├── pages/            # ページコンポーネント
│   │   ├── hooks/            # カスタムフック (API呼び出し)
│   │   ├── api/              # API クライアント・型定義
│   │   ├── App.tsx           # ルーティング設定
│   │   └── main.tsx          # エントリーポイント
│   ├── package.json
│   └── Dockerfile
│
├── docker-compose.yml        # 開発環境
├── docker-compose.prod.yml   # 本番環境
├── Makefile                  # 便利コマンド
└── README.md
```

## API エンドポイント

### Health
- `GET /api/health` - ヘルスチェック
- `GET /api/health/detailed` - 詳細ヘルスチェック

### Users
- `GET /api/users` - ユーザー一覧
- `GET /api/users/{id}` - ユーザー詳細
- `POST /api/users` - ユーザー作成
- `PUT /api/users/{id}` - ユーザー更新
- `DELETE /api/users/{id}` - ユーザー削除

### Models
- `GET /api/models` - モデル一覧
- `GET /api/models/{name}` - モデル詳細
- `GET /api/models/{name}/metrics` - モデルメトリクス
- `POST /api/models/{name}/activate` - モデル有効化

### Predictions
- `GET /api/predictions` - 予測一覧
- `GET /api/predictions/latest` - 最新予測
- `POST /api/predictions` - 予測実行

## 機能

### Dashboard
- システムヘルスステータス
- 最新モデル情報
- クイックアクセス

### Models
- モデル一覧表示
- バリデーションメトリクス表示
  - RankIC, P@K, NDCG, Spread, WIL
- モデル有効化

### Users
- ユーザー管理 (CRUD)
- 管理者権限設定

### Predictions (予定)
- 予測実行
- 予測履歴
- パフォーマンス追跡

## 環境変数

### Backend
```env
DATABASE_URL=postgresql+asyncpg://app:password@db:5432/app
MODEL_DIR=/workspace/gogooku3/gogooku5/output
SECRET_KEY=your-secret-key
```

### Frontend
```env
VITE_API_URL=http://localhost:8000
```

## 本番デプロイ

```bash
# ビルド
make build

# 起動
make start

# 停止
make stop
```

## 開発コマンド

```bash
# 全サービス起動
make dev

# ログ確認
make logs

# データベース初期化
make db-init

# クリーンアップ
make clean
```
