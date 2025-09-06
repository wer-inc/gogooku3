# ATFT-GAT-FAN 設定管理・モデル管理最適化計画

## 🎯 最適化目標

### 現在の問題点
1. **設定管理**: 移行された設定ファイルが未活用
2. **モデル管理**: パスが不統一、複数モデルの管理が困難
3. **設定分散**: ハードコードされた設定が散在

### 最適化目標
1. **統一された設定管理システム**の構築
2. **統合されたモデル管理システム**の構築
3. **設定の一元化**と**動的設定変更**の実現

## 📋 Phase 1: 設定管理最適化

### 1.1 現在の設定状況分析

#### 問題のある設定箇所
```python
# atft_inference.py - ハードコードされた設定
checkpoint_path = ATFT_PATH / "models/checkpoints/atft_gat_fan_final.pt"
config_path = ATFT_PATH / "configs/model/atft_gat_fan_v1.yaml"

# ハードコードされた設定構造
self.config = OmegaConf.create({
    "model": model_config,
    "data": {
        "sequence_length": 20,
        "prediction_horizons": [1, 2, 3, 5, 10],
        "num_features": 13
    },
    "train": {
        "batch": {
            "train_batch_size": 512,
            "val_batch_size": 1024
        }
    }
})
```

#### 移行済みの設定ファイル（未活用）
- `configs/atft/train/profiles/robust.yaml` - 成功した学習設定
- `configs/atft/model/atft_gat_fan_v1.yaml` - モデル設定
- `configs/atft/data/source.yaml` - データ設定
- `configs/atft/hydra/` - Hydra設定

### 1.2 設定管理最適化計画

#### Step 1: 設定マネージャーの作成
```python
# configs/atft_config_manager.py
class ATFTConfigManager:
    """ATFT-GAT-FAN設定管理システム"""

    def __init__(self, base_config_dir: str = "configs/atft"):
        self.base_config_dir = Path(base_config_dir)
        self.config_cache = {}

    def load_profile(self, profile_name: str) -> OmegaConf:
        """学習プロファイルを読み込み"""
        profile_path = self.base_config_dir / "train" / "profiles" / f"{profile_name}.yaml"
        return OmegaConf.load(profile_path)

    def load_model_config(self, model_name: str = "atft_gat_fan_v1") -> OmegaConf:
        """モデル設定を読み込み"""
        model_path = self.base_config_dir / "model" / f"{model_name}.yaml"
        return OmegaConf.load(model_path)

    def merge_configs(self, *configs) -> OmegaConf:
        """複数の設定を統合"""
        return OmegaConf.merge(*configs)
```

#### Step 2: 設定ファイルの統合
```yaml
# configs/atft/inference_config.yaml
defaults:
  - model/atft_gat_fan_v1
  - data/source
  - train/profiles/robust

inference:
  checkpoint_path: "models/best_model_v2.pth"
  device: "auto"
  batch_size: 512

  # 推論特有の設定
  return_confidence: true
  prediction_horizons: [1, 2, 3, 5, 10]
```

#### Step 3: 動的設定変更システム
```python
# scripts/config_override.py
class ConfigOverride:
    """実行時設定変更システム"""

    @staticmethod
    def override_from_env(config: OmegaConf) -> OmegaConf:
        """環境変数から設定を上書き"""
        for key, value in os.environ.items():
            if key.startswith("ATFT_"):
                config_key = key[5:].lower().replace("_", ".")
                OmegaConf.update(config, config_key, value)
        return config

    @staticmethod
    def override_from_args(config: OmegaConf, args: dict) -> OmegaConf:
        """コマンドライン引数から設定を上書き"""
        for key, value in args.items():
            OmegaConf.update(config, key, value)
        return config
```

### 1.3 設定最適化の実装計画

#### 実装順序
1. **設定マネージャー作成** (1日)
2. **設定ファイル統合** (1日)
3. **atft_inference.py修正** (0.5日)
4. **train_atft_wrapper.py廃止**（ARCHIVED、内製パイプラインへ統一） (0.5日)
5. **テスト・検証** (1日)

## 📋 Phase 2: モデル管理最適化

### 2.1 現在のモデル管理状況分析

#### 問題のあるモデルパス
```python
# 現在使用中（ATFT-GAT-FAN）
checkpoint_path = ATFT_PATH / "models/checkpoints/atft_gat_fan_final.pt"

# 移行されたファイル（未使用）
checkpoint_path = "models/best_model_v2.pth"
```

#### モデルファイルの分散
- `ATFT-GAT-FAN/models/checkpoints/atft_gat_fan_final.pt`
- `gogooku3/models/best_model_v2.pth`
- `gogooku3/models/checkpoints/` (空)

### 2.2 モデル管理最適化計画

#### Step 1: モデルマネージャーの作成
```python
# models/atft_model_manager.py
class ATFTModelManager:
    """ATFT-GAT-FANモデル管理システム"""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.model_registry = {}
        self.load_model_registry()

    def load_model_registry(self):
        """モデルレジストリを読み込み"""
        registry_path = self.models_dir / "model_registry.json"
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                self.model_registry = json.load(f)

    def register_model(self, name: str, path: str, metadata: dict):
        """モデルを登録"""
        self.model_registry[name] = {
            "path": path,
            "metadata": metadata,
            "registered_at": datetime.now().isoformat()
        }
        self.save_model_registry()

    def get_model_path(self, name: str) -> Path:
        """モデルパスを取得"""
        if name in self.model_registry:
            return Path(self.model_registry[name]["path"])
        else:
            # デフォルトモデル
            return self.models_dir / "best_model_v2.pth"

    def list_models(self) -> List[str]:
        """利用可能なモデル一覧を取得"""
        return list(self.model_registry.keys())
```

#### Step 2: モデルファイルの統合
```bash
# モデルファイル統合スクリプト
#!/bin/bash
# scripts/consolidate_models.sh

# モデルディレクトリ作成
mkdir -p models/checkpoints
mkdir -p models/archives

# 移行されたモデルを適切な場所に配置
cp models/best_model_v2.pth models/checkpoints/atft_gat_fan_best_v2.pt

# モデルレジストリ作成
cat > models/model_registry.json << 'EOF'
{
  "best_v2": {
    "path": "models/checkpoints/atft_gat_fan_best_v2.pt",
    "metadata": {
      "sharpe_ratio": 0.849,
      "training_date": "2025-08-25",
      "model_size": "77MB",
      "description": "Best performing ATFT-GAT-FAN model"
    }
  },
  "default": {
    "path": "models/checkpoints/atft_gat_fan_best_v2.pt",
    "metadata": {
      "description": "Default model for inference"
    }
  }
}
EOF
```

#### Step 3: モデルバージョン管理
```python
# models/model_versioning.py
class ModelVersioning:
    """モデルバージョン管理システム"""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.version_file = self.models_dir / "versions.json"

    def create_version(self, model_path: str, version: str, description: str):
        """新しいバージョンを作成"""
        versions = self.load_versions()
        versions[version] = {
            "path": model_path,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "sharpe_ratio": self.extract_sharpe_ratio(model_path)
        }
        self.save_versions(versions)

    def get_latest_version(self) -> str:
        """最新バージョンを取得"""
        versions = self.load_versions()
        if not versions:
            return "default"
        return max(versions.keys(), key=lambda x: versions[x]["created_at"])
```

### 2.3 モデル管理最適化の実装計画

#### 実装順序
1. **モデルマネージャー作成** (1日)
2. **モデルファイル統合** (0.5日)
3. **atft_inference.py修正** (0.5日)
4. **モデルバージョン管理** (1日)
5. **テスト・検証** (1日)

## 📋 Phase 3: 統合システム構築

### 3.1 統合設定・モデル管理システム

#### 統合マネージャー
```python
# scripts/atft_system_manager.py
class ATFTSystemManager:
    """ATFT-GAT-FAN統合システム管理"""

    def __init__(self):
        self.config_manager = ATFTConfigManager()
        self.model_manager = ATFTModelManager()
        self.versioning = ModelVersioning()

    def setup_inference(self,
                       model_name: str = "default",
                       profile_name: str = "robust",
                       device: str = "auto") -> ATFTInference:
        """推論システムをセットアップ"""

        # 設定読み込み
        config = self.config_manager.load_profile(profile_name)
        model_config = self.config_manager.load_model_config()

        # モデルパス取得
        model_path = self.model_manager.get_model_path(model_name)

        # 推論システム初期化
        return ATFTInference(
            checkpoint_path=str(model_path),
            config=config,
            device=device
        )

    def setup_training(self,
                      profile_name: str = "robust",
                      config_overrides: dict = None) -> dict:
        """学習システムをセットアップ"""

        # 設定読み込み
        config = self.config_manager.load_profile(profile_name)

        # 設定上書き
        if config_overrides:
            config = ConfigOverride.override_from_args(config, config_overrides)

        return config
```

### 3.2 新しい使用方法

#### 推論実行
```python
# 新しい使用方法
from scripts.atft_system_manager import ATFTSystemManager

# システムマネージャー初期化
system = ATFTSystemManager()

# 推論システムセットアップ
atft = system.setup_inference(
    model_name="best_v2",
    profile_name="robust"
)

# 推論実行
predictions = atft.predict(features)
```

#### 学習実行
```python
# 学習システムセットアップ
config = system.setup_training(
    profile_name="robust",
    config_overrides={
        "train.batch.train_batch_size": 256,
        "trainer.max_epochs": 30
    }
)

# 学習実行
python scripts/train_atft.py --config-path=configs/atft/inference_config.yaml
```

## 📅 実装スケジュール

### Week 1: 設定管理最適化
- **Day 1-2**: 設定マネージャー作成
- **Day 3**: 設定ファイル統合
- **Day 4**: atft_inference.py修正
- **Day 5**: テスト・検証

### Week 2: モデル管理最適化
- **Day 1-2**: モデルマネージャー作成
- **Day 3**: モデルファイル統合
- **Day 4**: モデルバージョン管理
- **Day 5**: テスト・検証

### Week 3: 統合システム構築
- **Day 1-2**: 統合システムマネージャー作成
- **Day 3**: 新しい使用方法の実装
- **Day 4-5**: 総合テスト・ドキュメント作成

## 🎯 期待される効果

### 設定管理最適化
- ✅ **設定の一元化**: すべての設定が`configs/atft/`に集約
- ✅ **動的設定変更**: 実行時の設定変更が可能
- ✅ **設定の再利用**: 学習・推論で同じ設定を使用
- ✅ **設定の検証**: 設定の整合性チェック

### モデル管理最適化
- ✅ **モデルの統一**: すべてのモデルが`models/`に集約
- ✅ **バージョン管理**: モデルの履歴管理
- ✅ **メタデータ管理**: モデルの性能情報管理
- ✅ **自動選択**: 最適なモデルの自動選択

### 統合効果
- ✅ **使いやすさ向上**: シンプルなAPI
- ✅ **保守性向上**: 設定・モデルの一元管理
- ✅ **拡張性向上**: 新しい設定・モデルの追加が容易
- ✅ **性能維持**: Sharpe比 0.849を完全維持

## 🚀 次のステップ

1. **Phase 1開始**: 設定管理最適化の実装
2. **段階的移行**: 既存システムを壊さずに段階的に移行
3. **テスト強化**: 各段階での十分なテスト
4. **ドキュメント更新**: 新しい使用方法のドキュメント作成

この最適化により、ATFT-GAT-FANの設定管理とモデル管理が大幅に改善され、より使いやすく保守しやすいシステムになります。
