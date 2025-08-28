#!/usr/bin/env python3
"""
MLflow Model Registry Integration
モデルレジストリ管理
"""

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRegistry:
    """MLflow Model Registry 管理クラス"""

    def __init__(self, tracking_uri: str = None):
        """
        Initialize Model Registry

        Args:
            tracking_uri: MLflow tracking server URI
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            mlflow.set_tracking_uri(
                os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            )

        self.client = MlflowClient()

    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: Dict[str, str] = None,
        description: str = None,
    ) -> str:
        """
        モデルを登録

        Args:
            model_uri: モデルURI (runs:/run_id/model)
            name: 登録名
            tags: タグ
            description: 説明

        Returns:
            モデルバージョン
        """
        # Register model
        model_details = mlflow.register_model(
            model_uri=model_uri,
            name=name,
        )

        # Update model version
        version = model_details.version

        # Set description
        if description:
            self.client.update_model_version(
                name=name,
                version=version,
                description=description,
            )

        # Set tags
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(
                    name=name,
                    version=version,
                    key=key,
                    value=value,
                )

        logger.info(f"Model {name} version {version} registered")
        return version

    def transition_model_stage(
        self,
        name: str,
        version: str,
        stage: str,
        archive_existing: bool = True,
    ):
        """
        モデルステージを遷移

        Args:
            name: モデル名
            version: バージョン
            stage: ステージ ("Staging", "Production", "Archived")
            archive_existing: 既存モデルをアーカイブ
        """
        valid_stages = ["None", "Staging", "Production", "Archived"]
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {valid_stages}")

        self.client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing,
        )

        logger.info(f"Model {name} version {version} transitioned to {stage}")

    def get_model(
        self,
        name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> Any:
        """
        モデルを取得

        Args:
            name: モデル名
            version: バージョン（指定しない場合は最新）
            stage: ステージ（"Staging", "Production"）

        Returns:
            モデルオブジェクト
        """
        if version:
            model_uri = f"models:/{name}/{version}"
        elif stage:
            model_uri = f"models:/{name}/{stage}"
        else:
            model_uri = f"models:/{name}/latest"

        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Loaded model from {model_uri}")

        return model

    def list_models(self) -> List[Dict[str, Any]]:
        """
        登録されているモデル一覧を取得

        Returns:
            モデル情報のリスト
        """
        models = []
        for model in self.client.search_registered_models():
            model_info = {
                "name": model.name,
                "creation_time": datetime.fromtimestamp(
                    model.creation_timestamp / 1000
                ),
                "last_updated": datetime.fromtimestamp(
                    model.last_updated_timestamp / 1000
                ),
                "description": model.description,
                "tags": model.tags,
                "versions": [],
            }

            # Get versions
            for version in model.latest_versions:
                version_info = {
                    "version": version.version,
                    "stage": version.current_stage,
                    "status": version.status,
                    "creation_time": datetime.fromtimestamp(
                        version.creation_timestamp / 1000
                    ),
                }
                model_info["versions"].append(version_info)

            models.append(model_info)

        return models

    def get_model_versions(
        self,
        name: str,
        stages: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        モデルのバージョン一覧を取得

        Args:
            name: モデル名
            stages: フィルタするステージリスト

        Returns:
            バージョン情報のリスト
        """
        filter_string = f"name='{name}'"
        versions_info = []

        for version in self.client.search_model_versions(filter_string):
            if stages and version.current_stage not in stages:
                continue

            version_info = {
                "version": version.version,
                "stage": version.current_stage,
                "status": version.status,
                "creation_time": datetime.fromtimestamp(
                    version.creation_timestamp / 1000
                ),
                "run_id": version.run_id,
                "tags": version.tags,
                "description": version.description,
            }

            # Get run metrics
            if version.run_id:
                run = self.client.get_run(version.run_id)
                version_info["metrics"] = run.data.metrics
                version_info["params"] = run.data.params

            versions_info.append(version_info)

        return versions_info

    def compare_models(
        self,
        name: str,
        versions: List[str] = None,
        metrics: List[str] = None,
    ) -> Dict[str, Any]:
        """
        モデルバージョンを比較

        Args:
            name: モデル名
            versions: 比較するバージョンリスト
            metrics: 比較するメトリック名リスト

        Returns:
            比較結果
        """
        all_versions = self.get_model_versions(name)

        if versions:
            all_versions = [v for v in all_versions if v["version"] in versions]

        comparison = {
            "model_name": name,
            "versions": [],
        }

        for version_info in all_versions:
            version_data = {
                "version": version_info["version"],
                "stage": version_info["stage"],
                "metrics": {},
                "params": version_info.get("params", {}),
            }

            # Filter metrics
            if metrics and "metrics" in version_info:
                version_data["metrics"] = {
                    k: v for k, v in version_info["metrics"].items() if k in metrics
                }
            elif "metrics" in version_info:
                version_data["metrics"] = version_info["metrics"]

            comparison["versions"].append(version_data)

        return comparison

    def delete_model_version(self, name: str, version: str):
        """
        モデルバージョンを削除

        Args:
            name: モデル名
            version: バージョン
        """
        self.client.delete_model_version(name=name, version=version)
        logger.info(f"Deleted model {name} version {version}")

    def delete_model(self, name: str):
        """
        モデル全体を削除

        Args:
            name: モデル名
        """
        self.client.delete_registered_model(name=name)
        logger.info(f"Deleted model {name}")

    def create_model_signature(
        self,
        input_columns: List[tuple],
        output_columns: List[tuple],
    ) -> ModelSignature:
        """
        モデルシグネチャを作成

        Args:
            input_columns: [(name, type), ...] の入力カラム定義
            output_columns: [(name, type), ...] の出力カラム定義

        Returns:
            ModelSignature オブジェクト
        """
        input_schema = Schema([ColSpec(dtype, name) for name, dtype in input_columns])

        output_schema = Schema([ColSpec(dtype, name) for name, dtype in output_columns])

        return ModelSignature(inputs=input_schema, outputs=output_schema)

    def promote_model(
        self,
        name: str,
        from_stage: str = "Staging",
        to_stage: str = "Production",
        archive_existing: bool = True,
    ):
        """
        モデルを昇格

        Args:
            name: モデル名
            from_stage: 元のステージ
            to_stage: 移行先ステージ
            archive_existing: 既存をアーカイブ
        """
        # Get latest version in from_stage
        versions = self.client.get_latest_versions(name, stages=[from_stage])

        if not versions:
            raise ValueError(f"No model found in {from_stage} stage")

        latest_version = versions[0]

        # Transition to new stage
        self.transition_model_stage(
            name=name,
            version=latest_version.version,
            stage=to_stage,
            archive_existing=archive_existing,
        )

        logger.info(f"Promoted {name} from {from_stage} to {to_stage}")

    def rollback_model(
        self,
        name: str,
        stage: str = "Production",
        target_version: Optional[str] = None,
    ):
        """
        モデルをロールバック

        Args:
            name: モデル名
            stage: ステージ
            target_version: ロールバック先バージョン（省略時は前バージョン）
        """
        # Get current production version
        current_versions = self.client.get_latest_versions(name, stages=[stage])
        if not current_versions:
            raise ValueError(f"No model in {stage}")

        current_version = current_versions[0].version

        if target_version is None:
            # Find previous version
            all_versions = self.get_model_versions(name)
            all_versions.sort(key=lambda x: x["version"], reverse=True)

            for v in all_versions:
                if v["version"] < current_version:
                    target_version = v["version"]
                    break

        if target_version is None:
            raise ValueError("No previous version found for rollback")

        # Archive current and promote target
        self.transition_model_stage(name, current_version, "Archived", False)
        self.transition_model_stage(name, target_version, stage, False)

        logger.info(f"Rolled back {name} from v{current_version} to v{target_version}")


def test_model_registry():
    """Model Registry のテスト"""
    print("Testing MLflow Model Registry")
    print("=" * 50)

    registry = ModelRegistry()

    # List models
    models = registry.list_models()
    print(f"\nRegistered models: {len(models)}")

    for model in models:
        print(f"\n  Model: {model['name']}")
        print(f"    Description: {model.get('description', 'N/A')}")
        print(f"    Versions: {len(model['versions'])}")

        for version in model["versions"]:
            print(f"      - v{version['version']}: {version['stage']}")

    # Test model signature creation
    signature = registry.create_model_signature(
        input_columns=[
            ("feature1", "double"),
            ("feature2", "double"),
            ("feature3", "long"),
        ],
        output_columns=[
            ("prediction", "double"),
            ("confidence", "double"),
        ],
    )

    print("\n✓ Model signature created")
    print(f"  Input schema: {signature.inputs}")
    print(f"  Output schema: {signature.outputs}")


if __name__ == "__main__":
    test_model_registry()
