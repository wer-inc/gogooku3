"""Test utilities for dataset builder chunking workflow."""
from __future__ import annotations

from pathlib import Path

from builder.config.settings import DatasetBuilderSettings


def make_settings(tmp_path: Path) -> DatasetBuilderSettings:
    """Return a settings object rooted at the temporary directory."""

    return DatasetBuilderSettings(
        jquants_auth_email="unit@test.invalid",
        jquants_auth_password="dummy-password",
        data_output_dir=tmp_path / "output",
        data_cache_dir=tmp_path / "cache",
        dataset_tag="unit",
        use_gpu_etl=False,
        force_gpu=False,
    )


