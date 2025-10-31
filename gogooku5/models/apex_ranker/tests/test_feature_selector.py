from __future__ import annotations

from pathlib import Path

from apex_ranker.data import FeatureSelector


def _config_path() -> Path:
    test_dir = Path(__file__).resolve().parent
    package_root = test_dir.parents[0]
    return package_root / "configs" / "feature_groups.yaml"


def test_feature_selector_core50_group_contains_features() -> None:
    selector = FeatureSelector(_config_path())
    result = selector.select(groups=["core50"])

    assert result.features, "core50 group should provide feature columns"


def test_feature_selection_with_suffix_retains_existing_names() -> None:
    selector = FeatureSelector(_config_path())
    result = selector.select(groups=["core50"])
    sample = result.features[:3]

    renamed = result.with_suffix("_cs_z", existing={f"{col}_cs_z" for col in sample})

    for col in sample:
        expected = f"{col}_cs_z"
        assert expected in renamed, "suffix was not applied to existing columns"
