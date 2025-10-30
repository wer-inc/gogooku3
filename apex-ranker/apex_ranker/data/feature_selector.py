from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class FeatureSelectionResult:
    """Container for selected feature and mask columns."""

    features: list[str]
    masks: list[str]

    def with_suffix(
        self, suffix: str, *, existing: Iterable[str] | None = None
    ) -> list[str]:
        """Return feature names with the given suffix appended.

        Args:
            suffix: Suffix to append (e.g., ``_cs_z``).
            existing: Optional iterable of column names that already exist.

        Returns:
            List of renamed features. If ``existing`` is provided the suffix is only
            applied when ``{feature}{suffix}`` exists in the iterable.
        """
        if existing is None:
            return [f"{col}{suffix}" for col in self.features]

        existing_set = set(existing)
        renamed: list[str] = []
        for col in self.features:
            candidate = f"{col}{suffix}"
            renamed.append(candidate if candidate in existing_set else col)
        return renamed


class FeatureSelector:
    """Feature selection helper driven by YAML configuration."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Feature groups config not found: {config_path}")

        with self.config_path.open("r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp)

        if not isinstance(data, dict) or "groups" not in data:
            raise ValueError(f"Invalid feature group config: {config_path}")

        self.groups: dict[str, dict[str, list[str]]] = data["groups"]

    def available_groups(self) -> list[str]:
        return sorted(self.groups.keys())

    def select(
        self,
        *,
        groups: Iterable[str],
        optional_groups: Iterable[str] | None = None,
        exclude_features: Iterable[str] | None = None,
        metadata_path: str | Path | None = None,
    ) -> FeatureSelectionResult:
        """Select feature columns for the provided groups.

        Args:
            groups: Required feature groups to include.
            optional_groups: Optional feature groups to include.
            exclude_features: List of feature names to exclude from selection.
            metadata_path: Path to metadata file for validation.

        Returns:
            FeatureSelectionResult with selected features and masks.
        """

        ordered_features: list[str] = []
        ordered_masks: list[str] = []

        def _extend(group_name: str) -> None:
            group_cfg = self.groups.get(group_name)
            if not group_cfg:
                raise KeyError(
                    f"Feature group '{group_name}' is not defined in {self.config_path}"
                )

            for key in ("include", "masks"):
                if key not in group_cfg:
                    continue
                values = group_cfg[key] or []
                target = ordered_features if key == "include" else ordered_masks
                for item in values:
                    if item not in target:
                        target.append(item)

        for g in groups:
            _extend(g)
        if optional_groups:
            for g in optional_groups:
                _extend(g)

        # Apply exclusions
        if exclude_features:
            exclude_set = set(exclude_features)
            ordered_features = [f for f in ordered_features if f not in exclude_set]
            # Log excluded features for transparency
            excluded_count = len([f for f in exclude_features if f in exclude_set])
            if excluded_count > 0:
                print(
                    f"[FeatureSelector] Excluded {excluded_count} features from selection"
                )

        if metadata_path:
            self._validate_against_metadata(
                metadata_path, ordered_features, ordered_masks
            )

        return FeatureSelectionResult(features=ordered_features, masks=ordered_masks)

    def _validate_against_metadata(
        self,
        metadata_path: str | Path,
        features: list[str],
        masks: list[str],
    ) -> None:
        path = Path(metadata_path)
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with path.open("r", encoding="utf-8") as fp:
            doc = json.load(fp)

        available_columns: set[str] = set()
        if isinstance(doc, dict):
            for _key, value in doc.items():
                if isinstance(value, list):
                    available_columns.update(value)

        missing = [col for col in (*features, *masks) if col not in available_columns]
        if missing:
            raise ValueError(
                f"Columns missing from metadata {metadata_path}: {missing}"
            )
