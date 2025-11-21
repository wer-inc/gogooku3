"""Helpers to locate shared dataset artifacts for the APEX-Ranker package."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path


def _find_project_root(markers: Sequence[str] = ("MIGRATION_PLAN.md",)) -> Path:
    """Return the gogooku5 repository root by walking upwards from this file."""

    current = Path(__file__).resolve()
    for parent in current.parents:
        if any((parent / marker).exists() for marker in markers):
            return parent
    raise RuntimeError("Failed to locate gogooku5 project root. Ensure MIGRATION_PLAN.md exists.")


def _candidate_paths(
    raw_path: str | Path | None,
    defaults: Iterable[str],
    *,
    extra_bases: Iterable[Path] | None = None,
) -> list[Path]:
    """Build a list of candidate paths to probe for an artifact."""

    candidates: list[Path] = []
    project_root = _find_project_root()
    workspace_root = project_root.parent
    bases: list[Path] = [Path.cwd(), project_root, workspace_root]
    if extra_bases:
        bases.extend(extra_bases)

    if raw_path:
        path = Path(raw_path)
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.extend(base / path for base in bases)

    candidates.extend(project_root / default for default in defaults)
    candidates.extend(workspace_root / default for default in defaults)
    return candidates


def resolve_artifact_path(
    raw_path: str | Path | None,
    defaults: Iterable[str],
    *,
    kind: str,
    extra_bases: Iterable[Path] | None = None,
) -> Path:
    """Resolve an artifact path using user-provided and default locations."""

    candidates = _candidate_paths(raw_path, defaults, extra_bases=extra_bases)
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Unable to locate {kind}. Checked: {', '.join(str(path) for path in candidates)}")


def resolve_dataset_path(
    parquet_path: str | Path | None = None,
    *,
    extra_bases: Iterable[Path] | None = None,
) -> Path:
    """Resolve the dataset parquet path, probing shared defaults when necessary."""

    return resolve_artifact_path(
        parquet_path,
        (
            "data/output/datasets/ml_dataset_latest_full.parquet",
            "data/output/datasets/ml_dataset_latest.parquet",
            "data/output/ml_dataset_latest_full.parquet",
            "data/output/ml_dataset_latest.parquet",
            "data/output/ml_dataset.parquet",
            "output/ml_dataset_latest_full.parquet",
            "ml_dataset_latest_full.parquet",
        ),
        kind="dataset parquet",
        extra_bases=extra_bases,
    )


def resolve_metadata_path(
    metadata_path: str | Path | None = None,
    *,
    extra_bases: Iterable[Path] | None = None,
) -> Path:
    """Resolve the metadata JSON path used for feature validation."""

    return resolve_artifact_path(
        metadata_path,
        (
            "data/output/dataset_features_detail.json",
            "dataset_features_detail.json",
        ),
        kind="dataset metadata JSON",
        extra_bases=extra_bases,
    )


def ensure_directory(path: str | Path) -> Path:
    """Create the directory (and parents) when it does not already exist."""

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target
