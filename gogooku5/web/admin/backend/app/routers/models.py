"""ML Model management endpoints."""

import os
from pathlib import Path

from app.models.ml_model import MLModel, MLModelMetrics
from fastapi import APIRouter, HTTPException

router = APIRouter()

# Model directory path
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/workspace/gogooku3/gogooku5/output"))


@router.get("", response_model=list[MLModel])
async def list_models():
    """List all available ML models."""
    models = []

    # Scan for .pt files
    if MODEL_DIR.exists():
        for pt_file in MODEL_DIR.glob("*.pt"):
            stat = pt_file.stat()
            models.append(
                MLModel(
                    name=pt_file.stem,
                    path=str(pt_file),
                    size_mb=stat.st_size / 1024 / 1024,
                    created_at=stat.st_mtime,
                    status="available",
                )
            )

    # Sort by creation time (newest first)
    models.sort(key=lambda x: x.created_at, reverse=True)
    return models


@router.get("/{model_name}", response_model=MLModel)
async def get_model(model_name: str):
    """Get details of a specific model."""
    model_path = MODEL_DIR / f"{model_name}.pt"

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    stat = model_path.stat()
    return MLModel(
        name=model_name,
        path=str(model_path),
        size_mb=stat.st_size / 1024 / 1024,
        created_at=stat.st_mtime,
        status="available",
    )


@router.get("/{model_name}/metrics", response_model=MLModelMetrics)
async def get_model_metrics(model_name: str):
    """Get validation metrics for a model."""
    import numpy as np

    npz_path = MODEL_DIR / f"{model_name}_val_perday.npz"

    if not npz_path.exists():
        raise HTTPException(status_code=404, detail="Metrics not found")

    data = np.load(npz_path, allow_pickle=True)

    # Extract h20 metrics (20-day horizon)
    metrics = {}
    for key in data.files:
        if key.startswith("h20_"):
            metric_name = key[4:]  # Remove 'h20_' prefix
            arr = data[key]
            if isinstance(arr, np.ndarray) and arr.size > 0:
                metrics[metric_name] = float(np.nanmean(arr))

    # Calculate delta metrics
    delta_p = metrics.get("p_at_k_pos", 0) - metrics.get("p_at_k_pos_rand", 0)
    delta_ndcg = metrics.get("ndcg", 0) - metrics.get("ndcg_rand", 0)

    return MLModelMetrics(
        model_name=model_name,
        rank_ic=metrics.get("rank_ic", 0),
        p_at_k=metrics.get("p_at_k_pos", 0),
        p_at_k_rand=metrics.get("p_at_k_pos_rand", 0),
        delta_p_at_k=delta_p,
        ndcg=metrics.get("ndcg", 0),
        ndcg_rand=metrics.get("ndcg_rand", 0),
        delta_ndcg=delta_ndcg,
        topk_overlap=metrics.get("topk_overlap", 0),
        spread=metrics.get("spread", 0),
        wil=metrics.get("wil", 0),
        validation_days=len(data.get("h20_rank_ic", [])),
    )


@router.post("/{model_name}/activate")
async def activate_model(model_name: str):
    """Set a model as the active model for predictions."""
    model_path = MODEL_DIR / f"{model_name}.pt"

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    # TODO: Implement model activation logic
    return {"message": f"Model {model_name} activated", "status": "active"}
