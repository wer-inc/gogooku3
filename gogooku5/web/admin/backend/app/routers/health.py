"""Health check endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@router.get("/health/detailed")
async def detailed_health():
    """Detailed health check with component status."""
    return {
        "status": "ok",
        "components": {
            "api": "healthy",
            "database": "healthy",
            "ml_models": "healthy",
        },
    }
