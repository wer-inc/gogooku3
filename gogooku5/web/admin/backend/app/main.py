"""FastAPI Admin Backend Application."""

from contextlib import asynccontextmanager

from app.db.database import init_db
from app.routers import health, models, predictions, users
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    await init_db()
    yield
    # Shutdown
    pass


app = FastAPI(
    title="APEX Ranker Admin API",
    description="Admin API for APEX Ranker ML System",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative dev port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(users.router, prefix="/api/users", tags=["Users"])
app.include_router(models.router, prefix="/api/models", tags=["Models"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["Predictions"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "APEX Ranker Admin API", "docs": "/docs"}
