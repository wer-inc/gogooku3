"""FastAPI server exposing inference and optimisation endpoints for APEX-Ranker."""
from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date as Date
from pathlib import Path
from threading import Lock
from typing import Any

import torch
import yaml
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from apex_ranker.backtest import (
    CostCalculator,
    OptimizationConfig,
    generate_target_weights,
    select_by_percentile,
)
from apex_ranker.backtest.inference import (
    BacktestInferenceEngine,
    compute_weight_turnover,
    ensure_date,
)
from apex_ranker.data import FeatureSelector
from apex_ranker.data.loader import load_backtest_frame


class RateLimitExceeded(Exception):
    """Raised when a caller exceeds the configured rate limit."""


class RateLimiter:
    """In-memory token bucket rate limiter keyed by caller identity."""

    def __init__(
        self, *, limit: int, window_seconds: int, logger: logging.Logger
    ) -> None:
        self.limit = max(1, limit)
        self.window = max(1, window_seconds)
        self.logger = logger
        self._hits: defaultdict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def consume(self, identity: str) -> None:
        now = time.monotonic()
        with self._lock:
            bucket = self._hits[identity]
            while bucket and now - bucket[0] > self.window:
                bucket.popleft()
            if len(bucket) >= self.limit:
                retry_after = (
                    max(0.0, self.window - (now - bucket[0])) if bucket else self.window
                )
                self.logger.warning(
                    "rate_limit_exceeded identity=%s limit=%s window=%ss",
                    identity,
                    self.limit,
                    self.window,
                )
                raise RateLimitExceeded(
                    f"Rate limit exceeded. Retry after {retry_after:.1f}s"
                )
            bucket.append(now)


PREDICTION_LATENCY = Histogram(
    "apex_ranker_prediction_latency_seconds",
    "Latency for prediction requests",
)
OPTIMIZATION_LATENCY = Histogram(
    "apex_ranker_optimization_latency_seconds",
    "Latency for optimisation requests",
)
REQUEST_COUNTER = Counter(
    "apex_ranker_requests_total",
    "Total API requests",
    labelnames=("endpoint",),
)


class PredictionRequest(BaseModel):
    date: str = Field(..., description="Target trading date (YYYY-MM-DD)")
    horizon: int = Field(20, description="Prediction horizon in trading days")
    top_k: int = Field(50, description="Candidate top-K to return")


class PredictionRecord(BaseModel):
    rank: int
    code: str
    score: float


class PredictionResponse(BaseModel):
    date: str
    horizon: int
    predictions: list[PredictionRecord]


class OptimisationPrediction(BaseModel):
    code: str
    score: float


class OptimisationRequest(BaseModel):
    predictions: list[OptimisationPrediction]
    current_weights: dict[str, float] = Field(default_factory=dict)
    volumes: dict[str, float] | None = None
    portfolio_value: float = 1.0


class OptimisationResponse(BaseModel):
    target_weights: dict[str, float]
    optimisation: dict[str, Any]
    turnover_estimate: float


class RebalanceRequest(BaseModel):
    date: str
    horizon: int = 20
    top_k: int = 50
    current_weights: dict[str, float] = Field(default_factory=dict)
    volumes: dict[str, float] | None = None
    portfolio_value: float = 1.0


class RebalanceResponse(BaseModel):
    predictions: list[PredictionRecord]
    target_weights: dict[str, float]
    optimisation: dict[str, Any]
    turnover_estimate: float


@dataclass(slots=True)
class Settings:
    model_path: Path
    config_path: Path
    data_path: Path
    panel_cache_dir: Path
    optimisation: OptimizationConfig
    top_k_default: int = 50
    horizon_default: int = 20
    api_keys: set[str] = field(default_factory=set)
    rate_limit: int = 120
    rate_window_seconds: int = 60
    selection_k_ratio: float | None = None
    selection_k_min: int | None = None
    selection_sign: int = 1


def _load_settings_from_env() -> Settings:
    model_path = Path(
        os.environ.get("APEX_MODEL_PATH", "models/apex_ranker_v0_enhanced.pt")
    )
    config_path = Path(
        os.environ.get("APEX_CONFIG_PATH", "apex-ranker/configs/v0_base.yaml")
    )
    data_path = Path(
        os.environ.get("APEX_DATA_PATH", "data/ml_dataset_latest_full.parquet")
    )
    panel_cache_dir = Path(os.environ.get("APEX_PANEL_CACHE_DIR", "cache/panel"))

    optimisation = OptimizationConfig(
        target_top_k=int(os.environ.get("APEX_TARGET_TOPK", 35)),
        candidate_multiplier=float(os.environ.get("APEX_CANDIDATE_MULTIPLIER", 2.0)),
        min_weight=float(os.environ.get("APEX_MIN_WEIGHT", 0.02)),
        turnover_limit=float(os.environ.get("APEX_TURNOVER_LIMIT", 0.35)),
        cost_penalty=float(os.environ.get("APEX_COST_PENALTY", 1.0)),
        min_alpha=float(os.environ.get("APEX_MIN_ALPHA", 0.1)),
    )

    api_keys_env = os.environ.get("APEX_API_KEYS", "")
    api_keys = {key.strip() for key in api_keys_env.split(",") if key.strip()}

    selection_k_ratio_env = os.environ.get("APEX_SELECTION_K_RATIO")
    selection_k_ratio = (
        float(selection_k_ratio_env)
        if selection_k_ratio_env is not None
        else None
    )
    selection_k_min_env = os.environ.get("APEX_SELECTION_K_MIN")
    selection_k_min = (
        int(selection_k_min_env) if selection_k_min_env is not None else None
    )
    selection_sign = int(os.environ.get("APEX_SELECTION_SIGN", "1"))
    if selection_sign not in (-1, 1):
        selection_sign = 1

    return Settings(
        model_path=model_path,
        config_path=config_path,
        data_path=data_path,
        panel_cache_dir=panel_cache_dir,
        optimisation=optimisation,
        top_k_default=int(os.environ.get("APEX_TOPK", 50)),
        horizon_default=int(os.environ.get("APEX_HORIZON", 20)),
        api_keys=api_keys,
        rate_limit=int(os.environ.get("APEX_RATE_LIMIT", 120)),
        rate_window_seconds=int(os.environ.get("APEX_RATE_WINDOW", 60)),
        selection_k_ratio=selection_k_ratio,
        selection_k_min=selection_k_min,
        selection_sign=selection_sign,
    )


def _extract_feature_columns(config: dict) -> list[str]:
    data_cfg = config["data"]
    selector = FeatureSelector(data_cfg["feature_groups_config"])
    groups = list(data_cfg.get("feature_groups", []))
    if data_cfg.get("use_plus30"):
        groups.append("plus30")
    selection = selector.select(
        groups=groups,
        optional_groups=data_cfg.get("optional_groups", []),
        exclude_features=data_cfg.get("exclude_features"),
    )
    return list(selection.features)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or _load_settings_from_env()

    with settings.config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    feature_cols = _extract_feature_columns(config)
    lookback = config["data"]["lookback"]

    frame = load_backtest_frame(
        settings.data_path,
        start_date=None,
        end_date=None,
        feature_cols=feature_cols,
        lookback=lookback,
    )

    engine = BacktestInferenceEngine(
        model_path=settings.model_path,
        config=config,
        frame=frame,
        feature_cols=feature_cols,
        dataset_path=settings.data_path,
        panel_cache_dir=settings.panel_cache_dir,
    )

    app = FastAPI(title="APEX-Ranker API", version="0.2.0")

    optimisation_config = settings.optimisation
    cost_calculator = CostCalculator()
    logger = logging.getLogger("apex_ranker.api")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    rate_limiter = RateLimiter(
        limit=settings.rate_limit,
        window_seconds=settings.rate_window_seconds,
        logger=logger,
    )

    def derive_identity(request: Request, api_key: str | None) -> str:
        if api_key:
            return api_key
        return request.headers.get("x-client-id") or (
            request.client.host if request.client else "anonymous"
        )

    def authenticate(request: Request) -> str:
        api_key = request.headers.get("x-api-key")
        if settings.api_keys:
            if not api_key or api_key not in settings.api_keys:
                raise HTTPException(
                    status_code=401, detail="Invalid or missing API key"
                )
        return derive_identity(request, api_key)

    def check_rate_limit(identity: str) -> None:
        try:
            rate_limiter.consume(identity)
        except RateLimitExceeded as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc

    @app.middleware("http")
    async def logging_middleware(request: Request, call_next):  # type: ignore[override]
        start = time.perf_counter()
        identity = derive_identity(request, request.headers.get("x-api-key"))
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "path=%s method=%s status=%s identity=%s duration_ms=%.2f",
                request.url.path,
                request.method,
                locals().get("status_code", "n/a"),
                identity,
                duration_ms,
            )

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics_endpoint() -> Response:
        data = generate_latest()
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)

    @app.get("/available-dates")
    def available_dates() -> dict[str, list[str]]:
        dates = sorted(int_to_str(d) for d in engine.available_dates())
        return {"dates": dates}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(
        req: PredictionRequest, identity: str = Depends(authenticate)
    ) -> PredictionResponse:
        check_rate_limit(identity)
        REQUEST_COUNTER.labels("predict").inc()
        with PREDICTION_LATENCY.time():
            target_date = ensure_date(req.date)
            prediction_df = engine.predict(
                target_date=target_date,
                horizon=req.horizon,
                top_k=req.top_k,
            )

        records = [
            PredictionRecord(
                rank=int(row["Rank"]), code=row["Code"], score=float(row["Score"])
            )
            for row in prediction_df.iter_rows(named=True)
        ]
        return PredictionResponse(
            date=req.date,
            horizon=req.horizon,
            predictions=records,
        )

    def build_config_override(top_k: int | None = None) -> OptimizationConfig:
        target_top_k = optimisation_config.target_top_k
        if top_k is not None:
            target_top_k = min(target_top_k, top_k)
        return OptimizationConfig(
            target_top_k=target_top_k,
            candidate_multiplier=optimisation_config.candidate_multiplier,
            min_weight=optimisation_config.min_weight,
            turnover_limit=optimisation_config.turnover_limit,
            cost_penalty=optimisation_config.cost_penalty,
            min_alpha=optimisation_config.min_alpha,
        )

    def optimise_predictions(
        predictions: dict[str, float],
        current_weights: dict[str, float],
        volumes: dict[str, float] | None,
        portfolio_value: float,
        config_override: OptimizationConfig,
    ) -> tuple[dict[str, float], dict[str, Any], float]:
        codes = list(predictions.keys())
        scores_tensor = torch.tensor(
            [predictions[code] for code in codes], dtype=torch.float32
        )
        if scores_tensor.numel() > 0:
            ratio = (
                settings.selection_k_ratio
                if settings.selection_k_ratio is not None
                else config_override.target_top_k / max(1, scores_tensor.numel())
            )
            ratio = float(max(0.0, min(1.0, ratio)))
            min_required = settings.selection_k_min or config_override.target_top_k
            min_required = max(1, int(min_required))
            idx_tensor, threshold_signed, gate_fallback = select_by_percentile(
                scores_tensor,
                k_ratio=ratio,
                k_min=min_required,
                sign=settings.selection_sign,
            )
            if idx_tensor.numel() == 0:
                gate_fallback = True
                fallback_count = min(scores_tensor.numel(), min_required)
                if fallback_count > 0:
                    idx_tensor = torch.arange(fallback_count, dtype=torch.long)
                    signed_scores = scores_tensor * float(settings.selection_sign)
                    threshold_signed = float(signed_scores[idx_tensor[-1]].item())
            if idx_tensor.numel() > 0:
                idx_tensor = torch.unique(idx_tensor, sorted=True)
                selected_codes = [codes[i] for i in idx_tensor.tolist()]
                selection_threshold = float(
                    threshold_signed * float(settings.selection_sign)
                )
            else:
                selected_codes = []
                selection_threshold = None
            gated_predictions = {
                code: predictions[code] for code in selected_codes
            }
        else:
            ratio = settings.selection_k_ratio
            min_required = settings.selection_k_min or config_override.target_top_k
            gate_fallback = False
            selection_threshold = None
            gated_predictions = {}

        with OPTIMIZATION_LATENCY.time():
            target_weights, optimisation = generate_target_weights(
                gated_predictions,
                current_weights,
                portfolio_value=portfolio_value,
                config=config_override,
                cost_calculator=cost_calculator,
                volumes=volumes,
            )

        optimisation_dict = optimisation.to_dict()
        optimisation_dict["selection"] = {
            "k_ratio": ratio,
            "threshold": selection_threshold,
            "fallback_used": bool(gate_fallback),
            "candidate_total": len(codes),
            "candidate_kept": len(gated_predictions),
            "selection_sign": settings.selection_sign,
        }
        turnover_estimate = compute_weight_turnover(current_weights, target_weights)
        return target_weights, optimisation_dict, turnover_estimate

    @app.post("/optimize", response_model=OptimisationResponse)
    def optimise(
        req: OptimisationRequest, identity: str = Depends(authenticate)
    ) -> OptimisationResponse:
        check_rate_limit(identity)
        REQUEST_COUNTER.labels("optimize").inc()
        predictions = {item.code: item.score for item in req.predictions}
        if not predictions:
            raise HTTPException(status_code=400, detail="No predictions supplied")

        config_override = build_config_override(top_k=len(predictions))
        target_weights, optimisation_dict, turnover_estimate = optimise_predictions(
            predictions,
            req.current_weights,
            req.volumes,
            req.portfolio_value,
            config_override,
        )

        return OptimisationResponse(
            target_weights=target_weights,
            optimisation=optimisation_dict,
            turnover_estimate=turnover_estimate,
        )

    @app.post("/rebalance", response_model=RebalanceResponse)
    def rebalance(
        req: RebalanceRequest, identity: str = Depends(authenticate)
    ) -> RebalanceResponse:
        check_rate_limit(identity)
        REQUEST_COUNTER.labels("rebalance").inc()
        with PREDICTION_LATENCY.time():
            target_date = ensure_date(req.date)
            prediction_df = engine.predict(
                target_date=target_date,
                horizon=req.horizon,
                top_k=req.top_k,
            )

        predictions = {
            row["Code"]: float(row["Score"])
            for row in prediction_df.iter_rows(named=True)
        }
        if not predictions:
            raise HTTPException(
                status_code=404, detail="No predictions available for date"
            )

        config_override = build_config_override(top_k=req.top_k)
        target_weights, optimisation_dict, turnover_estimate = optimise_predictions(
            predictions,
            req.current_weights,
            req.volumes,
            req.portfolio_value,
            config_override,
        )

        records = [
            PredictionRecord(
                rank=int(row["Rank"]), code=row["Code"], score=float(row["Score"])
            )
            for row in prediction_df.iter_rows(named=True)
        ]

        return RebalanceResponse(
            predictions=records,
            target_weights=target_weights,
            optimisation=optimisation_dict,
            turnover_estimate=turnover_estimate,
        )

    return app


def int_to_str(value: Date) -> str:
    return value.strftime("%Y-%m-%d")


app = create_app()
