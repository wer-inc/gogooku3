from __future__ import annotations

"""FastAPI server exposing Forecast/Detect/Index endpoints (flat I/O).

Endpoints (JSON in/out):
- POST /forecast/predict  { model: "timesfm"|"tft", horizons: [..], obs: [...] }
- POST /detect/score      { obs: [...], fcst: [...], h: 1, min_len: 2, perc: 0.95, labels?: [...] }
- POST /index/esvi        { fcst: [...], name?: "ESVI_JP", weights?: [...] }
"""

from typing import List, Literal, Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.gogooku3.forecast import timesfm_predict, TFTAdapter
from src.gogooku3.detect import (
    residual_q_score,
    change_point_score,
    spectral_residual_score,
    stack_and_score,
    score_to_ranges,
    evaluate_vus_pr,
)
from src.gogooku3.detect.ranges import evaluate_vus_pr_iou, RangeLabel
from src.gogooku3.decide import build_pseudo_vix


app = FastAPI(title="Gogooku3 API", version="0.1")


class Record(BaseModel):
    id: str
    ts: str
    y: Optional[float] = None
    # arbitrary additional fields allowed
    class Config:
        extra = "allow"


class PredictRequest(BaseModel):
    model: Literal["timesfm", "tft"] = "timesfm"
    horizons: List[int] = Field(default_factory=lambda: [1, 5, 10, 20, 30])
    obs: List[Record]
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model": "tft",
                    "horizons": [1, 5, 20],
                    "obs": [
                        {"id": "7203.T", "ts": "2025-09-10", "y": 0.012, "feat1": 1.0},
                        {"id": "7203.T", "ts": "2025-09-11", "y": -0.004, "feat1": 1.1}
                    ]
                }
            ]
        }
    }


@app.post("/forecast/predict")
def forecast_predict(req: PredictRequest):
    df_obs = pd.DataFrame([r.model_dump() for r in req.obs])
    if req.model == "timesfm":
        out = timesfm_predict(df_obs, horizons=req.horizons)
    else:
        model = TFTAdapter(horizons=req.horizons)
        try:
            model.fit(df_obs)
            out = model.predict(df_obs)
        except Exception:
            out = timesfm_predict(df_obs, horizons=req.horizons)
    return out.to_dict(orient="records")


class DetectRequest(BaseModel):
    obs: List[Record]
    fcst: List[dict]
    h: int = 1
    min_len: int = 2
    perc: float = 0.95
    labels: Optional[List[dict]] = None
    eval_iou: Optional[float] = None
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "obs": [
                        {"id": "7203.T", "ts": "2025-09-10", "y": 0.01},
                        {"id": "7203.T", "ts": "2025-09-11", "y": -0.02}
                    ],
                    "fcst": [
                        {"id": "7203.T", "ts": "2025-09-10", "h": 1, "y_hat": 0.005},
                        {"id": "7203.T", "ts": "2025-09-11", "h": 1, "y_hat": -0.01}
                    ],
                    "h": 1,
                    "min_len": 2,
                    "perc": 0.95
                }
            ]
        }
    }


@app.post("/detect/score")
def detect_score(req: DetectRequest):
    df_obs = pd.DataFrame(req.obs)
    df_fcst = pd.DataFrame(req.fcst)
    r = residual_q_score(df_obs, df_fcst, horizon=req.h)
    cp = change_point_score(df_obs)
    sr = spectral_residual_score(df_obs)
    ens = stack_and_score([r, cp, sr])
    ranges = score_to_ranges(ens, min_len=req.min_len, perc=req.perc)
    payload = [
        {"id": x.id, "start": x.start.isoformat(), "end": x.end.isoformat(), "score": x.score, "type": x.type}
        for x in ranges
    ]
    res = {"ranges": payload}
    if req.labels:
        gold = [RangeLabel(id=str(x.get("id")), start=pd.to_datetime(x["start"]), end=pd.to_datetime(x["end"])) for x in req.labels]
        if req.eval_iou is not None:
            vs = evaluate_vus_pr_iou(ranges, gold, min_iou=req.eval_iou)
        else:
            vs = evaluate_vus_pr(ranges, gold)
        res["vus_pr"] = vs
    return res


class IndexRequest(BaseModel):
    fcst: List[dict]
    name: str = "ESVI_JP"
    weights: Optional[List[dict]] = None
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "fcst": [
                        {"id": "7203.T", "ts": "2025-09-12", "h": 30, "y_hat": 0.012},
                        {"id": "6758.T", "ts": "2025-09-12", "h": 30, "y_hat": 0.015}
                    ],
                    "name": "ESVI_JP"
                }
            ]
        }
    }


@app.post("/index/esvi")
def index_esvi(req: IndexRequest):
    df_fcst = pd.DataFrame(req.fcst)
    w = pd.DataFrame(req.weights) if req.weights else None
    out = build_pseudo_vix(df_fcst, index_name=req.name, weights=w)
    return out.to_dict(orient="records")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
