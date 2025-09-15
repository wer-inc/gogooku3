import pytest
from fastapi.testclient import TestClient

from gogooku3.api.server import app


@pytest.mark.unit
def test_healthz():
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


@pytest.mark.unit
def test_forecast_predict_tft_minimal():
    client = TestClient(app)
    payload = {
        "model": "tft",
        "horizons": [1, 5],
        "obs": [
            {"id": "A", "ts": "2025-01-01", "y": 0.0},
            {"id": "A", "ts": "2025-01-02", "y": 0.1},
        ],
    }
    r = client.post("/forecast/predict", json=payload)
    assert r.status_code == 200
    out = r.json()
    assert isinstance(out, list) and len(out) >= 1


@pytest.mark.unit
def test_detect_score_minimal():
    client = TestClient(app)
    obs = [
        {"id": "A", "ts": "2025-01-01", "y": 0.0},
        {"id": "A", "ts": "2025-01-02", "y": 0.1},
    ]
    fcst = [
        {"id": "A", "ts": "2025-01-01", "h": 1, "y_hat": 0.0},
        {"id": "A", "ts": "2025-01-02", "h": 1, "y_hat": 0.05},
    ]
    r = client.post("/detect/score", json={"obs": obs, "fcst": fcst, "h": 1})
    assert r.status_code == 200
    res = r.json()
    assert "ranges" in res

