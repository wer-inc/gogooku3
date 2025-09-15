"""
API Smoke Tests - E2E基本動作確認
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.gogooku3.api.server import app, forecast_models, detection_engine, esvi_calculator

# Mock the global models for testing
@pytest.fixture(scope="module")
def test_app():
    """Create a test FastAPI app with mocked models"""
    from src.gogooku3.forecast.timesfm_adapter import TimesFMAdapter
    from src.gogooku3.forecast.tft_adapter import TFTAdapter
    from src.gogooku3.detect.ensemble import DetectionEnsemble
    from src.gogooku3.decide.pseudo_vix import ESVICalculator

    # Mock initialization
    forecast_models["champion"] = TimesFMAdapter(horizons=[1, 5, 10, 20], context=512)
    forecast_models["challenger"] = TFTAdapter(horizons=[1, 5, 10, 20])

    global detection_engine, esvi_calculator
    detection_engine = DetectionEnsemble()
    esvi_calculator = ESVICalculator()

    return app

@pytest.fixture(scope="module")
def client(test_app):
    """Create test client"""
    with TestClient(test_app) as client:
        yield client

class TestAPISmoke:
    """API基本動作のスモークテスト"""

    def test_health_check(self, client):
        """Health check endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "endpoints" in data

    def test_forecast_predict(self, client):
        """Forecast prediction endpoint"""
        request_data = {
            "symbol": "7203",
            "horizons": [1, 5],
            "model": "champion",
            "quantiles": [0.1, 0.5, 0.9]
        }
        response = client.post("/forecast/predict", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "7203"
        assert data["model"] == "champion"
        assert "predictions" in data
        assert "metadata" in data

    def test_detect_score(self, client):
        """Detection scoring endpoint"""
        request_data = {
            "symbol": "7203",
            "data": [
                {"timestamp": 1640995200, "value": 100.0},
                {"timestamp": 1641081600, "value": 105.0},
                {"timestamp": 1641168000, "value": 95.0}
            ],
            "method": "ensemble",
            "threshold": 0.25
        }
        response = client.post("/detect/score", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "7203"
        assert "anomaly_score" in data
        assert "vus_pr_score" in data
        assert "is_anomaly" in data

    def test_index_esvi(self, client):
        """ESVI index calculation endpoint"""
        request_data = {
            "symbols": ["7203", "6758", "8306"],
            "window": 30,
            "method": "esvi"
        }
        response = client.post("/index/esvi", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "esvi_value" in data
        assert data["component_count"] == 3
        assert "metadata" in data

    def test_model_status(self, client):
        """Model status endpoint"""
        response = client.get("/models/status")
        assert response.status_code == 200
        data = response.json()
        assert "champion" in data
        assert "challenger" in data
        assert data["champion"]["model"] == "TimesFM"

    def test_invalid_model(self, client):
        """Invalid model handling"""
        request_data = {
            "symbol": "7203",
            "model": "nonexistent"
        }
        response = client.post("/forecast/predict", json=request_data)
        assert response.status_code == 404

if __name__ == "__main__":
    pytest.main([__file__, "-v"])