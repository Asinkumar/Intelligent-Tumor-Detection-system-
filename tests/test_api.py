from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_home_endpoint() -> None:

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "Running Successfully"
    assert data["version"] == "1.0.0"


def test_health_endpoint() -> None:

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "Healthy"
    assert (
        data["message"]
        == "API is running successfully"
    )


def test_predict_rejects_wrong_feature_count() -> None:

    response = client.post(
        "/predict",
        json={
            "features": [
                1.0,
                2.0,
                3.0,
            ]
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "Error"

    assert (
        data["message"]
        == "Exactly 30 input features are required."
    )