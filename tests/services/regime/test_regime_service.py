"""
Service-level tests for services/regime/app.py.

Input validation runs before any query, so those stay in the CI subset.
Anything hitting the DB carries the `db` marker.
"""
import pytest

from services.regime import app as _module


@pytest.fixture
def client():
    _module.app.config["TESTING"] = True
    with _module.app.test_client() as c:
        yield c


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_regime_missing_as_of_returns_400(client):
    response = client.get("/api/regime")
    assert response.status_code == 400
    assert "'as_of'" in response.get_json()["error"]


def test_regime_invalid_as_of_returns_400(client):
    response = client.get("/api/regime?as_of=not-a-date")
    assert response.status_code == 400
    assert "is not a date" in response.get_json()["error"]


def test_history_missing_dates_returns_400(client):
    response = client.get("/api/regime/history?start=2026-01-01")
    assert response.status_code == 400
    assert "'end'" in response.get_json()["error"]


def test_backtest_invalid_horizons_returns_400(client):
    response = client.get(
        "/api/regime/backtest?start=2026-01-01&end=2026-01-31&horizons=1,abc"
    )
    assert response.status_code == 400
    assert "horizons" in response.get_json()["error"]


@pytest.mark.db
def test_regime_returns_score_and_posterior(client):
    response = client.get("/api/regime?as_of=2026-08-01")
    assert response.status_code == 200
    body = response.get_json()
    assert 0.0 <= body["score"] <= 100.0
    assert set(body["posterior"]) == {"risk_off", "soft_patch", "risk_on"}


@pytest.mark.db
def test_backtest_returns_bucket_stats(client):
    response = client.get(
        "/api/regime/backtest?start=2020-01-01&end=2025-12-31&horizons=1,5,20"
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["bucket_stats"]
    assert body["n_days"] > 0