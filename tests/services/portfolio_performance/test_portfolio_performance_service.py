"""services/portfolio_performance — validation and shaping, no database."""
import pandas as pd
import pytest

from services.portfolio_performance import app as _module


@pytest.fixture
def client():
    _module.app.config["TESTING"] = True
    with _module.app.test_client() as c:
        yield c


GOOD = {"tickers": ["AAPL", "MSFT"], "positions": ["LONG", "SHORT"],
        "start_date": "2025-01-01", "end_date": "2025-06-30"}


@pytest.mark.parametrize("patch,fragment", [
    ({"tickers": []}, "'tickers'"),
    ({"positions": ["LONG"]}, "one entry per ticker"),
    ({"positions": ["LONG", "FLAT"]}, "LONG or SHORT"),
    ({"start_date": None}, "'start_date'"),
    ({"end_date": "soon"}, "not a date"),
])
def test_invalid_input_is_400(client, patch, fragment):
    r = client.post("/api/factor_portfolio_performance", json={**GOOD, **patch})
    assert r.status_code == 400
    assert fragment in r.get_json()["error"]


def test_no_data_is_404(client, monkeypatch):
    monkeypatch.setattr(_module, "build_performance", lambda *a, **k: None)
    r = client.post("/api/factor_portfolio_performance", json=GOOD)
    assert r.status_code == 404


def test_shape_and_nan_handling(client, monkeypatch):
    dates = pd.date_range("2025-01-02", periods=2, freq="B")
    fake = {
        "cum_returns": pd.DataFrame({"Long Leg": [0.01, 0.02]}, index=dates),
        "stats": [{"Strategy": "Long Leg", "Sharpe": float("nan"), "Annual Vol": 0.0}],
    }
    monkeypatch.setattr(_module, "build_performance", lambda *a, **k: fake)
    r = client.post("/api/factor_portfolio_performance", json=GOOD)
    assert r.status_code == 200
    body = r.get_json()
    assert body["cum_returns"][0]["date"] == "2025-01-02"
    assert body["stats"][0]["Sharpe"] is None          # NaN never reaches the wire
