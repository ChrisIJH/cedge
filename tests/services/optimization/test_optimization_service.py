"""services/optimization — no database: covariance builders are replaced."""
import numpy as np
import pytest

from services.optimization import app as _module


@pytest.fixture
def client():
    _module.app.config["TESTING"] = True
    with _module.app.test_client() as c:
        yield c


def _diag_cov(n: float) -> np.ndarray:
    return np.diag([0.01] * int(n))


def test_missing_strategy_is_400(client):
    r = client.post("/api/optimize", json={"cov_method": "sample", "weights": {"A": 1}})
    assert r.status_code == 400
    assert "strategy" in r.get_json()["error"]


def test_bad_cov_method_is_400(client):
    r = client.post("/api/optimize", json={"strategy": "mean_variance", "cov_method": "nope",
                                           "weights": {"A": 1}})
    assert r.status_code == 400


def test_sample_cov_weights_keyed_by_request_tickers(client, monkeypatch):
    monkeypatch.setattr(_module, "_build_sample_cov", lambda tickers, payload: _diag_cov(len(tickers)))
    body = {"strategy": "mean_variance", "cov_method": "sample",
            "weights": {"AAA": 0.5, "BBB": -0.5, "CCC": 0.2},
            "gross_target": 1.2, "net_target": 0.2, "w_bounds": 0.6}
    r = client.post("/api/optimize", json=body)
    assert r.status_code == 200, r.get_json()
    out = r.get_json()
    assert set(out["weights"]) == {"AAA", "BBB", "CCC"}
    assert out["dropped_tickers"] == []
    assert sum(out["weights"].values()) == pytest.approx(0.2, abs=1e-6)


def test_factor_model_drop_keeps_weights_on_the_right_tickers(client, monkeypatch):
    """Regression: the response used to label result.weights with the REQUEST
    ticker order, so after a drop every weight shifted onto the wrong name."""
    def fake_factor_cov(tickers, payload):
        aligned = [t for t in tickers if t != "BBB"]        # BBB has no factor data
        return _diag_cov(len(aligned)), aligned

    monkeypatch.setattr(_module, "_build_factor_cov", fake_factor_cov)
    body = {"strategy": "mean_variance", "cov_method": "factor_model", "asof_date": "2026-01-02",
            "weights": {"AAA": 0.9, "BBB": -0.9, "CCC": 0.1},
            "gross_target": 1.0, "net_target": 1.0, "w_bounds": 0.9,
            "lambda_turnover": 0.0}
    r = client.post("/api/optimize", json=body)
    assert r.status_code == 200, r.get_json()
    out = r.get_json()
    assert out["dropped_tickers"] == ["BBB"]
    assert set(out["weights"]) == {"AAA", "CCC"}          # BBB must not reappear
    # min-variance with equal diagonal risk and net=1: both legs ~0.5
    assert out["weights"]["AAA"] == pytest.approx(0.5, abs=1e-3)
    assert out["weights"]["CCC"] == pytest.approx(0.5, abs=1e-3)