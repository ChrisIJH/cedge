"""
Service-level tests for services/risk/app.py.

These test the HTTP boundary — validation, serialization, status codes —
not the risk math itself, which is already covered by the known-answer
tests in tests/cedge_core/risk/. The one exception is the known-answer
test below, which pins the service to core: if the endpoint ever stops
being a thin pass-through and starts doing its own arithmetic, that test
fails.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cedge_core.risk.param_var import rolling_parametric_var_confidence

# services/ is not an installed package (each service is its own deployable
# unit, not a library), so the module is loaded by path rather than imported.
_APP_PATH = Path(__file__).resolve().parents[3] / "services" / "risk" / "app.py"
_spec = importlib.util.spec_from_file_location("risk_service_app", _APP_PATH)
_module = importlib.util.module_from_spec(_spec)
sys.modules["risk_service_app"] = _module
_spec.loader.exec_module(_module)


@pytest.fixture
def client():
    _module.app.config["TESTING"] = True
    with _module.app.test_client() as c:
        yield c


@pytest.fixture
def returns():
    """A fixed pseudo-random return series — same seed every run."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 0.01, 300).round(6).tolist()

@pytest.fixture
def fat_tail_returns():
    """Student-t(4) — real fat tails"""
    rng = np.random.default_rng(42)
    return rng.standard_t(4, 500).tolist()


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_parametric_matches_core_directly(client, returns):
    """The endpoint must return exactly what core returns — no rounding,
    no rescaling, no reinterpretation of the confidence level."""
    response = client.post("/api/var_es", json={
        "returns": returns, "method": "parametric",
        "window": 60, "confidence": 0.99,
    })
    assert response.status_code == 200
    body = response.get_json()

    expected = rolling_parametric_var_confidence(
        pd.Series(returns), confidence=0.99, window=60, vol_method="sample"
    )
    got_var = [v for v in body["var"] if v is not None]
    want_var = expected["var"].dropna().tolist()
    assert got_var == pytest.approx(want_var)


def test_warmup_days_serialize_as_null(client, returns):
    """NaN is not valid JSON; the warm-up window must come back as null."""
    body = client.post("/api/var_es", json={
        "returns": returns, "method": "parametric", "window": 60,
    }).get_json()
    assert body["var"][:60] == [None] * 60
    assert body["var"][-1] is not None


def test_es_is_at_least_var(client, returns):
    """ES averages losses beyond VaR, so it can never be the smaller number."""
    for method, level in (("parametric", {"confidence": 0.99}), ("fhs", {"alpha": 0.01})):
        body = client.post("/api/var_es", json={
            "returns": returns, "method": method, "window": 60, **level,
        }).get_json()
        pairs = [(v, e) for v, e in zip(body["var"], body["es"])
                 if v is not None and e is not None]
        assert pairs, f"{method} produced no usable rows"
        assert all(e >= v for v, e in pairs), f"{method} returned ES < VaR"


def test_backtest_well_calibrated_var_is_not_rejected(client, returns):
    """A 95% VaR tested at alpha=0.05 on its own data should pass Kupiec.

    Note the wording: failing to reject H0 is not evidence the model is
    correct — it only means this sample gives no reason to doubt it.
    """
    var_body = client.post("/api/var_es", json={
        "returns": returns, "method": "parametric", "window": 60, "confidence": 0.95,
    }).get_json()

    body = client.post("/api/var_backtest", json={
        "returns": returns, "var_est": var_body["var"], "alpha": 0.05,
    }).get_json()

    assert body["kupiec"]["reject"] is False
    assert body["n_observations"] == 240      # 300 returns - 60 warm-up days


@pytest.mark.parametrize("payload,fragment", [
    ({"returns": [], "method": "parametric"}, "non-empty"),
    ({"returns": [0.01] * 100, "window": 60, "method": "bogus"}, "'method'"),
    ({"returns": [0.01] * 100, "window": 60, "method": "parametric", "confidence": 1.5}, "'confidence'"),
    ({"returns": [0.01] * 100, "window": 60, "method": "fhs", "alpha": 0}, "'alpha'"),
    ({"returns": [0.01] * 10, "window": 252}, "'window'"),
    ({"returns": [0.01] * 100, "window": 60, "method": "parametric", "vol_method": "psychic"}, "'vol_method'"),
])


def test_invalid_input_returns_400_with_reason(client, payload, fragment):
    response = client.post("/api/var_es", json=payload)
    assert response.status_code == 400
    assert fragment in response.get_json()["error"]


def test_backtest_rejects_length_mismatch(client):
    response = client.post("/api/var_backtest", json={
        "returns": [0.01] * 100, "var_est": [0.02] * 50,
    })
    assert response.status_code == 400
    assert "length" in response.get_json()["error"]


def test_full_backtest_returns_all_requested_cells(client, fat_tail_returns):
    response = client.post("/api/full_backtest", json={
        "returns": fat_tail_returns, "window": 100,
        "levels": [0.05, 0.01], "methods": ["parametric", "historic"],
    })
    assert response.status_code == 200
    cells = response.get_json()["cells"]
    assert len(cells) == 4
    got = {(c["method"], c["alpha"]) for c in cells}
    assert got == {("parametric", 0.05), ("parametric", 0.01),
                   ("historic", 0.05), ("historic", 0.01)}
    for cell in cells:
        assert len(cell["var"]) == len(cell["es"]) == len(cell["breaches"]) == 400


def test_full_backtest_defaults_cover_both_methods_and_levels(client, fat_tail_returns):
    response = client.post("/api/full_backtest", json={
        "returns": fat_tail_returns, "window": 100,
    })
    assert len(response.get_json()["cells"]) == 4


def test_full_backtest_parametric_underestimates_fat_tail_risk(client, fat_tail_returns):
    """The whole point of offering both methods: at a deep confidence level,
    parametric (normal-distribution) VaR should reject more readily than
    historic (empirical-quantile) VaR on fat-tailed data."""
    response = client.post("/api/full_backtest", json={
        "returns": fat_tail_returns, "window": 100, "levels": [0.01],
    })
    cells = {c["method"]: c for c in response.get_json()["cells"]}
    assert cells["parametric"]["kupiec"]["reject"] is True
    assert cells["historic"]["kupiec"]["reject"] is False


@pytest.mark.parametrize("payload,fragment", [
    ({"returns": [0.01] * 200, "window": 100, "levels": []}, "'levels'"),
    ({"returns": [0.01] * 200, "window": 100, "levels": [1.5]}, "'levels'"),
    ({"returns": [0.01] * 200, "window": 100, "methods": ["bogus"]}, "unknown method"),
    ({"returns": [0.01] * 200, "window": 100, "n_boot": 10}, "'n_boot'"),
])
def test_full_backtest_invalid_input_returns_400(client, payload, fragment):
    response = client.post("/api/full_backtest", json=payload)
    assert response.status_code == 400
    assert fragment in response.get_json()["error"]