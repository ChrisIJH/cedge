"""Unit tests for apps/cedge_mcp/tools.py with a fake HTTP session. No network, no MCP SDK."""
from __future__ import annotations

import pytest

pytest.importorskip("requests")

from apps.cedge_mcp import tools
from apps.cedge_mcp.client import CedgeClient, ServiceError

URLS = {"regime": "http://r", "marketdata": "http://m", "risk": "http://k",
        "portfolio_performance": "http://p", "optimization": "http://o"}

CELL = {
    "method": "parametric", "alpha": 0.01, "eval_idx": [0], "r_eval": [0.0], "var": [0.02],
    "es": [0.03], "breaches": [0],
    "kupiec": {"LR": 3.555, "p_value": 0.0594, "reject": False, "N": 6, "T": 250, "breach_rate": 0.024},
    "christoffersen": {"LR_CC": 3.855, "p_value": 0.1455, "reject": False,
                       "transitions": {"n00": 238, "n01": 6, "n10": 5, "n11": 0}},
    "acerbi": {"Z2": -0.2, "p_value": 0.3, "reject": False},
    "avg_pred_es": 0.03, "avg_real_loss": 0.036,
}


class FakeResp:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, str(body)

    def json(self):
        return self._body


class FakeSession:
    """Routes (METHOD, URL) -> (status, body). Records calls."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def _hit(self, method, url, **kw):
        self.calls.append((method, url, kw))
        status, body = self.routes[(method, url)]
        return FakeResp(status, body)

    def get(self, url, params=None, timeout=None):
        return self._hit("GET", url, params=params)

    def post(self, url, json=None, timeout=None):
        return self._hit("POST", url, json=json)


def test_get_regime_passes_as_of_and_records_provenance():
    sess = FakeSession({("GET", "http://r/api/regime"): (200, {"as_of": "2026-09-30", "score": 0.4})})
    out = tools.get_regime(CedgeClient(sess, URLS), "2026-09-30")
    assert sess.calls[0][2]["params"] == {"as_of": "2026-09-30"}
    assert out["data"]["score"] == 0.4
    assert out["provenance"]["calls"][0]["path"] == "/api/regime"


def test_bad_date_never_calls_service():
    sess = FakeSession({})
    with pytest.raises(ValueError):
        tools.get_regime(CedgeClient(sess, URLS), "09/30/2026")
    assert sess.calls == []


def test_history_rejects_inverted_range():
    with pytest.raises(ValueError):
        tools.get_regime_history(CedgeClient(FakeSession({}), URLS), "2026-10-01", "2026-09-01")


def test_history_truncates_long_results():
    rows = [{"as_of_date": f"d{i}"} for i in range(300)]
    sess = FakeSession({("GET", "http://r/api/regime/history"): (200, {"history": rows})})
    out = tools.get_regime_history(CedgeClient(sess, URLS), "2025-01-01", "2026-01-01")
    assert out["data"]["n_rows"] == 300
    assert out["data"]["truncated_to_last"] == tools.MAX_HISTORY_ROWS
    assert out["data"]["history"][0]["as_of_date"] == "d40"


def test_service_404_becomes_service_error_with_message():
    sess = FakeSession({("GET", "http://r/api/regime"): (404, {"error": "no macro price data"})})
    with pytest.raises(ServiceError) as ei:
        tools.get_regime(CedgeClient(sess, URLS), "1990-01-01")
    assert ei.value.status == 404 and "no macro price data" in str(ei.value)


def _var_routes(n_obs=500):
    return {
        ("GET", "http://m/api/portfolio_weights"): (200, {"pf_name": "BOOK", "trade_date": "2026-09-30",
                                                         "weights": {"A": 0.5, "B": -0.5}}),
        ("POST", "http://m/api/portfolio_returns"): (200, {"dates": ["x"] * n_obs, "returns": [0.001] * n_obs,
                                                          "n_observations": n_obs, "gross": 1.0, "net": 0.0,
                                                          "beta": None, "weighting": "fixed weights ..."}),
        ("POST", "http://k/api/full_backtest"): (200, {"cells": [CELL], "convention": "loss>0"}),
    }


def test_var_backtest_chains_three_calls_and_drops_arrays():
    sess = FakeSession(_var_routes())
    out = tools.var_backtest_summary(CedgeClient(sess, URLS), "BOOK", "2026-09-30",
                                     "2024-01-01", "2026-09-30")
    assert [c[1] for c in sess.calls] == ["http://m/api/portfolio_weights",
                                          "http://m/api/portfolio_returns",
                                          "http://k/api/full_backtest"]
    assert sess.calls[0][2]["params"]["as_of_date"] == "2026-09-30"
    cell = out["data"]["cells"][0]
    assert "r_eval" not in cell and "breaches" not in cell
    assert out["provenance"]["calls"][2]["args"]["returns"] == "<list of 500>"
    assert any("in force on 2026-09-30" in c for c in out["data"]["caveats"])


def test_var_backtest_window_too_long_stops_before_risk_call():
    sess = FakeSession(_var_routes(n_obs=100))
    with pytest.raises(ValueError):
        tools.var_backtest_summary(CedgeClient(sess, URLS), "BOOK", "2026-09-30",
                                   "2026-05-01", "2026-09-30", window=252)
    assert all("full_backtest" not in c[1] for c in sess.calls)