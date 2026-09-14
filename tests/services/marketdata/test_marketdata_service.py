"""
Service-level tests for services/marketdata/app.py.

Split by what they need. Input validation runs before any query, so those
tests exercise the real endpoint with no database and stay in the CI subset.
Anything that must actually read prices carries the `db` marker and is
excluded from CI — CI has no MySQL to talk to.
"""

import pytest

from services.marketdata import app as _module

@pytest.fixture
def client():
    _module.app.config["TESTING"] = True
    with _module.app.test_client() as c:
        yield c


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


@pytest.mark.parametrize("payload,fragment", [
    ({}, "'weights'"),
    ({"weights": {}}, "'weights'"),
    ({"weights": []}, "'weights'"),
    ({"weights": {"SPY": "heavy"}}, "not a number"),
    ({"weights": {"SPY": float("inf")}}, "finite"),
    ({"weights": {"  ": 1.0}}, "empty ticker"),
])
def test_invalid_weights_return_400(client, payload, fragment):
    payload.setdefault("start_date", "2020-01-01")
    payload.setdefault("end_date", "2020-12-31")
    response = client.post("/api/portfolio_returns", json=payload)
    assert response.status_code == 400
    assert fragment in response.get_json()["error"]


@pytest.mark.parametrize("payload,fragment", [
    ({"weights": {"SPY": 1.0}, "end_date": "2020-12-31"}, "'start_date'"),
    ({"weights": {"SPY": 1.0}, "start_date": "2020-01-01"}, "'end_date'"),
    ({"weights": {"SPY": 1.0}, "start_date": "not-a-date",
      "end_date": "2020-12-31"}, "is not a date"),
])
def test_invalid_dates_return_400(client, payload, fragment):
    response = client.post("/api/portfolio_returns", json=payload)
    assert response.status_code == 400
    assert fragment in response.get_json()["error"]


def test_non_json_body_returns_400(client):
    response = client.post("/api/portfolio_returns", data="weights=SPY")
    assert response.status_code == 400
    assert "JSON" in response.get_json()["error"]


def test_missing_pf_name_returns_400(client):
    response = client.get("/api/portfolio_weights")
    assert response.status_code == 400
    assert "'pf_name'" in response.get_json()["error"]


def test_weights_are_normalized_to_uppercase():
    """Tickers are stored uppercase; a lowercase request must still match.

    Calling the validator directly — the endpoint would need a database to
    get any further than this.
    """
    cleaned = _module._require_weights({"weights": {" mtum ": 1, "spy": -1}})
    assert cleaned == {"MTUM": 1.0, "SPY": -1.0}


@pytest.mark.db
def test_dollar_neutral_book_has_near_zero_market_beta(client):
    """MTUM long against SPY short should carry almost no market exposure.

    This is the check that catches a sign or alignment error: if the short
    leg were added rather than subtracted, beta would land near 2, not 0.
    """
    response = client.post("/api/portfolio_returns", json={
        "weights": {"MTUM": 1.0, "SPY": -1.0},
        "start_date": "2015-01-01", "end_date": "2025-12-31",
    })
    assert response.status_code == 200
    body = response.get_json()

    assert body["gross"] == pytest.approx(2.0)
    assert body["net"] == pytest.approx(0.0)
    assert abs(body["beta"]["beta"]) < 0.2
    assert body["n_observations"] == len(body["returns"]) == len(body["dates"])


@pytest.mark.db
def test_unknown_ticker_returns_404(client):
    response = client.post("/api/portfolio_returns", json={
        "weights": {"NOSUCHTICKER": 1.0},
        "start_date": "2015-01-01", "end_date": "2025-12-31",
    })
    assert response.status_code == 404
    assert "NOSUCHTICKER" in response.get_json()["error"]


@pytest.mark.db
def test_portfolios_listing_has_the_four_key_columns(client):
    """A book is keyed by four columns, not by pf_name alone — the UI needs
    all four to fetch weights back."""
    body = client.get("/api/portfolios").get_json()
    assert body["portfolios"]
    for key in ("pf_name", "experiment_id", "branch", "source"):
        assert key in body["portfolios"][0]
