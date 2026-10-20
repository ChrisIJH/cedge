"""services/common.py — the conventions every service inherits."""
import math

import pytest
from flask import Flask, jsonify

from services.common import (
    BadRequest,
    NotFound,
    ServiceError,
    json_safe,
    json_safe_dict,
    register_error_handlers,
    require_date,
    require_json,
    require_tickers,
    require_weights,
)


@pytest.fixture
def client():
    app = Flask("probe")
    register_error_handlers(app)

    @app.route("/boom/<kind>", methods=["POST"])
    def boom(kind):
        if kind == "bad":
            raise BadRequest("bad input")
        if kind == "missing":
            raise NotFound("no data")
        if kind == "broken":
            raise ServiceError("solver died")
        return jsonify(require_json())

    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_healthz_is_registered(client):
    assert client.get("/healthz").get_json() == {"status": "ok"}


@pytest.mark.parametrize("kind,status,fragment", [
    ("bad", 400, "bad input"), ("missing", 404, "no data"), ("broken", 500, "solver died"),
])
def test_exceptions_map_to_status_codes(client, kind, status, fragment):
    r = client.post(f"/boom/{kind}", json={})
    assert r.status_code == status
    assert r.get_json() == {"error": fragment}


def test_require_json_rejects_non_json_body(client):
    r = client.post("/boom/echo", data="a=1")
    assert r.status_code == 400
    assert "JSON" in r.get_json()["error"]


def test_json_safe_turns_nan_into_null():
    assert json_safe([1, float("nan"), 2.5]) == [1.0, None, 2.5]
    assert json_safe_dict({"a": math.nan, "b": "x", "c": 1.0}) == {"a": None, "b": "x", "c": 1.0}


def test_require_date():
    assert require_date({"d": "2026-01-02"}, "d") == "2026-01-02"
    with pytest.raises(BadRequest, match="required"):
        require_date({}, "d")
    with pytest.raises(BadRequest, match="not a date"):
        require_date({"d": "yesterday"}, "d")


def test_tickers_and_weights_are_upper_cased_and_validated():
    assert require_tickers({"tickers": [" spy", "mtum "]}) == ["SPY", "MTUM"]
    assert require_weights({"weights": {" mtum ": 1, "spy": "-1"}}) == {"MTUM": 1.0, "SPY": -1.0}
    for bad in ({"tickers": []}, {"tickers": "SPY"}, {"tickers": [" "]}):
        with pytest.raises(BadRequest):
            require_tickers(bad)
    for bad in ({"weights": {}}, {"weights": {"SPY": "heavy"}}, {"weights": {"SPY": math.inf}}):
        with pytest.raises(BadRequest):
            require_weights(bad)
