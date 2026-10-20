"""services/model_portfolio — no database: the price loader returns a seeded panel."""
import numpy as np
import pandas as pd
import pytest

from services.model_portfolio import app as _module

ASOF = "2026-10-13"
LAST = {"AGG": 94.71, "EFA": 103.47, "IEMG": 80.22, "IJR": 137.60, "IVV": 781.54, "TIP": 104.56}


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range(end=ASOF, periods=300)
    cols = {}
    for t, last in LAST.items():
        level = np.exp(np.cumsum(rng.normal(0.0003, 0.008, len(dates))))
        cols[t] = np.round(last * level / level[-1], 2)
    return pd.DataFrame(cols, index=[d.date() for d in dates])


@pytest.fixture
def client(monkeypatch):
    close = _panel()
    adj = close * 1.0
    monkeypatch.setattr(_module, "_fetch_panels", lambda asof: (close, adj))
    _module._load_panels.cache_clear()
    _module._attribution_payload.cache_clear()
    _module.app.config["TESTING"] = True
    with _module.app.test_client() as c:
        yield c
    _module._load_panels.cache_clear()
    _module._attribution_payload.cache_clear()


def _get(client, path, **params):
    return client.get(path, query_string={"asof": ASOF, "n": 300, **params})


def test_healthz(client):
    assert client.get("/healthz").get_json() == {"status": "ok"}


def test_models_come_with_their_benchmarks(client):
    body = client.get("/api/model_portfolio/models").get_json()
    assert set(body["models"]) == {"MOD_6040", "MOD_8020"}
    for m in body["models"].values():
        assert sum(m["benchmark"]["weights"].values()) == pytest.approx(1.0)


@pytest.mark.parametrize("params,fragment", [
    ({"asof": None}, "'asof'"),
    ({"band": "wide"}, "'band' must be a number"),
    ({"band": 0.9}, "'band' must be between"),
    ({"n": 0}, "'n' must be between"),
])
def test_bad_parameters_are_400(client, params, fragment):
    query = {"asof": ASOF, **params}
    query = {k: v for k, v in query.items() if v is not None}
    r = client.get("/api/model_portfolio/run", query_string=query)
    assert r.status_code == 400
    assert fragment in r.get_json()["error"]


def test_stale_asof_is_404(client):
    r = client.get("/api/model_portfolio/run", query_string={"asof": "2026-10-14"})
    assert r.status_code == 404
    assert "latest 2026-10-13" in r.get_json()["error"]


def test_run_reconciles_and_is_json_clean(client):
    body = _get(client, "/api/model_portfolio/run").get_json()
    c = body["counts"]
    assert body["status"] == "OK"
    assert c["n_target"] == len(body["accounts"]) == 310          # 300 synthetic + 10 edge
    assert c["n_target"] == c["n_processed"] + c["n_exception"] + c["n_skipped"]
    assert body["rerun_n_processed"] == 0
    assert len(body["trades"]) == body["n_orders"]
    bad = next(a for a in body["accounts"] if a["account_id"] == "E03_NEG_SHARES")
    assert bad["status"] == "EXC_INPUT" and bad["total_value"] is None   # NaN -> null
    cash = [a["post_cash"] for a in body["accounts"] if a["post_cash"] is not None]
    assert all(round(x, 2) == x for x in cash)                          # cents on the wire


def test_bands_never_trigger_more_when_wider(client):
    rows = _get(client, "/api/model_portfolio/bands").get_json()["bands"]
    triggered = [r["n_triggered"] for r in rows]
    assert [r["band"] for r in rows] == [0.01, 0.02, 0.03, 0.05]
    assert triggered == sorted(triggered, reverse=True)


def test_account_detail_for_the_legacy_holder(client):
    body = _get(client, "/api/model_portfolio/account", account_id="E02_LEGACY").get_json()
    rows = {r["ticker"]: r for r in body["rows"]}
    assert body["status"] == "TRADED"
    assert rows["IEMG"]["order"] == -352 and rows["IEMG"]["shares_after"] == 0
    assert sum(r["weight_after"] for r in body["rows"]) == pytest.approx(1.0)


def test_account_detail_for_bad_inputs_and_unknown_ids(client):
    bad = _get(client, "/api/model_portfolio/account", account_id="E07_FRACTIONAL").get_json()
    assert bad["status"] == "EXC_INPUT" and bad["reason"] == "BAD_HOLDINGS"
    r = _get(client, "/api/model_portfolio/account", account_id="NOPE")
    assert r.status_code == 404


def test_attribution_parts_add_up(client):
    body = _get(client, "/api/model_portfolio/attribution").get_json()
    assert sum(r["accounts"] for r in body["by_size"]) == 300
    for a in body["accounts"]:
        parts = (a["allocation"] + a["selection"] + a["interaction"] + a["cash"] + a["rounding"])
        assert parts == pytest.approx(a["active"], abs=1e-12)
