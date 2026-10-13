"""
scripts/fake_services.py

Canned stand-ins for the regime, marketdata, and risk services, for exercising
the MCP server without a database. Response shapes follow services/*/app.py.

    python scripts/fake_services.py      # serves on 18004, 18002, 18001
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

CELL = {
    "method": "parametric", "alpha": 0.01, "eval_idx": [0], "r_eval": [0.0], "var": [0.02],
    "es": [0.03], "breaches": [0],
    "kupiec": {"LR": 3.555, "p_value": 0.0594, "reject": False, "N": 6, "T": 250,
               "breach_rate": 0.024, "expected_rate": 0.01},
    "christoffersen": {"LR_CC": 3.855, "p_value": 0.1455, "reject": False,
                       "transitions": {"n00": 238, "n01": 6, "n10": 5, "n11": 0}},
    "acerbi": {"Z2": -0.2, "p_value": 0.3, "reject": False},
    "avg_pred_es": 0.03, "avg_real_loss": 0.036,
}


def make_handler(routes):
    class H(BaseHTTPRequestHandler):
        def _send(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            fn = routes.get(("GET", u.path))
            if fn is None:
                return self._send(404, {"error": f"no route {u.path}"})
            self._send(*fn({k: v[0] for k, v in parse_qs(u.query).items()}))

        def do_POST(self):
            u = urlparse(self.path)
            fn = routes.get(("POST", u.path))
            if fn is None:
                return self._send(404, {"error": f"no route {u.path}"})
            n = int(self.headers.get("Content-Length", 0))
            self._send(*fn(json.loads(self.rfile.read(n) or b"{}")))

        def log_message(self, *a):
            pass
    return H


def regime(q):
    if q.get("as_of", "") < "2000-01-01":
        return 404, {"error": f"no macro price data before {q.get('as_of')}"}
    return 200, {"as_of": q["as_of"], "score": 0.42,
                 "posterior": {"risk_on": 0.41, "soft_patch": 0.26, "risk_off": 0.33},
                 "policy": {"gross": 103.2, "beta_cap": 0.33}}


ROUTES = {
    18004: {("GET", "/api/regime"): regime,
            ("GET", "/api/regime/history"): lambda q: (200, {"history": [
                {"as_of_date": "2026-09-29", "score": 0.40}, {"as_of_date": "2026-09-30", "score": 0.42}]})},
    18002: {("GET", "/api/portfolios"): lambda q: (200, {"portfolios": [
                {"pf_name": "SPX_LS_FACTOR_V1", "experiment_id": None, "branch": None,
                 "source": "backtest", "start_date": "2017-10-11", "end_date": "2026-09-30"}]}),
            ("GET", "/api/portfolio_weights"): lambda q: (200, {
                "pf_name": q.get("pf_name"), "trade_date": "2026-09-30",
                "weights": {"AAA": 0.0667, "BBB": -0.0667}}),
            ("POST", "/api/portfolio_returns"): lambda p: (200, {
                "dates": ["d"] * 500, "returns": [0.001] * 500, "n_observations": 500,
                "gross": 0.1334, "net": 0.0, "beta": None,
                "weighting": "fixed weights applied across the whole window; not a rebalanced backtest"})},
    18001: {("POST", "/api/full_backtest"): lambda p: (200, {"cells": [CELL], "convention": "VaR>0 is a loss"})},
}


def serve_all(block=True):
    servers = []
    for port, routes in ROUTES.items():
        s = ThreadingHTTPServer(("127.0.0.1", port), make_handler(routes))
        threading.Thread(target=s.serve_forever, daemon=True).start()
        servers.append(s)
    if block:
        threading.Event().wait()
    return servers


if __name__ == "__main__":
    serve_all()