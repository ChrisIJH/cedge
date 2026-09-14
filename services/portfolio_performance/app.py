"""
services/portfolio_performance/app.py

Minimal, independently-deployable Flask service exposing
cedge_core.portfolio.performance.build_performance() over HTTP.

No sys.path manipulation: cedge_core is installed as a package
(pip install -e .), so this works identically on a dev machine and inside
a container. Extracted from the CHRiskAnalytics app_prod.py monolith
(3,900 lines, pulling in pymc/arviz/scikit-learn/MongoDB) so this service
carries only the dependencies it actually needs.

Run locally:
    python services/portfolio_performance/app.py
"""
import json

import pandas as pd
from cedge_core.portfolio.performance import build_performance
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/healthz", methods=["GET"])
def healthz():
    """Liveness check for container orchestration (Docker HEALTHCHECK, k8s probes)."""
    return jsonify({"status": "ok"}), 200


@app.route("/api/factor_portfolio_performance", methods=["POST"])
def factor_portfolio_performance():
    ts_json = json.loads(request.data.decode("utf-8"))

    weights_df = pd.DataFrame({
        "ticker": ts_json["tickers"],
        "position": ts_json["positions"],
    })

    result = build_performance(weights_df, ts_json["start_date"], ts_json["end_date"])

    if result is None:
        return jsonify({"error": "no data for given tickers/date range"}), 404

    cum_returns = result["cum_returns"].reset_index().rename(columns={"index": "date"})
    cum_returns["date"] = cum_returns["date"].astype(str)

    return jsonify({
        "cum_returns": cum_returns.to_dict(orient="records"),
        "stats": result["stats"],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
