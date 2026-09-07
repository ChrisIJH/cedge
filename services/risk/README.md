# risk service

VaR / ES computation and backtesting over HTTP. All calculation lives in
`cedge_core.risk`; this service only validates input, calls core, and
serializes the result.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz` | Liveness probe |
| POST | `/api/var_es` | Rolling VaR/ES — parametric or filtered historical simulation |
| POST | `/api/var_backtest` | Kupiec UC, Christoffersen CC, Acerbi-Székely Z2 |

### Risk-level convention

The two methods take different risk-level parameters, matching the
convention each one uses in `cedge_core`:

- `method: "parametric"` takes **`confidence`** (e.g. `0.99` for 99% VaR)
- `method: "fhs"` takes **`alpha`** (tail probability, e.g. `0.01` for 99% VaR)

These are not unified on purpose — see the module docstring in
`core/cedge_core/risk/param_var.py`.

VaR and ES are returned as **positive loss** numbers: larger means riskier.
Warm-up days, where a rolling estimator has no value yet, come back as `null`.

## Examples

```bash
curl -X POST localhost:8000/api/var_es \
  -H 'Content-Type: application/json' \
  -d '{"returns": [...], "method": "parametric", "window": 60, "confidence": 0.99}'

curl -X POST localhost:8000/api/var_backtest \
  -H 'Content-Type: application/json' \
  -d '{"returns": [...], "var_est": [...], "alpha": 0.05}'