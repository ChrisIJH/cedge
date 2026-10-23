# model_portfolio

Model weights → whole-share orders across 1,010 synthetic accounts
(1,000 seeded + 10 hand-made edge cases), valued at real closes.

## Pipeline

```
validate  →  targets  →  trades  →  checks  →  run
                                               ├─ run_id  = hash of every input
                                               └─ accounts = processed + exceptions + skipped
then:  implementation effect (account vs model)  →  Brinson-Fachler attribution (model vs benchmark)
```

| Module | What it does |
|---|---|
| `models.py` | Versioned model weights; weights must sum to 1, no shorts |
| `accounts.py` | Account: holdings in shares, cash, last rebalance date |
| `validate.py` | Abort the run if a model fund has no usable price; exclude one account for its own bad input (unknown model, bad holdings, unpriced holding); abort if more than 5% are excluded |
| `trades.py` | One account → orders (rules below) |
| `run.py` | All accounts → one run: `run_id`, per-account status, reconciliation, post-trade holdings |
| `batch.py` | Vectorized version of `trades.py`, checked against it in tests |
| `monitor.py` | Band comparison over a book; one account before and after |
| `synthetic.py` | Seeded synthetic book: each account bought the model at a real close 1–12 months back |
| `implementation.py` | Account vs model since the last rebalance: drift, cash, rounding (exact split) |
| `attribution.py` | Account vs benchmark: allocation, selection, interaction (Brinson-Fachler) + cash, rounding |

## Trade rules (`trades.py`)

1. **Target** = model weight × (1 − cash buffer); a holding not in the model has target 0.
2. **Band:** trade only if some holding is more than `band` away from its target; then trade every holding back to target, not to the band edge.
3. **Whole shares, rounded toward zero:** buys floor, sells ceil — never overbuy, oversell or go short. A zero-target holding is sold as exactly the shares held; share counts are never re-derived from dollar amounts.
4. **Minimum ticket:** orders below `min_trade` dollars are dropped.
5. **Cash:** sales fund purchases and the buffer is kept. If buys exceed what is available, they are scaled down pro rata and floored (`CASH_SHORT`), then leftover cash buys one share at a time of the most underweight holding, never above its unscaled order — so a rerun at the same prices trades nothing.
6. **Checks:** no negative cash, no shorts, value conserved to the cent, integer shares. A violation becomes status `EXC_INVARIANT`, not an exception. Each account gets exactly one status — `TRADED`, `SKIPPED_IN_BAND`, `EXC_TOO_SMALL`, `EXC_INVARIANT` (plus `EXC_INPUT` from validation) — and any number of flags (`CASH_SHORT`, `STILL_OUT_OF_BAND`).

## Run it

```bash
python scripts/run_model_portfolio.py --asof 2026-10-16                # new run → runs/<run_id>/
python scripts/run_model_portfolio.py --asof 2026-10-16 --band 0.03    # same accounts, another band
python scripts/run_model_portfolio.py --replay runs/<run_id>           # recompute the id from saved inputs, rerun, compare orders
```

Options: `--n` (accounts, default 1000), `--seed`, `--band`, `--cash-buffer`, `--out`.
A run saves `inputs.json`, `prices.json`, `holdings.json`, `accounts.parquet`,
`trades.parquet`, `band_table.csv` and `summary.json`. Exit codes: 0 ok,
1 aborted or replay mismatch, 2 no close for the as-of date (stale prices).
Needs the price database (`CEDGE_DB_USER`, `CEDGE_DB_PASSWORD`).

## Results (2026-10-16 closes, 1,010 accounts, cash buffer 2%)

Band 5%: 1,010 = 77 processed + 8 exceptions + 925 skipped; rerun after fills: 0 orders.

| Band | Accounts traded | Too small | Orders | Book turnover |
|---|---|---|---|---|
| 1% | 797 | 10 | 3,246 | 2.46% |
| 2% | 559 | 8 | 2,343 | 2.08% |
| 3% | 253 | 5 | 1,077 | 1.14% |
| 5% | 77 | 4 | 318 | 0.42% |

Implementation effect (account vs model, since each account's last rebalance;
1,000 synthetic accounts without deposits; medians, so columns do not add up):

| Account size | Accounts | Starting cash | Cash | Rounding | Total | Worst |
|---|---|---|---|---|---|---|
| < $25k | 117 | 6% | −19.8 bp | −4.4 bp | −25.3 bp | −275 bp |
| $25k–100k | 259 | 3% | −12.4 bp | −0.8 bp | −11.0 bp | −53 bp |
| $100k–500k | 421 | 2% | −10.0 bp | −0.3 bp | −6.4 bp | −35 bp |
| > $500k | 203 | 2% | −9.8 bp | 0.0 bp | −6.8 bp | −27 bp |

## Limits

- Synthetic accounts, not client data; benchmarks are illustrative (one proxy ETF per sleeve), not the benchmarks these models would be measured against.
- No transaction costs; cash earns no interest; distributions are not credited to cash.
- Orders are priced and assumed filled at the as-of close; a run is a one-day snapshot and does not carry holdings forward.
- Attribution covers one holding period (buy-and-hold since the last rebalance), so no multi-period linking.
- The band is in weight points, not tracking error.

## Tests

```bash
pytest tests/cedge_core/model_portfolio tests/services/model_portfolio   # 57 tests, no database
pytest -m "not db"                                                       # what CI runs: 297 of 312
```
Known-answer cases worked by hand (e.g. the Brinson split, the implementation
split), loop vs vectorized agreement, rerun-is-a-no-op at four bands, and
reconciliation on every run.