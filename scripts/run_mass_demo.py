"""
scripts/run_mass_demo.py

Model portfolio lab: one rebalance run over synthetic accounts at real closes.
Imperative shell — the only place that reads the DB and writes files.

    python scripts/run_mass_demo.py --asof 2026-10-16
    python scripts/run_mass_demo.py --asof 2026-10-16 --band 0.03
    python scripts/run_mass_demo.py --replay runs/<run_id>

runs/<run_id>/
    inputs.json      models, rules, as-of, code version, generator settings   } inputs: enough to
    prices.json      the close used for every account                         } rerun without the DB
    holdings.json    every account before the run (shares, cash, model)       }
    accounts.parquet one row per account: status, reason, flags, drift         } results
    trades.parquet   one row per order                                         }
    band_table.csv   band 1/2/3/5% comparison                                  }
    summary.json     counts, reconciliation, idempotence check                 }

Same inputs -> same run_id -> the existing run is reused. --replay reloads the
three input files, recomputes the run_id, reruns, and checks the orders match.
"""
import argparse
import datetime as dt
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from cedge_core.db import ch_engine
from cedge_core.marketdata.prices import load_prices_close
from cedge_core.model_portfolio.accounts import Account
from cedge_core.model_portfolio.models import MODELS, ModelPortfolio
from cedge_core.model_portfolio.monitor import band_table
from cedge_core.model_portfolio.run import RunInputs, RunResult, make_run_id, run
from cedge_core.model_portfolio.synthetic import edge_accounts, make_accounts
from cedge_core.model_portfolio.trades import Rules
from cedge_core.model_portfolio.validate import validate_inputs

LEGACY = "IEMG"     # held by one edge account, not in any model
MONEY_COLUMNS = ("total_value", "post_cash", "notional")


def code_version() -> str:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True, check=False).stdout.strip() or "unknown"
    dirty = subprocess.run(["git", "status", "--porcelain"],
                           capture_output=True, text=True, check=False).stdout.strip()
    return sha + ("-dirty" if dirty else "")


# --- input snapshot: write and read back ------------------------------------

def save_inputs(out: Path, inputs: RunInputs, prices: Dict[str, float],
                accounts: List[Account], generator: Dict[str, object]) -> None:
    (out / "inputs.json").write_text(json.dumps({
        "asof": inputs.asof,
        "code_version": inputs.code_version,
        "rules": asdict(inputs.rules),
        "models": {k: {"version": m.version, "weights": dict(m.weights)}
                   for k, m in inputs.models.items()},
        "generator": generator,
    }, indent=2))
    (out / "prices.json").write_text(json.dumps(prices, indent=2, sort_keys=True))
    (out / "holdings.json").write_text(json.dumps([asdict(a) for a in accounts]))


def load_inputs(folder: Path) -> Tuple[RunInputs, Dict[str, float], List[Account]]:
    cfg = json.loads((folder / "inputs.json").read_text())
    models = {k: ModelPortfolio(k, m["version"], m["weights"]) for k, m in cfg["models"].items()}
    inputs = RunInputs(models, cfg["asof"], Rules(**cfg["rules"]), cfg["code_version"])
    prices = json.loads((folder / "prices.json").read_text())
    accounts = [Account(**a) for a in json.loads((folder / "holdings.json").read_text())]
    return inputs, prices, accounts


# --- results ----------------------------------------------------------------

def _cents(df: pd.DataFrame) -> pd.DataFrame:
    """Money to the cent on the way out; the engine itself computes in float64."""
    out = df.copy()
    for col in MONEY_COLUMNS:
        if col in out.columns:
            out[col] = out[col].round(2)
    return out


def summarize(res: RunResult, inputs: RunInputs, rerun: RunResult) -> Dict[str, object]:
    c = res.counts
    flags = res.accounts.get("flag_codes", pd.Series(dtype=str)).fillna("")
    book_value = float(res.accounts["total_value"].sum())
    gross = float(res.trades["notional"].abs().sum())
    return {
        "run_id": res.run_id,
        "status": res.status,
        "asof": inputs.asof,
        "code_version": inputs.code_version,
        "rules": asdict(inputs.rules),
        "counts": c,
        "reconciliation": (f"{c['n_target']} = {c['n_processed']} processed + "
                           f"{c['n_exception']} exceptions + {c['n_skipped']} skipped"),
        "flags": {f: int(flags.str.contains(f).sum()) for f in ("CASH_SHORT", "STILL_OUT_OF_BAND")},
        "n_orders": len(res.trades),
        "gross_notional": round(gross, 2),
        "book_value": round(book_value, 2),
        "book_turnover": round(gross / 2 / book_value, 6) if book_value else 0.0,
        "rerun_n_processed": rerun.counts.get("n_processed"),
    }


def write_results(out: Path, res: RunResult, bands: pd.DataFrame, summary: Dict[str, object]) -> None:
    _cents(res.accounts).to_parquet(out / "accounts.parquet")
    _cents(res.trades).to_parquet(out / "trades.parquet")
    bands.to_csv(out / "band_table.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


# --- commands ---------------------------------------------------------------

def new_run(args: argparse.Namespace) -> int:
    tickers = sorted({t for m in MODELS.values() for t in m.weights} | {LEGACY})
    start = (dt.date.fromisoformat(args.asof) - dt.timedelta(days=550)).isoformat()
    close = load_prices_close(ch_engine(), tickers, start, args.asof)
    last = close.index.max()
    if str(last) != args.asof:
        print(f"no close for {args.asof} (latest {last}); not trading on stale prices")
        return 2
    prices = {t: float(p) for t, p in close.loc[last].dropna().items()}

    accounts = make_accounts(close, list(MODELS.values()), args.asof, args.n, args.seed)
    model_tickers = {t for m in MODELS.values() for t in m.weights}
    if model_tickers <= set(prices):          # edge accounts are priced at the run-day close
        accounts += edge_accounts(MODELS, prices, LEGACY)

    inputs = RunInputs(MODELS, args.asof, Rules(band=args.band, cash_buffer=args.cash_buffer),
                       code_version())
    res = run(inputs, accounts, prices)

    out = Path(args.out) / res.run_id
    if (out / "summary.json").exists():
        print(f"run {res.run_id} already exists, reusing {out}")
        print((out / "summary.json").read_text())
        return 0
    out.mkdir(parents=True, exist_ok=True)
    save_inputs(out, inputs, prices, accounts,
                {"n": args.n, "seed": args.seed, "legacy": LEGACY, "price_start": start})

    if res.abort_reason:
        summary = {"run_id": res.run_id, "status": res.status, "abort_reason": res.abort_reason,
                   "asof": args.asof, "code_version": inputs.code_version, "counts": res.counts}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return 1

    rerun = run(inputs, res.post_accounts, prices)
    valid = validate_inputs(inputs.models, accounts, prices).valid
    bands = band_table(valid, inputs.models, prices, inputs.rules)
    summary = summarize(res, inputs, rerun)
    write_results(out, res, bands, summary)

    print(json.dumps(summary, indent=2))
    print(bands.to_string(index=False))
    print(f"written to {out}")
    return 0


def replay(folder: Path) -> int:
    inputs, prices, accounts = load_inputs(folder)
    rid = make_run_id(inputs, accounts, prices)
    if rid != folder.name:
        print(f"replay FAILED: inputs hash to {rid}, folder is {folder.name} (inputs were changed)")
        return 1
    res = run(inputs, accounts, prices)
    stored = pd.read_parquet(folder / "trades.parquet")
    again = _cents(res.trades)
    same = again.reset_index(drop=True).equals(stored.reset_index(drop=True))
    print(f"run_id {rid} matches the stored inputs; "
          f"{len(again)} orders {'identical' if same else 'DIFFER'} to the stored run")
    if inputs.code_version.endswith("-dirty"):
        print("note: the stored run came from uncommitted code; an identical replay "
              "needs that same working tree")
    return 0 if same else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", help="trade date (YYYY-MM-DD) for a new run")
    ap.add_argument("--replay", help="runs/<run_id> folder to rerun from its saved inputs")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--band", type=float, default=0.05)
    ap.add_argument("--cash-buffer", type=float, default=0.02)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()
    if args.replay:
        return replay(Path(args.replay))
    if not args.asof:
        ap.error("give --asof for a new run or --replay for a saved one")
    return new_run(args)


if __name__ == "__main__":
    sys.exit(main())