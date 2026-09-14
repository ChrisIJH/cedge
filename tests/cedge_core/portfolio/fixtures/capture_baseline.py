"""
Snapshots known-answer outputs of cedge_core.portfolio.analytics against the
real DB, for regression testing.

IMPORTANT: the committed known_answer_baseline.pkl was captured from the
ORIGINAL pre-refactor code in CHResearch. Do not regenerate it casually —
validating migrated code against a baseline produced by that same migrated
code proves nothing. Re-run this only when the reference dataset itself
must change, and say so explicitly in the commit.

Run with:
    conda activate research3
    cd ~/work/cedge
    python -m tests.cedge_core.portfolio.fixtures.capture_baseline
"""

import pickle
from pathlib import Path

import pandas as pd
from cedge_core.portfolio.analytics import (
    build_performance,
    calc_return_stats,
    calc_stats,
    get_daily_prices,
)

START = "2025-01-01"
END = "2025-06-30"


def main():
    raw_aapl_msft = get_daily_prices(["AAPL", "MSFT"], START, END)
    raw_spy = get_daily_prices(["SPY"], START, END, )

    weights_df = pd.DataFrame({
        "ticker": ["AAPL", "MSFT"],
        "position": ["LONG", "SHORT"],
    })
    perf_before = build_performance(weights_df, START, END)

    aapl_only = raw_aapl_msft[raw_aapl_msft["ticker"] == "AAPL"].copy()
    aapl_only["date"] = pd.to_datetime(aapl_only["date"])
    aapl_px = aapl_only.set_index("date")["adj_close_price"].sort_index()
    aapl_ret = aapl_px.pct_change().dropna()

    calc_stats_before = calc_stats(aapl_ret)
    calc_return_stats_before = calc_return_stats(aapl_ret, "AAPL")

    baseline = {
        "raw_aapl_msft": raw_aapl_msft,
        "raw_spy": raw_spy,
        "weights_df": weights_df,
        "perf_before": perf_before,
        "aapl_ret": aapl_ret,
        "calc_stats_before": calc_stats_before,
        "calc_return_stats_before": calc_return_stats_before,
    }

    out_path = Path(__file__).parent / "known_answer_baseline.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(baseline, f)

    print(f"Baseline written to {out_path}")
    print(f"raw_aapl_msft rows: {len(raw_aapl_msft)}")
    print(f"raw_spy rows: {len(raw_spy)} (expected 0 — build_performance's internal SPY call "
          f"doesn't pass instrument_type, so it still defaults to 'stock' and SPY, stored as "
          f"'etf', is excluded)")
    print(f"perf_before is None: {perf_before is None}")
    print(f"calc_stats_before: {calc_stats_before}")


if __name__ == "__main__":
    main()