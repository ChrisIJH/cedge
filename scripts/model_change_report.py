"""
scripts/model_change_report.py

Model-level report on real closes: what a proposed model change does, and the
closest variant of each model under a client's rules. Imperative shell: reads
adjusted closes from the DB; every number comes from model_portfolio/model_change.py.

    python scripts/model_change_report.py --asof 2026-10-16
    python scripts/model_change_report.py --asof 2026-10-16 --shift IJR:TIP:0.02 --aum 5e9
    python scripts/model_change_report.py --asof 2026-10-16 --exclude IJR --max-weight 0.45

Exit codes: 0 ok, 1 bad input or a gap in prices, 2 no close for the as-of date.
"""
import argparse
import datetime as dt
import sys
from typing import Dict, Tuple

import pandas as pd
from cedge_core.db import ch_engine
from cedge_core.marketdata.prices import load_prices_adjclose
from cedge_core.model_portfolio.attribution import BENCHMARKS
from cedge_core.model_portfolio.model_change import (
    annualized_covariance,
    compare_models,
    make_variant,
)
from cedge_core.model_portfolio.models import MODELS, ModelPortfolio


def parse_shift(text: str) -> Tuple[str, str, float]:
    """'IJR:TIP:0.02' -> move 2 points of weight from IJR to TIP."""
    try:
        src, dst, size = text.split(":")
        return src, dst, float(size)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--shift must look like FROM:TO:SIZE, got {text!r}")


def shifted(model: ModelPortfolio, src: str, dst: str, size: float) -> ModelPortfolio:
    w: Dict[str, float] = dict(model.weights)
    if w.get(src, 0.0) < size:
        raise ValueError(f"{model.model_id} holds {w.get(src, 0.0):.2%} of {src}, can't move {size:.2%}")
    w[src] -= size
    w[dst] = w.get(dst, 0.0) + size
    return ModelPortfolio(model.model_id, f"{model.version}+{src}->{dst}", {t: x for t, x in w.items() if x > 0})


def pct(x: float) -> str:
    return f"{x:.3%}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", required=True, help="last close used for the covariance (YYYY-MM-DD)")
    ap.add_argument("--years", type=int, default=5, help="years of daily returns behind the covariance")
    ap.add_argument("--model", default="MOD_6040", choices=sorted(MODELS))
    ap.add_argument("--shift", type=parse_shift, default=parse_shift("IJR:TIP:0.02"))
    ap.add_argument("--aum", type=float, default=5e9, help="assets following the model, dollars")
    ap.add_argument("--exclude", nargs="+", default=["IJR"])
    ap.add_argument("--max-weight", type=float)
    ap.add_argument("--keep-sleeves", action="store_true")
    ap.add_argument("--grid", type=float, default=0.005)
    args = ap.parse_args()

    tickers = sorted({t for m in [*MODELS.values(), *BENCHMARKS.values()] for t in m.weights})
    start = (dt.date.fromisoformat(args.asof) - dt.timedelta(days=round(365.25 * args.years))).isoformat()
    adj = load_prices_adjclose(ch_engine(), tickers, start, args.asof)
    last = adj.index.max()
    if str(last) != args.asof:
        print(f"no close for {args.asof} (latest {last}); not using stale prices")
        return 2
    try:
        cov = annualized_covariance(adj)
    except ValueError as e:
        print(f"cannot build the covariance: {e}")
        return 1
    print(f"covariance: {len(adj) - 1} daily returns, {adj.index.min()} .. {last}, annualized x252\n")

    # 1. one proposed model change
    old = MODELS[args.model]
    src, dst, size = args.shift
    try:
        new = shifted(old, src, dst, size)
    except ValueError as e:
        print(e)
        return 1
    ch = compare_models(old, new, cov, benchmark=BENCHMARKS[args.model], aum=args.aum)
    print(f"== model change: {args.model}, {size:.1%} from {src} to {dst} (AUM ${args.aum:,.0f})")
    print(f"turnover {ch.turnover:.2%} one-way, {ch.n_changed} funds change")
    print(f"tracking error new vs old {pct(ch.te_old_new)}; vs benchmark {pct(ch.te_old_bench or 0.0)}"
          f" -> {pct(ch.te_new_bench or 0.0)}")
    print("by sleeve:", ", ".join(f"{s} {x:+.2%}" for s, x in ch.diff_by_sleeve.items()))
    flows = ch.flows if ch.flows is not None else pd.Series(dtype=float)
    print("flows:", ", ".join(f"{t} ${x:+,.0f}" for t, x in flows[flows != 0].items()), "\n")

    # 2. the closest variant of every model under the same rules
    rules = f"exclude {', '.join(args.exclude)}" + \
        (f", max weight {args.max_weight:.0%}" if args.max_weight else "") + \
        (", keep sleeves" if args.keep_sleeves else "")
    print(f"== variants: {rules}, rounded to {args.grid:.1%}")
    rows = []
    for mid, base in MODELS.items():
        v = make_variant(base, cov, exclude=args.exclude, max_weight=args.max_weight,
                         keep_sleeves=args.keep_sleeves, grid=args.grid)
        if v.status != "optimal" or v.rounded is None or v.te_share is None:
            rows.append({"model": mid, "status": v.status})
            continue
        top = v.te_share.idxmax()
        rows.append({"model": mid, "status": v.status,
                     **{t: f"{base.weights.get(t, 0):.1%} -> {x:.1%}" for t, x in v.rounded.items()},
                     "te": pct(v.te or 0.0), "te_rounded": pct(v.te_rounded or 0.0),
                     "most_te": f"{top} {v.te_share[top]:.0%}",
                     "binding": ", ".join(v.binding) or "-", "violations": ", ".join(v.violations) or "-"})
    print(pd.DataFrame(rows).set_index("model").T.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())