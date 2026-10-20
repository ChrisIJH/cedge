"""Known answers for the Brinson-Fachler split, worked by hand.

Benchmark: E1 60%, B1 40%.  Model: E1 50%, E2 20%, B1 30%.
Period growth: E1 +10%, E2 +20%, B1 +2%.  Sleeves: equity = {E1, E2}, bonds = {B1}.

  benchmark: equity 0.60 @ 10%, bonds 0.40 @ 2%   -> B = 0.068
  model:     equity 0.70 @ (0.05 + 0.04)/0.70 = 12.857%, bonds 0.30 @ 2%  -> 0.096
  active = 0.028
  allocation:  equity (0.70-0.60)(0.10-0.068) = 0.0032,  bonds (0.30-0.40)(0.02-0.068) = 0.0048
  selection:   equity 0.60 * (0.128571-0.10) = 0.017143, bonds 0
  interaction: equity 0.10 * (0.128571-0.10) = 0.002857, bonds 0
  0.008 + 0.017143 + 0.002857 = 0.028
"""
import pytest
from cedge_core.model_portfolio.attribution import (
    BENCHMARKS,
    attribution_table,
    brinson,
)
from cedge_core.model_portfolio.models import MODELS
from cedge_core.model_portfolio.synthetic import make_accounts

SLEEVE_OF = {"E1": "equity", "E2": "equity", "B1": "bonds"}
GROWTH = {"E1": 0.10, "E2": 0.20, "B1": 0.02}
MODEL = {"E1": 0.5, "E2": 0.2, "B1": 0.3}
BENCH = {"E1": 0.6, "B1": 0.4}


def test_hand_worked_brinson():
    t = brinson(MODEL, BENCH, GROWTH, SLEEVE_OF).set_index("sleeve")
    assert t.loc["equity", "allocation"] == pytest.approx(0.0032, abs=1e-12)
    assert t.loc["bonds", "allocation"] == pytest.approx(0.0048, abs=1e-12)
    assert t.loc["equity", "selection"] == pytest.approx(0.6 * (0.09 / 0.7 - 0.10), abs=1e-12)
    assert t.loc["bonds", "selection"] == pytest.approx(0.0, abs=1e-12)
    assert t.loc["equity", "interaction"] == pytest.approx(0.1 * (0.09 / 0.7 - 0.10), abs=1e-12)
    total = t[["allocation", "selection", "interaction"]].to_numpy().sum()
    assert total == pytest.approx(0.096 - 0.068, abs=1e-12)


def test_model_equal_to_benchmark_has_no_effects():
    t = brinson(BENCH, BENCH, GROWTH, SLEEVE_OF)
    assert t[["allocation", "selection", "interaction"]].abs().to_numpy().max() == 0.0


def test_shipped_benchmarks_cover_the_shipped_models():
    assert set(BENCHMARKS) == set(MODELS)
    for m in MODELS.values():
        assert abs(sum(BENCHMARKS[m.model_id].weights.values()) - 1.0) < 1e-12


def test_account_gap_splits_exactly_into_decision_and_implementation(close_panel):
    book = make_accounts(close_panel, list(MODELS.values()), "2026-10-13", n=200,
                         deposit_prob=0.0)
    t = attribution_table(book, MODELS, close_panel, "2026-10-13")
    assert len(t) == 200
    decision = t["allocation"] + t["selection"] + t["interaction"]
    implementation = t["cash"] + t["rounding"]
    assert ((t["model"] - t["benchmark"]) - decision).abs().max() < 1e-12
    assert ((t["account"] - t["model"]) - implementation).abs().max() < 1e-12
    assert (t["active"] - decision - implementation).abs().max() < 1e-12