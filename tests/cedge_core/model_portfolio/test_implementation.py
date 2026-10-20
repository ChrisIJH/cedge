"""Known answers for implementation_effect, worked by hand.

Two assets, model 50/50, three closes:  A 100 -> 110 -> 121,  B 100 -> 100 -> 100.
Account: $1,000 at start, bought 4 A + 5 B, $100 cash left.

  model_rebalanced = 1.05 * 1.05 - 1                    = 0.1025
  model_buy_hold   = 0.5 * 1.21 + 0.5 * 1.00 - 1        = 0.1050
  account          = (4*121 + 5*100 + 100) / 1000 - 1    = 0.0840
  c0 = 0.10, invested = (484 + 500) / 900 - 1            = 0.093333...
  drift    =  0.1050 - 0.1025                            = +0.0025
  cash     = -0.10 * 0.1050                              = -0.0105
  rounding =  0.90 * (0.093333 - 0.1050)                 = -0.0105
  total    =  0.0840 - 0.1025                            = -0.0185
"""
import pandas as pd
import pytest
from cedge_core.model_portfolio.accounts import Account
from cedge_core.model_portfolio.implementation import (
    by_account_size,
    implementation_effect,
    implementation_table,
)
from cedge_core.model_portfolio.models import MODELS, ModelPortfolio
from cedge_core.model_portfolio.synthetic import make_accounts

CLOSE = pd.DataFrame({"A": [100.0, 110.0, 121.0], "B": [100.0, 100.0, 100.0]},
                     index=pd.bdate_range("2026-01-05", periods=3))
MODEL = {"A": 0.5, "B": 0.5}
ACCT = Account("X", "M", {"A": 4.0, "B": 5.0}, 100.0, "2026-01-05")


def test_hand_worked_decomposition():
    e = implementation_effect(ACCT, MODEL, CLOSE, "2026-01-07")
    assert e.model_rebalanced == pytest.approx(0.1025, abs=1e-12)
    assert e.model_buy_hold == pytest.approx(0.1050, abs=1e-12)
    assert e.account == pytest.approx(0.0840, abs=1e-12)
    assert e.cash_start == pytest.approx(0.10, abs=1e-12)
    assert e.drift_effect == pytest.approx(0.0025, abs=1e-12)
    assert e.cash_effect == pytest.approx(-0.0105, abs=1e-12)
    assert e.rounding_effect == pytest.approx(-0.0105, abs=1e-12)
    assert e.total == pytest.approx(-0.0185, abs=1e-12)


def test_exact_model_holder_has_no_cash_or_rounding_effect():
    exact = Account("Y", "M", {"A": 5.0, "B": 5.0}, 0.0, "2026-01-05")
    e = implementation_effect(exact, MODEL, CLOSE, "2026-01-07")
    assert e.cash_effect == 0.0
    assert e.rounding_effect == pytest.approx(0.0, abs=1e-12)
    assert e.total == pytest.approx(e.drift_effect, abs=1e-12)


def test_growth_can_come_from_adjusted_prices():
    # B pays a 2% distribution: raw close stays at 100, adjusted close rises to 102
    adj = CLOSE.assign(B=[100.0, 101.0, 102.0])
    e = implementation_effect(ACCT, MODEL, CLOSE, "2026-01-07", returns_from=adj)
    assert e.cash_start == pytest.approx(0.10, abs=1e-12)          # still valued at the paid close
    assert e.model_buy_hold == pytest.approx(0.5 * 1.21 + 0.5 * 1.02 - 1, abs=1e-12)
    assert e.account == pytest.approx((400 * 1.21 + 500 * 1.02 + 100) / 1000 - 1, abs=1e-12)


def test_needs_a_start_date():
    with pytest.raises(ValueError, match="last_rebalanced"):
        implementation_effect(Account("Z", "M", {}, 1.0), MODEL, CLOSE, "2026-01-07")


def test_parts_add_up_across_a_synthetic_book(close_panel):
    book = make_accounts(close_panel, list(MODELS.values()), "2026-10-13", n=200,
                         deposit_prob=0.0)
    table = implementation_table(book, MODELS, close_panel, "2026-10-13")
    assert len(table) == 200
    parts = table["drift_effect"] + table["cash_effect"] + table["rounding_effect"]
    assert (parts - table["total"]).abs().max() < 1e-12
    # cash earns nothing, so holding it costs exactly when the model went up
    assert (table["cash_effect"] * table["model_buy_hold"] <= 0).all()


def test_size_summary_has_one_row_per_bucket(close_panel):
    book = make_accounts(close_panel, list(MODELS.values()), "2026-10-13", n=200,
                         deposit_prob=0.0)
    summary = by_account_size(implementation_table(book, MODELS, close_panel, "2026-10-13"))
    assert summary["accounts"].sum() == 200
    assert {"drift_bp", "cash_bp", "rounding_bp", "total_bp", "worst_total_bp"} <= set(summary)


def test_unknown_model_is_skipped(close_panel):
    other = ModelPortfolio("OTHER", "1", {"IVV": 1.0})
    a = Account("Q", "MOD_X", {"IVV": 1.0}, 0.0, "2026-10-01")
    assert implementation_table([a], {"OTHER": other}, close_panel, "2026-10-13").empty