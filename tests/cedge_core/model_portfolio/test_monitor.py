import pytest
from cedge_core.model_portfolio.accounts import Account
from cedge_core.model_portfolio.models import MODELS
from cedge_core.model_portfolio.monitor import account_detail, band_table
from cedge_core.model_portfolio.synthetic import make_accounts
from cedge_core.model_portfolio.trades import Rules


def test_wider_band_never_triggers_more(close_panel, prices):
    book = make_accounts(close_panel, list(MODELS.values()), "2026-10-13", n=300)
    tbl = band_table(book, MODELS, prices, Rules())
    assert list(tbl["band"]) == [0.01, 0.02, 0.03, 0.05]
    assert tbl["n_triggered"].is_monotonic_decreasing
    assert (tbl["n_triggered"] == tbl["n_traded"] + tbl["n_too_small"]).all()


def test_account_detail_matches_case1():
    prices = {"EQ": 187.37, "BOND": 96.42}
    acct = Account("C1", "M", {"EQ": 352.0, "BOND": 318.0}, 3_400.0)
    res, t = account_detail(acct, {"EQ": 0.6, "BOND": 0.4}, prices,
                            Rules(band=0.05, cash_buffer=0.05, min_trade=250))
    t = t.set_index("ticker")
    assert res.trades == {"EQ": -47, "BOND": 74}
    assert t.loc["EQ", "order"] == -47 and t.loc["EQ", "shares_after"] == 305
    assert t.loc["EQ", "weight_after"] == pytest.approx(0.5714, abs=1e-4)
    assert t.loc["BOND", "weight_after"] == pytest.approx(0.3779, abs=1e-4)
    assert t.loc["CASH", "weight_after"] == pytest.approx(0.0507, abs=1e-4)
    assert t["weight_before"].sum() == pytest.approx(1.0, abs=1e-12)
    assert t["weight_after"].sum() == pytest.approx(1.0, abs=1e-12)
