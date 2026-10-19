from dataclasses import replace

import pytest
from cedge_core.model_portfolio.models import MODELS
from cedge_core.model_portfolio.run import ABORTED, OK, RunInputs, run
from cedge_core.model_portfolio.synthetic import edge_accounts, make_accounts
from cedge_core.model_portfolio.trades import Rules

ASOF = "2026-10-13"
INPUTS = RunInputs(MODELS, ASOF, Rules(band=0.05, cash_buffer=0.02), "test")


@pytest.fixture(scope="module")
def book(close_panel, prices):
    return make_accounts(close_panel, list(MODELS.values()), ASOF) + \
        edge_accounts(MODELS, prices)


def test_reconciles(book, prices):
    res = run(INPUTS, book, prices)
    c = res.counts
    assert res.status == OK
    assert c["n_target"] == len(book) == 1010
    assert c["n_target"] == c["n_processed"] + c["n_exception"] + c["n_skipped"]
    assert len(res.accounts) == len(book)
    assert res.accounts["account_id"].is_unique


def test_edge_accounts_land_where_expected(book, prices):
    st = run(INPUTS, book, prices).accounts.set_index("account_id")
    assert st.loc["E01_ZERO_CASH", "status"] == "TRADED"
    assert st.loc["E02_LEGACY", "status"] == "TRADED"
    assert st.loc["E03_NEG_SHARES", "status"] == "EXC_INPUT"
    assert st.loc["E04_TINY", "status"] == "EXC_TOO_SMALL"
    assert st.loc["E04B_SMALL", "status"] == "TRADED"
    assert "STILL_OUT_OF_BAND" in st.loc["E04B_SMALL", "flag_codes"]
    assert st.loc["E05_DEPOSIT", "status"] == "TRADED"
    assert st.loc["E06_NO_PRICE", "reason"] == "PRICE_MISSING"
    assert st.loc["E07_FRACTIONAL", "reason"] == "BAD_HOLDINGS"
    assert st.loc["E08_AT_TARGET", "status"] == "SKIPPED_IN_BAND"
    assert st.loc["E09_UNKNOWN_MODEL", "reason"] == "UNKNOWN_MODEL"


def test_legacy_position_fully_exited(book, prices):
    res = run(INPUTS, book, prices)
    t = res.trades.set_index(["account_id", "ticker"])
    assert t.loc[("E02_LEGACY", "IEMG"), "shares"] == -352


@pytest.mark.parametrize("band", [0.01, 0.02, 0.03, 0.05])
def test_rerun_after_fills_trades_nothing(book, prices, band):
    inputs = replace(INPUTS, rules=Rules(band=band, cash_buffer=0.02))
    first = run(inputs, book, prices)
    again = run(inputs, first.post_accounts, prices)
    assert first.counts["n_processed"] > 0
    assert again.counts["n_processed"] == 0
    assert again.trades.empty


def test_no_invariant_failures_and_cash_non_negative(book, prices):
    res = run(INPUTS, book, prices)
    assert res.counts["status_EXC_INVARIANT"] == 0
    traded = res.accounts[res.accounts["status"] == "TRADED"]
    assert (traded["post_cash"] >= 0).all()


def test_run_id_tracks_every_input(book, prices):
    base = run(INPUTS, book, prices).run_id
    assert run(INPUTS, book, prices).run_id == base
    assert run(replace(INPUTS, rules=Rules(band=0.03)), book, prices).run_id != base
    assert run(replace(INPUTS, code_version="other"), book, prices).run_id != base
    assert run(INPUTS, book, {**prices, "IVV": prices["IVV"] + 0.01}).run_id != base
    assert run(INPUTS, book[:-1], prices).run_id != base


def test_missing_model_price_aborts_run(book, prices):
    px = {t: p for t, p in prices.items() if t != "TIP"}
    res = run(INPUTS, book, px)
    assert res.status == ABORTED
    assert res.trades.empty