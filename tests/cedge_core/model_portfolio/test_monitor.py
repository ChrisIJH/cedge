from cedge_core.model_portfolio.models import MODELS
from cedge_core.model_portfolio.monitor import band_table
from cedge_core.model_portfolio.synthetic import make_accounts
from cedge_core.model_portfolio.trades import Rules


def test_wider_band_never_triggers_more(close_panel, prices):
    book = make_accounts(close_panel, list(MODELS.values()), "2026-10-13", n=300)
    tbl = band_table(book, MODELS, prices, Rules())
    assert list(tbl["band"]) == [0.01, 0.02, 0.03, 0.05]
    assert tbl["n_triggered"].is_monotonic_decreasing
    assert (tbl["n_triggered"] == tbl["n_traded"] + tbl["n_too_small"]).all()