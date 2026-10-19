from cedge_core.model_portfolio.models import MODELS
from cedge_core.model_portfolio.synthetic import make_accounts


def test_same_seed_same_book(close_panel):
    a = make_accounts(close_panel, list(MODELS.values()), "2026-10-13", n=200)
    b = make_accounts(close_panel, list(MODELS.values()), "2026-10-13", n=200)
    assert a == b


def test_book_is_whole_shares_and_non_negative(close_panel):
    book = make_accounts(close_panel, list(MODELS.values()), "2026-10-13", n=200)
    for acct in book:
        assert acct.cash >= 0
        assert all(q >= 0 and float(q).is_integer() for q in acct.shares.values())