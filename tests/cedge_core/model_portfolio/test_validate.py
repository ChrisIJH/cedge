from cedge_core.model_portfolio.accounts import Account
from cedge_core.model_portfolio.validate import (
    BAD_HOLDINGS,
    PRICE_MISSING,
    UNKNOWN_MODEL,
    validate_inputs,
)

GOOD = Account("G", "MOD_6040", {"IVV": 10.0}, 1_000.0)


def test_missing_model_price_aborts(models, prices):
    px = {t: p for t, p in prices.items() if t != "AGG"}
    v = validate_inputs(models, [GOOD], px)
    assert v.abort_reason == "ABORTED_PRICE_MISSING: AGG"


def test_account_level_problems_are_excluded(models, prices):
    accts = [GOOD,
             Account("N", "MOD_6040", {"IVV": -1.0}, 0.0),
             Account("F", "MOD_6040", {"IVV": 1.5}, 0.0),
             Account("C", "MOD_6040", {}, -5.0),
             Account("Z", "MOD_6040", {"ZZZ": 3.0}, 0.0),
             Account("U", "MOD_X", {}, 100.0)]
    v = validate_inputs(models, accts, prices, max_excluded_frac=1.0)
    assert [a.account_id for a in v.valid] == ["G"]
    assert v.excluded == {"N": BAD_HOLDINGS, "F": BAD_HOLDINGS, "C": BAD_HOLDINGS,
                          "Z": PRICE_MISSING, "U": UNKNOWN_MODEL}
    assert v.abort_reason is None


def test_too_many_exclusions_abort(models, prices):
    accts = [GOOD] + [Account(f"N{i}", "MOD_6040", {"IVV": -1.0}, 0.0) for i in range(2)]
    v = validate_inputs(models, accts, prices)          # 2/3 > 5%
    assert v.abort_reason.startswith("ABORTED_TOO_MANY_EXCLUDED")