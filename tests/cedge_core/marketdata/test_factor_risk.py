import numpy as np
import pandas as pd
import pytest
from cedge_core.marketdata.factor_risk import SqlFactorRiskRepository


class _FakeEngine:
    """Minimal test double for SQLAlchemy engine."""
    def begin(self):
        class _Ctx:
            def __enter__(self):
                return None
            def __exit__(self, *exc):
                return False
        return _Ctx()

def test_load_factor_cov_mirrors_upper_triangle(monkeypatch):
    """Triangle-only rows must come back as a full symmetric matrix."""
    rows = pd.DataFrame([
        ("mkt", "mkt", 0.04), ("mkt", "mom", 0.01), ("mom", "mom", 0.02),
    ], columns=["factor_i", "factor_j", "cov_ij"])
    monkeypatch.setattr(pd, "read_sql", lambda *a, **k: rows)

    repo = SqlFactorRiskRepository(engine=_FakeEngine())
    sigma_f = repo.load_factor_cov("2026-09-18")

    assert sigma_f.loc["mom", "mkt"] == pytest.approx(0.01)
    assert sigma_f.loc["mkt", "mom"] == pytest.approx(0.01)
    assert not sigma_f.isna().any().any()
    np.testing.assert_allclose(sigma_f.to_numpy(), sigma_f.to_numpy().T)


def test_load_factor_cov_rejects_missing_variance(monkeypatch):
    """A factor present only as an off-diagonal partner has no variance."""
    rows = pd.DataFrame([
        ("mkt", "mkt", 0.04), ("mkt", "mom", 0.01),   # no (mom, mom)
    ], columns=["factor_i", "factor_j", "cov_ij"])
    monkeypatch.setattr(pd, "read_sql", lambda *a, **k: rows)

    repo = SqlFactorRiskRepository(engine=_FakeEngine())
    with pytest.raises(LookupError, match="no positive variance"):
        repo.load_factor_cov("2026-09-18")