from __future__ import annotations

import numpy as np
import pytest
from cedge_core.estimators import realized_beta


class TestRealizedBeta:
    # @pytest.mark.xfail(reason="Not Implemented yet")
    def test_exact_beta_one(self):
        """
        asset = market -> beta = 1, alpha = 0, r2 = 1
        """
        rng = np.random.default_rng(0)
        m = rng.normal(0, 0.01, 500)

        r = realized_beta(m.copy(), m)
        assert r["beta"]  == pytest.approx(1.0, abs=1e-9)
        assert r["alpha"] == pytest.approx(0.0, abs=1e-12)
        assert r["r2"]    == pytest.approx(1.0, abs=1e-9)
        assert r["n"] == 500

    def test_known_linear_no_noise(self):
        """
        asset = 0.0005 + 1.3*market (결정론) → 정확히 복원, r2=1.
        """
        rng = np.random.default_rng(0)
        m = rng.normal(0, 0.02, 400)
        a = 0.0005 + 1.3 * m

        r = realized_beta(a, m)

        assert r['beta'] == pytest.approx(1.3, abs=1e-9)
        assert r['alpha'] == pytest.approx(0.0005, abs=1e-9)
        assert r['r2'] == pytest.approx(1.0, abs=1e-9)

    def test_negative_beta(self):
        """
        asset = -market
        """
        rng = np.random.default_rng(0)
        m = rng.normal(0, 0.01, 300)
        a = -m.copy()
        r = realized_beta(a, m)

        assert r['beta'] == pytest.approx(-1.0, abs=1e-9)
        assert r['alpha'] == pytest.approx(0, abs=1e-9)


    def test_scaled_beta(self):
        m = np.random.default_rng(3).normal(0, 0.01, 300)
        assert realized_beta(2.5 * m, m)["beta"] == pytest.approx(2.5, abs=1e-9)



    def test_beta_equals_cov_over_var(self):
        """정의 일치: beta == Cov(a,m)/Var(m) (population). 노이즈 → 0<r2<1."""
        rng = np.random.default_rng(4)
        m = rng.normal(0, 0.01, 600)
        a = 0.3 * m + rng.normal(0, 0.005, 600)
        r = realized_beta(a, m)
        cov = np.mean((m - m.mean()) * (a - a.mean()))
        var = np.mean((m - m.mean()) ** 2)
        assert r["beta"] == pytest.approx(cov / var, rel=1e-9)
        assert 0.0 < r["r2"] < 1.0

    def test_zero_beta_when_asset_constant(self):
        """market 평균0, asset 상수 → Cov=0 → beta=0."""
        m = np.array([-1.0, 1.0] * 100)     # mean 0
        a = np.ones(200)                    # constant → cov 0
        assert realized_beta(a, m)["beta"] == pytest.approx(0.0, abs=1e-12)

    def test_nan_rows_dropped(self):
        """어느 쪽이든 NaN인 행 제거, n은 살아남은 쌍."""
        m = np.array([0.01, 0.02, np.nan, 0.03, -0.01])
        a = np.array([0.01, np.nan, 0.05, 0.03, -0.01])
        r = realized_beta(a, m)              # 살아남는 행: 0,3,4 → a==m
        assert r["n"] == 3
        assert r["beta"] == pytest.approx(1.0, abs=1e-9)

    def test_alpha_intercept_recovered(self):
        m = np.random.default_rng(5).normal(0, 0.01, 300)
        a = 0.001 + 0.8 * m
        r = realized_beta(a, m)
        assert r["alpha"] == pytest.approx(0.001, abs=1e-9)
        assert r["beta"]  == pytest.approx(0.8,   abs=1e-9)