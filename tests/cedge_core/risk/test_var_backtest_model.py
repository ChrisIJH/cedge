from __future__ import annotations

import numpy as np
import pytest
from cedge_core.risk.var_backtest_model import (
    acerbi_szekely_z2,
    christoffersen_cc,
    kupiec_uc,
    run_backtest,
)


class TestKupiecUC:
    def test_no_breach_small_T_passes(self):
        """T가 작으면 breach 0개도 정상 (데이터 부족, 기각 안 함)."""
        b = np.zeros(20, dtype=int)
        r = kupiec_uc(b, alpha=0.05)
        assert r["N"] == 0
        assert r["reject"] == False        # LR ≈ 2.05 < 3.841

    def test_no_breach_large_T_rejects(self):
        """T가 크면 breach 0개 = 과대보수 → 기각 (양측 검정)."""
        b = np.zeros(500, dtype=int)
        r = kupiec_uc(b, alpha=0.05)
        assert r["N"] == 0
        assert r["reject"] == True         # LR ≈ 51.29 > 3.841
    
    def test_perfect_rate_passes(self):
        """정확히 5% breach --> LR ~ 0 -> pass"""
        b = np.zeros(500)
        b[:25] = 1
        r = kupiec_uc(b)
        print(r)
        assert r['breach_rate'] == pytest.approx(0.05)
        assert r['LR'] == pytest.approx(0.0, abs=1e-6)
        assert r['reject'] == False

    def test_too_many_breaches_rejects(self):
        """15% breach --> 과소추정 --> reject"""
        b = np.zeros(500)
        b[:75] = 1
        r = kupiec_uc(b)
        assert r["reject"] == True
        assert r["LR"] > 3.841

    def test_lr_is_nonnegative(self):
        rng = np.random.default_rng(0)
        b = (rng.random(500) < 0.05).astype(int)
        assert kupiec_uc(b)["LR"] >= 0


class TestChristoffersenCC:
    def test_clustered_breached_reject_even_if_rate_ok(self):
        """frequency 5% but clustered. Kupiec pass but CC reject"""
        b = np.zeros(500)
        b[100:125]=1
        cc = christoffersen_cc(b)
        assert cc["pi_11"] > cc["pi_01"]     # 몰림 신호
        assert cc["reject"] == True

    def test_cc_ge_uc(self):
        """LR_CC = LR_UC + LR_IND ≥ LR_UC ."""
        rng = np.random.default_rng(1)
        b = (rng.random(500) < 0.05).astype(int)
        cc = christoffersen_cc(b, 0.05)
        assert cc['LR_CC'] >= cc["LR_UC"] - 1e-9


class TestRunBacktest:
    def test_good_model_passes_both(self):
        rng = np.random.default_rng(0)
        T, alpha = 750, 0.05
        from scipy import stats
        sigma = 0.01
        r = rng.normal(0, sigma, T)
        var_est = np.full(T, -(stats.norm.ppf(alpha) * sigma))
        out = run_backtest(r, var_est, alpha=alpha)
        assert out["kupiec"]["reject"] == False
        assert out["christoffersen"]["reject"] == False

    def test_fixed_var_ignoring_crisis_rejects(self):
        """위기 무시한 고정 VaR → 기각 (R-1 핵심 케이스)."""
        rng = np.random.default_rng(0)
        from scipy import stats
        T, alpha = 750, 0.05
        sigma_t = np.full(T, 0.01)
        sigma_t[300:380] = 0.05
        r = rng.normal(0, sigma_t, T)
        var_fixed = np.full(T, -(stats.norm.ppf(alpha) * 0.01))
        out = run_backtest(r, var_fixed, alpha=alpha)
        assert out["kupiec"]["reject"] == True
        assert out["christoffersen"]["reject"] == True

class TestAcerbiSzekely:
    def test_returns_z2_and_keys(self):
        rng = np.random.default_rng(0)
        from scipy import stats
        T, alpha = 750, 0.05
        sigma = 0.01
        r = rng.normal(0, sigma, T)
        z = stats.norm.ppf(alpha)
        var_est = np.full(T, -(z * sigma))
        es_est = np.full(T, sigma * stats.norm.pdf(z) / alpha)
        out = acerbi_szekely_z2(r, var_est, es_est, alpha, n_boot=2000)
        assert "Z2" in out and "p_value" in out
        assert 0.0 <= out["p_value"] <= 1.0