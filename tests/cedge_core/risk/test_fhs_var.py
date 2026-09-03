"""Known-answer gates for cedge_core.risk.fhs_var (Filtered Historical Simulation).

All values pre-verified 2026-07-30 (see study_materials FT02 handoff, Part 2).
Gates are the correctness spec — if one fails, fix the implementation,
never the gate.

alpha convention here is TAIL PROBABILITY (matches var_es_model.py in this
package) — NOT confidence level (contrast param_var.py).
"""
import numpy as np
import pytest
from scipy import stats

from cedge_core.risk.fhs_var import ewma_filter, fhs_quantile, rolling_fhs_var, rolling_fhs_es


class TestEWMAFilter:
    def test_gate_a_hand_computed_recursion(self):
        lam = 0.94
        sigma2_init = 4.0e-4  # sigma_0 = 0.02
        r = np.array([0.0100, -0.0200, 0.0150, -0.0300, 0.0050])

        EXPECTED_SIGMA2 = np.array([
            4.000000000000e-04,
            3.820000000000e-04,
            3.830800000000e-04,
            3.735952000000e-04,
            4.051794880000e-04,
        ])
        sigma, z = ewma_filter(r, lam=lam, sigma2_init=sigma2_init)
        assert np.allclose(sigma ** 2, EXPECTED_SIGMA2, rtol=1e-12), sigma ** 2

    def test_sigma2_init_required(self):
        with pytest.raises(ValueError):
            ewma_filter(np.array([0.01, -0.02, 0.01]))


class TestFHSQuantile:
    def test_gate_b_degenerates_to_normal(self):
        """If the residual pool IS Normal, FHS must equal parametric exactly."""
        n = 100_000
        i = np.arange(1, n + 1)
        z_grid = stats.norm.ppf((i - 0.5) / n)  # deterministic grid, no RNG

        q = fhs_quantile(z_grid, alpha=0.01)
        assert abs(q - stats.norm.ppf(0.01)) < 1e-3, q

    def test_gate_c_student_t_ratio(self):
        """Fat tail does NOT always mean bigger VaR — there's a crossover
        between alpha=0.05 (FHS < parametric) and alpha=0.01 (FHS > parametric)."""
        nu = 5
        i = np.arange(1, 100_001)
        t_grid = stats.t.ppf((i - 0.5) / 100_000, df=nu) / np.sqrt(nu / (nu - 2))

        expected = {0.01: 1.120410, 0.05: 0.948929}
        for a, exp_ratio in expected.items():
            q_fhs = fhs_quantile(t_grid, alpha=a)
            q_norm = stats.norm.ppf(a)
            assert abs(q_fhs / q_norm - exp_ratio) < 5e-3, (a, q_fhs / q_norm, exp_ratio)


class TestConvergence:
    def test_gate_d_bootstrap_converges_to_direct(self):
        rng = np.random.default_rng(7)
        z_pool = rng.standard_t(df=5, size=2000) / np.sqrt(5 / 3)
        truth = fhs_quantile(z_pool, alpha=0.01, n_bootstrap=None)

        ses = []
        for B in [100, 1_000, 10_000]:
            ests = np.array([
                fhs_quantile(z_pool, 0.01, n_bootstrap=B, rng=np.random.default_rng(s))
                for s in range(200)
            ])
            ses.append(ests.std())

        assert ses[0] > ses[1] > ses[2], ses
        assert abs(ests.mean() - truth) < 0.05  # last batch = B=10000


class TestLookAhead:
    def test_gate_e_mutation_future_return_must_not_change_past_var(self):
        rng = np.random.default_rng(11)
        r = rng.normal(0, 0.01, 300)
        T = 200
        r2 = r.copy()
        r2[T] = r[T] * 50  # mutate ONE future point only

        v1 = rolling_fhs_var(r, window=100, alpha=0.01)
        v2 = rolling_fhs_var(r2, window=100, alpha=0.01)

        mask = ~np.isnan(v1[:T + 1])
        assert mask.sum() > 0, "comparison must be non-vacuous"
        assert np.array_equal(v1[:T + 1][mask], v2[:T + 1][mask]), "LOOK-AHEAD DETECTED"

    def test_es_mutation_future_return_must_not_change_past_es(self):
        rng = np.random.default_rng(11)
        r = rng.normal(0, 0.01, 300)
        T = 200
        r2 = r.copy()
        r2[T] = r[T] * 50

        e1 = rolling_fhs_es(r, window=100, alpha=0.01)
        e2 = rolling_fhs_es(r2, window=100, alpha=0.01)

        mask = ~np.isnan(e1[:T + 1])
        assert mask.sum() > 0, "comparison must be non-vacuous"
        assert np.array_equal(e1[:T + 1][mask], e2[:T + 1][mask]), "LOOK-AHEAD DETECTED"


class TestEdgeCases:
    def test_window_larger_than_data_is_all_nan(self):
        r = np.random.default_rng(0).normal(0, 0.01, 50)
        out = rolling_fhs_var(r, window=100, alpha=0.01)
        assert np.all(np.isnan(out))

    def test_alpha_out_of_range_raises(self):
        z = np.random.default_rng(0).normal(0, 1, 100)
        with pytest.raises(ValueError):
            fhs_quantile(z, alpha=1.5)

    def test_nan_in_pool_raises(self):
        z = np.array([0.1, np.nan, -0.2])
        with pytest.raises(ValueError):
            fhs_quantile(z, alpha=0.05)

    def test_var_output_is_loss_positive(self):
        """Regression guard for the sign bug found during review."""
        rng = np.random.default_rng(3)
        r = rng.normal(0, 0.01, 400)
        v = rolling_fhs_var(r, window=100, alpha=0.01)
        valid = v[~np.isnan(v)]
        assert (valid > 0).all(), "VaR must be a positive loss magnitude"
