"""
Known-answer and property-based tests for solve_mean_variance.
"""
import numpy as np
import pytest
from cedge_core.optimization.mean_variance import solve_mean_variance


def test_two_asset_min_variance_gross_binds():
    """Gross_target binds when turnover cost forces w away from w_prev."""
    cov = np.diag([0.04, 0.04])
    w_prev = np.array([1.0, -1.0])  # Non-zero starting point
    
    result = solve_mean_variance(
        cov, w_prev, gross_target=2.0, net_target=0.0, w_bounds=1.0,
        lambda_turnover=0.1,  # Turnover cost to force movement
    )
    
    assert result.status in ("optimal", "optimal_inaccurate")
    assert result.gross_realized == pytest.approx(2.0, abs=1e-3)


def test_turnover_penalty_pulls_solution_toward_w_prev():
    cov = np.eye(3) * 0.01
    w_prev = np.array([0.3, -0.3, 0.0])

    no_penalty = solve_mean_variance(cov, w_prev, lambda_turnover=0.0, w_bounds=1.0)
    high_penalty = solve_mean_variance(cov, w_prev, lambda_turnover=1000.0, w_bounds=1.0)

    dist_no_penalty = np.sum(np.abs(no_penalty.weights - w_prev))
    dist_high_penalty = np.sum(np.abs(high_penalty.weights - w_prev))
    assert dist_high_penalty < dist_no_penalty
    assert high_penalty.turnover < no_penalty.turnover

def test_factor_cap_is_actually_enforced():
    """Regression guard for factor_cap constraint."""
    n = 3
    cov = np.eye(n) * 0.0001
    w_prev = np.zeros(n)
    # Only first two assets have mkt exposure; third is neutral
    factor_exposure = np.array([[1.0], [1.0], [0.0]])
    factor_names = ["mkt"]
    alpha = np.array([1.0, 1.0, 0.0])
    
    uncapped = solve_mean_variance(
        cov, w_prev, alpha=alpha, mu_alpha=50.0, w_bounds=1.0,
        gross_target=2.0, net_target=0.0,
        factor_exposure=factor_exposure, factor_names=factor_names,
    )
    capped = solve_mean_variance(
        cov, w_prev, alpha=alpha, mu_alpha=50.0, w_bounds=1.0,
        gross_target=2.0, net_target=0.0,
        factor_exposure=factor_exposure, factor_names=factor_names,
        factor_limits={"mkt": 0.5},
    )
    
    uncapped_mkt = abs(uncapped.factor_realized["mkt"])
    capped_mkt = abs(capped.factor_realized["mkt"])
    
    assert uncapped_mkt > 0.5, f"uncapped mkt exposure {uncapped_mkt} should exceed cap baseline"
    assert capped_mkt <= 0.5 + 1e-3, f"capped mkt exposure {capped_mkt} should respect 0.5 cap"
    assert uncapped_mkt > capped_mkt, "cap should reduce factor exposure"

def test_factor_realized_reports_uncapped_factors_too():
    """factor_realized must include every named factor, not just the ones
    that had a limit — diagnostics shouldn't silently drop information."""
    n = 2
    cov = np.eye(n) * 0.01
    w_prev = np.zeros(n)
    factor_exposure = np.array([[1.0, 0.5], [1.0, -0.5]])
    factor_names = ["mkt", "value"]

    result = solve_mean_variance(
        cov, w_prev, w_bounds=1.0,
        factor_exposure=factor_exposure, factor_names=factor_names,
        factor_limits={"mkt": 0.5},   # only "mkt" capped
    )

    assert set(result.factor_realized) == {"mkt", "value"}


def test_unknown_factor_limit_name_raises():
    with pytest.raises(ValueError, match="unknown factor"):
        solve_mean_variance(
            np.eye(2) * 0.01, np.zeros(2), w_bounds=1.0,
            factor_exposure=np.ones((2, 1)), factor_names=["mkt"],
            factor_limits={"typo_mkt": 0.05},
        )


def test_factor_names_required_when_factor_exposure_given():
    with pytest.raises(ValueError, match="factor_names is required"):
        solve_mean_variance(
            np.eye(2) * 0.01, np.zeros(2), w_bounds=1.0,
            factor_exposure=np.ones((2, 1)),
        )


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="cov shape"):
        solve_mean_variance(np.eye(2), np.zeros(3))


def test_infeasible_net_target_raises_not_silently_falls_back():
    """net_target=1.0 with w_bounds too small to reach it must fail loudly —
    this is the feasibility gap the original (net_target hardcoded to 0)
    never had to handle."""
    with pytest.raises(ValueError, match="infeasible by construction"):
        solve_mean_variance(
            np.eye(3) * 0.01, np.zeros(3),
            net_target=1.0, w_bounds=0.1,   # 3 * 0.1 = 0.3 < 1.0
        )


def test_gross_less_than_net_raises():
    with pytest.raises(ValueError, match="infeasible by construction"):
        solve_mean_variance(
            np.eye(3) * 0.01, np.zeros(3),
            net_target=1.0, gross_target=0.5, w_bounds=1.0,
        )


def test_no_alpha_is_minimum_variance():
    """alpha=None must not error and must produce a valid solve — this is
    the 'no alpha signal source yet' path cedge actually uses today."""
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    result = solve_mean_variance(cov, np.zeros(2), alpha=None, w_bounds=1.0)
    assert result.status in ("optimal", "optimal_inaccurate")


def test_non_psd_covariance_does_not_crash():
    """A covariance built from noisy estimates (e.g. a short window) can
    end up with a tiny negative eigenvalue from floating-point error alone
    — ensure_psd must absorb that rather than the solver rejecting it as
    non-convex."""
    cov = np.array([
        [1.0, 0.9999, 0.9999],
        [0.9999, 1.0, 0.9999],
        [0.9999, 0.9999, 1.0],
    ])
    cov[0, 0] -= 1e-8   # nudge just enough to risk a negative eigenvalue

    result = solve_mean_variance(cov, np.zeros(3), w_bounds=1.0, ensure_psd=True)
    assert result.status in ("optimal", "optimal_inaccurate")


def test_all_solvers_failing_raises_runtime_error(monkeypatch):
    """If every solver in the fallback chain fails, the function must
    raise — not return a w_prev result dressed up as a solve."""
    import cedge_core.optimization.mean_variance as mv
    monkeypatch.setattr(mv, "_SOLVERS", ())   # empty chain -> nothing to try

    with pytest.raises(RuntimeError, match="no solver in"):
        solve_mean_variance(np.eye(2) * 0.01, np.zeros(2), w_bounds=1.0)
