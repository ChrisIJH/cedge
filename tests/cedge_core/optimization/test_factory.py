import numpy as np
import pytest
from cedge_core.optimization.factory import solve
from cedge_core.optimization.mean_variance import OptimizeResult


def test_dispacth_to_mean_variance():
    """Factory correctly dispatches to mean_variance"""
    cov = np.eye(2) * 0.01
    w_prev = np.zeros(2)

    result = solve("mean_variance", cov=cov, w_prev=w_prev)

    assert isinstance(result, OptimizeResult)


def test_unknown_strategy_raises():
    """Unknown strategy name raises ValueError with available list."""
    with pytest.raises(ValueError, match="unknown strategy.*available:.*mean_variance"):
        solve("nonexistent")