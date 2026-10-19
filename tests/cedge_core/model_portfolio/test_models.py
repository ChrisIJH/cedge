import pytest
from cedge_core.model_portfolio.models import MOD_6040, MOD_8020, ModelPortfolio


def test_shipped_models_sum_to_one():
    for m in (MOD_6040, MOD_8020):
        assert sum(m.weights.values()) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("weights", [{}, {"A": 0.5, "B": 0.4}, {"A": 1.2, "B": -0.2},
                                     {"A": float("nan"), "B": 1.0}])
def test_bad_model_rejected(weights):
    with pytest.raises(ValueError):
        ModelPortfolio("BAD", "0", weights)