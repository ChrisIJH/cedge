import pandas as pd
import pytest
from cedge_core.regime.backtest import (
    bucket_stats_by_label,
    forward_returns,
    label_from_posterior,
)


def test_forward_returns_h1_is_same_day_return():
    ret = pd.Series([0.0, 0.01, 0.02, -0.01, 0.03])
    fwd = forward_returns(ret, horizons=(1,))
    assert fwd["fwd_1"].tolist() == pytest.approx(ret.tolist())


def test_forward_returns_h2_compounds_two_days():
    ret = pd.Series([0.0, 0.01, 0.02, -0.01, 0.03])
    fwd = forward_returns(ret, horizons=(2,))
    expected = [0.01, 0.0302, 0.0098, 0.0197, None]
    for got, exp in zip(fwd["fwd_2"].tolist(), expected):
        if exp is None:
            assert pd.isna(got)
        else:
            assert got == pytest.approx(exp, abs=1e-6)


def test_label_from_posterior():
    assert label_from_posterior({"risk_off": 0.1, "soft_patch": 0.7, "risk_on": 0.2}) == "soft_patch"
    assert label_from_posterior({"risk_off": 0.8, "soft_patch": 0.15, "risk_on": 0.05}) == "risk_off"


def test_bucket_stats_by_label_known_values():
    regime_history = pd.DataFrame({
        "posterior_risk_off":   [0.8, 0.7, 0.05, 0.2],
        "posterior_soft_patch": [0.15, 0.2, 0.05, 0.6],
        "posterior_risk_on":    [0.05, 0.1, 0.9, 0.2],
    })
    fwd_returns_df = pd.DataFrame({"fwd_1": [0.01, -0.02, 0.03, 0.00]})

    stats = bucket_stats_by_label(regime_history, fwd_returns_df)

    assert stats.loc["risk_off", "fwd_1_mean"] == pytest.approx(-0.005)
    assert stats.loc["risk_off", "fwd_1_count"] == 2
    assert stats.loc["risk_on", "fwd_1_mean"] == pytest.approx(0.03)
    assert stats.loc["risk_on", "fwd_1_count"] == 1
    assert stats.loc["soft_patch", "fwd_1_mean"] == pytest.approx(0.0)
    assert stats.loc["soft_patch", "fwd_1_count"] == 1