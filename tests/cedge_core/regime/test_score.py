import math

import pandas as pd
import pytest
from cedge_core.regime.score import (
    composite_score,
    credit_asym,
    pc1_score,
    s_curve_penalty,
)


def test_s_curve_penalty_neutral():
    assert s_curve_penalty(0.0) == pytest.approx(-10.0)


def test_s_curve_penalty_none():
    assert s_curve_penalty(None) == 0.0


def test_s_curve_penalty_saturates():
    assert s_curve_penalty(100.0) == pytest.approx(-20.0, abs=1e-6)
    assert s_curve_penalty(-100.0) == pytest.approx(0.0, abs=1e-6)


def test_credit_asym_risk_on_and_off():
    assert credit_asym(1.0) == pytest.approx(5.0)
    assert credit_asym(-1.0) == pytest.approx(-15.0)
    assert credit_asym(None) == 0.0


def test_pc1_score_perfectly_correlated_columns():
    z_history = pd.DataFrame({"vix_z": [1.0, 2.0, 3.0, 4.0],
                               "other_z": [1.0, 2.0, 3.0, 4.0]})
    result = pc1_score(z_history, anchor_col="vix_z")
    assert result == pytest.approx(4 * math.sqrt(2))


def test_pc1_score_anchor_sign_convention():
    # vix_z and other_z anti-correlated; anchor loading forced positive
    z_history = pd.DataFrame({"vix_z": [4.0, 3.0, 2.0, 1.0],
                               "other_z": [1.0, 2.0, 3.0, 4.0]})
    result = pc1_score(z_history, anchor_col="vix_z")
    assert result == pytest.approx(-3 / math.sqrt(2))


def test_composite_score_neutral_inputs_is_40_not_50():
    # Known issue: fully neutral macro backdrop still scores below 50.
    score = composite_score(vix_z=0.0, dxy_z=0.0, ief_z=0.0, credit_z=0.0, pca=0.0)
    assert score == pytest.approx(40.0)


def test_composite_score_credit_sign_invariance_bug():
    # Known issue: credit_z=+1 (risk-on) and credit_z=-1 (risk-off)
    # produce the IDENTICAL score — the combined credit term is -5*|z|
    # regardless of sign.
    score_pos = composite_score(vix_z=0.0, dxy_z=0.0, ief_z=0.0, credit_z=1.0, pca=0.0)
    score_neg = composite_score(vix_z=0.0, dxy_z=0.0, ief_z=0.0, credit_z=-1.0, pca=0.0)
    assert score_pos == pytest.approx(35.0)
    assert score_neg == pytest.approx(35.0)
    assert score_pos == score_neg