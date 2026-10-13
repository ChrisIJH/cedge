"""Unit tests for apps/cedge_mcp/summarize.py. No network, no MCP SDK needed."""
from __future__ import annotations

import pytest

from apps.cedge_mcp import summarize

CELL = {  # shape of one /api/full_backtest cell
    "method": "parametric", "alpha": 0.01, "eval_idx": [0, 1], "r_eval": [0.0, -0.05],
    "var": [0.02, 0.02], "es": [0.03, 0.03], "breaches": [0, 1],
    "kupiec": {"test": "Kupiec UC", "LR": 3.555, "df": 1, "crit_5pct": 3.841,
               "p_value": 0.0594, "reject": False, "N": 6, "T": 250,
               "breach_rate": 0.024, "expected_rate": 0.01},
    "christoffersen": {"test": "Christoffersen CC", "LR_UC": 3.555, "LR_IND": 0.3, "LR_CC": 3.855,
                       "df": 2, "crit_5pct": 5.991, "p_value": 0.1455, "reject": False,
                       "pi_01": 0.02, "pi_11": 0.0,
                       "transitions": {"n00": 238, "n01": 6, "n10": 5, "n11": 0}},
    "acerbi": {"test": "Acerbi-Szekely Z2 (fixed)", "Z2": -0.2, "p_value": 0.3,
               "reject": False, "n_breach": 6, "boot_mean": 0.0},
    "avg_pred_es": 0.03, "avg_real_loss": 0.036,
}


def test_summarize_cell_known_answer():
    s = summarize.summarize_backtest_cell(CELL)
    assert s["confidence"] == 0.99
    assert s["n_observations"] == 250 and s["n_breaches"] == 6
    assert s["expected_breaches"] == 2.5
    assert s["kupiec"] == {"LR": 3.555, "p_value": 0.0594, "reject_5pct": False}
    assert s["christoffersen"]["consecutive_breaches_n11"] == 0
    assert s["es_ratio_realized_to_predicted"] == pytest.approx(1.2)


def test_summarize_cell_drops_daily_arrays():
    s = summarize.summarize_backtest_cell(CELL)
    for key in ("r_eval", "var", "es", "breaches", "eval_idx"):
        assert key not in s


def test_summarize_cell_nan_becomes_none_and_no_zero_division():
    cell = {**CELL, "kupiec": {**CELL["kupiec"], "p_value": float("nan")}, "avg_pred_es": 0.0}
    s = summarize.summarize_backtest_cell(cell)
    assert s["kupiec"]["p_value"] is None
    assert s["es_ratio_realized_to_predicted"] is None


def test_summarize_weights():
    s = summarize.summarize_weights({"A": 0.5, "B": -0.3, "C": 0.2}, top_n=2)
    assert s["n_long"] == 2 and s["n_short"] == 1
    assert s["gross"] == pytest.approx(1.0) and s["net"] == pytest.approx(0.4)
    assert [p["ticker"] for p in s["top_positions_by_abs_weight"]] == ["A", "B"]