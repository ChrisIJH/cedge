"""
apps/cedge_mcp/summarize.py

Pure functions that shrink service responses before they reach an LLM.
No HTTP, no cedge_core imports.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List


def _num(x: Any) -> Any:
    """JSON-safe number: NaN/inf become None."""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def summarize_backtest_cell(cell: Dict[str, Any]) -> Dict[str, Any]:
    """One /api/full_backtest cell -> compact dict.
    Keys follow cedge_core.risk.var_backtest_model:
      kupiec: LR, p_value, reject, N (breaches), T (observations), breach_rate
      christoffersen: LR_CC, p_value, reject, transitions.n11
      acerbi: Z2, p_value, reject
    """
    k, c, a = cell["kupiec"], cell["christoffersen"], cell["acerbi"]
    alpha = float(cell["alpha"])
    n_obs, n_breach = int(k["T"]), int(k["N"])
    avg_pred_es = cell.get("avg_pred_es")
    avg_real_loss = cell.get("avg_real_loss")
    es_ratio = None
    if isinstance(avg_pred_es, (int, float)) and avg_pred_es and isinstance(avg_real_loss, (int, float)):
        es_ratio = avg_real_loss / avg_pred_es

    return {
        "method": cell["method"],
        "alpha": alpha,
        "confidence": round(1.0 - alpha, 6),
        "n_observations": n_obs,
        "n_breaches": n_breach,
        "expected_breaches": round(alpha * n_obs, 4),
        "breach_rate": _num(k["breach_rate"]),
        "kupiec": {"LR": _num(k["LR"]), "p_value": _num(k["p_value"]), "reject_5pct": bool(k["reject"])},
        "christoffersen": {
            "LR_CC": _num(c["LR_CC"]), "p_value": _num(c["p_value"]),
            "reject_5pct": bool(c["reject"]),
            "consecutive_breaches_n11": int(c["transitions"]["n11"]),
        },
        "acerbi_szekely": {"Z2": _num(a["Z2"]), "p_value": _num(a["p_value"]), "reject_5pct": bool(a["reject"])},
        "avg_predicted_es": _num(avg_pred_es),
        "avg_realized_loss_on_breach": _num(avg_real_loss),
        "es_ratio_realized_to_predicted": _num(es_ratio),
    }


def summarize_backtest(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [summarize_backtest_cell(c) for c in cells]


def summarize_weights(weights: Dict[str, float], top_n: int = 10) -> Dict[str, Any]:
    """Book composition without dumping every position."""
    items = sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return {
        "n_positions": len(weights),
        "n_long": sum(1 for w in weights.values() if w > 0),
        "n_short": sum(1 for w in weights.values() if w < 0),
        "gross": sum(abs(w) for w in weights.values()),
        "net": sum(weights.values()),
        "top_positions_by_abs_weight": [{"ticker": t, "weight": w} for t, w in items[:top_n]],
    }