"""
cedge_core/regime/policy.py

Maps a regime posterior to portfolio policy limits. Ported as-is from
CHResearch's `regime_dashboard`; NOT reconciled with the optimizer's
consumer, which expects a different scale.
"""
from __future__ import annotations

from typing import Dict


def policy_from_posterior(posterior: Dict[str, float],
                           gross_base: float = 100.0, gross_weight: float = 40.0,
                           beta_base: float = 0.3, beta_weight: float = 0.4) -> Dict[str, float]:
    """
    gross_target: percent-of-NAV units (100.0 = fully invested gross
      exposure), range ~[gross_base - gross_weight, gross_base + gross_weight].
    beta_cap: beta units (dimensionless), range ~[beta_base - beta_weight,
      beta_base + beta_weight] — can go negative when risk_off dominates;
      not clamped at zero here (ported as-is).
    """
    tilt = posterior.get("risk_on", 0.0) - posterior.get("risk_off", 0.0)
    return {
        "gross_target": float(gross_base + gross_weight * tilt),
        "beta_cap": float(beta_base + beta_weight * tilt),
    }