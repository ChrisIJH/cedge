"""
cedge_core/regime/model.py

Orchestrates signals -> score -> posterior -> policy for a single as-of
date, given a wide macro price panel. Pure function — no DB access (see
regime/prices.py for the repository supplying `px_macro`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from cedge_core.regime.policy import policy_from_posterior
from cedge_core.regime.posterior import mixture_posterior
from cedge_core.regime.score import ScoreWeights, composite_score, pc1_score
from cedge_core.regime.signals import (
    credit_ratio,
    rolling_zscore_series,
    trailing_zscore,
)

_ZSCORE_WINDOW = 252


@dataclass(frozen=True)
class RegimeResult:
    score: float
    posterior: Dict[str, float]
    policy: Dict[str, float]
    vix_z: Optional[float]
    dxy_z: Optional[float]
    ief_z: Optional[float]
    credit_z: Optional[float]
    pca_factor: Optional[float]


def compute_regime(px_macro: pd.DataFrame,
                    weights: Optional[ScoreWeights] = None,
                    window: int = _ZSCORE_WINDOW) -> RegimeResult:
    """
    px_macro: wide DataFrame, index=date, columns >= {'^VIX','UUP','HYG','IEF'},
    values=adj_close_price. Needs >= `window` trading days of history for
    z-scores to resolve (else terms fall back to 0/None).
    """
    weights = weights or ScoreWeights()
    df = px_macro.dropna()

    vix_z = trailing_zscore(df["^VIX"], window=window)
    dxy_z = trailing_zscore(df["UUP"], window=window)
    ief_z = trailing_zscore(df["IEF"], window=window)

    ratio = credit_ratio(df["HYG"], df["IEF"])
    credit_z = trailing_zscore(ratio, window=window)

    z_history = pd.DataFrame({
        "vix_z": rolling_zscore_series(df["^VIX"], window=window),
        "dxy_z": rolling_zscore_series(df["UUP"], window=window),
        "ief_z": rolling_zscore_series(df["IEF"], window=window),
        "credit_z": rolling_zscore_series(ratio, window=window),
    }).dropna()

    pca = pc1_score(z_history, anchor_col="vix_z")

    score = composite_score(vix_z, dxy_z, ief_z, credit_z, pca, weights)
    posterior = mixture_posterior(score)
    policy = policy_from_posterior(posterior)

    return RegimeResult(
        score=score, posterior=posterior, policy=policy,
        vix_z=vix_z, dxy_z=dxy_z, ief_z=ief_z, credit_z=credit_z,
        pca_factor=pca,
    )