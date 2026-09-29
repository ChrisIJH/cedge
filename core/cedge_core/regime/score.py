"""
cedge_core/regime/score.py

Pure functions combining macro z-scores into a single 0-100 composite
regime score. Ported as-is from the original repo's ch_api/ch_metric.py
(`regime_dashboard`), including its known issues — see composite_score
docstring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def s_curve_penalty(z: Optional[float], amplitude: float = 20.0,
                     slope: float = 1.2, center: float = 0.0) -> float:
    """Logistic penalty, negative in the risk-off direction (z > center).
    0.0 at z=None; approaches -amplitude as z -> +inf, 0 as z -> -inf."""
    if z is None:
        return 0.0
    val = 1.0 / (1.0 + np.exp(-slope * (z - center)))
    return float(-amplitude * val)


def credit_asym(z: Optional[float], widen_weight: float = 15.0,
                 tighten_weight: float = 5.0) -> float:
    """
    Asymmetric credit-spread term: rising HYG/IEF ratio (z >= 0, credit
    tightening / risk-on) contributes +tighten_weight * z; falling ratio
    (z < 0, credit widening / risk-off) contributes -widen_weight * |z|.
    """
    if z is None:
        return 0.0
    if z >= 0:
        return tighten_weight * z
    return -widen_weight * abs(z)

def pc1_score(z_history: pd.DataFrame, 
              anchor_col: str = "vix_z") -> Optional[float]:
    """
    First principal component of a macro z-score panel, projected onto
    the most recent row. Sign is fixed so `anchor_col`'s loading is
    positive.
    """
    df = z_history.dropna(how="any")
    if df.empty:
        return None

    x = df.to_numpy()
    cov = np.cov(x, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    pc1_vec = eigvecs[:, np.argmax(eigvals)]

    if anchor_col in df.columns:
        anchor_idx = df.columns.get_loc(anchor_col)
        if pc1_vec[anchor_idx] < 0:
            pc1_vec = -pc1_vec

    norm = np.linalg.norm(pc1_vec)
    if norm > 0:
        pc1_vec = pc1_vec / norm

    score = float(np.dot(x[-1, :], pc1_vec))
    if np.isnan(score) or np.isinf(score):
        return None
    return score


@dataclass(frozen=True)
class ScoreWeights:
    """Linear weights on each macro z-score, plus the PCA term weight and
    base score. Matches CHResearch's `regime_dashboard` defaults exactly."""
    vix: float = -12.0
    dxy: float = -8.0
    credit: float = -10.0
    ief: float = -6.0
    pca: float = -6.0
    base: float = 50.0


def composite_score(vix_z: Optional[float], dxy_z: Optional[float],
                     ief_z: Optional[float], credit_z: Optional[float],
                     pca: Optional[float],
                     weights: ScoreWeights = ScoreWeights()) -> float:
    """
    0-100 composite regime score: base + linear z-score terms + credit
    asymmetry + VIX s-curve penalty + PCA term, clamped to [0, 100].

    """
    lin = (weights.vix * (vix_z or 0.0)
           + weights.dxy * (dxy_z or 0.0)
           + weights.credit * (credit_z or 0.0)
           + weights.ief * (ief_z or 0.0))

    score = (weights.base
             + lin
             + credit_asym(credit_z)
             + s_curve_penalty(vix_z)
             + weights.pca * (pca or 0.0))

    return float(max(0.0, min(100.0, score)))
    