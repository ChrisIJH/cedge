"""
cedge_core/regime/posterior.py

Maps a 0-100 regime score to a posterior over {risk_off, soft_patch,
risk_on} under a FIXED 3-component Gaussian mixture — not a fitted or
learned posterior. Every parameter (component means, shared sigma,
prior) is a literal constant; "Bayesian" here describes the update rule
(likelihood x prior, normalized), not a trained model.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

_LABELS = ("risk_off", "soft_patch", "risk_on")
_DEFAULT_MUS = (20.0, 50.0, 80.0)
_DEFAULT_SIGMA = 15.0

def mixture_posterior(score: Optional[float],
                       mus: Sequence[float] = _DEFAULT_MUS,
                       sigma: float = _DEFAULT_SIGMA,
                       prior: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Returns a uniform posterior (1/3 each) if `score` is None/NaN, or if
    all three component likelihoods underflow to zero.
    """
    prior = prior or {label: 1 / 3 for label in _LABELS}

    if score is None or np.isnan(score):
        return {label: 1 / 3 for label in _LABELS}

    def gaussian_pdf(x: float, mu: float, sd: float) -> float:
        return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))

    unnorm = {
        label: prior[label] * gaussian_pdf(score, mu, sigma)
        for label, mu in zip(_LABELS, mus)
    }
    total = sum(unnorm.values())
    if total <= 0 or np.isnan(total):
        return {label: 1 / 3 for label in _LABELS}

    return {label: v / total for label, v in unnorm.items()}