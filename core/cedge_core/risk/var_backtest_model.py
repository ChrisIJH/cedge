from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class BacktestConfig:
    alpha: float = 0.05     # VaR level
    crit_uc: float=3.841    # chi2(1) 95%
    cirt_cc: float=5.99     # chi2(2) 95%

def lruc(N, T, alpha=0.05):
    p_hat = N / T if T else 0.0
    if N == 0:
        lr = -2.0 * (T * np.log(1 - alpha))
    elif N == T:
        lr = -2.0 * (T * np.log(alpha))
    else:
        ll_null = N * np.log(alpha) + (T - N) * np.log(1 - alpha)
        ll_alt = N * np.log(p_hat) + (T - N) * np.log(1 - p_hat)
        lr = -2.0 * (ll_null - ll_alt)
    return lr

def kupiec_uc(breaches: np.ndarray, alpha: float=0.05) -> Dict[str, object]:
    """
    H0: p = alpha
    LR_UC = -2 ln[ (1-alpha)^(T-N) alpha^N / (1-p_hat)^(T-N) p_hat^N ] ~ chi2(1)
    p_hat = N/T  (= breach's MLE)
    """
    breaches = np.asarray(breaches).astype(int)
    T = len(breaches)
    N = int(breaches.sum())

    p_hat = N / T if T else 0.0

    lr = lruc(N, T, alpha)

    return {
        "test": "Kupiec UC",
        "LR": lr,
        "df": 1,
        "crit_5pct": 3.841,
        "p_value": float(1 - stats.chi2.cdf(lr, df=1)),
        "reject": bool(lr > 3.841),
        "N": N,
        "T": T,
        "breach_rate": p_hat,
        "expected_rate": alpha,
    }

def christoffersen_cc(breaches: np.ndarray, alpha: float=0.05) -> Dict[str, object]:
    """
    Kupiec + clustering 

    n_ij = (yesterday i -> today j).
    LR_IND: pi vs pi_01, pi_11 
    LR_CC = LR_UC + LR_IND ~ chi2(2), reject when >5.991.
    """
    b = np.asarray(breaches).astype(int)
    T = len(b)
    prev, cur = b[:-1], b[1:]
    n00 = int(np.sum((prev==0)&(cur==0)))
    n01 = int(np.sum((prev==0)&(cur==1)))
    n10 = int(np.sum((prev==1)&(cur==0)))
    n11 = int(np.sum((prev==1)&(cur==1)))

    pi01 = n01/(n01+n00) if (n00+n01) > 0 else 0.0
    pi11 = n11/(n11+n10) if (n10+n11) > 0 else 0.0
    pi = (n10 + n11) / ( n00 + n01 + n10 + n11) if T > 1 else 0.0

    def _xlogx(n: int, p: float):
        return -2 * n * np.log(p) if (n > 0 and p > 0) else 0.0

    ll_ind_null = _xlogx(n00 + n10, 1 - pi) + _xlogx(n01 + n11, pi)
    ll_ind_alt = (_xlogx(n00, 1 - pi01) + _xlogx(n01, pi01)
                  + _xlogx(n10, 1 - pi11) + _xlogx(n11, pi11))
    lr_ind = (ll_ind_null - ll_ind_alt)

    uc = kupiec_uc(b, alpha)
    lr_cc = uc["LR"] + lr_ind

    return {
        "test": "Christoffersen CC",
        "LR_UC": uc["LR"],
        "LR_IND": lr_ind,
        "LR_CC": lr_cc,
        "df": 2,
        "crit_5pct": 5.991,
        "p_value": float(1 - stats.chi2.cdf(lr_cc, df=2)),
        "reject": bool(lr_cc > 5.991),
        "pi_01": pi01,
        "pi_11": pi11,
        "transitions": {"n00": n00, "n01": n01, "n10": n10, "n11": n11},
    }


def acerbi_szekely_z2(returns, var_est, es_est, alpha=0.05, n_boot=10000, seed=42):
    """
    ES(ETL) check.
    Real return / ES

    Z2 = (1/T) sum_{t: breach} r_t / (alpha * ES_t) + 1
    H0에서 E[Z2] ~ 0. 
    Z2 < 0 reject. --> Loss > es --> expected loss not enought
    Z2 > 0 reject --> loos < es --> expected loss too much
    """
    r = np.asarray(returns, float)
    var_est = np.asarray(var_est, float)
    es_est = np.asarray(es_est, float)
    T = len(r)

    def _z2(rr):
        br = rr < -var_est
        if not br.any():
            return 0.0
        return float(np.sum(rr[br] / (alpha * es_est[br])) / T + 1.0)

    z2_obs = _z2(r)

    rng = np.random.default_rng(seed)
    scale = np.maximum(es_est - var_est, 1e-12)   # Exp mean of tail excess (>0)
    noise_scale = np.maximum(var_est * 0.5, 1e-12)
    z2_boot = np.empty(n_boot)
    for i in range(n_boot):
        is_b = rng.random(T) < alpha
        loss = np.zeros(T)
        loss[is_b] = var_est[is_b] + rng.exponential(scale[is_b])   # all >= VaR
        sim_r = np.where(is_b, -loss, np.abs(rng.normal(0, noise_scale)))
        z2_boot[i] = _z2(sim_r)

    # two-sided would be cleaner, but keep left-tail (ES-too-small) consistent w/ original
    p = float(np.mean(z2_boot <= z2_obs))
    return {"test": "Acerbi-Szekely Z2 (fixed)", "Z2": z2_obs, "p_value": p,
            "reject": bool(p < 0.05), "n_breach": int((r < -var_est).sum()),
            "boot_mean": float(z2_boot.mean())}





def run_backtest(
    returns: np.ndarray,
    var_est: np.ndarray,
    es_est: Optional[np.ndarray] = None,
    alpha: float = 0.05,
) -> Dict[str, object]:
    """returns vs rolling VaR/ES 예측 -> 세 검정 한 번에."""
    r = np.asarray(returns, float)
    var_est = np.asarray(var_est, float)
    breaches = (r < -var_est).astype(int)

    out: Dict[str, object] = {
        "alpha": alpha,
        "kupiec": kupiec_uc(breaches, alpha),
        "christoffersen": christoffersen_cc(breaches, alpha),
        "breaches": breaches,
    }
    if es_est is not None:
        out["acerbi_szekely"] = acerbi_szekely_z2(r, var_est, np.asarray(es_est, float), alpha)
    return out
