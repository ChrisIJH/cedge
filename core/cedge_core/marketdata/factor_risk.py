"""
cedge_core/marketdata/factor_risk.py

The three tables a factor risk model is assembled from:
factor_betas_bayes (B), factor_covariance (Sigma_f), factor_resid_var (D).

Loading only — the assembly itself is
cedge_core.optimization.covariance.factor_model_covariance.
"""
from __future__ import annotations

from typing import List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from cedge_core.db import ch_engine

DEFAULT_LOOKBACK = 252
DEFAULT_MODEL_VERSION = "bayes_ridge_v1"
DEFAULT_COV_METHOD = "ledoit_wolf"
DEFAULT_RESID_METHOD = "ols_sigma2"

class FactorRiskRepository(Protocol):
    """Port — the three loads a factor risk model needs."""

    def load_factor_betas(self, asof_date: str, tickers: List[str], *,
                          lookback_window: int, model_version: str) -> pd.DataFrame:
        ...

    def load_factor_cov(self, asof_date: str, *, lookback_window: int,
                        model_version: str, method: str) -> pd.DataFrame:
        ...

    def load_resid_var(self, asof_date: str, tickers: List[str], *,
                       lookback: int, method: str) -> pd.Series:
        ...


class SqlFactorRiskRepository:
    """Adapter — reads the factor tables through whatever engine is given."""
    def __init__(self, engine: Optional[Engine] = None):
        self._engine = engine

    def _resolve_engine(self) -> Engine:
        return self._engine or ch_engine()

    def load_factor_betas(
            self,
            asof_date: str,
            tickers: List[str],
            lookback_window: int = DEFAULT_LOOKBACK,
            model_version: str = DEFAULT_MODEL_VERSION,
    ) -> pd.DataFrame:
        """B: index=ticker, columns=factor_name, values=beta_mean."""
        if not tickers:
            raise ValueError("tickers empty")

        placeholders = ", ".join(f":t{i}" for i in range(len(tickers)))
        sql = text(f"""
            select ticker, factor_name, beta_mean
            from factor_betas_bayes
            where asof_date = :asof
              and lookback_window = :lookback
              and model_version = :model_version
              and ticker in ({placeholders})
            """)
        params = {"asof": asof_date, "lookback": int(lookback_window),
                  "model_version": model_version, 
                  **{f"t{i}": t for i, t in enumerate(tickers)}}
        
        with self._resolve_engine().begin() as conn:
            df = pd.read_sql(sql, conn, params=params)
        if df.empty:
            raise LookupError(
                f"no factor betas for asof_date={asof_date} "
                f"lookback_window={lookback_window} model_version={model_version!r}"
            )

        betas = df.pivot_table(index="ticker", columns="factor_name",
                               values="beta_mean", aggfunc="last")

        return betas.sort_index().sort_index(axis=1).astype(float)

    def load_factor_cov(
            self, asof_date: str, *,
            lookback_window: int = DEFAULT_LOOKBACK,
            model_version: str = DEFAULT_MODEL_VERSION,
            method: str = DEFAULT_COV_METHOD,
        ) -> pd.DataFrame:
            """Sigma_f, fully symmetric.

            factor_covariance stores one triangle only (verified: 21 rows for 6
            factors, no mirrored pairs). Both sides are written explicitly here
            rather than symmetrising with 0.5 * (S + S.T) — against
            triangle-only storage that expression leaves every off-diagonal NaN.
            """
            sql = """
                select factor_i, factor_j, cov_ij
                from factor_covariance
                where asof_date = :asof
                and lookback_window = :lookback
                and model_version = :model_version
                and method = :method
            """
            params = {"asof": asof_date, "lookback": int(lookback_window),
                    "model_version": model_version, "method": method}
            with self._resolve_engine().begin() as conn:
                df = pd.read_sql(text(sql), conn, params=params)

            if df.empty:
                raise LookupError(
                    f"no factor covariance for asof_date={asof_date} "
                    f"lookback_window={lookback_window} "
                    f"model_version={model_version!r} method={method!r}"
                )

            factors = sorted(set(df["factor_i"]) | set(df["factor_j"]))
            sigma_f = pd.DataFrame(0.0, index=factors, columns=factors)
            for factor_i, factor_j, cov_ij in df.itertuples(index=False):
                sigma_f.loc[factor_i, factor_j] = float(cov_ij)
                sigma_f.loc[factor_j, factor_i] = float(cov_ij)

            variances = pd.Series(np.diag(sigma_f.to_numpy()), index=factors)
            degenerate = variances[variances <= 0.0]
            if not degenerate.empty:
                raise LookupError(
                    f"factor_covariance has no positive variance for "
                    f"{list(degenerate.index)} on {asof_date} — Sigma_f would be singular"
                )
            return sigma_f

    def load_resid_var(
        self, asof_date: str, tickers: List[str], *,
        lookback: int = DEFAULT_LOOKBACK,
        method: str = DEFAULT_RESID_METHOD,
    ) -> pd.Series:
        """D's diagonal: index=ticker, values=resid_var."""
        if not tickers:
            raise ValueError("tickers is empty")
        placeholders = ", ".join(f":t{i}" for i in range(len(tickers)))

        sql = text(f"""
            select ticker, resid_var
            from factor_resid_var
            where asof_date = :asof
              and method_ = :method
              and lookback = :lookback
              and ticker in ({placeholders})
        """)
        params = {"asof": asof_date, "method": method,
                  "lookback": int(lookback), 
                  **{f"t{i}": t for i, t in enumerate(tickers)}}
        with self._resolve_engine().begin() as conn:
            df = pd.read_sql(sql, conn, params=params)

        if df.empty:
            raise LookupError(
                f"no residual variance for asof_date={asof_date} "
                f"lookback={lookback} method_={method!r}"
            )
        return (df.drop_duplicates("ticker").set_index("ticker")["resid_var"]
                  .astype(float).sort_index())

    def load_aligned(
        self, asof_date: str, tickers: List[str], *,
        lookback_window: int = DEFAULT_LOOKBACK,
        model_version: str = DEFAULT_MODEL_VERSION,
        cov_method: str = DEFAULT_COV_METHOD,
        resid_method: str = DEFAULT_RESID_METHOD,
        min_assets: int = 2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """(B, Sigma_f, resid_var) with indexes already intersected.

        factor_model_covariance rejects misaligned inputs rather than
        intersecting them itself.
        """
        betas = self.load_factor_betas(
            asof_date, tickers,
            lookback_window=lookback_window, model_version=model_version)
        sigma_f = self.load_factor_cov(
            asof_date, lookback_window=lookback_window,
            model_version=model_version, method=cov_method)
        resid_var = self.load_resid_var(
            asof_date, tickers, lookback=lookback_window, method=resid_method)

        factors = sorted(set(betas.columns) & set(sigma_f.index))
        if not factors:
            raise LookupError(
                f"no factor overlap between factor_betas_bayes "
                f"{sorted(betas.columns)} and factor_covariance {sorted(sigma_f.index)}"
            )
        betas = betas.loc[:, factors]

        # A ticker is written with all its factors in one pass, so a partial
        # row means the write itself was incomplete — drop it rather than
        # imputing a zero exposure.
        complete = betas.dropna(axis=0, how="any")
        common = sorted(set(complete.index) & set(resid_var.index))
        if len(common) < min_assets:
            raise LookupError(
                f"only {len(common)} ticker(s) survive alignment on {asof_date} "
                f"(min_assets={min_assets}); requested={len(tickers)}, "
                f"betas={len(betas)}, complete_beta_rows={len(complete)}, "
                f"resid_var={len(resid_var)}"
            )
        return complete.loc[common, factors], sigma_f.loc[factors, factors], resid_var.loc[common]