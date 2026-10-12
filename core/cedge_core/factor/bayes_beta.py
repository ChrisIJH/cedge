"""
cedge_core/factor/bayes_beta.py

Closed-form Bayesian ridge regression for per-ticker factor loadings.
Ported from CHResearch's ch_api/ch_factor/bayes_factor_model.py.

Model: y = X @ beta + eps, eps ~ N(0, sigma2 * I)
Prior: beta ~ N(0, tau2 * I)  -- isotropic, ZERO-MEAN, shared across all
factors and all tickers (tau2 is a fixed hyperparameter)
This is NOT Vasicek-style shrinkage
toward a cross-sectional mean beta: every factor loading is pulled
toward 0, and the shrinkage *strength* varies per ticker only through
that ticker's own X^T X / sigma2 (its regression precision) -- not
through a per-ticker prior. Structurally this is a multivariate Bayesian
ridge regression, not a univariate normal-normal shrinkage of a single
beta.

"""