from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from cedge_core.risk import var_es_model
from cedge_core.risk.var_es_model import (
    _cell_summary,
    backtest_one,
    full_backtest,
    historic_es,
    historic_var,
    multi_backtest,
    parametric_es,
    parametric_var,
    rolling_var_es,
)
from scipy import stats

# ---------------------------------------------------------------------------
# Point estimators (parametric / historic)
# ---------------------------------------------------------------------------

class TestParametricVarEs:
    def test_matches_closed_form_normal(self):
        """N(0, sigma) 표본 -> VaR/ES가 이론값에 수렴 (대표본)."""
        rng = np.random.default_rng(0)
        sigma, alpha = 0.02, 0.05
        w = rng.normal(0, sigma, 200_000)
        z = stats.norm.ppf(alpha)

        var = parametric_var(w, alpha)
        es = parametric_es(w, alpha)

        assert var == pytest.approx(-z * sigma, rel=0.05)
        assert es == pytest.approx(sigma * stats.norm.pdf(z) / alpha, rel=0.05)

    def test_es_ge_var_for_normal(self):
        """정규분포 하에서 ES(더 깊은 꼬리 평균) >= VaR (같은 부호, loss 기준)."""
        rng = np.random.default_rng(1)
        w = rng.normal(0, 0.01, 5_000)
        assert parametric_es(w) >= parametric_var(w)

    def test_shift_mean_left_increases_both(self):
        """평균 수익률이 나빠지면(더 마이너스) VaR/ES(손실) 모두 커져야 함."""
        rng = np.random.default_rng(2)
        base = rng.normal(0, 0.01, 5_000)
        bad = base - 0.05  # 평행이동으로 손실 확대

        assert parametric_var(bad) > parametric_var(base)
        assert parametric_es(bad) > parametric_es(base)


class TestHistoricVarEs:
    def test_var_matches_empirical_quantile(self):
        w = np.array([-0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.04])
        alpha = 0.05
        expected = -np.quantile(w, alpha)
        assert historic_var(w, alpha) == pytest.approx(expected)

    def test_es_is_mean_of_left_tail_beyond_var(self):
        """ES = -mean(w[w < quantile]); tail 평균이 quantile 자체보다 더 나쁨(<=)."""
        rng = np.random.default_rng(3)
        w = rng.normal(0, 0.02, 2_000)
        alpha = 0.05
        var = historic_var(w, alpha)
        es = historic_es(w, alpha)
        assert es >= var  # tail 평균 손실 >= quantile 손실

    def test_es_falls_back_to_quantile_when_tail_empty(self):
        """분위수보다 엄격히 작은 관측치가 하나도 없으면 (이산 데이터) -a로 폴백."""
        w = np.array([0.0, 0.0, 0.0, 0.0])  # quantile == 0, tail(<0) 비어있음
        alpha = 0.5
        a = np.quantile(w, alpha)
        assert historic_es(w, alpha) == pytest.approx(-a)

    def test_alpha_moves_monotonically(self):
        """alpha가 작을수록(더 깊은 꼬리) VaR/ES(손실)가 커지거나 같아야 함."""
        rng = np.random.default_rng(4)
        w = rng.normal(0, 0.02, 5_000)
        assert historic_var(w, 0.01) >= historic_var(w, 0.05)
        assert historic_es(w, 0.01) >= historic_es(w, 0.05)


# ---------------------------------------------------------------------------
# rolling_var_es — look-ahead 차단이 핵심 계약
# ---------------------------------------------------------------------------

class TestRollingVarEs:
    def test_output_shapes_and_index_alignment(self):
        window, n = 50, 200
        r = np.random.default_rng(5).normal(0, 0.01, n)
        idx, var, es = rolling_var_es(r, window, 0.05, "historic")

        assert list(idx) == list(range(window, n))
        assert len(var) == len(idx)
        assert len(es) == len(idx)

    def test_no_lookahead_spike_excluded_until_next_step(self):
        """t 시점의 spike는 idx==t 평가엔 안 보이고, idx==t+1부터 window에 들어와야 함."""
        window, n, spike_pos = 50, 200, 100
        rng = np.random.default_rng(6)
        r = rng.normal(0, 0.001, n)  # 작은 노이즈로 spike를 압도적으로 만듦
        r[spike_pos] = -0.9

        idx, var, _ = rolling_var_es(r, window, 0.05, "historic")

        var_before = var[list(idx).index(spike_pos)]      # window = r[50:100], spike 미포함
        var_after = var[list(idx).index(spike_pos + 1)]    # window = r[51:101], spike 포함

        assert var_before < 0.1          # spike 반영 전 -> 정상 수준의 VaR
        assert var_after > var_before    # spike 반영 후 -> VaR 급등

    def test_unknown_method_raises_keyerror(self):
        r = np.random.default_rng(0).normal(0, 0.01, 100)
        with pytest.raises(KeyError):
            rolling_var_es(r, 50, 0.05, "bogus_method")


# ---------------------------------------------------------------------------
# backtest_one — 단일 (method, alpha) 셀
# ---------------------------------------------------------------------------

class TestBacktestOne:
    def test_keys_present(self):
        rng = np.random.default_rng(7)
        r = rng.normal(0, 0.01, 400)
        out = backtest_one(r, window=100, alpha=0.05, method="parametric", n_boot=200)

        for key in ("method", "alpha", "eval_idx", "r_eval", "var", "es",
                    "breaches", "kupiec", "christoffersen", "acerbi",
                    "avg_pred_es", "avg_real_loss"):
            assert key in out

        assert out["method"] == "parametric"
        assert out["alpha"] == 0.05

    def test_breaches_consistent_with_var(self):
        rng = np.random.default_rng(8)
        r = rng.normal(0, 0.01, 400)
        out = backtest_one(r, window=100, alpha=0.05, method="historic", n_boot=200)

        expected = (out["r_eval"] < -out["var"]).astype(int)
        assert np.array_equal(out["breaches"], expected)

    def test_avg_metrics_nan_when_no_breach(self):
        """VaR을 항상 통과하는 초저변동성 구간 -> breach 없음 -> avg_* == nan."""
        r = np.full(400, 0.001)  # 상수 수익률: 분산 0에 가까움, quantile로 손실 없음
        out = backtest_one(r, window=100, alpha=0.05, method="historic", n_boot=50)
        assert out["breaches"].sum() == 0
        assert np.isnan(out["avg_pred_es"])
        assert np.isnan(out["avg_real_loss"])

    def test_good_model_does_not_reject(self):
        """모델이 데이터를 생성한 분포와 일치하면 kupiec/christoffersen 모두 통과해야 함."""
        rng = np.random.default_rng(9)
        r = rng.normal(0, 0.01, 1500)
        out = backtest_one(r, window=250, alpha=0.05, method="parametric", n_boot=500)
        assert out["kupiec"]["reject"] is False
        assert out["christoffersen"]["reject"] is False


# ---------------------------------------------------------------------------
# full_backtest / _cell_summary — grid 및 flatten 계약
# ---------------------------------------------------------------------------

class TestFullBacktest:
    def test_grid_covers_all_method_level_combos(self):
        rng = np.random.default_rng(10)
        r = rng.normal(0, 0.01, 400)
        levels = (0.05, 0.01)
        methods = ("parametric", "historic")
        res = full_backtest(r, window=100, levels=levels, methods=methods, n_boot=100)

        assert set(res.keys()) == {(m, a) for a in levels for m in methods}
        for (m, a), cell in res.items():
            assert cell["method"] == m
            assert cell["alpha"] == a


class TestCellSummary:
    def test_flattens_expected_fields(self):
        rng = np.random.default_rng(11)
        r = rng.normal(0, 0.01, 400)
        cell = backtest_one(r, window=100, alpha=0.05, method="historic", n_boot=200)
        summary = _cell_summary(cell)

        expected_keys = {
            "breach_rate", "breach_n", "kupiec_reject", "christ_reject", "n11",
            "es_real", "es_pred", "es_ratio", "acerbi_Z2", "acerbi_reject",
        }
        assert set(summary.keys()) == expected_keys
        assert isinstance(summary["kupiec_reject"], bool)
        assert isinstance(summary["breach_n"], int)

    def test_es_ratio_nan_when_es_pred_zero_or_nan(self):
        rng = np.random.default_rng(12)
        r = rng.normal(0, 0.01, 400)
        cell = backtest_one(r, window=100, alpha=0.05, method="historic", n_boot=100)
        cell["avg_pred_es"] = float("nan")
        summary = _cell_summary(cell)
        assert np.isnan(summary["es_ratio"])


# ---------------------------------------------------------------------------
# multi_backtest — DB 접근을 mock으로 대체한 통합 테스트
# ---------------------------------------------------------------------------

def _make_price_panel(tickers, n=400, seed=0, start="2020-01-01"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    data = {}
    for i, t in enumerate(tickers):
        rets = rng.normal(0.0002, 0.01, n)
        data[t] = 100 * np.cumprod(1 + rets)
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def patch_price_loader(monkeypatch):
    """load_prices_adjclose를 in-memory panel로 대체 -> DB 없이 multi_backtest 테스트."""
    def _apply(panel):
        monkeypatch.setattr(
            var_es_model, "load_prices_adjclose",
            lambda engine, tickers, start, end: panel[[t for t in tickers if t in panel.columns]],
        )
    return _apply


class TestMultiBacktest:
    def test_single_portfolio_row_shape(self, patch_price_loader):
        panel = _make_price_panel(["AAPL", "MSFT", "SPY"], n=400, seed=1)
        patch_price_loader(panel)

        portfolios = {"book1": {"AAPL": 0.6, "MSFT": -0.4}}
        rows = multi_backtest(portfolios, window=100, n_boot=100, engine=object())

        assert len(rows) == 1
        row = rows[0]
        for key in ("name", "period_start", "period_end", "obs", "gross", "net",
                    "beta", "r2", "alpha", "beta_n", "cells"):
            assert key in row

        assert row["name"] == "book1"
        assert row["gross"] == pytest.approx(1.0)
        assert row["net"] == pytest.approx(0.2)
        assert set(row["cells"].keys()) == {(m, a) for a in (0.05, 0.01)
                                             for m in ("parametric", "historic")}

    def test_multiple_portfolios_independent_rows(self, patch_price_loader):
        panel = _make_price_panel(["AAPL", "MSFT", "SPY"], n=400, seed=2)
        patch_price_loader(panel)

        portfolios = {
            "long_only": {"AAPL": 1.0},
            "long_short": {"AAPL": 0.5, "MSFT": -0.5},
        }
        rows = multi_backtest(portfolios, window=100, n_boot=50, engine=object())

        names = {r["name"] for r in rows}
        assert names == set(portfolios)

    def test_common_window_intersects_native_periods(self, patch_price_loader):
        """한 종목이 늦게 상장 -> common_window=True면 두 포트폴리오의 obs가 같아야 함."""
        panel = _make_price_panel(["AAPL", "MSFT", "SPY"], n=400, seed=3)
        late_ticker_panel = panel.copy()
        late_ticker_panel.loc[late_ticker_panel.index[:100], "MSFT"] = np.nan
        patch_price_loader(late_ticker_panel)

        portfolios = {
            "early": {"AAPL": 1.0},
            "late": {"MSFT": 1.0},
        }
        rows = multi_backtest(portfolios, window=100, n_boot=50,
                               common_window=True, engine=object())

        obs = {r["name"]: r["obs"] for r in rows}
        assert obs["early"] == obs["late"]

    def test_native_window_default_keeps_full_history_per_book(self, patch_price_loader):
        """common_window=False(기본)이면 늦게 시작한 종목이 다른 book까지 truncate하면 안 됨."""
        panel = _make_price_panel(["AAPL", "MSFT", "SPY"], n=400, seed=4)
        late_ticker_panel = panel.copy()
        late_ticker_panel.loc[late_ticker_panel.index[:100], "MSFT"] = np.nan
        patch_price_loader(late_ticker_panel)

        portfolios = {"early": {"AAPL": 1.0}, "late": {"MSFT": 1.0}}
        rows = multi_backtest(portfolios, window=100, n_boot=50,
                               common_window=False, engine=object())

        obs = {r["name"]: r["obs"] for r in rows}
        assert obs["early"] > obs["late"]

    def test_beta_nan_when_market_not_in_panel(self, patch_price_loader):
        """market 컬럼이 패널에 없으면 beta/alpha/r2 = nan, beta_n = 0으로 안전하게 폴백."""
        panel = _make_price_panel(["AAPL", "MSFT"], n=400, seed=5)  # SPY 없음
        patch_price_loader(panel)

        portfolios = {"book1": {"AAPL": 1.0}}
        rows = multi_backtest(portfolios, window=100, n_boot=50, engine=object())

        row = rows[0]
        assert np.isnan(row["beta"])
        assert np.isnan(row["r2"])
        assert np.isnan(row["alpha"])
        assert row["beta_n"] == 0
