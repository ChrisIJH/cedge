import pandas as pd
from cedge_core.regime.history import get_regime_history


class FakeScoreRepo:
    def get_regime_history(self, start_date, end_date, score_name):
        return pd.DataFrame({
            "as_of_date": ["2026-01-02"],
            "score": [55.0],
            "posterior_risk_on": [0.5],
            "posterior_soft_patch": [0.4],
            "posterior_risk_off": [0.1],
        })


def test_get_regime_history_passes_through():
    df = get_regime_history("2026-01-01", "2026-01-31", repo=FakeScoreRepo())
    assert df["score"].iloc[0] == 55.0