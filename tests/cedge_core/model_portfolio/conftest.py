import numpy as np
import pandas as pd
import pytest
from cedge_core.model_portfolio.models import MODELS

TICKERS = ["AGG", "EFA", "IEMG", "IJR", "IVV", "TIP"]
LAST_CLOSE = {"AGG": 94.71, "EFA": 103.47, "IEMG": 80.22, "IJR": 137.60,
              "IVV": 781.54, "TIP": 104.56}
ASOF = "2026-10-13"


@pytest.fixture(scope="session")
def close_panel() -> pd.DataFrame:
    """300 business days of seeded random-walk closes ending at the real 2026-10-13 closes."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range(end=ASOF, periods=300)
    vol = {"AGG": 0.003, "TIP": 0.003, "EFA": 0.010, "IEMG": 0.012, "IJR": 0.013, "IVV": 0.010}
    cols = {}
    for t in TICKERS:
        daily = rng.normal(0.0003, vol[t], len(dates))      # log returns
        level = np.exp(np.cumsum(daily))
        cols[t] = np.round(LAST_CLOSE[t] * level / level[-1], 2)  # last row = real close
    return pd.DataFrame(cols, index=dates)


@pytest.fixture(scope="session")
def prices(close_panel) -> dict:
    return close_panel.iloc[-1].to_dict()


@pytest.fixture(scope="session")
def models():
    return MODELS