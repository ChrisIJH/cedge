"""
core/cedge_core/model_portfolio/models.py

Model portfolio definition: id, version, target weights (securities only, sum to 1)
"""
import math
from dataclasses import dataclass
from typing import Dict, Mapping

WEIGHT_TOL = 1e-9

@dataclass(frozen=True)
class ModelPortfolio:
    model_id: str 
    version: str
    weights: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError(f"{self.model_id}: empty model")
        bad = {t: w for t, w in self.weights.items() if not math.isfinite(w) or w < 0}
        if bad:
            raise ValueError(f"{self.model_id}: negative or non-finite weights {bad}")

        total = sum(self.weights.values())
        if abs(total - 1.0) >  WEIGHT_TOL:
            raise ValueError(f"{self.model_id}: weights sum to {total}, not 1")


MOD_6040 = ModelPortfolio("MOD_6040", "1.0",
                          {"IVV": 0.42, "IJR": 0.08, "EFA": 0.10, "AGG": 0.30, "TIP": 0.10})
MOD_8020 = ModelPortfolio("MOD_8020", "1.0",
                          {"IVV": 0.56, "IJR": 0.12, "EFA": 0.12, "AGG": 0.15, "TIP": 0.05})

MODELS: Dict[str, ModelPortfolio] = {m.model_id: m for m in (MOD_6040, MOD_8020)}

