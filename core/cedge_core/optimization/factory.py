"""
core/cedge_core/optimization/factory.py
Factory for Optimizer.

"""

from __future__ import annotations

from typing import Callable, Dict

from cedge_core.optimization.mean_variance import OptimizeResult, solve_mean_variance

_STRATEGIES: Dict[str, Callable[..., OptimizeResult]] = {
    'mean_variance': solve_mean_variance
}

def solve(strategy: str, **kwargs) -> OptimizeResult:
    """Dispatch to the named strategy

    Args:
        strategy: Name of the strategy ("mean_variance", ..)
        **kwargs: Passed through to the strategy function unchanged.
    
    Returns:
        OptimizeResult from the chosen strategy
    
    Raises:
        ValueError: If strategy name is unknown.

    """
    try:
        fn = _STRATEGIES[strategy]
    except KeyError:
        raise ValueError(
                    f"unknown strategy {strategy!r}; available: {sorted(_STRATEGIES)}"
                ) from None
    return fn(**kwargs)
    

