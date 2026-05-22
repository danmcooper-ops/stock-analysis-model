"""Shared valuation infrastructure: input validation + the Valuation result type.

The other model modules use `_validate_numeric` internally before every
division or discount calculation so failure modes are loud, not silent.

`Valuation` is a result envelope: every fair-value path can package its
output as (value, method, confidence, warnings, inputs_used) so that
silent fallbacks do not pose as authoritative numbers downstream.

Note: as of this commit the existing models still return `float | None`
for the legacy call sites in scripts/analyze_stock.py. The Valuation
type is available here for callers that want the richer envelope and as
a target for the gradual migration of return types.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Valuation:
    """Result envelope for a fair-value computation.

    Attributes
    ----------
    value : float or None
        Headline number. None means the model could not compute and the
        reason is in `warnings`.
    method : str
        Which path produced the number ('two_stage_ev_ggm',
        'monte_carlo_dcf', 'epv_zero_growth', ...). Distinguishes
        fallback paths from primary paths.
    confidence : float
        0.0-1.0. Penalize for terminal-value dominance, low-fit beta,
        leverage-inflated ROE, Monte Carlo clip rate, etc. Treat
        anything below 0.5 as needs-review.
    warnings : tuple of str
        Concrete caveats that would otherwise be lost to a downstream
        round.
    inputs_used : dict
        Post-validation values fed to the math, so an output can be
        re-derived or audited without re-fetching upstream data.
    """
    value: Optional[float]
    method: str
    confidence: float = 1.0
    warnings: Tuple[str, ...] = field(default_factory=tuple)
    inputs_used: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.value is not None

    def __float__(self) -> float:
        return float(self.value) if self.value is not None else float('nan')

    @classmethod
    def invalid(cls, method: str, reason: str,
                inputs: Optional[Dict[str, Any]] = None) -> 'Valuation':
        return cls(value=None, method=method, confidence=0.0,
                   warnings=(reason,), inputs_used=inputs or {})


def _validate_numeric(name, value, *, positive=False, allow_zero=True,
                      finite=True, low=None, high=None):
    """Validate a single numeric input.

    Returns the value as float or raises ValueError with a specific reason.
    Use before every division or discount calculation so failure modes are
    loud, not silent.

    Parameters
    ----------
    name : str
        Used in the error message so the caller knows which input failed.
    value : Any
        The input to validate.
    positive : bool
        Require strictly > 0.
    allow_zero : bool
        When False, reject value == 0.
    finite : bool
        Reject NaN and inf.
    low, high : float or None
        Optional bounds. Inclusive — value < low or value > high raises.
    """
    if value is None:
        raise ValueError(f"{name} is None")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} is not numeric: {value!r}")
    if finite and not np.isfinite(v):
        raise ValueError(f"{name} is not finite: {v}")
    if positive and v <= 0:
        raise ValueError(f"{name} must be > 0, got {v}")
    if not allow_zero and v == 0:
        raise ValueError(f"{name} must be non-zero")
    if low is not None and v < low:
        raise ValueError(f"{name} below {low}: {v}")
    if high is not None and v > high:
        raise ValueError(f"{name} above {high}: {v}")
    return v


def _validate_returns(name, arr, min_obs=24):
    """Validate a 1-D return series for beta / correlation work.

    Raises ValueError on wrong shape, too few observations, or non-finite
    values. Returns the array as a clean 1-D float numpy array.
    """
    a = np.asarray(arr, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    if a.size < min_obs:
        raise ValueError(f"{name} has {a.size} observations, need at least {min_obs}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a
