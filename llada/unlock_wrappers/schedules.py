"""Step-dependent schedules for lambda(t) and gamma(t) multipliers.

`step_in_block` and `steps_in_block` are already threaded through the
scheduler signatures (they arrive as function arguments), so a schedule is
just a function of fractional progress t = step_in_block / max(1, steps_in_block - 1).
"""

from __future__ import annotations

import math
from typing import Callable


def _fraction(step_in_block: int, steps_in_block: int) -> float:
    denom = max(1, int(steps_in_block) - 1)
    t = float(step_in_block) / float(denom)
    return max(0.0, min(1.0, t))


def constant(_value: float, _step_in_block: int, _steps_in_block: int) -> float:
    return float(_value)


def linear_decay(base: float, step_in_block: int, steps_in_block: int) -> float:
    return float(base) * (1.0 - _fraction(step_in_block, steps_in_block))


def linear_growth(base: float, step_in_block: int, steps_in_block: int) -> float:
    return float(base) * _fraction(step_in_block, steps_in_block)


def cosine(base: float, step_in_block: int, steps_in_block: int) -> float:
    # Smooth 0 -> base over the block, matching linear_growth at endpoints.
    t = _fraction(step_in_block, steps_in_block)
    return float(base) * 0.5 * (1.0 - math.cos(math.pi * t))


_SCHEDULES: dict[str, Callable[[float, int, int], float]] = {
    "constant": constant,
    "linear_decay": linear_decay,
    "linear_growth": linear_growth,
    "cosine": cosine,
}


def resolve(name: str) -> Callable[[float, int, int], float]:
    """Look up a schedule by name. Unknown names fall back to constant to
    preserve the neutral-config guarantee rather than hard-failing during a run."""
    return _SCHEDULES.get(str(name).strip().lower(), constant)


# --- Interval-interpolating schedules used for the per-step budget factor. ---
#
# These take two endpoints (floor, ceil) and produce a value that moves
# between them as the block progresses. The scheduler multiplies
# `multiplier_after_warmup` by the output, so neutral is floor=ceil=1.0.


def _lerp(a: float, b: float, t: float) -> float:
    return float(a) + (float(b) - float(a)) * float(t)


def budget_constant(floor: float, ceil: float, step: int, steps: int) -> float:
    # Neutral: returns the geometric midpoint-like anchor. For a true neutral
    # pass floor=ceil=1.0.
    return float(floor)


def budget_ramp_up(floor: float, ceil: float, step: int, steps: int) -> float:
    return _lerp(floor, ceil, _fraction(step, steps))


def budget_ramp_down(floor: float, ceil: float, step: int, steps: int) -> float:
    return _lerp(ceil, floor, _fraction(step, steps))


def budget_cosine_up(floor: float, ceil: float, step: int, steps: int) -> float:
    t = _fraction(step, steps)
    eased = 0.5 * (1.0 - math.cos(math.pi * t))
    return _lerp(floor, ceil, eased)


def budget_cosine_down(floor: float, ceil: float, step: int, steps: int) -> float:
    t = _fraction(step, steps)
    eased = 0.5 * (1.0 - math.cos(math.pi * t))
    return _lerp(ceil, floor, eased)


_BUDGET_SCHEDULES: dict[str, Callable[[float, float, int, int], float]] = {
    "constant": budget_constant,
    "ramp_up": budget_ramp_up,
    "ramp_down": budget_ramp_down,
    "cosine_up": budget_cosine_up,
    "cosine_down": budget_cosine_down,
}


def resolve_budget(name: str) -> Callable[[float, float, int, int], float]:
    """Look up a budget schedule by name; unknown names fall back to constant."""
    return _BUDGET_SCHEDULES.get(str(name).strip().lower(), budget_constant)


def budget_factor_by_t(name: str, floor: float, ceil: float, t: float) -> float:
    """Direct t-based budget factor, for progress signals that aren't derived
    from step_in_block (e.g. masked-fraction)."""
    name = str(name).strip().lower()
    t = max(0.0, min(1.0, float(t)))
    if name == "ramp_up":
        return _lerp(floor, ceil, t)
    if name == "ramp_down":
        return _lerp(ceil, floor, t)
    if name == "cosine_up":
        eased = 0.5 * (1.0 - math.cos(math.pi * t))
        return _lerp(floor, ceil, eased)
    if name == "cosine_down":
        eased = 0.5 * (1.0 - math.cos(math.pi * t))
        return _lerp(ceil, floor, eased)
    return float(floor)
