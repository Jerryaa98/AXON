"""Environment-variable-backed configuration for the unlock wrapper layer.

Every knob is opt-in: unset or neutral values produce behavior byte-identical
to the unwrapped path. Reading is done once at activation and the resolved
values are stored on a single module-level Config object that the rest of
the wrapper imports.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return default if value is None else str(value).strip()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name, "")
    if raw == "":
        return default
    return raw.lower() in {"1", "true", "t", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = _env(name, "")
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    raw = _env(name, "")
    return default if raw == "" else raw


@dataclass
class Config:
    # Top-level dispatch mode.
    # "baseline"           : no wrapping, behave exactly like the unwrapped code
    # "wrapped"            : scheduler wrapper active, original generate()
    # "wrapped_with_cache" : scheduler wrapper active + KV-cached generate()
    impl: str = "baseline"

    # Step 2 - gain-driven adaptive stop.
    # gain_stop_ratio >= 1.0 -> neutral (falls back to original hard-cap behavior).
    gain_stop_ratio: float = 1.0

    # Step 3 - step-dependent lambda(t), gamma(t) schedules.
    # "constant" is neutral.
    lambda_schedule: str = "constant"   # {constant, linear_decay, linear_growth, cosine}
    gamma_schedule: str = "constant"

    # Step 4 - warm-start greedy coverage across denoising steps.
    greedy_warm_start: bool = False

    # Step 5 - saliency-weighted target uncertainty (entropy of logits).
    saliency_weighting: bool = False

    # Step 6 - sparsify the attention kernel before the conflict matrix.
    # threshold 0.0 is neutral (no zeroing).
    sparse_conflict_threshold: float = 0.0

    # Step 7 - dynamic per-step budget. The scheduler multiplies the
    # incoming `multiplier_after_warmup` by a schedule that interpolates
    # between `budget_floor` and `budget_ceil` across each block.
    # Neutral: budget_schedule="constant" with floor=ceil=1.0 (identity).
    budget_schedule: str = "constant"   # {constant, ramp_up, ramp_down, cosine_up, cosine_down}
    budget_floor: float = 1.0
    budget_ceil: float = 1.0

    # Progress signal for the budget schedule.
    #   "step"   : t = step_in_block / (steps_in_block - 1)  [original behavior]
    #   "masked" : t = 1 - num_masked / num_masked_at_block_start
    # Use "masked" when blocks exit early — the step-based fraction never
    # approaches 1.0 in that regime, so the ramp never fully triggers.
    budget_progress: str = "step"

    # Step 8 - one-shot top-k + conflict-filter greedy. Replaces the
    # Python while loop with a single argsort + linear conflict walk.
    batch_greedy: bool = False

    # Step 9 - submodular gain transform.
    #   "linear": gain = sum_i [max(cover[i], influence[i,j]) - cover[i]]
    #   "log":    gain = sum_i [log(1+max(cover[i], influence[i,j])) - log(1+cover[i])]
    # Log saturates faster, rewarding breadth over mass concentration.
    gain_fn: str = "linear"             # {linear, log}

    # Step 10 - confidence-gated free-commit pass (DAWN-style).
    # Before the submod/greedy selection, unconditionally unmask every masked
    # token whose confidence >= free_commit_tau. These are "free-lunch" unmasks:
    # they don't consume the budget cap and allow later steps to be skipped.
    # Neutral: free_commit_tau >= 1.0 (impossible to reach, so pass is a no-op).
    free_commit_tau: float = 1.0        # {0.0 .. 1.0}; 1.0 = disabled

    @property
    def use_kv_cache(self) -> bool:
        # KV-cache composition is driven by the dispatch mode.
        return self.impl == "wrapped_with_cache"

    def is_neutral(self) -> bool:
        """True iff every wrapper-introduced knob is at a neutral setting.

        When True, the wrapped code path must produce outputs identical to
        the unwrapped path (Gate 1 of the correctness ladder). Note: the
        cached mode is intentionally NOT neutral here — enabling it trades
        prefix-cache approximation for wall-clock."""
        budget_neutral = (
            self.budget_schedule == "constant"
            and float(self.budget_floor) == 1.0
            and float(self.budget_ceil) == 1.0
        )
        return (
            self.gain_stop_ratio >= 1.0
            and self.lambda_schedule == "constant"
            and self.gamma_schedule == "constant"
            and not self.greedy_warm_start
            and not self.saliency_weighting
            and float(self.sparse_conflict_threshold) <= 0.0
            and budget_neutral
            and not self.batch_greedy
            and str(self.gain_fn).strip().lower() == "linear"
        )


_CONFIG: Optional[Config] = None


def load_from_env() -> Config:
    """Resolve the singleton Config from environment variables. Cached."""
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG
    _CONFIG = Config(
        impl=_env_str("UNLOCK_IMPL", "baseline"),
        gain_stop_ratio=_env_float("UNLOCK_GAIN_STOP_RATIO", 1.0),
        lambda_schedule=_env_str("UNLOCK_LAMBDA_SCHEDULE", "constant"),
        gamma_schedule=_env_str("UNLOCK_GAMMA_SCHEDULE", "constant"),
        greedy_warm_start=_env_bool("UNLOCK_GREEDY_WARM_START", False),
        saliency_weighting=_env_bool("UNLOCK_SALIENCY_WEIGHTING", False),
        sparse_conflict_threshold=_env_float("UNLOCK_SPARSE_CONFLICT_THRESHOLD", 0.0),
        budget_schedule=_env_str("UNLOCK_BUDGET_SCHEDULE", "constant"),
        budget_floor=_env_float("UNLOCK_BUDGET_FLOOR", 1.0),
        budget_ceil=_env_float("UNLOCK_BUDGET_CEIL", 1.0),
        budget_progress=_env_str("UNLOCK_BUDGET_PROGRESS", "step"),
        batch_greedy=_env_bool("UNLOCK_BATCH_GREEDY", False),
        gain_fn=_env_str("UNLOCK_GAIN_FN", "linear"),
    )
    return _CONFIG


def get() -> Config:
    """Return the active Config, loading from env on first access."""
    return load_from_env() if _CONFIG is None else _CONFIG


def reset() -> None:
    """Drop the cached Config so the next get() re-reads env. For tests."""
    global _CONFIG
    _CONFIG = None
