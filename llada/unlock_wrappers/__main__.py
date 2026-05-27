"""Entrypoint that runs a target script under the unlock wrapper.

Usage (from within the llada/ directory):
    UNLOCK_IMPL=wrapped python -m unlock_wrappers quick_eval_gsm8k.py [args...]

Mechanism:
  1. Import the prerequisite modules (`generate`, `eval_llada`) so that the
     names we want to rebind exist in their modules' globals.
  2. Activate the wrapper — this is the point where monkey-patching happens.
  3. runpy the target script as `__main__` with the remaining argv.

Originals (`generate.py`, `eval_llada.py`, `gdllm_utils.py`) are never edited.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _usage() -> None:
    sys.stderr.write(
        "Usage: python -m unlock_wrappers <target_script.py> [args...]\n"
        "Activation is gated on UNLOCK_IMPL (unset/'baseline' runs unwrapped).\n"
    )


def main() -> int:
    if len(sys.argv) < 2:
        _usage()
        return 2

    # Pre-import so that the names we rebind exist in their modules.
    import generate  # noqa: F401
    import eval_llada  # noqa: F401

    # Activate the wrapper after prerequisites are loaded.
    from . import activate, get_config
    resolved = activate()
    cfg = get_config()
    if resolved != "baseline":
        sys.stderr.write(
            f"[unlock_wrappers] activated impl={resolved} "
            f"use_kv_cache={cfg.use_kv_cache} "
            f"gain_stop_ratio={cfg.gain_stop_ratio} "
            f"lambda_schedule={cfg.lambda_schedule} "
            f"gamma_schedule={cfg.gamma_schedule} "
            f"sparse_conflict_threshold={cfg.sparse_conflict_threshold} "
            f"warm_start={cfg.greedy_warm_start} "
            f"batch_greedy={cfg.batch_greedy} "
            f"gain_fn={cfg.gain_fn} "
            f"budget_schedule={cfg.budget_schedule} "
            f"budget_floor={cfg.budget_floor} "
            f"budget_ceil={cfg.budget_ceil} "
            f"budget_progress={cfg.budget_progress}\n"
        )

    target = sys.argv[1]
    target_path = Path(target)
    if not target_path.exists():
        sys.stderr.write(f"[unlock_wrappers] target not found: {target}\n")
        return 2

    sys.argv = [target] + sys.argv[2:]
    runpy.run_path(str(target_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
