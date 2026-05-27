"""Outside-in wrapper layer for the UNLOCK submodular scheduler.

Activation is env-var-gated via `UNLOCK_IMPL`:
  - unset / "baseline"       : inert, nothing is patched
  - "wrapped"                : scheduler monkey-patched for adaptive/scheduled UNLOCK
  - "wrapped_with_cache"     : scheduler + KV-cached generate()

Nothing in this package edits `gdllm_utils.py`, `generate.py`, or
`eval_llada.py`. All new behavior is opt-in at the launcher level.
"""

from __future__ import annotations

from .config import Config, get as get_config, load_from_env, reset as reset_config
from .dispatch import activate, is_activated

__all__ = [
    "Config",
    "get_config",
    "load_from_env",
    "reset_config",
    "activate",
    "is_activated",
]


# No auto-activation at import time: the patch targets (names in the
# `generate` and `eval_llada` modules) must already be imported when we
# rebind them. Activation happens from the `__main__` entrypoint after
# prerequisites are loaded. Importing this package alone is always safe.
