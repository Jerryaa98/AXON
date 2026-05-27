"""Monkey-patch activation gated on UNLOCK_IMPL.

The wrapper is inert by default. Only when `activate()` is called AND
`UNLOCK_IMPL` resolves to a wrapping mode do we rebind names in the
existing modules. The originals remain untouched in gdllm_utils; we only
rebind the names imported into the `generate` and `eval_llada` modules.
"""

from __future__ import annotations

import importlib

from . import config as _cfg


_ACTIVATED = False


def activate(impl: str | None = None) -> str:
    """Install monkey-patches for the requested implementation.

    Returns the resolved impl string ("baseline" | "wrapped" | "wrapped_with_cache").
    Safe to call multiple times — only the first call patches.
    """
    global _ACTIVATED
    c = _cfg.get()
    if impl is not None:
        c.impl = str(impl).strip().lower() or c.impl

    if _ACTIVATED or c.impl == "baseline":
        return c.impl

    # Scheduler-name rebinding applies to both wrapped modes.
    generate_mod = importlib.import_module("generate")
    from . import scheduler as _scheduler
    generate_mod.get_transfer_index_unlock = _scheduler.get_transfer_index_unlock_wrapped
    generate_mod.get_transfer_index_unlock_clean = _scheduler.get_transfer_index_unlock_clean_wrapped

    if c.impl == "wrapped_with_cache":
        # Replace the `generate` symbol that eval_llada imported at load time.
        eval_mod = importlib.import_module("eval_llada")
        from . import generate_cached as _gc
        eval_mod.generate = _gc.generate_with_cache_and_unlock

    _ACTIVATED = True
    return c.impl


def is_activated() -> bool:
    return _ACTIVATED
